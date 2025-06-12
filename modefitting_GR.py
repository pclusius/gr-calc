import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.optimize import curve_fit
from operator import itemgetter
import matplotlib.dates as mdates
import json

def timestamp_indexing(timestamps):
    '''
    Creates list of indices corresponding to given timestamp list.
    Needed when there are gaps in time lists.
    '''
    start_time = timestamps[0]
    time_interval = 30 #minutes
    indices = []

    for time in timestamps:
        mins_since_start = abs((time - start_time).total_seconds() / 60) #for each timestamp
        index = int(mins_since_start / time_interval) #divide by interval
        indices.append(index)
    
    return np.array(indices)
def closest(list, number):
    '''
    Finds closest element in a list to a given value.
    Returns the index of that element.
    '''
    value = []
    for i in list:
        value.append(abs(number-i))
    return value.index(min(value))
def cal_mape(x,y,popt):
    y_predicted = linear(x, *popt)
    absolute_error = np.abs(y - y_predicted)
    mape = np.mean(absolute_error / y) * 100
    return mape
def linear(x,k,b):
    return k*x + b
def points_in_existing_line(unfinished_lines, point, nearby_point=None):
    ''' Calculates in how many line a datapoint is.'''
    if nearby_point is None:
        score = sum([1 for line in unfinished_lines if point in line])
    else:
        score = sum([1 for line in unfinished_lines if point in line])
        score += sum([1 for line in unfinished_lines if nearby_point in line])
    return score
def extract_data(line,exclude_start=0,exclude_end=0):
    return (timestamp_indexing([point[0] for point in line[exclude_start:len(line)-exclude_end]]),  #x values
            [point[1] for point in line[exclude_start:len(line)-exclude_end]])  #y values

def find_growth(df_peaks,a,gret):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with wanted times and diameters for plotting growth rates.
    
    gret = growth rate error threshold for filtering bigger changes in gr when adding new points to lines
    mape = mean average precentage error
    ''' 
    #extract times and diameters from df
    times = df_peaks.index
    diams = df_peaks['peak_diameter']
    
    #combine to the same list and sort data by diameter
    data_sorted = np.array(sorted(zip(times, diams), key=itemgetter(0,1))) #[[time1,diam1],[time2,diam2]...]
    
    #init
    unfinished_lines = []
    finalized_lines = []
    results_dict = {}
    df_mapes = pd.DataFrame()

    #iterate over each datapoint
    for i, datapoint in enumerate(data_sorted):
        time0, diam0 = datapoint
        datapoint = tuple(datapoint)

        #iterate over timestamps after current datapoint and look for the nearest datapoint
        for ii in range(1,3): #allows one missing datapoint in between
            timestamp = time0 + timedelta(minutes=30)*ii

            #search for datapoints in this timestamp that fulfill the base diam requirements
            base_low_diam_limit = diam0-10*ii #nm*timestep
            base_high_diam_limit = diam0+10*ii
            
            ts_points = [point for point in data_sorted if point[0] == timestamp]  
            closest_ts_points = [point for point in ts_points  \
                                 if point[1] >= base_low_diam_limit and point[1] <= base_high_diam_limit]
            if not closest_ts_points: #skip if no nearby datapoints in timestamp
                # print("no nearby datapoints")
                # breakpoint()
                continue
            
            #closest datapoint next in list
            if points_in_existing_line(unfinished_lines,datapoint) == 0:
                nearby_datapoint = tuple(min(closest_ts_points, key=lambda point: abs(point[1] - diam0)))
                time1, diam1 = nearby_datapoint
                # print("nearby datapoint with base diam diff")
            elif points_in_existing_line(unfinished_lines,datapoint) > 1: #datapoint in many lines (convergence)
                # print("convergence")
                # [print(line) for line in unfinished_lines if datapoint in line]
                
                converging_lines = [line for line in unfinished_lines if datapoint in line]
                nearby_datapoint = tuple(min(closest_ts_points, key=lambda point: abs(point[1] - diam0)))
                
                #continue the line that best fits the next datapoint
                #if len(closest_ts_points) == 1: #only if there is one nearby datapoint and at least one of the lines is longer than 2 datapoints
                #minimize MAPE
                mapes = []
                for line in converging_lines:
                    line_with_new_point = line + [nearby_datapoint]
                    x, y = extract_data(line_with_new_point) #x=times,y=diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    mapes.append(mape)

                min_mape_i = mapes.index(min(mapes))
                #nearby_datapoint = tuple(closest_ts_points[0])
                line_before = converging_lines[min_mape_i]
                iii = unfinished_lines.index(line_before)
                # breakpoint()

            else:
                #minimize MAPE when choosing the new point
                # print("nearby datapoint with stricter diam diff")
                iii, line_before = [(i,line) for i,line in enumerate(unfinished_lines) if datapoint in line][0]
                # print("line before:",line_before)

                mapes = []
                for point in closest_ts_points:
                    line_with_new_point = line_before + [point]
                    x, y = extract_data(line_with_new_point) #x=times,y=diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    mapes.append(mape)

                min_mape_i = mapes.index(min(mapes))
                nearby_datapoint = tuple(closest_ts_points[min_mape_i])

                time1, diam1 = nearby_datapoint
                diam_diff = abs(nearby_datapoint[1] - diam0)
                time_diff = (time1-time0).total_seconds() / 60 / 60 #hours

                #more strict diameter range when finding the 3rd/4th point
                if len(line_before) == 2 or len(line_before) == 3:
                    #calculate growth rate
                    x, y = extract_data(line_before) #x=times,y=diams
                    popt, pcov = curve_fit(linear, x, y)
                    GR = popt[0] * 2 #nm/h

                    b = 2 if diam1 < 20 else 0.1 * diam1 #2nm if dp<20nm, else 10% of new peak 
                    if GR >= 0:
                        low_diam_limit = (GR * time_diff)/1.5 - b + diam0 #nm
                        high_diam_limit = 1.5 * GR * time_diff + b + diam0
                    else:
                        low_diam_limit = 1.5 * GR * time_diff - b + diam0
                        high_diam_limit = (GR * time_diff)/1.5 + b + diam0
                        
                    #if nearby datapoint is not in the diameter limit
                    if diam1 <= low_diam_limit or diam1 >= high_diam_limit:
                        unfinished_lines[iii] = [datapoint,nearby_datapoint]
                        # print("strict diam limit not met (shift one point forward)")
                        # breakpoint()
                        break


            ### add new point to a line ###
            if points_in_existing_line(unfinished_lines,datapoint) == 0:
                unfinished_lines.append([datapoint,nearby_datapoint])
                # print("no points in existing line")
                break

            elif points_in_existing_line(unfinished_lines,datapoint) > 0:  
                unfinished_lines[iii] = line_before + [nearby_datapoint]
                
                # print("points in existing line")
                # [print(line) for line in unfinished_lines if datapoint in line]

            #make sure datapoints in every line are sorted by time
            unfinished_lines = [list(set(line)) for line in unfinished_lines]
            unfinished_lines = [sorted(sublist, key=lambda x: x[0]) for sublist in unfinished_lines] 
            
            
            ### make a linear fit to check mape for line with new datapoint ###
            iii, line_after = [(i,line) for i,line in enumerate(unfinished_lines) if datapoint in line and nearby_datapoint in line][0]
            
            #define variables for linear fit
            x, y = extract_data(line_after) #x=times,y=diams
            x_last_excluded, y_last_excluded = extract_data(line_after,exclude_end=1)
            x_first_4, y_first_4 = extract_data(line_after,exclude_end=max(len(line_after)-4, 0))
            x_last_4, y_last_4 = extract_data(line_after,exclude_start=max(len(line_after)-4, 0))

            ### check mae and gr error thresholds ###    
            mape_threshold = a*len(line_after)**(-1) #a*x^(-1)
            min_line_length = 4
        

            if len(line_after) <= min_line_length:
                popt, pcov = curve_fit(linear, x, y)
                mape = cal_mape(x,y,popt)

                if mape > mape_threshold:
                    # print("mape surpassed (len <= 4), remove first point",line_after)

                    unfinished_lines[iii] = line_after[1:] #remove first point
                    # breakpoint()
                    break
 
                # print("mape ok! (len <= 4)",line_after)
                # breakpoint()
                break
            else:
                # #calculate mape of last 4 points
                # x_last4 = timestamp_indexing([datapoint[0] for datapoint in line_after[-4:]])
                # y_last4 = [datapoint[1] for datapoint in line_after[-4:]]
                # popt_last4, pcov = curve_fit(linear, x_last4, y_last4)
                # mape_last4 = cal_mape(x_last4,y_last4,popt_last4)

                #calculate growth rates of first 4 and last 4 points
                popt_first_4, pcov = curve_fit(linear, x_first_4, y_first_4)
                GR_first_4 = popt_first_4[0] * 2
                popt_last_4, pcov = curve_fit(linear, x_last_4, y_last_4)
                GR_last_4 = popt_last_4[0] * 2
                
                #calculate mape of the whole line
                popt, pcov = curve_fit(linear, x, y)
                mape = cal_mape(x,y,popt)

                #if growth rate is under 1nm/h error is +-0.5nm/h, otherwise gret
                gr_error_threshold = 0.5 if GR <= 1 else gret 
                gr_abs_precentage_error = abs(GR_first_4-GR_last_4) / abs(GR_last_4) * 100

                if mape > mape_threshold:
                    
                    # print("mape surpassed, remove last point",line_after)
                    
                    unfinished_lines = [line for line in unfinished_lines if line != line_after]
                    finalized_lines.append(line_after[:-1])
                    unfinished_lines.append([datapoint,nearby_datapoint]) #new line starts with end of previous one
                    # #calculate mae without the first and then last datapoint 
                    # popt, pcov = curve_fit(linear, x_first_excluded, y_first_excluded)
                    # mape_no_first = cal_mape(x_first_excluded,y_first_excluded,popt)

                    # popt, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)
                    # mape_no_last = cal_mape(x_last_excluded,y_last_excluded,popt)

                    # #remove last or first point based on mae comparison
                    # if mape_no_last <= mape_no_first:
                    #     unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                    #     finalized_lines.append(line_after[:-1])
                    #     unfinished_lines.append([datapoint,nearby_datapoint]) #new line starts with end of previous one
                    # else:
                    #     unfinished_lines[iii] = line_after[1:]
                    # breakpoint()
                    break
                
                elif len(line_after) >= 4 and gr_abs_precentage_error > gr_error_threshold:
                    # print("gr error,remove last point",line_after)

                    #remove last point if threshold is exceeded
                    unfinished_lines = [line for line in unfinished_lines if line != line_after]
                    finalized_lines.append(line_after[:-1])
                    unfinished_lines.append([datapoint,nearby_datapoint])
                    # breakpoint()
                    break
  
                # breakpoint()
                break
    
    #add rest of the lines to finalized lines and by timestamp
    unfinished_lines = [line for line in unfinished_lines if len(line) >= min_line_length]
    finalized_lines.extend(unfinished_lines)
    finalized_lines = [sorted(line, key=lambda x: x[0]) for line in finalized_lines] 
    
    '''
    #try splitting line into two parts from the middle to lower mape
    for i, finalized_line in enumerate(finalized_lines):
        if len(finalized_line) >= 7: #at least 7 datapoints needed  
            middle_index = len(finalized_line)//2
            line_1st_half = finalized_line[:middle_index+1] #overlap +1
            line_2nd_half = finalized_line[middle_index:]
            
            #calculate if mape lowered in both halves
            #whole line
            x = timestamp_indexing([datapoint[0] for datapoint in finalized_line])
            y = [datapoint[1] for datapoint in finalized_line] #diams
            popt, pcov = curve_fit(linear, x, y)
            mape = cal_mape(x,y,popt)
            
            #1st half
            x = timestamp_indexing([datapoint[0] for datapoint in line_1st_half])
            y = [datapoint[1] for datapoint in line_1st_half] #diams
            popt, pcov = curve_fit(linear, x, y)
            mape1 = cal_mape(x,y,popt)
            
            #2nd half
            x = timestamp_indexing([datapoint[0] for datapoint in line_2nd_half])
            y = [datapoint[1] for datapoint in line_2nd_half]
            popt, pcov = curve_fit(linear, x, y)
            mape2 = cal_mape(x,y,popt)
            
            if mape1 < mape and mape2 <= mape:
                #remove the second half of current line and add it as its own line to finalized lines
                finalized_lines[i] = line_1st_half
                finalized_lines.append(line_2nd_half)  
            
            # #relative change in mape
            # rel_diff1 = ((mape1-mape)/mape) * 100
            # rel_diff2 = ((mape2-mape)/mape) * 100
            
            # print("rel_diff1",rel_diff1,"rel_diff2",rel_diff2)
            # print("mape",mape,"mape1",mape1,"mape2",mape2)
            # print()
            
            # #if mape improves (decreases) by 40% split line in two
            # if rel_diff1 <= -40 or rel_diff2 <= -40:
            #     #remove the second half of current line and add it as its own line to finalized lines
            #     finalized_lines[i] = line_1st_half
            #     finalized_lines.append(line_2nd_half)  
    '''
    
    #calculate mapes and growth rates
    for i, finalized_line in enumerate(finalized_lines):
        x = timestamp_indexing([datapoint[0] for datapoint in finalized_line]) #time
        y = [datapoint[1] for datapoint in finalized_line] #diams
        popt, pcov = curve_fit(linear, x, y)
        mape = cal_mape(x,y,popt) #%
        GR = popt[0] * 2 #nm/h
        
        results_dict[f'line{str(i)}'] = {'points': finalized_line, 'growth rate': GR, 'mape': mape}
        
        new_row = pd.DataFrame({"length": [len(x)], "mape": [mape]})
        df_mapes = pd.concat([df_mapes,new_row],ignore_index=True)
        
    df_mapes.to_csv('./df_mapes_modefitting.csv', sep=',', header=True, index=True, na_rep='nan')

    return results_dict, df_mapes
def filter_lines(combined):
    '''
    Filter datapoints of lines that are too short or
    with too big of an error.
    '''
    
    #filter lines shorter than 4
    filtered_lines = [subpoints for subpoints in combined if len(subpoints) >= 1]
    
    #filter lines with too high of a growth rate
    #???
    
    return filtered_lines
def cal_modefit_growth(df,a,gret):
    results_dict, df_mapes = find_growth(df,a=a,gret=gret)
    print('Periods of growth found! (1/2)')
    filtered_lines = filter_lines(results_dict)
    print('Filtering done! (2/2)')    
 
    return results_dict, df_mapes


import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.optimize import curve_fit
from operator import itemgetter
import matplotlib.dates as mdates

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
    score = False
    if nearby_point is None:
        if any(point in line for line in unfinished_lines):
            score = True
    else:
        if any(point in line for line in unfinished_lines) or any(nearby_point in line for line in unfinished_lines):
            score = True
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
    df_mapes = pd.DataFrame()
    
    max_time_diff = 60 #mins = 1h between current and nearby point, i.e. max one point missing
    
    #iterate over each datapoint to find suitable pairs of mode fitting datapoints
    for i, datapoint in enumerate(data_sorted):
        time0, diam0 = datapoint
        datapoint = tuple(datapoint)
        
        #diam difference in channels changes in a logarithmic scale
        #nextnext_diam = df1.columns[closest(df1.columns,diam0)+2] #diameter in the channel one after
        #max_diam_diff = abs(df1.columns[closest(df1.columns,diam0)]-nextnext_diam) #max two diameter channels empty in between

        #iterate over timestamps after current datapoint and look for the nearest datapoint
        for ii in range(1,3): #allows one missing datapoint in between
            timestamp = time0 + timedelta(minutes=30)*ii

            #search for datapoints in this timestamp
            ts_points = [point for point in data_sorted if point[0] == timestamp]  
            if not ts_points: #skip if no datapoints in current timestamp
                continue
            
            #closest datapoint next in list
            nearby_datapoint = tuple(min(ts_points, key=lambda point: abs(point[1] - diam0)))
            time1, diam1 = nearby_datapoint
            diam_diff = abs(nearby_datapoint[1] - diam0)
            time_diff = (time1-time0).total_seconds() / 60 / 60 #hours

            #define maximum diameter difference between points
            low_diam_limit = diam0-10*ii #nm*timestep
            high_diam_limit = diam0+10*ii
            
            closest_ts_points = [point for point in ts_points if point[1] >= low_diam_limit and point[1] <= high_diam_limit]
            if not closest_ts_points: 
                continue

            if points_in_existing_line(unfinished_lines,datapoint):
                #find index of that line
                for line in unfinished_lines:
                    if datapoint in line and len(line) > 2:
                        #calculate growth rate
                        x, y = extract_data(line) #x=times,y=diams
                        popt, pcov = curve_fit(linear, x, y)
                        GR = popt[0] * 2 #nm/h

                        b = 0.1 * diam1 #10% of new peak
                        if GR >= 0:
                            low_diam_limit = (GR * time_diff)/1.5 - b + diam0 #nm
                            high_diam_limit = 1.5 * GR * time_diff + b + diam0
                        else:
                            low_diam_limit = 1.5 * GR * time_diff - b + diam0
                            high_diam_limit = (GR * time_diff)/1.5 + b + diam0
                    
                        #minimize MAPE when choosing the new point
                        mapes = []
                        for point in closest_ts_points:
                            line_with_new_point = line + [point]
                            x, y = extract_data(line_with_new_point) #x=times,y=diams
                            popt, pcov = curve_fit(linear, x, y)
                            mape = cal_mape(x,y,popt)
                            mapes.append(mape)

                        min_mape_i = mapes.index(min(mapes))
                        nearby_datapoint = tuple(closest_ts_points[min_mape_i])
                        #print("GR",GR,"diam1", diam1,"time diff",time_diff,  low_diam_limit,high_diam_limit)
                        break
            
            if i > 200 and i < 280:
                print(time0)
                print(unfinished_lines)
                
            
            
            if diam1 >= low_diam_limit and diam1 <= high_diam_limit: 
                ### add new point to a line ###
                if not points_in_existing_line(unfinished_lines,datapoint,nearby_datapoint):
                    unfinished_lines.append([datapoint,nearby_datapoint])

                elif points_in_existing_line(unfinished_lines,datapoint):
                    #find index of that line
                    for line in unfinished_lines:
                        if datapoint in line:
                            line.append(nearby_datapoint)
                            break

                #make sure datapoints in every line are sorted by time
                unfinished_lines = [list(set(line)) for line in unfinished_lines]
                unfinished_lines = [sorted(sublist, key=lambda x: x[0]) for sublist in unfinished_lines] 
                
                ### make a linear fit to check mape for line with new datapoint ###
                iii, current_line = [(i,line) for i,line in enumerate(unfinished_lines) if nearby_datapoint in line][0]
                
                #define variables for linear fit
                x, y = extract_data(current_line) #x=times,y=diams
                x_last_excluded, y_last_excluded = extract_data(current_line,exclude_end=1)
                x_first_excluded, y_first_excluded = extract_data(current_line,exclude_start=1)
                
                if len(current_line) <= 2:
                    break #proceed to next datapoint
                elif len(current_line) == 3:
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                else:
                    # #calculate mape of last 4 points
                    # x_last4 = timestamp_indexing([datapoint[0] for datapoint in current_line[-4:]])
                    # y_last4 = [datapoint[1] for datapoint in current_line[-4:]]
                    # popt_last4, pcov = curve_fit(linear, x_last4, y_last4)
                    # mape_last4 = cal_mape(x_last4,y_last4,popt_last4)

                    #fit to line excluding last datapoint, but calculate mape with full line
                    popt_last_excluded, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)
                    GR_last_excluded = popt_last_excluded[0] * 2
                    
                    #fit to full line
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    GR = popt[0] * 2

                    gr_abs_precentage_error = abs(GR-GR_last_excluded) / abs(GR_last_excluded) * 100
                    
                    
                ### check mae and gr error thresholds ###    
                mape_threshold = a*len(current_line)**(-1) #a*x^(-1)
                gr_error_threshold = gret

                if mape > mape_threshold:
                    #calculate mae without the first and then last datapoint 
                    popt, pcov = curve_fit(linear, x_first_excluded, y_first_excluded)
                    mape_no_first = cal_mape(x_first_excluded,y_first_excluded,popt)

                    popt, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)
                    mape_no_last = cal_mape(x_last_excluded,y_last_excluded,popt)

                    #remove last or first point based on mae comparison
                    if mape_no_last <= mape_no_first and len(current_line) > 4:
                        unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                        finalized_lines.append(current_line[:-1])
                        unfinished_lines.append([datapoint,nearby_datapoint]) #new line starts with end of previous one
                    else:
                        unfinished_lines[iii] = current_line[1:] #another chance for shorter lines
                
                elif len(current_line) >= 5 and gr_abs_precentage_error > gr_error_threshold:
                    #remove last point if threshold is exceeded
                    unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                    finalized_lines.append(current_line[:-1])
                    unfinished_lines.append([datapoint,nearby_datapoint])
                
                break
            else: #keep looking for next datapoint until end of points if the next one isnt suitable
                continue
    
    #add rest of the lines to finalized lines and by timestamp
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
    
    #calculate mapes to show on plot
    for finalized_line in finalized_lines:
        x = timestamp_indexing([datapoint[0] for datapoint in finalized_line])
        y = [datapoint[1] for datapoint in finalized_line] #diams
        popt, pcov = curve_fit(linear, x, y)
        mape = cal_mape(x,y,popt)
        
        new_row = pd.DataFrame({"length": [len(x)], "mape": [mape]})
        df_mapes = pd.concat([df_mapes,new_row],ignore_index=True)
        
    df_mapes.to_csv('./df_mapes_modefitting.csv', sep=',', header=True, index=True, na_rep='nan')
    
    return finalized_lines
def filter_lines(combined):
    '''
    Filter datapoints of lines that are too short or
    with too big of an error.
    '''
    
    #filter lines shorter than 4
    filtered_lines = [subpoints for subpoints in combined if len(subpoints) >= 3]
    
    #filter lines with too high of a growth rate
    #???
    
    return filtered_lines
def cal_modefit_growth(df,a,gret):
    growth_lines = find_growth(df,a=a,gret=gret)
    print('Periods of growth found! (1/2)')
    filtered_lines = filter_lines(growth_lines)
    print('Filtering done! (2/2)')    
 
    return filtered_lines


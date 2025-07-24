import numpy as np
import statsmodels.api as sm
from datetime import timedelta
from scipy.optimize import curve_fit
from operator import itemgetter
from matplotlib.dates import date2num

################# USEFUL FUNCTIONS ##################
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
    '''Calculates mean absolute percentage error (%).'''
    y_predicted = linear(x, *popt)
    absolute_error = np.abs(y - y_predicted)
    mape = np.mean(absolute_error / y) * 100
    return mape
def linear(x,k,b):
    return k*x + b
def robust_fit(x,y):
        '''Robust linear regression using statsmodels with HuberT weighting.'''
        x_linear = np.linspace(np.min(x), np.max(x),num=len(y))
        rlm_HuberT = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT())
        rlm_results = rlm_HuberT.fit()
        
        #predict data of estimated models
        y_rlm = rlm_results.predict(sm.add_constant(x_linear))
        y_params = rlm_results.params
        
        #for printing fitting results
        # print("Statsmodel robus linear model results: \n",rlm_results.summary())
        # print("\nparameters: ",rlm_results.params)
        # print(help(sm.RLM.fit))

        return x_linear, y_rlm, y_params
def points_in_existing_line(unfinished_lines, point, nearby_point=None):
    ''' Calculates in how many line a datapoint is.'''
    if nearby_point is None:
        score = sum([1 for line in unfinished_lines if point in line])
    else:
        score = sum([1 for line in unfinished_lines if point in line])
        score += sum([1 for line in unfinished_lines if nearby_point in line])
    return score
def extract_data(line,exclude_start=0,exclude_end=0):
    '''Extracts wanted data points from line.'''
    return (date2num([point[0] for point in line[exclude_start:len(line)-exclude_end]]),  #x values
            [point[1] for point in line[exclude_start:len(line)-exclude_end]])  #y values

#####################################################
def find_growth(df_peaks,a,gret):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with wanted times and diameters for plotting growth rates.
    
    a = factor for MAPE threshold function a*x⁻¹
    gret = growth rate error threshold for filtering bigger changes in gr when adding new points to lines
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
                continue
            
            #closest datapoint next in list
            if points_in_existing_line(unfinished_lines,datapoint) == 0: #datapoint not in any line
                nearby_datapoint = tuple(min(closest_ts_points, key=lambda point: abs(point[1] - diam0)))
                time1, diam1 = nearby_datapoint
            elif points_in_existing_line(unfinished_lines,datapoint) > 1: #datapoint in many lines (convergence)
                converging_lines = [line for line in unfinished_lines if datapoint in line]
                nearby_datapoint = tuple(min(closest_ts_points, key=lambda point: abs(point[1] - diam0)))
                
                #continue the line that best fits the next datapoint by minimizing MAPE
                mapes = []
                for line in converging_lines:
                    line_with_new_point = line + [nearby_datapoint]
                    x, y = extract_data(line_with_new_point) #x=times,y=diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    mapes.append(mape)

                min_mape_i = mapes.index(min(mapes))
                line_before = converging_lines[min_mape_i]
                iii = unfinished_lines.index(line_before)

            else:
                #minimize MAPE when choosing the new point
                iii, line_before = [(i,line) for i,line in enumerate(unfinished_lines) if datapoint in line][0]

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
                    GR = popt[0]/24 #nm/h

                    b = 2 if diam1 < 20 else 0.1 * diam1 #2nm if dp<20nm, else 10% of new peak 
                    if GR >= 0:
                        low_diam_limit = (GR * time_diff)/1.5 - b + diam0 #nm
                        high_diam_limit = 1.5 * GR * time_diff + b + diam0
                    else:
                        low_diam_limit = 1.5 * GR * time_diff - b + diam0
                        high_diam_limit = (GR * time_diff)/1.5 + b + diam0
                        
                    #if nearby datapoint is not in the diameter limit
                    if diam1 <= low_diam_limit or diam1 >= high_diam_limit:
                        unfinished_lines[iii] = line_before[1:] + [nearby_datapoint]
                        break


            ### add new point to a line ###
            if points_in_existing_line(unfinished_lines,datapoint) == 0: #not in any line
                unfinished_lines.append([datapoint,nearby_datapoint])
                break

            elif points_in_existing_line(unfinished_lines,datapoint) > 0: #in line(s)
                unfinished_lines[iii] = line_before + [nearby_datapoint]

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
                    unfinished_lines[iii] = line_after[1:] #remove first point
                    break
 
                break
            else:
                #calculate growth rates of first 4 and last 4 points
                popt_first_4, pcov = curve_fit(linear, x_first_4, y_first_4)
                GR_first_4 = popt_first_4[0] * 2
                popt_last_4, pcov = curve_fit(linear, x_last_4, y_last_4)
                GR_last_4 = popt_last_4[0] * 2
                
                #calculate mape of the whole line
                popt, pcov = curve_fit(linear, x, y)
                mape = cal_mape(x,y,popt)

                #if growth rate is under 1nm/h error is +-0.5nm/h, otherwise gret
                if abs(GR) <= 1:
                    gr_error_threshold = 0.5
                    gr_error = abs(GR_first_4-GR_last_4) #nm
                else:
                    gr_error_threshold = gret
                    gr_error = abs(GR_first_4-GR_last_4) / abs(GR_last_4) * 100 #%


                if mape > mape_threshold:
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
                    break
                
                elif len(line_after) >= 4 and gr_error > gr_error_threshold:
                    #remove last point if threshold is exceeded
                    unfinished_lines = [line for line in unfinished_lines if line != line_after]
                    finalized_lines.append(line_after[:-1])
                    unfinished_lines.append([datapoint,nearby_datapoint])
                    break

                break
    
    #add rest of the lines to finalized lines and by timestamp
    unfinished_lines = [line for line in unfinished_lines if len(line) >= min_line_length]
    finalized_lines.extend(unfinished_lines)
    finalized_lines = [sorted(line, key=lambda x: x[0]) for line in finalized_lines] 
    
    #robust fit, calculate mapes and growth rates
    for i, finalized_line in enumerate(finalized_lines):
        x = date2num([datapoint[0] for datapoint in finalized_line]) #time days
        y = [datapoint[1] for datapoint in finalized_line] #diams nm
        x_fit, y_fit, params = robust_fit(x,y)
        mape = cal_mape(x,y,[params[1],params[0]]) #%
        GR = params[1]/24 #nm/h
        
        fitted_points = [(time,diam) for time,diam in zip(x_fit,y_fit)]
        finalized_line = [(time,diam) for time,diam in zip(x,y)] #to change time from timestamps to days
        results_dict[f'line{str(i)}'] = {'points': finalized_line, 'fitted points': fitted_points, 'growth rate': GR, 'mape': mape}

    return results_dict



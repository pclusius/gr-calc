import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from operator import itemgetter
from datetime import timedelta
from scipy.optimize import curve_fit
from matplotlib.dates import num2date, date2num, DateFormatter


#################### FUNCTIONS #####################
#useful functions
def closest(list, number):
    '''
    Finds closest element in a list to a given value.
    Returns the index of that element.
    '''
    value = []
    for i in list:
        try:
            value.append(abs(number-i))
        except TypeError:
            value.append(tuple(abs(np.subtract(i,number))))
        
    return value.index(min(value))
def cal_mae(x,y,popt):
    '''Calculates mean absolute error (hours).'''
    x = np.array(x)
    y = np.array(y)
    y_predicted = linear(x, *popt)
    mean_absolute_error = np.mean(np.abs(y - y_predicted)) * 24
    return mean_absolute_error
def cal_derivative(dataframe):
    '''
    Calculates 1st derivatives between neighbouring datapoints. 
    Returns dataframe with derivatives, unit: cm⁻³/h
    '''
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns) 
    
    #convert time to days 
    time_days = date2num(dataframe.index)
    
    for i in dataframe.columns: 
        N = dataframe[i] #concentration
        time = time_days * 24 #change days to hours
        dNdt = np.diff(N)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = dNdt #add calculated derivatives to dataframe
    return df_derivatives
def average_filter(dataframe,window):         
    '''Smoothens data in dataframe with average filter and given window.'''
    smoothed_df = dataframe.copy()
    for i in dataframe.columns:
       smoothed_df[i] = smoothed_df[i].rolling(window=window, center=True).mean()
    smoothed_df.dropna()
    return smoothed_df 

#mathematical functions for fitting
def gaussian(x,a,x0,sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def logistic(x,L,x0,k): 
    return L / (1 + np.exp(-k*(x-x0))) 
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
        
        #print("Statsmodel robus linear model results: \n",rlm_results.summary())
        #print("\nparameters: ",rlm_results.params)
        #print(help(sm.RLM.fit))

        return x_linear, y_rlm, y_params

#################### METHODS #######################
def find_peak_areas(df_filtered,df_deriv,mpd,derivative_threshold):
    '''
    Finds peak areas with derivative threshold.
    Returns dataframe with found peak areas.
    
    mtd = maximum time difference between peaks to be considered the same horizontal peak area
    '''
    #initialize variables
    df_peak_areas = pd.DataFrame()
    start_times_list = []
    maxima_list = []

    #define a boolean dataframe where the derivative threshold has been surpassed
    df_surpassed = df_deriv > derivative_threshold

    #find start points and shift them one timestamp earlier
    df_start_points = df_surpassed & (~df_surpassed.shift(1,fill_value=False)) 
    df_start_points = df_start_points.shift(-1, fill_value=False)

    #find local concentration maxima
    df_left = df_filtered.shift(-1)
    df_right = df_filtered.shift(1)
    df_maxima = (df_left < df_filtered) & (df_filtered > df_right)

    max_peak_diff = timedelta(hours=mpd) #max time difference between peaks to be considered the same peak

    #iterate over diameter channels
    for diam in df_deriv.columns:
        
        #define start times and end time
        df_start_points_only = df_start_points[diam][df_start_points[diam].values]
        start_times = df_start_points_only.index
        end_time = None
        
        #save for use in channel plotting
        start_concs = [df_filtered.loc[start_time,diam] for start_time in start_times]
        start_times_list.append((diam,start_times,start_concs)) 
        
        #iterate over start times
        for start_time in start_times:
            try:
                if start_time < end_time:
                    continue
            except TypeError:
                pass
            
            #find end time after local maximum concentration 
            try:
                subset_end = start_time + timedelta(hours=15) #15 hours ahead
            except IndexError:
                subset_end = df_filtered.index[-1]  
                
            df_subset = df_filtered.loc[start_time:subset_end,diam]
            df_subset_maxima = df_maxima.loc[start_time:subset_end,diam].copy()

            #check if channel has any peaks
            if not df_subset_maxima.any():
                continue
            
            #make df with maxima only
            df_subset_only_maxima = df_subset_maxima[df_subset_maxima.values].copy()
            df_all_maxima = df_subset_only_maxima.copy()

            #check if peaks are nearby and choose higher one
            for i in range(len(df_subset_only_maxima)):
                try:
                    max_time1 = df_subset_only_maxima.index[i]
                    max_time2 = df_subset_only_maxima.index[i+1]
                    max_conc1 = df_filtered.loc[max_time1,diam]
                    max_conc2 = df_filtered.loc[max_time2,diam]
                    
                    diff_mins = max_time2-max_time1
                    
                    if diff_mins <= max_peak_diff and max_conc1 > max_conc2:
                        df_maxima.loc[max_time2,diam] = False
                        df_subset_maxima.loc[max_time2] = False
                    elif diff_mins <= max_peak_diff and max_conc1 < max_conc2:
                        df_maxima.loc[max_time1,diam] = False
                        df_subset_maxima.loc[max_time1] = False
                except IndexError:
                    break
            
            #update subset_only_maxima
            df_subset_only_maxima = df_subset_maxima[df_subset_maxima.values].copy()

            #save for use in channel plotting
            maxima_times = df_subset_only_maxima.index
            maximum_concs = [df_filtered.loc[max_time,diam] for max_time in maxima_times]  
            maxima_list.append((diam,maxima_times,maximum_concs)) 

            #choose closest maximum after start time
            for i in range(len(df_subset_maxima.values)):
                if df_subset_maxima.index[i] > start_time and df_subset_maxima.values[i] == True:
                    closest_maximum = df_subset.values[i]
                    break
            else: #if no maxima after start time skip this start time
                break
            
            #define concentration threshold for ending point
            min_conc = np.nanmin(df_filtered[diam].values) #global minimum
            end_conc = (closest_maximum + min_conc) * 0.5 #(N_max + N_min)/2 

            #estimate time for maximum concentration
            max_conc_time = df_subset.index[df_subset == closest_maximum].tolist()[0] #rough estimate of maximum concentration time 
            max_conc_time_i = df_filtered.index.get_loc(max_conc_time) #index of max_conc_time
            
            #choose next start time after closest local maximum to the maximum
            if any(df_all_maxima.index < max_conc_time):
                closest_local_max = df_all_maxima.index[df_all_maxima.index < max_conc_time].max()  
                start_time = start_times[start_times > closest_local_max].min()

            #define subset from maximum concentration time to end
            df_subset_maxcon = df_filtered.loc[max_conc_time:subset_end,diam] 
            
            #iterate over concentrations after maximum to find ending time
            for i, (time,conc) in enumerate(df_subset_maxcon.items()):
                #check for another maximum along the way
                if i != 0 and df_subset_maxima.loc[time]:
                    timestep_before_next_peak = df_subset_maxcon.index[i-1]
                    end_conc = min(df_subset_maxcon.loc[max_conc_time:timestep_before_next_peak]) #end point after peak
                    end_time = df_subset_maxcon.index[df_subset_maxcon.values == end_conc][0] #end time found!
                    break
 
                #check if concentration drops under the threshold
                if conc < end_conc:
                    end_time = time
                    break
            else:
                #in case we reach the end of the subset without falling under the end conc value
                end_time = df_subset_maxcon.index[-1]
            
            # #attempt to filter peaks that are not at least 1000 cm⁻³ higher than their min value
            # min_conc = min(dataframe.loc[start_time:max_conc_time,diam])
            # if closest_maximum - min_conc < mcd:
            #     continue

            #save peak areas longer than 3 datapoints
            subset = df_filtered[diam].loc[start_time:end_time]
            if len(subset.values) > 3:
                #df_peak_areas.loc[subset.index,diam] = subset #fill dataframe
                new_row = pd.DataFrame({"start_time": [start_time], "end_time": [end_time], "diameter": [diam]})
                df_peak_areas = pd.concat([df_peak_areas,new_row],ignore_index=True)
                    
    df_peak_areas.to_csv('./find_peak_areas.csv', sep=',', header=True, index=True, na_rep='nan')
    return df_peak_areas, derivative_threshold, start_times_list, maxima_list
def maximum_concentration(df,df_peak_areas): 
    '''Calculates maximum concentration in peak areas with gaussian fit.'''
    
    #create lists for results
    df_mc = pd.DataFrame()
    fitting_params = []
    area_edges = []

    #extract values from dataframe
    area_values = df_peak_areas.values
    start_times = area_values[:,0]
    end_times = area_values[:,1]
    area_diams = area_values[:,2]
    
    #gaussian fit to every horizontal area of growth
    for i in range(len(start_times)):
        start_time = start_times[i]
        end_time = end_times[i]
        diam = area_diams[i]
        
        #find values from the dataframe
        subset = df.loc[start_time:end_time,diam]
        x = date2num(subset.index) #time in days
        y = subset.values #concentration
    
        #rescaling for more stable fitting
        x_min, y_min = min(x), min(y)
        x = x - x_min
        y = y - y_min
        
        #initial guess for parameters
        mu=np.mean(x)
        sigma = np.std(x)
        a = max(y)

        try: #gaussian fit
            popt,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma],bounds=((0,0,-np.inf),(max(y),np.inf,np.inf)))
            
            #checking that the peak is within time range and not at the edges
            if ((popt[1]<=x.min()) | (popt[1]>=x.max()) | (popt[1]<x[1]) | (popt[1]>x[-2])): 
                pass
            else:
                #save results to df
                new_row = pd.DataFrame({"timestamp": [num2date(popt[1]+x_min).replace(tzinfo=None)], \
                                        "peak_diameter": [diam], "max_concentration": [popt[0]+y_min]})
                df_mc = pd.concat([df_mc,new_row],ignore_index=True)

                #save gaussian fit parameters and peak area edges for channel plotting 
                fitting_params.append([diam,y_min,popt[0],popt[1]+x_min,popt[2]]) #[diam,ymin for scale,a,mu,sigma]
                area_edges.append([diam,start_time,end_time]) #[diameter,start time, end time]
                                           
        except:
            pass
            #print("Diverges. Skipping.")

    return df_mc, fitting_params, area_edges
def appearance_time(df,mc_params,mc_area_edges):
    '''Calculates appearance times in peak areas with logistic fit.'''
    
    #create lists for results
    df_at = pd.DataFrame()
    fitting_params = []
    area_edges = []
    
    #logistic fit to every horizontal area of growth
    for mc_edge, mc_param in zip(mc_area_edges,mc_params):
        diam, start_time, end_time = mc_edge
        diam, _, a, mu, __ = mc_param
        
        #find values from the dataframe
        subset = df.loc[start_time:end_time,diam]
        x = date2num(subset.index) #time in days
        y = subset.values #concentration
    
        #rescaling for more stable fitting
        x_min, y_min = min(x), min(y)
        x = x - x_min
        y = y - y_min
        mu = mu - x_min
        
        #limit x and y to values between start time of peak area and maximum concentration time in peak area
        max_conc_time = mu
        max_conc_index = closest(x, max_conc_time)
        x_sliced, y_sliced = x[:max_conc_index+1], y[:max_conc_index+1]

        #logistic fit for more than 2 datapoints
        if len(y_sliced) > 2: 
            #initial guess for parameters
            L = a #maximum value of gaussian fit
            x0 = np.nanmean(x_sliced) #midpoint x value
            k = 1.0 #growth rate

            try: #logistic fit
                popt,pcov = curve_fit(logistic,x_sliced,y_sliced,p0=[L,x0,k],bounds=((L*0.999,0,-np.inf),(L,np.inf,np.inf)))
                if ((popt[1]>=x_sliced.max()) | (popt[1]<=x_sliced.min()) | (popt[1]<x_sliced[1])): #checking that the mid point is within time range   
                    pass
                else:
                    #save results to df
                    new_row = pd.DataFrame({"timestamp": [num2date(popt[1]+x_min).replace(tzinfo=None)], \
                                            "diameter": [diam], "mid_concentration": [popt[0]/2+y_min]})
                    df_at = pd.concat([df_at,new_row],ignore_index=True) #appearance time concentration (~50% maximum concentration), L/2

                    #save logistic fit parameters and peak area edges for channel plotting 
                    fitting_params.append([diam, y_min, popt[0], popt[1]+x_min, popt[2]]) #[diam,y min for scale,L,x0,k]
                    area_edges.append([diam,
                        num2date(x_sliced[0] + x_min).replace(tzinfo=None),
                        num2date(x_sliced[-1] + x_min).replace(tzinfo=None)])
            except RuntimeError:
                continue
                          
    return df_at, fitting_params, area_edges
def disappearance_time(df,df_at,mc_area_edges):
    '''Calculates disappearance times in peak areas
    
    NOT FINALIZED
    - fitting reversed logistic?
    '''
    
    #create lists for results
    df_dt = pd.DataFrame()

    #find matching disappearance times for appearance times
    for at_point in df_at.values:
        x0_time, x0_diam, x0_conc = at_point[0], at_point[1], at_point[2]
        channel_mc_areas = [area for area in mc_area_edges if area[0] == x0_diam]
        mc_area_with_at = [area for area in channel_mc_areas if (area[1]<=x0_time<=area[2])][0]
        area_diam, area_start, area_end = mc_area_with_at

        max_i = np.argmax(df.loc[area_start:area_end,x0_diam].values)
        max_ts = df.loc[area_start:area_end,x0_diam].index[max_i]
        
        try: #next area after the current one with appearance time 
            next_mc_area = [area for area in channel_mc_areas if area[1]>x0_time][0]
            next_area_diam, next_area_start, next_area_end = next_mc_area
            
            next_max_i = np.argmax(df.loc[next_area_start:next_area_end,x0_diam].values)
            next_max_ts = df.loc[next_area_start:next_area_end,x0_diam].index[next_max_i]
        except IndexError:
            next_max_ts = df.index[-1]

        #iterate over timesteps after maximum concentration
        subset = df.loc[max_ts:,x0_diam]
        for ts,conc in zip(subset.index,subset.values):
            #stop if disappearance is not found before next area
            if ts >= next_max_ts:
                break
            
            #disappearance found!
            if conc < x0_conc:
                #save results to df
                new_row = pd.DataFrame({"timestamp": [ts], \
                                        "diameter": [x0_diam], "concentration": [conc]})
                df_dt = pd.concat([df_dt,new_row],ignore_index=True)
                break
                      
    return df_dt
def init_methods(df,mpd,mdc,derivative_threshold):
    '''Initialize all functions.'''

    #crop dataframe by allowed mdc (maximum diameter channel)
    df = df[df.columns[df.columns <= mdc]]

    #smoothen data, calculate derivative and define peak areas
    df_interpolated = df.interpolate(method='time')
    df_filtered = average_filter(df_interpolated,window=3)
    df_deriv = cal_derivative(df_filtered) 
    df_peak_areas, derivative_threshold, start_times_list, maxima_list = find_peak_areas(df_filtered,df_deriv,mpd,derivative_threshold)
    
    #methods
    df_mc, mc_params, mc_area_edges = maximum_concentration(df_interpolated,df_peak_areas)
    df_at, at_params, at_area_edges = appearance_time(df_interpolated,mc_params,mc_area_edges)
    df_dt = disappearance_time(df_interpolated,df_at,mc_area_edges)

    #find points that are poorly defined
    time_edges = [df.index[0],df.index[1],df.index[-2],df.index[-1]]
    i_to_remove_mc, i_to_remove_at = [], []
    
    for times, areas, i_to_remove in [(df_mc['timestamp'],mc_area_edges,i_to_remove_mc),(df_at['timestamp'],at_area_edges,i_to_remove_at)]:
        for t, area in zip(times,areas):
            diam, start, end = area
            if start in time_edges or end in time_edges:
                i_to_remove.append(list(times).index(t))

    #lines close to the edges of the dataset
    incomplete_mc_xyz = (
        [[df_mc['timestamp'][i] for i in i_to_remove_mc],
        [df_mc['peak_diameter'][i] for i in i_to_remove_mc],
        [df_mc['max_concentration'][i] for i in i_to_remove_mc]]
        if i_to_remove_mc else [])

    incomplete_at_xyz = (
        [[df_at['timestamp'][i] for i in i_to_remove_at],
        [df_at['diameter'][i] for i in i_to_remove_at],
        [df_at['mid_concentration'][i] for i in i_to_remove_at]]
        if i_to_remove_at else [])

    return df_mc, df_at, df_dt, incomplete_mc_xyz, incomplete_at_xyz, mc_area_edges, at_area_edges, mc_params, at_params, derivative_threshold, start_times_list, maxima_list

################## GROWTH RATES ####################
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
    return ([point[1] for point in line[exclude_start:len(line)-exclude_end]],  #x values
            [point[0] for point in line[exclude_start:len(line)-exclude_end]])  #y values
    
def find_growth(df,times,diams,mgsc,a,gret):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with times and diameters for plotting growth rates.
    
    gret = growth rate error threshold for filtering bigger changes in gr when adding new points to lines
    mtd = maximum time difference between current point and nearby point to add to line
    mgsc = maximum growth start channel, i.e. highest diameter channel where the start of growth lines are allowed
    mae = mean average error
    '''
    #convert time to days
    times = date2num(times)
    
    #combine to the same list and sort data by diameter
    data_sorted = np.array(sorted(zip(times, diams), key=itemgetter(1,0))) #[[time1,diam1],[time2,diam2]...]

    #init
    unfinished_lines = []
    finalized_lines = []
    results_dict = {}
    mtd = 2.5 #h, initial maximum time difference
    
    #iterate over each datapoint to find suitable pairs of mode fitting datapoints
    for datapoint in data_sorted:
        time0, diam0 = datapoint
        datapoint = tuple(datapoint)
        
        #iterate over diameter channels after current datapoint and look for the nearest datapoint
        for ii in range(1,3): #allows one channel in between
            try:
                diam_channel = df.columns.values[df.columns.values >= diam0][ii] #diam in current bin
            except IndexError:
                break #reached diameter of 1000nm
            
            #allowed time difference depends on growth rate and diam difference between last and new point
            base_low_time_limit = time0-mtd/24 #days
            base_high_time_limit = time0+mtd/24
            
            channel_points = [point for point in data_sorted if point[1] == diam_channel]
            closest_channel_points = [point for point in channel_points  \
                                      if point[0] >= base_low_time_limit and point[0] <= base_high_time_limit]
            if not closest_channel_points: #skip if no nearby datapoints in diam channel
                continue
            
            #closest datapoint next in list
            if points_in_existing_line(unfinished_lines,datapoint) == 0:
                nearby_datapoint = tuple(min(closest_channel_points, key=lambda point: abs(point[0] - time0)))
                time1, diam1 = nearby_datapoint
            elif points_in_existing_line(unfinished_lines,datapoint) > 1: #datapoint in many lines (convergence)
                converging_lines = [line for line in unfinished_lines if datapoint in line]
                nearby_datapoint = tuple(min(closest_channel_points, key=lambda point: abs(point[0] - time0)))
                
                #continue the line that best fits the next datapoint
                #minimize MAE
                maes = []
                for line in converging_lines:
                    line_with_new_point = line + [nearby_datapoint]
                    x, y = extract_data(line_with_new_point) #x=diams,y=times
                    popt, pcov = curve_fit(linear, x, y)
                    mae = cal_mae(x,y,popt)
                    maes.append(mae)

                min_mae_i = maes.index(min(maes))
                line_before = converging_lines[min_mae_i]
                iii = unfinished_lines.index(line_before)
            else:
                #minimize MAE when choosing the new point
                iii, line_before = [(i,line) for i,line in enumerate(unfinished_lines) if datapoint in line][0]

                maes = []
                for point in closest_channel_points:
                    line_with_new_point = line_before + [point]
                    x, y = extract_data(line_with_new_point) #x=diams,y=times
                    popt, pcov = curve_fit(linear, x, y)
                    mae = cal_mae(x,y,popt)
                    maes.append(mae)

                min_mae_i = maes.index(min(maes))
                nearby_datapoint = tuple(closest_channel_points[min_mae_i])
                
                time1, diam1 = nearby_datapoint
                diam_diff = diam_channel-diam0
                
                #more strict diameter range when finding the 3rd/4th point
                if len(line_before) == 2 or len(line_before) == 3:
                    #calculate growth rate
                    x, y = extract_data(line_before) #x=diams,y=times
                    popt, pcov = curve_fit(linear, x, y)
                    GR = 1/(popt[0]) #nm/days

                    b = 0 #1 hour in days
                    if GR >= 0:
                        low_time_limit = 1/(GR * 3) * diam_diff - b + time0 #days
                        high_time_limit = 3/GR * diam_diff + b + time0
                    else:
                        low_time_limit = 3/GR * diam_diff - b + time0 #days
                        high_time_limit = 1/(GR * 3) * diam_diff + b + time0
                        
                    #if nearby datapoint is not in the time limits
                    if time1 <= low_time_limit or time1 >= high_time_limit:
                        unfinished_lines[iii] = line_before[1:] + [nearby_datapoint]
                        break
            
            ### add new point to a line ###
            if points_in_existing_line(unfinished_lines,datapoint) == 0:
                if diam0 > mgsc or diam1 > mgsc:
                    break
                unfinished_lines.append([datapoint,nearby_datapoint])
                break
                
            elif points_in_existing_line(unfinished_lines,datapoint) > 0:
                unfinished_lines[iii] = line_before + [nearby_datapoint]
            
            #make sure datapoints in every line are sorted by diameter
            unfinished_lines = [list(set(line)) for line in unfinished_lines]
            unfinished_lines = [sorted(sublist, key=lambda x: x[1]) for sublist in unfinished_lines] 


            ### make a linear fit to check mae for line with new datapoint ###
            iii, line_after = [(i,line) for i,line in enumerate(unfinished_lines) if datapoint in line and nearby_datapoint in line][0]
            
            #define variables for linear fit
            x, y = extract_data(line_after) #x=diams,y=times
            x_first_4, y_first_4 = extract_data(line_after,exclude_end=max(len(line_after)-4, 0))
            x_last_4, y_last_4 = extract_data(line_after,exclude_start=max(len(line_after)-4, 0))
            
            ### check mae and gr error thresholds ###    
            mae_threshold = a*len(line_after)**(-1) #a*x^(-1)
            min_line_length = 4


            if len(line_after) <= min_line_length:
                popt, pcov = curve_fit(linear, x, y)
                mae = cal_mae(x,y,popt)
                
                if mae > mae_threshold:
                    unfinished_lines[iii] = line_after[1:] #remove first point  
                break
            else:
                #calculate growth rates of first 4 and last 4 points
                popt_first_4, pcov = curve_fit(linear, x_first_4, y_first_4)
                GR_first_4 = 1/(popt_first_4[0]*24)
                popt_last_4, pcov = curve_fit(linear, x_last_4, y_last_4)
                GR_last_4 = 1/(popt_last_4[0]*24)
                
                #fit to full line
                popt, pcov = curve_fit(linear, x, y)
                mae = cal_mae(x,y,popt)
                
                #if growth rate is under 1nm/h error is +-0.5nm/h, otherwise gret
                if abs(GR) <= 1:
                    gr_error_threshold = 0.5
                    gr_error = abs(GR_first_4-GR_last_4) #nm
                else:
                    gr_error_threshold = gret
                    gr_error = abs(GR_first_4-GR_last_4) / abs(GR_last_4) * 100 #%
                

                if mae > mae_threshold:
                    unfinished_lines = [line for line in unfinished_lines if line != line_after]
                    finalized_lines.append(line_after[:-1])
                    
                    if diam0 > mgsc or diam1 > mgsc:
                        break
                    unfinished_lines.append([datapoint,nearby_datapoint]) #new line starts with end of previous one

                elif len(line_after) >= 4 and gr_error > gr_error_threshold:
                    #remove last point if threshold is exceeded
                    unfinished_lines = [line for line in unfinished_lines if line != line_after]
                    finalized_lines.append(line_after[:-1])
 
                    if diam0 > mgsc or diam1 > mgsc:
                        break
                    unfinished_lines.append([datapoint,nearby_datapoint])
                
                break

    
    #add rest of the lines to finalized lines and by diameter
    unfinished_lines = [line for line in unfinished_lines if len(line) >= min_line_length]
    finalized_lines.extend(unfinished_lines)
    finalized_lines = [sorted(line, key=lambda x: x[1]) for line in finalized_lines] 
    
    #robust fit, calculate maes and growth rates
    for i, finalized_line in enumerate(finalized_lines):
        x, y = extract_data(finalized_line) #x=diams,y=times
        x_fit, y_fit, params = robust_fit(x,y)
        mae = cal_mae(x,y,[params[1],params[0]]) #h
        GR = 1/(params[1]*24) #nm/h
        
        fitted_points = [(time,diam) for diam,time in zip(x_fit,y_fit)]
        results_dict[f'line{str(i)}'] = {'points': finalized_line, 'fitted points': fitted_points, 'growth rate': GR, 'mae': mae}

    return results_dict
def init_find(df,df_mc,df_AT,mgsc,a,gret):
    '''
    Initialize functions.
    Format of results:
    results = {'line0': {'points': [(t1,d1),(t2,d2),...], 'fitted points': ..., 'growth rate': ... etc.}}
    '''
    
    #find consecutive datapoints
    mc_results = find_growth(df,times=df_mc['timestamp'],diams=df_mc['peak_diameter'],mgsc=mgsc,a=a,gret=gret) #maximum concentration
    at_results = find_growth(df,times=df_AT['timestamp'],diams=df_AT['diameter'],mgsc=mgsc,a=a,gret=gret) #appearance time
    
    return mc_results, at_results 
    
#################### PLOTTING ######################
def plot_channel(df,diameter_list_i,mpd,mdc,threshold_deriv,show_start_times_and_maxima):
    '''
    Plots chosen diameter channels with various useful information.
    
    mpd = maximum peak difference
    mdc = maximum diameter channel 
    '''   
    
    '''1 assemble all datasets'''
    df_mc, df_at, df_dt, incomplete_mc_xyz, incomplete_mc_xyz, peak_area_edges_gaussian, peak_area_edges_logistic, \
        fitting_parameters_gaus, fitting_parameters_logi,  \
        threshold_deriv, start_times_list, maxima_list = init_methods(df,mpd,mdc,threshold_deriv)

    '''2 define lists and their shapes'''
    xy_maxcon =  []             #[(max con diameter, max con time (UTC), max con), ...]
    fitting_params_gaus = []    #[(max con diameter, *params...), ...]
    fitting_params_logi = []    #[(appearance time diameter, y min for scale, *params...), ...]
    appearances = []            #[(diameter, time (UTC), concentration), ...] 
    peak_area_edges_gaus = []
    peak_area_edges_logi = []

    '''3 find data in datasets with chosen diameters'''
    diameter_list = [df.columns[i] for i in diameter_list_i]
    
    for diam in diameter_list:
        #MAXIMUM CONCENTRATION & TIME
        indices = [i for i, a in enumerate(df_mc['peak_diameter']) if a == diam] #indices of datapoints with wanted diameter
        xy_maxcons = [(df_mc['peak_diameter'][b],df_mc['timestamp'][b],df_mc['max_concentration'][b]) for b in indices]
        [xy_maxcon.append(i) for i in xy_maxcons]

        #FITTING PARAMETERS
        [fitting_params_gaus.append(params) for params in fitting_parameters_gaus if params[0] == diam]
        [fitting_params_logi.append(params) for params in fitting_parameters_logi if params[0] == diam]

        #APPEARANCE TIME & CONCENTRATION
        indices = [i for i, a in enumerate(df_at['diameter']) if a == diam]
        appearance = [(df_at['diameter'][b],df_at['timestamp'][b],df_at['mid_concentration'][b]) for b in indices]
        [appearances.append(i) for i in appearance]
        
        #PEAK AREA EDGES
        [peak_area_edges_gaus.append(peak_area) for peak_area in peak_area_edges_gaussian if peak_area[0] == diam]
        [peak_area_edges_logi.append(peak_area) for peak_area in peak_area_edges_logistic if peak_area[0] == diam]


    '''4 plotting'''
    fig, ax1 = plt.subplots(len(diameter_list),2,figsize=(9, 4.7), dpi=300)
    fig.subplots_adjust(wspace=0.38, hspace=0.29) #adjust spaces between subplots
    ax1 = np.atleast_2d(ax1) #to avoid problems with plotting only one channel
    lines_and_labels = set() #later use for legends
    
    #parameters
    #define x and y for the whole channel
    x = df.index #time
    y_list = [df[diam] for diam in diameter_list] #concentrations
    
    #also smoothed concentrations
    df_interpolated = df.interpolate(method='time')
    df_filtered = average_filter(df_interpolated,window=3)
    x_smooth = df_filtered.index #times
    y_smooth = [df_filtered[diam] for diam in diameter_list] #concentrations

    #PLOTS ON THE LEFT
    #row_num keeps track of which row of figure we are plotting in
    for row_num, y in enumerate(y_list):
        #fig.text(0.51,0.9, f'dp: ≈{diameter_list[row_num]:.2f} nm', ha='center', va='baseline',fontsize=8)
        ax1[row_num,0].set_title(f'≈{diameter_list[row_num]:.2f} nm', loc='right', fontsize=6) #diameter titles

        #left axis (smoothed)
        smooth_conc, = ax1[row_num,0].plot(x_smooth, y_smooth[row_num], color="cadetblue", lw=1)
        lines_and_labels.add((smooth_conc,"smoothed concentration"))
        
        #left axis (normal scale)
        color1 = "royalblue"
        ax1[row_num,0].set_ylabel("dN/dlogDp (cm⁻³)", color=color1, fontsize=8)
        original_conc, = ax1[row_num,0].plot(x, y, color=color1, lw=1)
        lines_and_labels.add((original_conc,"unsmoothed concentration"))
        
        for item in ([ax1[row_num,0].title, ax1[row_num,0].xaxis.label, ax1[row_num,0].yaxis.label] + ax1[row_num,0].get_xticklabels() + ax1[row_num,0].get_yticklabels()):
            item.set_fontsize(8)
            item.set_fontweight("bold")
        ax1[row_num,0].tick_params(axis='y', labelcolor=color1)
            
        #all possible start times of peak areas
        if show_start_times_and_maxima:
            
            for start in start_times_list: #start points
                start_diam, start_times, start_concs = start
                
                if start_diam == diameter_list[row_num]:
                    #ax1[row_num,0].vlines(x=start_times,ymin=0,ymax=100, color='black', linestyle='-', lw=0.8)
                    starts = ax1[row_num,0].scatter(start_times, start_concs, s=5, c='black',marker='>', alpha=0.6, zorder=10)
                    lines_and_labels.add((starts,"possible peak area starting points"))
            
            for maximum in maxima_list: #maximas
                max_diam, max_times, max_concs = maximum
                
                if max_diam == diameter_list[row_num]:  
                    #ax1[row_num,0].vlines(x=start_times,ymin=0,ymax=100, color='black', linestyle='-', lw=0.8)
                    maxs = ax1[row_num,0].scatter(max_times, max_concs, s=5, c='black',marker='.', alpha=0.6, zorder=10)
                    lines_and_labels.add((maxs,"possible peak area maxima"))
        
        '''
        #start and end points of modes
        found_ranges = []
        for edges in mode_edges:
            diam, start, end = edges
            if diam == diameter_list[row_num] and edges not in found_ranges:
                line1 = ax1[row_num,0].axvspan(start, end, alpha=0.13, color='darkorange')
                found_ranges.append(edges) #plot the same range once
        '''
        
        #gaussian fit 
        for params, edges in zip(fitting_params_gaus,peak_area_edges_gaus):
            diam_params, y_min, a, mu, sigma = params
            diam_gaus, start_time, end_time = edges 
            peak_area_times_UTC = df_interpolated.loc[start_time:end_time,diam].index
            
            if diam_params == diameter_list[row_num] and diam_gaus == diameter_list[row_num]: #check that plotting happens in the right channel
                line2, = ax1[row_num,0].plot(peak_area_times_UTC, gaussian(date2num(peak_area_times_UTC),a,mu,sigma)+y_min, '--', color="mediumturquoise",lw=1.2)
                lines_and_labels.add((line2,"gaussian fit"))
                #ax2.plot(peak_area_times_UTC, gaussian(peak_area_times,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
        
        #logistic fit
        for params, edges_logi in zip(fitting_params_logi,peak_area_edges_logi):
            diam_params, y_min, L, x0, k = params
            diam_logi, start_time_logi, end_time_logi = edges_logi
            peak_area_times_UTC = df_interpolated.loc[start_time_logi:end_time_logi,diam].index
     
            if diam_params == diameter_list[row_num] and diam_logi == diameter_list[row_num]:
                line3, = ax1[row_num,0].plot(peak_area_times_UTC, logistic(date2num(peak_area_times_UTC),L,x0,k)+y_min, '--', color="gold",lw=1.2)
                lines_and_labels.add((line3,"logistic fit"))
                #ax2.plot(peak_area_times_UTC, logistic(peak_area_times,L,x0,k), '--', color="gold",lw=1.2)

        #maximum concentration
        for i in xy_maxcon:
            diam, x_maxcon, y_maxcon = i
            if diam == diameter_list[row_num]:
                line4, = ax1[row_num,0].plot(x_maxcon, y_maxcon, '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6)
                lines_and_labels.add((line4,"maximum concentration"))
                #ax2.plot(x_maxcon, y_maxcon, '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6)

        #appearance time
        for i in appearances:
            diam, time, conc = i
            if diam == diameter_list[row_num]:
                line5, = ax1[row_num,0].plot(time, conc, '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6)
                lines_and_labels.add((line5,"appearance time"))
                #ax2.plot(time, conc, '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6)
        
        ax1[row_num,0].set_xlim(df.index[0],df.index[-1])
        ax1[row_num,0].set_facecolor("lightgray")
        #ax1[row_num,0].xaxis.set_tick_params(rotation=30)
        ax1[row_num,0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ax1[row_num,0].xaxis.set_major_formatter(DateFormatter("%H"))
        #if row_num == 0:
        #    ax1[row_num,0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    #PLOTS ON THE RIGHT
    df_1st_derivatives = cal_derivative(df_interpolated)

    x = df_1st_derivatives.index #time
    y_list = [df_1st_derivatives[diam] for diam in diameter_list] #concentrations

    #row_num keeps track of which row of figure we are plotting in
    for row_num, y in enumerate(y_list):
        #left axis
        color1 = "royalblue"
        ax1[row_num,1].set_ylabel("d²N/dlogDpdt (cm⁻³/h)", color=color1, fontsize=8)
        ax1[row_num,1].plot(x, y, color=color1, lw=1)
        ax1[row_num,1].scatter(x, y, s=2, c=color1)
        
        for item in ([ax1[row_num,1].title, ax1[row_num,1].xaxis.label, ax1[row_num,1].yaxis.label] + ax1[row_num,1].get_xticklabels() + ax1[row_num,1].get_yticklabels()):
            item.set_fontsize(8)
            item.set_fontweight("bold")
        ax1[row_num,1].tick_params(axis='y', labelcolor=color1)
        line6 = ax1[row_num,1].axhline(y=threshold_deriv, color="royalblue", linestyle='--', lw=1)
        lines_and_labels.add((line6,f"threshold = {str(threshold_deriv)} cm⁻³/h"))
        
        #start and end points of peak areas
        found_ranges = []
        for edges in peak_area_edges_gaus:
            diam, start, end = edges
            if diam == diameter_list[row_num] and edges not in found_ranges:
                line7 = ax1[row_num,1].axvspan(start, end, alpha=0.18, color='darkorange')
                lines_and_labels.add((line7,"peak area edges"))
                found_ranges.append(edges) #plot the same range once 

        #maximum concentration
        for i in xy_maxcon:
            diam, x_maxcon, y_maxcon = i
            y_maxcon = y_maxcon*0 #to place the start lower where y = 0
            if diam == diameter_list[row_num]:
                ax1[row_num,1].plot(x_maxcon, y_maxcon, '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6)  
        
        #appearance time
        for i in appearances:
            diam, time, conc = i
            conc = conc*0 #to place the start lower where y = 0
            if diam == diameter_list[row_num]:
                ax1[row_num,1].plot(time, conc, '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6)
                #ax2.plot(time, conc, '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6)

        ax1[row_num,1].set_xlim(df.index[1],df.index[-1])
        ax1[row_num,1].set_ylim(-50,threshold_deriv*1.8)
        ax1[row_num,1].set_facecolor("lightgray")
        #ax1[row_num,1].xaxis.set_tick_params(rotation=30) #rotating x-axis labels
        ax1[row_num,1].xaxis.set_major_formatter(DateFormatter("%H"))


    ### ADJUSTMENTS ###
    #common titles
    ax1[0,0].set_title("Concentration", fontsize=10, fontweight="bold") 
    ax1[0,1].set_title("Derivative", fontsize=10, fontweight="bold")
    ax1[len(diameter_list)-1,0].set_xlabel("time (h)", fontsize=8, fontweight="bold")
    ax1[len(diameter_list)-1,1].set_xlabel("time (h)", fontsize=8, fontweight="bold")
    
    #LEGEND#
    #unzip the valid entries into separate lists for the legend
    if len(lines_and_labels) <= 2:
        return print("This diameter channel(s) has no fits!\n**************************")

    #filter duplicates
    lines_and_labels = {entry[1]: entry for entry in lines_and_labels}
    lines_and_labels = set(lines_and_labels.values())

    #left plots
    lines_and_labels1 = [elem for elem in lines_and_labels if elem[1] in ["smoothed concentration", "unsmoothed concentration","possible peak area starting points","possible peak area maxima",\
                                                                            "gaussian fit","logistic fit","maximum concentration","appearance time"]]
    valid_lines, valid_labels = zip(*lines_and_labels1)
    legend_1 = ax1[0, 0].legend(valid_lines, valid_labels, fancybox=False, framealpha=0.9, fontsize=4, loc="upper right")
    
    #right plots
    lines_and_labels2 = [elem for elem in lines_and_labels if elem[1] in ["maximum concentration","appearance time","peak area edges",f"threshold = {str(threshold_deriv)} cm⁻³/h"]] 
    valid_lines, valid_labels = zip(*lines_and_labels2)
    legend_2 = ax1[0, 1].legend(valid_lines, valid_labels, fancybox=False, framealpha=0.9, fontsize=4, loc="upper right")


    #plot line when day changes
    new_day = None
    for i in df.index:
        day = i.strftime("%d")  
        if day != new_day:
            i = i - timedelta(minutes=15) #shift back 15mins due to resolution change
            for row_num in range(len(y_list)):
                ax1[row_num,0].axvline(x=i, color='black', linestyle='-', lw=0.8)
                ax1[row_num,1].axvline(x=i, color='black', linestyle='-', lw=0.8)
        new_day = day
    
    print("Drawing diameter channel(s):",diameter_list)

from GR_calculator_unfer_v2_modified import * 
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg") #backend changes the plotting style
import matplotlib.dates as mdates
from operator import itemgetter
from collections import defaultdict
import statsmodels.api as sm
from sklearn import linear_model
from itertools import cycle

'''
This code assumes dmps data.
- 1st column (skipping the first 0 value):              time in days
- 2nd column (skipping the first 0 value):              total concentration in current timestamp (row)
- 1st row (skipping first two 0 values):                diameters in meters, ~3nm-1000nm
- 3rd column onwards till the end (under diameters):    concentrations, dN/dlog(dp)
'''

################ DATA FORMATTING ################
folder = r"./dmps Nesrine/" #folder where data files are stored, should be in the same directory as this code
file_names = file_names #copy paths from the file "GR_calculator_unfer_v2_modified"

#let's define three useful functions
def combine_data(): 
    '''
    Loads dmps data for given days.
    Returns one dataframe with all given data.
    '''
    dfs = []
    test = True
    
    #load all given data files and save them in a list
    for file_name in file_names:
        df = pd.DataFrame(pd.read_csv(folder + file_name,sep='\s+',engine='python'))
        
        #different dmps data files have slightly different diameter values although they represent the same diameter
        #name all other columns with the labels of the first one
        if test == True:
            diameter_labels = df.columns
            test = False
        df.rename(columns=dict(zip(df.columns, diameter_labels)), inplace=True)
        dfs.append(df) #add dataframes to a list of dataframes
    
    combined_data = pd.concat(dfs,axis=0,ignore_index=True) #combine dataframes
    return combined_data
def days_into_UTC(data):
    '''
    Changes time from days to UTC.
    Takes in data which can be a df, list or just one date.
    Returns the times in UTC.
    '''
    try:
        time_steps = data["time (d)"] - time_d[0] #calculate timesteps between every timestamp
        start_date_str = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00" #define start date
        start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S") #change to date time
        data["time (d)"] = [start_datetime + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
        return data
    except:
        time_steps = data - time_d[0]
        start_date_str = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00"
        start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        try: #converting timesteps to datetime
            data = [start_datetime + timedelta(days=i) for i in time_steps] 
        except: #in case the list has just one element
            data = start_datetime + timedelta(days=time_steps)
        return data
def avg_filter(dataframe,resolution):
    '''
    Smoothens data in dataframe with average filter and given resolution (minutes), 
    i.e. takes averages in a window without overlapping ranges.
    Discards blocks of time with incomplete timestamps.
    Returns smoothened dataframe and new time in days for that dataframe.
    '''

    dataframe.index = dataframe.index.round('10min') #change timestamps to be exactly 10min intervals

    #if average is taken of less than 3 datapoints, neglect that datapoint
    full_time_range = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq='10min')
    missing_timestamps = full_time_range.difference(dataframe.index) #missing timestamps
    blocks = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq=f'{resolution}min') #blocks of resolution
    
    dataframe = dataframe.resample(f'{resolution}min').mean() #change resolution and take average of values
    dataframe = dataframe.shift(1, freq=f'{int(resolution/2)}min') #set new timestamps to be in the middle of the new resolution

    irrelevant_ts = []
    irrelevant_i = []
    missing_ts_i = [] #needs to be in order from small to big (for insterting nan values)
    for timestamp in missing_timestamps:
        for i in range(len(blocks) - 1):
            if blocks[i] <= timestamp < blocks[i + 1]: #check if timestamp is in this 30min block
                irrelevant_ts.append(blocks[i] + pd.Timedelta(minutes=15))  #add 15 minutes to center the block
                irrelevant_i.append(dataframe.index.get_loc(blocks[i] + pd.Timedelta(minutes=15))) #save index of the block
                missing_ts_i.append(full_time_range.get_loc(timestamp)) #save indices of missing timestamps

    #remove irrelevant timestamps
    for dp in dataframe.columns:
        for ts in irrelevant_ts:
            dataframe.loc[ts,dp] = np.nan #set nan value for irrelevant datapoints
            dataframe = dataframe.dropna() #remove these rows

    #add nan values for missing timestamps
    time_d_res = time_d.copy()
    for k in missing_ts_i:
        time_d_res = np.insert(time_d_res,k,np.nan)
    
    time_d_res = [time_d_res[i] + 0.25/24 for i in range(0,len(time_d_res)-1,int(resolution/10))] #save time in days with new resolution
    time_d_res = [j for i, j in enumerate(time_d_res) if i not in irrelevant_i] #remove irrelevant times    

    return dataframe, np.array(time_d_res)

df = combine_data()
df.rename(columns=dict(zip(df.columns[[0,1]], ["time (d)", "total number concentration (N)"])), inplace=True) #rename first two columns
df = df.drop(['total number concentration (N)'], axis=1) #drop total N concentrations from the dataframe as they're not needed
time_d = df['time (d)'].values.astype(float) #save time as days before changing them to UTC
df = days_into_UTC(df)

#set new UTC timestamps as indices
df.rename(columns={'time (d)': 'time (UTC)'}, inplace=True)
df['time (UTC)']=pd.to_datetime(df['time (UTC)'], format="%Y-%m-%d %H:%M:%S")
df.index=df['time (UTC)']
df = df.drop(['time (UTC)'], axis=1)

df.columns = pd.to_numeric(df.columns) * 10**9 #set numerical diameters as column headers, units from m to nm
#df.to_csv('./data_original.csv', sep=',', header=True, index=True, na_rep='nan')
df, time_d_res = avg_filter(df,resolution=30) #filtering

#with this we can check the format
#df.to_csv('./data_filtered.csv', sep=',', header=True, index=True, na_rep='nan')

####################################################
#useful functions
def closest(list, number):
    '''
    Finds closest element in a list to a given value.
    Returns the index of that element.
    '''
    value = []
    for i in list:
        value.append(abs(number-i))
    return value.index(min(value))
def find_zero(dataframe):
    '''
    Finds all zeros in a dataframe and creates a new dataframe where zeros are kept and all other values are nans.
    Returns dataframe with zeros and nans.
    '''
    df_zero_deriv = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns)
    
    for i in dataframe.columns:
        zero_value = dataframe[i] == 0
        df_zero_deriv[i] = dataframe[i].where(zero_value, np.nan)
    return df_zero_deriv
def flatten_list(list):
    '''
    Takes in a list of lists and
    returns a flattened list. 
    '''
    return [x for xs in list for x in xs]
def combine_connected_pairs(list):
    ''' From AI
    Takes in a list of lists with pairs of datapoints and
    returns a list with lists pooled together containing
    overlapping elements. 
    '''
    # Convert arrays to tuples for hashing and build adjacency list
    graph = defaultdict(set)
    for p1, p2 in list:
        graph[tuple(p1)].add(tuple(p2))
        graph[tuple(p2)].add(tuple(p1))
    
    # Find connected components using DFS
    visited = set()
    combined_lists = []
    
    for node in graph:
        if node not in visited:
            stack = [node]
            component = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(graph[current] - visited)
            combined_lists.append(component)
    
    # Convert back to numpy arrays if needed
    return [[point for point in component] for component in combined_lists]

#define mathematical functions for fitting
def gaussian(x,a,x0,sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def logistic(x,L,x0,k): 
    return L / (1 + np.exp(-k*(x-x0))) 
def linear(x,k,b):
    return k*x + b
def logarithmic(x):
    return np.log(x)

def cal_1st_derivative(dataframe,time_days):
    '''
    Calculates 1st derivatives between neighbouring datapoints. 
    Takes in dataframe and wanted range of time in days (in case of smaller dataframes).
    Returns dataframe with derivatives.
    '''
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns) 
    
    for i in dataframe.columns: 
        N = dataframe[i] #concentration
        time = time_days * 24 * 60 * 60 #change days to seconds
        dNdt = np.diff(N)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = dNdt #add calculated derivatives to dataframe
    return  df_derivatives
def find_modes(dataframe,df_deriv,threshold_deriv):
    '''
    Finds modes with derivative threshold.
    Takes a dataframe with concentrations and time (UTC), 
    dataframe with time derivatives of the concentrations
    and the wanted derivative threshold.
    Returns dataframe with found modes.
    '''

    df_modes = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns) 

    #threshold that determines what is a high concentration
    threshold = abs(df_deriv) > threshold_deriv #checks which values surpass the threshold
    start_points = threshold & (~threshold.shift(1,fill_value=False)) #find start points (df)

    #finding values within start and end points
    start_time = None 
    end_time = None

    for diam in df_deriv.columns:
        for timestamp in df_deriv.index: #identify pairs of start and end times
            if start_time != None: #when a start time has been found find list of values between start/end times
                subset = dataframe[diam].loc[start_time:end_time]
                df_modes.loc[subset.index,diam] = subset #fill dataframe with mode ranges
                
                #restart initial values
                start_time = None 
                end_time = None                     
                                  
            elif start_points.loc[timestamp,diam] == True and df_deriv.loc[timestamp,diam] > 0: #checks also if the starting point derivative is neg or not
                start_time = timestamp
                start_conc = df_deriv.loc[timestamp,diam]

                #finding end time after local maximum concentration 
                '''
                try:
                    subset_end_i = dataframe.index.get_loc(start_time) + 14
                    subset_end = dataframe.index[subset_end_i] #time after 140mins from starting point
                except IndexError:
                    subset_end = dataframe.index[-1] #in case we meet the end of the range
                '''
                subset_end = dataframe.index[-1]
                
                df_subset = dataframe.loc[start_time:subset_end,diam] #subset from mode start to (mode start + 140mins) unless mode start is near the end of a range
                end_conc = ( max(df_subset.values) + start_conc ) / 2  #(N_max + N_start)/2
                max_conc_time = df_subset.index[df_subset == max(df_subset.values)].tolist()[0] #rough estimate of maximum concentration time 
                max_conc_time_i = dataframe.index.get_loc(max_conc_time) #index of max_conc_time
                
                try:
                    new_subset_end = dataframe.index[max_conc_time_i + (max_conc_time_i - dataframe.index.get_loc(start_time))] #new ending limit to find end concentration time, same distance from max conc time as start time is
                except IndexError:
                    new_subset_end = dataframe.index[-1] #limits to range aroung GRs
                
                end_conc_i = closest(dataframe.loc[max_conc_time:new_subset_end,diam],end_conc) #index of end point
                end_time = dataframe.index[end_conc_i+max_conc_time_i] 

                if start_time == end_time: #skip modes with same start/end time
                    start_time = None
                    end_time = None
                
            else:
                continue
    return df_modes
def find_ranges(): 
    '''
    Finds ranges around growth rates from previously calculated mode fitting data.
    Returns a list of dataframes with the wanted ranges and their growth rates.
    '''
    
    #pidennä ikkunaa kun loivempi??
    df_GR_values = df_GR_final #open calculated values in Gabi's code
    threshold = 0 #GR [nm/h], all growth rates

    df_ranges = []
    growth_rates =[]

    for i in df_GR_values.index: #go through every fitted growth rate
        growth_rate = df_GR_values.loc[i,"GR"]
        if abs(growth_rate) >= threshold: #find maximum concentration if growth rate is higher than threshold value
            row = df_GR_values.loc[i,:]
            
            #start and end times/diameters of fitted lines
            start_time = row["start"] 
            end_time = row["end"]

            if growth_rate < 0: #if growth rate is negative
                start_diam = row["d_final"]
                end_diam = row["d_initial"]
            else:
                start_diam = row["d_initial"]
                end_diam = row["d_final"]
            
            #make the ranges bigger with given parameters
            start_time = start_time - timedelta(hours=5) #5 hours
            end_time = end_time + timedelta(hours=5)
            
            start_diam = start_diam / 1.5 #factor of 1.5
            end_diam = end_diam * 1.5

            #make a df with wanted range and add them to the list
            df_mfit_con = df[(df.index >= start_time) & (df.index <= end_time)]
            df_ranges.append(df_mfit_con[df_mfit_con.columns[(df_mfit_con.columns >= start_diam) & (df_mfit_con.columns <= end_diam)]])
            growth_rates.append(growth_rate) #save also the growth rates

    return df_ranges, growth_rates

def maximum_concentration(dataframe): 
    '''
    Calculates the maximum concentration.
    Takes in dataframe from wanted area in the PSD.
    Returns:
    max_conc_time =         list of maximum concentration times (UTC)
    max_conc_diameter =     list of maximum concentration diameters (nm)
    max_conc =              list of maximum concentrations in corresponding datapoints
    maxcon_x_days =         list of time in days
    fitting_params =        list of gaussian fit parameters and more:
                            [[start time of mode in days, end time of mode in days,
                            parameter A, parameter mu, parameter sigma], ...]
    dfs_mode_start =        list of dfs with start of mode
    dfs_mode_end =          list of dfs with end of mode
    threshold_deriv =       chosen derivative threshold value
    '''
    #create lists for results
    max_conc_time = []
    max_conc_diameter = []
    max_conc = []
    fitting_params = []
    
    dfs_mode_start = []
    dfs_mode_end = []

    threshold_deriv = 0.05 #choose derivative threshold
    
    #times to days
    time_range = df.index.intersection(dataframe.index) #find matching indices
    indices = [df.index.get_loc(row) for row in time_range]
    time_d_range = time_d_res[indices] #define time in days again depending on chosen range
    
    df_deriv = cal_1st_derivative(dataframe,time_days=time_d_range) #calculate derivative
    df_modes = find_modes(dataframe,df_deriv,threshold_deriv=threshold_deriv) #find modes 

    #gaussian fit to every mode
    for dp_index in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for t_index in range(len(df_modes.index)):
            time = time_d_range[t_index]
            concentration = df_modes.iloc[t_index,dp_index] #find concentration values from the dataframe i.e range (y)

            if np.isnan(concentration) and len(y) != 0: #gaussian fit when all values of one mode have been added to the y list
                #gaussian fit to get maximum concentration values
                #rescaling for a more stable fitting
                x_min = np.min(x) 
                x = x - x_min
                y_min = np.min(y)
                y = y - y_min
                
                #initial guess for parameters
                mu=np.mean(x)
                sigma = np.std(x)
                a = np.max(y)

                try: #gaussian fit
                    params,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma],bounds=((0,0,-np.inf),(np.max(y),np.inf,np.inf)))
                    if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                        print("Peak outside range. Skipping.")
                    else:
                        max_conc_time = np.append(max_conc_time,params[1]+x_min) #make a list of times with the max concentration time in each diameter
                        max_conc_diameter = np.append(max_conc_diameter,float(df_modes.columns[dp_index])) #make list of diameters with max concentrations
                        max_conc = np.append(max_conc,params[0]+y_min) #maximum concentrations
                        
                        #create dfs with start and end points
                        df_mode_start = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns)
                        df_mode_end = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns)
                        
                        start_time_days = x[0]+x_min  #define start/end time (days)
                        end_time_days = x[-1]+x_min
                        
                        start_time_UTC = pd.Series(days_into_UTC(start_time_days-0.25/24)).dt.round('30min')[0] #UTC, rounded to the nearest 30min increment
                        end_time_UTC = pd.Series(days_into_UTC(end_time_days-0.25/24)).dt.round('30min')[0]
                        start_time_UTC += timedelta(minutes=15) #makes it easier to round
                        end_time_UTC += timedelta(minutes=15)
                        
                        df_mode_start.loc[start_time_UTC,df_modes.columns[dp_index]] = y[0]+y_min #replace nan value with concentration value at start/end point
                        df_mode_end.loc[end_time_UTC,df_modes.columns[dp_index]] = y[-1]+y_min
                        
                        dfs_mode_start.append(df_mode_start) #add to list of dfs
                        dfs_mode_end.append(df_mode_end)
                        
                        #gaussian fit parameters with start and end times and minimum concentration of horizontal mode
                        fitting_params.append([start_time_days, end_time_days, params[0]+y_min, params[1]+x_min, params[2]])                               
                except:
                    print("Diverges. Skipping.")
                                    
                x = [] #reset
                y = [] 
            elif not np.isnan(concentration): #separates mode values
                x = np.append(x,time)
                y = np.append(y,concentration)
            elif np.isnan(concentration): #skips nan values
                x = [] #reset
                y = [] 

    maxcon_x_days = max_conc_time #save maximum concentration time in days as well
    max_conc_time = np.array(days_into_UTC(max_conc_time)) #change days to UTC in the list of max concentration times

    return max_conc_time, max_conc_diameter, max_conc, maxcon_x_days, fitting_params, dfs_mode_start, dfs_mode_end, threshold_deriv
def appearance_time(dataframe):
    '''
    Calculates the appearance times.
    Takes in dataframe from wanted area in the PSD.
    Returns:
    appear_time =       list of appearance times (UTC)
    appear_diameter =   list of appearance time diameters (nm)
    mid_conc =          list of appearance time concentrations in corresponding datapoints
    appear_x_days =     list of time in days
    fitting_params =    list of logistic fit parameters and more:
                        [[start time of mode in days, end time of mode in days,
                        parameter L, parameter x0, parameter k], ...]
    '''
    #create lists for results
    appear_time = []
    appear_diameter = []
    mid_conc = []
    fitting_params = []
    
    #times to days
    time_range = df.index.intersection(dataframe.index) #find matching indices
    indices = [df.index.get_loc(row) for row in time_range]
    time_d_range = time_d_res[indices] #define time in days again depending on chosen range
    
    df_deriv = cal_1st_derivative(dataframe,time_days=time_d_range) #calculate derivative
    df_modes = find_modes(dataframe,df_deriv,threshold_deriv=0.05) #find modes

    #logistic fit to every mode
    for dp_index in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for t_index in range(len(df_modes.index)):
            time = time_d_range[t_index]
            concentration = df_modes.iloc[t_index,dp_index] #find concentration values from the dataframe (y)    

            if np.isnan(concentration) and len(y) != 0: #logistic fit when all values of one mode have been added to the y list 
                #gaussian fit to get maximum concentration values
                #rescaling for a more stable fitting
                x_min = np.min(x) 
                x = x - x_min
                y_min = np.min(y)
                y = y - y_min
                
                #initial guess for parameters
                mu=np.mean(x)
                sigma = np.std(x)
                a = np.max(y)

                try: 
                    params,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma],bounds=((0,0,-np.inf),(np.max(y),np.inf,np.inf)))
                    if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                        print("Peak outside range. Skipping.")
                    else:
                        max_conc_time = params[1] #from gaussian fit

                        #limit x and y to values between start time of mode and maximum concentration time in mode
                        max_conc_index = closest(x, max_conc_time)
                        x_sliced = x[:max_conc_index+1]
                        y_sliced = y[:max_conc_index+1]     
                        
                        #logistic fit
                        if len(y_sliced) != 0:
                            #initial guess for parameters
                            L = params[0] #maximum value of gaussian fit (concentration)
                            x0 = np.nanmean(x_sliced) #midpoint x value (appearance time)
                            k = 1.0 #growth rate

                            try:
                                params,pcov = curve_fit(logistic,x_sliced,y_sliced,p0=[L,x0,k],bounds=((params[0]*0.999,0,-np.inf),(params[0],np.inf,np.inf)))
                                if ((params[1]>=x_sliced.max()) | (params[1]<=x_sliced.min())): #checking that the peak is within time range   
                                    print("Peak outside range. Skipping.")
                                else:
                                    appear_time = np.append(appear_time,params[1]+x_min) #make a list of times with the appearance time in each diameter
                                    appear_diameter = np.append(appear_diameter,float(df_modes.columns[dp_index])) 
                                    mid_conc.append((params[0]+y_min)/2) #appearance time concentration (~50% maximum concentration), L/2, y_min to preserve shape of graph

                                    #logistic fit parameters with time range and minimum concentration of horizontal mode
                                    fitting_params.append([x[0]+x_min, x[-1]+x_min, params[0]+y_min, params[1]+x_min, params[2]])                                 
                            except:
                                print("Logistic diverges. Skipping.")
                        else:
                            print("NO Y VALUES IN THIS ROW")
                except:
                    print("Gaussian diverges. Skipping.")
                
                x = [] #reset
                y = [] 
            elif not np.isnan(concentration): #separates mode values
                x = np.append(x,time)
                y = np.append(y,concentration)
            elif np.isnan(concentration): #skips nan values
                x = [] #reset
                y = []

    appear_x_days = appear_time #save appearance time in days as well
    appear_time = np.array(days_into_UTC(appear_time)) #change days to UTC in the list of max concentration times

    return appear_time, appear_diameter, mid_conc, appear_x_days, fitting_params
def init_ranges():
    '''
    Goes through all ranges and calculates the points for maximum concentration
    and appearance time methods along with other useful information.
    x = time, y = diameter, z = concentration
    '''
    maxcon_xyz = [] #maximum concentration
    maxcon_x = []
    maxcon_y = []
    maxcon_z = []
    maxcon_x_days = []
    mc_fitting_params = []
    
    appear_xyz = [] #appearance time
    appear_x = []
    appear_y = []
    appear_z = []
    appear_x_days = []
    at_fitting_params = []
    
    GRs_mc = [] #growth rates of each range (around mode fitting)
    GRs_at = []
    dfs_mode_start = []
    dfs_mode_end = []

    df_ranges, growth_rates = find_ranges()

    for df_range,growth_rate in zip(df_ranges,growth_rates): #go through every range around GRs
        
        #THESE MIGHT NOT BE FOUND IN ALL RANGES, GR??, try except for mc and at separately
        mc_x, mc_y, mc_z, mc_days, mc_params, dfs_start, dfs_end, threshold_deriv = maximum_concentration(df_range)
        at_x, at_y, at_z, at_days, at_params = appearance_time(df_range)
        
        #add to lists
        maxcon_x = np.append(maxcon_x,mc_x) #maximum concentration
        maxcon_x_days = np.append(maxcon_x_days,mc_days)
        maxcon_y = np.append(maxcon_y,mc_y)
        maxcon_z = np.append(maxcon_z,mc_z)
        [mc_fitting_params.append(i) for i in mc_params]
        
        appear_x = np.append(appear_x,at_x) #appearance time
        appear_x_days = np.append(appear_x_days,at_days)
        appear_y = np.append(appear_y,at_y)
        appear_z = np.append(appear_z,at_z)
        [at_fitting_params.append(i) for i in at_params]
        
        GRs_mc = np.append(GRs_mc, growth_rate) #growth rates
        GRs_at = np.append(GRs_at, growth_rate)
        [dfs_mode_start.append(i) for i in dfs_start]
        [dfs_mode_end.append(i) for i in dfs_end]       
    
    #combine to same lists
    maxcon_xyz = [maxcon_x,maxcon_y,maxcon_z]
    appear_xyz = [appear_x,appear_y,appear_z]
    
    return maxcon_xyz, maxcon_x_days, mc_fitting_params, GRs_mc, appear_xyz, appear_x_days, at_fitting_params, GRs_at, dfs_mode_start, dfs_mode_end, threshold_deriv

xyz_maxcon, x_maxcon_days, fitting_parameters_gaus, GRs_maxcon, xyz_appear, x_appear_days, fitting_parameters_logi, GRs_appear, dfs_mode_start, dfs_mode_end, threshold_deriv = init_ranges()


def filter_duplicates(list,time_days):
    #TAKING A MEAN IS INACCURATE
    
    #filter points that are basically the same but found with a different range
    # to avoid many points in the same time/diam, take an average and set that as the new timestamp
    for i, diam in enumerate(df.columns.values): #loop through diameter values     
        channel = [[list[0][j],list[1][j],list[2][j]] for j, d in enumerate(list[1]) if d == diam] #choose points that are in diam channel

        if len(channel) > 1: #if there are several points in one channel, check time difference
            max_time_diff = 30/(60*24) #30min
            
            for elem in channel: #loop through points in diam channel
                time = elem[0]
                same_times = [point[0] for point in channel if abs(point[0]-time) < timedelta(days=max_time_diff)] #list of times close to current time

                #REMOVE DUPLICATES
                x, y, z = list
                t_d = time_days.copy()

                #identify indices of elements to remove
                mask_x = np.isin(x, same_times)
                
                if not any(mask_x): #if there are no True values in 'mask_x' 
                    continue #go to next iteration
                    
                #extract removed elements for averaging
                removed_times = x[mask_x]
                removed_z_values = z[mask_x]
                removed_time_days = time_days[mask_x]
                
                #remove the identified elements
                x = x[~mask_x]
                y = y[~mask_x]
                z = z[~mask_x]
                t_d = t_d[~mask_x]
                
                #calculate averages of removed elements
                #average_time = np.mean(removed_times).astype("datetime64[ms]")
                average_time = pd.to_datetime(pd.Series(removed_times)).mean()
                average_conc = np.mean(removed_z_values)
                average_time_days = np.mean(removed_time_days)

                #add the averages to the remaining data
                times = np.append(x, average_time)
                diams = np.append(y, diam)
                concs = np.append(z, average_conc)
                time_days_list = np.append(t_d,average_time_days)
                
                #update xyz and time_days
                list = [times, diams, concs]
                time_days = time_days_list

    return list, time_days

xyz_maxcon, x_maxcon_days = filter_duplicates(xyz_maxcon,x_maxcon_days)
xyz_appear, x_appear_days = filter_duplicates(xyz_appear,x_appear_days)


#################### GR DATA ######################

def robust_fit(time,diam,color):
    """
    Args:
        time (_type_): _description_
        diam (_type_): _description_
        
    Robust linear model with statsmodel package.
    Used the HuberT weighting function in iterative 
    robust estimation.
    
    
    """
    #do fit to linear data
    #diam_log = np.geomspace(min(diam), max(diam),num=len(time))[:, np.newaxis]
    diam_linear = np.linspace(min(diam), max(diam),num=len(time))[:, np.newaxis]
    time = np.array(time).reshape(-1,1)
    diam = np.array(diam).reshape(-1,1)
    
    #linear fit for comparison
    #lr = linear_model.LinearRegression().fit(diam, time) #x,y

    #robust fit
    #ransac = linear_model.RANSACRegressor().fit(diam, time)
    rlm_HuberT = sm.RLM(time, sm.add_constant(diam), M=sm.robust.norms.HuberT()) #statsmodel robust linear model
    rlm_results = rlm_HuberT.fit()
    
    #predict data of estimated models
    #t_linear = lr.predict(diam_linear)
    #t_ransac = ransac.predict(diam_linear)
    t_rlm = rlm_results.predict(sm.add_constant(diam_linear))
    t_params = rlm_results.params

    #change times to UTC
    #time_fit_linear = list(t_linear.reshape(-1))
    #time_UTC_linear = np.array(days_into_UTC(time_fit_linear)) #change days into UTC
    #time_fit_ransac = list(t_ransac.reshape(-1))
    #time_UTC_ransac = np.array(days_into_UTC(time_fit_ransac)) #change days into UTC
    time_fit_rlm = list(t_rlm.reshape(-1))
    time_UTC_rlm = np.array(days_into_UTC(time_fit_rlm)) #change days into UTC
    
    lw = 2
    #plt.plot(time_UTC, diam_linear, color="navy", linewidth=lw, label="Linear regressor")
    #plt.plot(time_UTC_ransac,diam_linear,color="cornflowerblue",linewidth=lw,label="RANSAC regressor")
    #plt.plot(time_UTC_linear, diam_linear, color="navy", linewidth=lw)
    #plt.plot(time_UTC_ransac, diam_linear,color="red",linewidth=lw)
    plt.plot(time_UTC_rlm, diam_linear,color=color,linewidth=lw)
    
    #growth rate annotation
    gr = 1/(t_params[1]*24) #unit to nm/h from time in days
    
    midpoint_idx = len(time_fit_rlm) // 2 #growth rate value
    midpoint_time = time_UTC_rlm[midpoint_idx]
    midpoint_value = diam[midpoint_idx]
    plt.annotate(f'{gr:.2f} nm/h', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8)
    
    #print("Statsmodel robus linear model results: \n",rlm_results.summary())
    #print("\nparameters: ",rlm_results.params)
    #print(help(sm.RLM.fit))

def find_dots_old(times,diams):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with wanted times and diameters for plotting growth rates.
    
    Parameters:
    times (list): List of time values.
    diams (list): List of corresponding diameter values.
    
    Returns:
    list: Combined lists of nearby datapoints.
    ''' 
    #combine to the same list and sort data by diameter
    data_sorted = np.array(sorted(zip(times, diams), key=itemgetter(1))) #[[time1,diam1],[time2,diam2]...]

    data_pairs = []
    base_max_time_diff = 90/(60*24) #max time difference in days = 90mins = 1,5h
    higher_max_time_diff = 120/(60*24) #120mins = 2h

    #iterate through each datapoint to find suitable pairs
    for i, datapoint in enumerate(data_sorted):
        for ii in range(1,len(data_sorted)-i):
            try: #in case we reach the end with no "next datapoint"
                next_datapoint = data_sorted[i+ii]
                time0, diam0 = datapoint
                time1, diam1 = next_datapoint
                time_diff = abs(time1-time0)
                diam_diff = abs(diam1-diam0)
                
                nextnext_diam = df.columns.values[df.columns.values > diam0][1] # 2 diameter points forward after current diameter
                max_diam_diff = abs(diam0-nextnext_diam) #max one diameter channel empty in between
                
                #at higher diameters allow longer time difference between starts
                max_time_diff = higher_max_time_diff if diam0 >= 22 else base_max_time_diff
                
                if time_diff <= max_time_diff and diam_diff <= max_diam_diff: #check time and diameter difference
                    #sub_data.append(next_datapoint) #add next valid datapoint to the sublist
                    data_pairs.append([datapoint,next_datapoint])
                    break #next point found 
                elif diam_diff > max_diam_diff: #if diam difference is already too big break loop
                    break
                else: #keep looking for next datapoint until end of points if the next one isnt suitable
                    continue
            except IndexError:
                continue
    
    #combine overlapping lists to create lists with nearby datapoints
    combined = combine_connected_pairs(data_pairs)          
    return combined
def find_dots(times,diams):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with wanted times and diameters for plotting growth rates.
    
    Parameters:
    times (list): List of time values.
    diams (list): List of corresponding diameter values.
    
    Returns:
    list: Combined lists of nearby datapoints.
    ''' 
    #combine to the same list and sort data by diameter
    data_sorted = np.array(sorted(zip(times, diams), key=itemgetter(1))) #[[time1,diam1],[time2,diam2]...]
    
    data_pairs = []
    base_max_time_diff = 90/(60*24) #max time difference in days = 90mins = 1,5h
    higher_max_time_diff = 120/(60*24) #120mins = 2h
    
    #iterate through each datapoint to find suitable pairs of mode fitting datapoints
    for i, datapoint in enumerate(data_sorted):
        for ii in range(1,len(data_sorted)-i):
            try: #in case we reach the end with no "next datapoint"
                next_datapoint = data_sorted[i+ii] #always after current datapoint
                time0, diam0 = datapoint
                time1, diam1 = next_datapoint
                time_diff = abs(time1-time0)
                diam_diff = abs(diam1-diam0)
                
                #diam difference in channels changes in a logarithmic scale, allow one empty channel in between
                nextnext_diam = df.columns.values[df.columns.values > diam0][1] #diameter in the channel one after
                next_diam = df.columns.values[df.columns.values > diam0][0]
                max_diam_diff = abs(next_diam-nextnext_diam) #max one diameter channel empty in between
                
                #at higher diameters allow longer time difference between starts
                max_time_diff = higher_max_time_diff if diam0 >= 22 else base_max_time_diff

                if diam_diff > max_diam_diff: #if diam difference is already too big break loop
                    break
                elif time_diff <= max_time_diff and diam_diff <= max_diam_diff: #check time and diameter difference
                    data_pairs.append([datapoint,next_datapoint]) #add nearby datapoint pairs to list
                    combined = combine_connected_pairs(data_pairs) #combine overlapping pairs to make lines
                    
                    #make a linear fit to check mape in each line of datapoints
                    for iii, line in enumerate(combined):
                        x = np.arange(len(line)) #diams
                        y = [datapoint[0] for datapoint in line] #times
                        
                        if len(x) == 1: #fitting curve wont work with only 1 datapoint
                            continue
                        
                        popt, pcov = curve_fit(linear, x, y)
                        absolute_error = np.abs(linear(x, *popt) - y)
                        mape = np.mean(absolute_error / y) * 100

                        #mape is not the same in different scales so here the threshold must be stricter
                        #allow mape of 5% when there are only 3 datapoints, otherwise 4%
                        mape_threshold = 0.025 if len(line) <= 3 else 0.02
                        
                        if mape > mape_threshold:
                            #calculate mape without first datapoint of line to see if mape is smaller
                            x = np.arange(len(line[1:]))
                            y = [datapoint[0] for datapoint in line[1:]] #diams of line
                            popt, pcov = curve_fit(linear, x, y)
                            absolute_error = np.abs(linear(x, *popt) - y)
                            mape = np.mean(absolute_error / y) * 100
                            
                            if mape > mape_threshold: #if mape is still too big
                                #delete recently added datapoint from both lists
                                combined = [[point for point in component if any(point != next_datapoint)] for component in combined] 
                                data_pairs = [pair for pair in data_pairs if not any(np.array_equal(next_datapoint, point) for point in pair)] 
                            else:
                                #remove the first datapoint of the line
                                first_datapoint = line[0]
                                combined[iii] = line[1:]
                                #remove also data pair with this datapoint
                                data_pairs = [pair for pair in data_pairs if not any(np.array_equal(first_datapoint, point) for point in pair)]
                            
                    break
                else: #keep looking for next datapoint until end of points if the next one isnt suitable
                    continue
            except IndexError:
                print("INDEX ERROR")
                continue
    
    combined = [sorted(component, key=lambda x: x[0]) for component in combined] # Sort each individual component by timestamp
    
    return combined

def filter_dots(datapoints):
    '''
    Filter datapoints of lines that are too short or
    with too big of an error.
    '''
    #check length of datapoints for each line
    datapoints = [subpoints for subpoints in datapoints if len(subpoints) >= 4] #length of at least 4 datapoints
    
    #check error of possible fitted linear curve
    filtered_datapoints = []
    for line in datapoints:
        i = 1
        while True:
            try:
                x = np.arange(len(line)) #diams
                y = [time[0] for time in line] #time as y
                
                popt, pcov = curve_fit(linear, x, y)
                absolute_error = np.abs(linear(x, *popt) - y)
                #print("linear(x, *popt) - y",linear(x, *popt),"-",y)
                mape = np.mean(absolute_error / y) * 100
                GR = 1/(popt[0]*24) #growth rate

                #mape is not the same in different scales so here the threshold must be stricter
                #maximum mape 1% and GR is not bigger than +-10nm/h
                if mape <= 1 and abs(GR) <= 10:
                    filtered_datapoints.append(line)
                    break
                else:
                    break
                #else:
                #    line = line[:-i] #exclude last elements one by one
                #    i += 1
                    #print("deleted",line[-1])
                    #removed_points.append(line[-1])    #add removed datapoint to another list
 
            except:
                print("Linear fit diverges.")

    return filtered_datapoints  
def init_find():
    
    #find consequtive datapoints
    mc_data = find_dots(times= x_maxcon_days,diams= xyz_maxcon[1]) #maximum concentration
    at_data = find_dots(times= x_appear_days,diams= xyz_appear[1]) #appearance time
    
    #filter series of datapoints that are too short or with high deviation
    mc_filtered = filter_dots(mc_data)
    at_filtered = filter_dots(at_data)
    
    #extract times and diameters
    time_mc = [[seg[0] for seg in mc_segment] for mc_segment in mc_filtered]
    diam_mc = [[seg[1] for seg in mc_segment] for mc_segment in mc_filtered]
    time_at = [[seg[0] for seg in at_segment] for at_segment in at_filtered]
    diam_at = [[seg[1] for seg in at_segment] for at_segment in at_filtered]
    
    return time_mc, diam_mc, time_at, diam_at
    
#################### PLOTTING ######################

def linear_fit(time,diam):
    '''
    y = time in days
    x = diameters in nm
    Flipped as the error is in time.
    '''
    #linear least square fits
    #params, pcov = curve_fit(linear, np.log(diam), time) #logarthmic diam
    params, pcov = curve_fit(linear, diam, time) #linear diam
    gr = 1/(params[0]*24) #unit to nm/h from time in days
    time_fit = params[0]*np.array(diam) + params[1]
    time_UTC = np.array(days_into_UTC(time_fit)) #change days into UTC

    #plotting
    plt.plot(time_UTC,diam,lw=3) #line
    
    midpoint_idx = len(time_fit) // 2 #growth rate value
    midpoint_time = time_UTC[midpoint_idx]
    midpoint_value = diam[midpoint_idx]
    plt.annotate(f'{gr:.2f} nm/h', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, fontweight='bold')
def plot_manual_GR(times,diams,time_range,diam_range):
    '''
    Plots growth rates from chosen range of either method's dots. 
    Takes in the x and y values of determined maximum concentrations / appearance times and their wanted range.
    time_range = [time1,time2]
    diam_range = [diam1,diam2]
    #plot_manual_GR(x_maxcon_days,xyz_maxcon[1],["2016-06-12 10:30:00","2016-06-12 12:30:00"],[9,25]) #maximum concentration
    #plot_manual_GR(x_appear_days,xyz_appear[1],["2016-06-12 08:00:00","2016-06-12 12:00:00"],[5.5,30]) #appearance time
    '''
    #change strings to days from start of measurement
    time_range_days = []
    for i in time_range:
        i = datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        start_of_day = datetime(i.year, i.month, i.day)
        days_from_start = (i - start_of_day) / timedelta(days=1)
        time_range_days.append(time_d[0] + days_from_start)

    #choose wanted dots
    sub_times = []
    sub_diams = []
    for time,diam in zip(times,diams):
        if time >= time_range_days[0] and time <= time_range_days[1] and diam >= diam_range[0] and diam <= diam_range[1]:
            sub_times = np.append(sub_times,time)
            sub_diams = np.append(sub_diams,diam)

    x = np.sort(sub_diams) #flip due to variability in time 
    y = np.sort(sub_times)

    #linear least square fits
    #params, pcov = curve_fit(linear, np.log(x), y) #logarthmic x
    params, pcov = curve_fit(linear, x, y) #linear x
    gr = 1/(params[0]*24) #unit to nm/h from time in days
    y_fit = params[0]*x + params[1] #TOOK LOG(x) AWAY
    x_UTC = np.array(days_into_UTC(y_fit)) #change days into UTC

    #plotting
    plt.plot(x_UTC,x,lw=3) #line
    
    midpoint_idx = len(y_fit) // 2 #growth rate value
    midpoint_time = x_UTC[midpoint_idx]
    midpoint_value = x[midpoint_idx]
    plt.annotate(f'{gr:.2f} nm/h', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, fontweight='bold')

def plot_PSD(dataframe):
    '''
    Plots dots for maximum concentration and 
    appearance time methods.
    Additionally does adjustments to PSD.
    '''
    #plot line when day changes
    new_day = None
    for i in dataframe.index:
        day = i.strftime("%d")  
        if day != new_day:
            i = i - timedelta(minutes=15) #shift back 15mins due to resolution change
            plt.axvline(x=i, color='black', linestyle='-', lw=1)
        new_day = day
    #fig, ax1 = plt.subplots(figsize=(9, 4.7))
    
    #dots
    plt.plot(xyz_maxcon[0], xyz_maxcon[1], '*', alpha=0.5, color='white', ms=5,label='maximum concentration') 
    plt.plot(xyz_appear[0], xyz_appear[1], '*', alpha=0.5, color='green', ms=5,label='appearance time')
    
    #growth rates
    
    time_mc, diam_mc, time_at, diam_at = init_find()
    for time_seg_mc, diam_seg_mc, time_seg_at, diam_seg_at in zip(time_mc,diam_mc,time_at,diam_at):
        #linear_fit(time_seg_mc, diam_seg_mc) #maximum concentration
        #linear_fit(time_seg_at, diam_seg_at) #appearance time
        robust_fit(time_seg_at, diam_seg_at,"green")
        robust_fit(time_seg_mc, diam_seg_mc,"white")
    
    #adjustments to plot
    plt.legend(fontsize=9,fancybox=False,framealpha=0.9)
    for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
        legend_handle.set_markeredgewidth(0.5)
        legend_handle.set_markeredgecolor("black")
    
    plt.xlim(dataframe.index[0],dataframe.index[-1])
    plt.ylim(dataframe.columns[0],dataframe.columns[-1])
    plt.ylabel("diameter (nm)",fontsize=14) #add y-axis label
    plt.xlabel("time",fontsize=14) #add y-axis label
plot_PSD(df)

def plot_channel(dataframe,diameter_list,choose_GR,draw_range_edges):
    '''
    Plots chosen diameter channels over UTC time, with thresholds and gaussian fit.
    ax[0,0] = whole channel over time, with ranges      ax[0,1] = derivative of concentrations
    ax[n,0] = n amount of channels                      ax[n,1] = derivative of concentrations
                                                ...
    Inputs dataframe with data and diameters (numerical).
    choose_GR = None or write wanted GR around which the info will be plotted
    range_edges = True, draws the edges of ranges around growth rates
    Assumes chosen channel has modes that have been found with the maximum concentration method!!!
    '''   
    
    '''1 assemble all datasets'''
    xyz_maxcon, x_maxcon_days, fitting_parameters_gaus, GRs_maxcon, xyz_appear, x_appear_days, fitting_parameters_logi, GRs_appear, dfs_mode_start, dfs_mode_end, threshold_deriv = init_ranges()

    '''1.5 (if a specific range is wanted) choose range that is plotted and modify datasets accodingly'''
    if choose_GR != None: 
        gr_indices = [i for i, x in enumerate(GRs_maxcon) if x == choose_GR] #maximum concentration 
        xyz_maxcon[0] = [xyz_maxcon[0][i] for i in gr_indices]
        xyz_maxcon[1] = [xyz_maxcon[1][i] for i in gr_indices]
        dfs_mode_start = [dfs_mode_start[i] for i in gr_indices]
        dfs_mode_end = [dfs_mode_end[i] for i in gr_indices]
        xyz_maxcon[2] = [xyz_maxcon[2][i] for i in gr_indices]
        fitting_parameters_gaus = [fitting_parameters_gaus[i] for i in gr_indices]

        gr_indices = [i for i, x in enumerate(GRs_appear) if x == choose_GR] #appearance time
        xyz_appear[0] = [xyz_appear[0][i] for i in gr_indices]
        xyz_appear[1] = [xyz_appear[1][i] for i in gr_indices]
        xyz_appear[2] = [xyz_appear[2][i] for i in gr_indices]
        fitting_parameters_logi = [fitting_parameters_logi[i] for i in gr_indices]

    '''2 define lists and their shapes'''
    mode_edges = []             #[(diameter,start_time (UTC),end_time (UTC)), ...]
    range_edges = []            #[(diameter,start_time (UTC),end_time (UTC)), ...]
    xy_maxcon =  []             #[(max con diameter, max con time (UTC), max con), ...]
    fitting_params_gaus = []    #[(max con diameter, mode start time (days), mode end time (days), *params...), ...]
    fitting_params_logi = []    #[(appearance time diameter, mode start time (days), mode end time (days), *params...), ...]
    mode_times_UTC = []         #[(diameter, time UTC), ...]
    appearances = []            #[(diameter, time (UTC), concentration), ...] 

    '''3 find data in datasets with chosen diameters'''
    for diam in diameter_list:
        #MAXIMUM CONCENTRATION & TIME
        indices = [i for i, a in enumerate(xyz_maxcon[1]) if a == diam] #indices of datapoints with wanted diameter
        xy_maxcons = [(xyz_maxcon[1][b],xyz_maxcon[0][b],xyz_maxcon[2][b]) for b in indices]
        [xy_maxcon.append(i) for i in xy_maxcons]

        #FITTING PARAMETERS FOR GAUSSIAN FIT
        indices = [i for i, a in enumerate(xyz_maxcon[1]) if a == diam]
        fittings = [(xyz_maxcon[1][b],fitting_parameters_gaus[b][0],fitting_parameters_gaus[b][1],fitting_parameters_gaus[b][2],fitting_parameters_gaus[b][3],fitting_parameters_gaus[b][4]) for b in indices]
        [fitting_params_gaus.append(i) for i in fittings]

        #FITTING PARAMETERS FOR LOGISTIC FIT
        indices = [i for i, a in enumerate(xyz_appear[1]) if a == diam]
        fittings = [(xyz_appear[1][b],fitting_parameters_logi[b][0],fitting_parameters_logi[b][1],fitting_parameters_logi[b][2],fitting_parameters_logi[b][3],fitting_parameters_logi[b][4]) for b in indices]
        [fitting_params_logi.append(i) for i in fittings]

        #APPEARANCE TIME & CONCENTRATION
        indices = [i for i, a in enumerate(xyz_appear[1]) if a == diam]
        appearance = [(xyz_appear[1][b],xyz_appear[0][b],xyz_appear[2][b]) for b in indices]
        [appearances.append(i) for i in appearance]
        
        for df_mode_start,df_mode_end in zip(dfs_mode_start,dfs_mode_end):
            try:
                #START/END TIME & TIME RANGE OF MODE
                start_time = df_mode_start[diam].notna().idxmax() #finding the start/end value in df
                end_time = df_mode_end[diam].notna().idxmax()
                mode_times = df.index.intersection(df_mode_start.loc[start_time:end_time].index) #UTC time for each range with wanted diameters
                
                if start_time != end_time: #checking that start time and end time aren't the same
                    mode_edges.append((diam,start_time,end_time))
                    mode_times_UTC.append((diam,mode_times)) 
                
                #START/END TIME OF RANGE
                start_time = df_mode_start.index[0]
                end_time = df_mode_end.index[-1]
                range_edges.append((diam,start_time,end_time))
                
            except KeyError:
                continue

    #find unique values
    #xy_maxcon = list(dict.fromkeys(xy_maxcon))


    '''4 plotting'''
    fig, ax1 = plt.subplots(len(diameter_list),2,figsize=(9, 4.7), dpi=300)
    fig.subplots_adjust(wspace=0.38, hspace=0.29) #adjust spaces between subplots
 #parameters
    #define x and y for the whole channel
    x = dataframe.index #time
    y_list = [dataframe[diam] for diam in diameter_list] #concentrations

    #PLOTS ON THE LEFT
    #row_num keeps track of which row of figure we are plotting in
    for row_num, y in enumerate(y_list):
        ax1[row_num,0].set_title(f'dp: ≈{diameter_list[row_num]:.2f} nm', loc='right', fontsize=8) #diameter titles

        #left axis (normal scale)
        color1 = "royalblue"
        ax1[row_num,0].set_ylabel("dN/dlogDp (cm⁻³)", color=color1, fontsize=8)
        ax1[row_num,0].plot(x, y, color=color1, lw=1)
        for item in ([ax1[row_num,0].title, ax1[row_num,0].xaxis.label, ax1[row_num,0].yaxis.label] + ax1[row_num,0].get_xticklabels() + ax1[row_num,0].get_yticklabels()):
            item.set_fontsize(8)
            item.set_fontweight("bold")
        ax1[row_num,0].tick_params(axis='y', labelcolor=color1)

        '''
        #start and end points of modes
        found_ranges = []
        for edges in mode_edges:
            diam, start, end = edges
            if diam == diameter_list[row_num] and edges not in found_ranges:
                line1 = ax1[row_num,0].axvspan(start, end, alpha=0.13, color='darkorange')
                found_ranges.append(edges) #plot the same range once
        '''

        #right axis (logarithmic scale)
        ax2 = ax1[row_num,0].twinx()
        color2 = "cadetblue"
        ax2.set_ylabel("log(dN/dlogDp) (cm⁻³)", color=color2, fontsize=8) 
        ax2.plot(x, y, color=color2, lw=1)
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(8)
            item.set_fontweight("bold")
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        #gaussian & logistic fit for both scales
        #we need time in days
        mode_times_days = []
        for i in mode_times_UTC:
            time_range = df.index.intersection(i[1]) #find matching indices
            indices = [df.index.get_loc(row) for row in time_range]
            mode_times_days.append(time_d_res[indices]) #define time in days again depending on chosen range

        
        #gaussian
        for params in fitting_params_gaus:
            diam, start_time, end_time, a, mu, sigma = params
            
            #change days into UTC
            start_time_UTC = pd.Series(days_into_UTC(start_time-0.25/24)).dt.round('30min')[0] 
            end_time_UTC = pd.Series(days_into_UTC(end_time-0.25/24)).dt.round('30min')[0]
            start_time_UTC += timedelta(minutes=15) #makes it easier to round
            end_time_UTC += timedelta(minutes=15)

            for time_UTC,time_days in zip(mode_times_UTC,mode_times_days):
                if diam == diameter_list[row_num] and time_UTC[1][0] == start_time_UTC and time_UTC[1][-1] == end_time_UTC: #check that plotting happens in the right mode
                    line2, = ax1[row_num,0].plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
                    ax2.plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
        
        #logistic
        for params in fitting_params_logi:
            diam, start_time, end_time, L, x0, k = params
            
            #change days into UTC
            start_time_UTC = pd.Series(days_into_UTC(start_time-0.25/24)).dt.round('30min')[0] 
            end_time_UTC = pd.Series(days_into_UTC(end_time-0.25/24)).dt.round('30min')[0]
            start_time_UTC += timedelta(minutes=15) #makes it easier to round
            end_time_UTC += timedelta(minutes=15)

            for time_UTC,time_days in zip(mode_times_UTC,mode_times_days):     
                if diam == diameter_list[row_num] and time_UTC[1][0] == start_time_UTC and time_UTC[1][-1] == end_time_UTC: #check that plotting happens in the right mode
                    line3, = ax1[row_num,0].plot(time_UTC[1], logistic(time_days,L,x0,k), '--', color="gold",lw=1.2)
                    ax2.plot(time_UTC[1], logistic(time_days,L,x0,k), '--', color="gold",lw=1.2)

        #maximum concentration
        for i in xy_maxcon:
            diam, x_maxcon, y_maxcon = i
            if diam == diameter_list[row_num]:
                line4, = ax1[row_num,0].plot(x_maxcon, y_maxcon, '*', color="white", ms=5, mew=0.6,alpha=0.8)
                ax2.plot(x_maxcon, y_maxcon, '*', color="white", ms=5, mew=0.6,alpha=0.8)

        #appearance time
        for i in appearances:
            diam, time, conc = i
            if diam == diameter_list[row_num]:
                line5, = ax1[row_num,0].plot(time, conc, '*', color="green", ms=5, mew=0.6,alpha=0.8)
                ax2.plot(time, conc, '*', color="green", ms=5, mew=0.6,alpha=0.8)
        
        ax1[row_num,0].set_xlim(dataframe.index[0],dataframe.index[-1])
        ax1[row_num,0].set_facecolor("lightgray")
        #ax1[row_num,0].xaxis.set_tick_params(rotation=30)
        ax1[row_num,0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ax1[row_num,0].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        if row_num == 0:
            ax1[row_num,0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    #PLOTS ON THE RIGHT
    df_1st_derivatives = cal_1st_derivative(df,time_days=time_d_res)

    x = df_1st_derivatives.index #time
    y_list = [df_1st_derivatives[diam] for diam in diameter_list] #concentrations

    #row_num keeps track of which row of figure we are plotting in
    for row_num, y in enumerate(y_list):
        if choose_GR != None:
            ax1[row_num,1].set_title(f'range: {choose_GR} nm/h', loc='left', fontsize=8) #range titles

        #left axis
        color1 = "royalblue"
        ax1[row_num,1].set_ylabel("d²N/dlogDpdt (cm⁻³s⁻¹)", color=color1, fontsize=8)
        ax1[row_num,1].plot(x, y, color=color1, lw=1)
        ax1[row_num,1].scatter(x, y, s=2, c=color1)
        
        for item in ([ax1[row_num,1].title, ax1[row_num,1].xaxis.label, ax1[row_num,1].yaxis.label] + ax1[row_num,1].get_xticklabels() + ax1[row_num,1].get_yticklabels()):
            item.set_fontsize(8)
            item.set_fontweight("bold")
        ax1[row_num,1].tick_params(axis='y', labelcolor=color1)
        line6 = ax1[row_num,1].axhline(y=threshold_deriv, color="royalblue", linestyle='--', lw=1)
        ax1[row_num,1].axhline(y= -threshold_deriv, color="royalblue", linestyle='--', lw=1)
        
        #start and end points of modes
        found_ranges = []
        for edges in mode_edges:
            diam, start, end = edges
            if diam == diameter_list[row_num] and edges not in found_ranges:
                line7 = ax1[row_num,1].axvspan(start, end, alpha=0.18, color='darkorange')
                found_ranges.append(edges) #plot the same range once              
        
        if draw_range_edges == True:
            #start and end points of ranges
            found_ranges = []
            for edges in range_edges:
                diam, start, end = edges
                if diam == diameter_list[row_num] and edges not in found_ranges:
                    line8 = ax1[row_num,1].axvspan(start, end, alpha=0.2, color='gray')
                    found_ranges.append(edges) #plot the same range once

        #maximum concentration
        for i in xy_maxcon:
            diam, x_maxcon, y_maxcon = i
            y_maxcon = y_maxcon*0 #to place the start lower where y = 0
            if diam == diameter_list[row_num]:
                ax1[row_num,1].plot(x_maxcon, y_maxcon, '*', color="white",ms=5, mew=0.6,alpha=0.8)  
        
        #appearance time
        for i in appearances:
            diam, time, conc = i
            conc = conc*0 #to place the start lower where y = 0
            if diam == diameter_list[row_num]:
                ax1[row_num,1].plot(time, conc, '*', color="green", ms=5, mew=0.6,alpha=0.8)
                ax2.plot(time, conc, '*', color="green", ms=5, mew=0.6,alpha=0.8)

        ax1[row_num,1].set_xlim(dataframe.index[1],dataframe.index[-1])
        ax1[row_num,1].set_ylim(-0.2,0.2)
        ax1[row_num,1].set_facecolor("lightgray")
        #ax1[row_num,1].xaxis.set_tick_params(rotation=30) #rotating x-axis labels
        ax1[row_num,1].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        if row_num == 0:
            ax1[row_num,1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        """ Logarithmic derivate
        #right axis
        ax2 = ax1[row_num,1].twinx()
        color2 = "green"
        ax2.set_ylabel("log10(d(dN/dlogDp)dt)", color=color2) 
        ax2.plot(x, y, color=color2,zorder=1)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        ax2.axhline(y=0.03, color='green', linestyle='--', lw=0.7, label="threshold = 0.03",zorder=1)
        ax2.axhline(y=-0.03, color='green', linestyle='--', lw=0.7,zorder=1)
        """


    ### FEW ADJUSTMENTS ###
    #common titles
    ax1[0,0].set_title("Concentration", fontsize=10, fontweight="bold") 
    ax1[0,1].set_title("Derivative", fontsize=10, fontweight="bold")
    ax1[len(diameter_list)-1,0].set_xlabel("time (h)", fontsize=8, fontweight="bold")
    ax1[len(diameter_list)-1,1].set_xlabel("time (h)", fontsize=8, fontweight="bold")
    
    #common legends
    #ax1[0,0].legend([line1,line2,line3,line4,line5],["mode edges","gaussian fit","logistic fit","maximum concentration","appearance time"],fancybox=False,framealpha=0.9,loc='upper center', bbox_to_anchor=(0.85, 1.8), fontsize=8)
    legend_1 = ax1[0,0].legend([line2,line3,line4,line5,line7],["gaussian fit","logistic fit","maximum concentration","appearance time","mode edges"],fancybox=False,framealpha=0.9, fontsize=4, loc="upper right")
    legend_1.remove() #to have the legend on top of graph lines
    legend_11 = ax2.add_artist(legend_1)
    
    if draw_range_edges == True:
        legend_2 = ax1[0,1].legend([line4,line5,line7,line8,line6],["maximum concentration","appearance time","mode edges","range",f"threshold = {str(threshold_deriv)}"],fancybox=False,framealpha=0.9, loc='upper right', fontsize=4)
    else:
        legend_2 = ax1[0,1].legend([line4,line5,line7,line6],["maximum concentration","appearance time","mode edges",f"threshold = {str(threshold_deriv)}"],fancybox=False,framealpha=0.9, loc='upper right', fontsize=4)
    
    #set black edges to star markers in the legend
    white_star11 = legend_11.legend_handles[2]
    green_star11 = legend_11.legend_handles[3]
    white_star2 = legend_2.legend_handles[0]
    green_star2 = legend_2.legend_handles[1]
    white_star11.set_markeredgewidth(0.4)
    green_star11.set_markeredgewidth(0.4)
    white_star11.set_markeredgecolor("black")
    green_star11.set_markeredgecolor("black")
    white_star2.set_markeredgewidth(0.4)
    green_star2.set_markeredgewidth(0.4)
    white_star2.set_markeredgecolor("black")
    green_star2.set_markeredgecolor("black")
    
    fig.tight_layout()
#plot_channel(df,[df.columns[7],df.columns[8]],choose_GR=None,draw_range_edges=True)

plt.show()

####################################################
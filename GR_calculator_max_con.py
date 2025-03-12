from GR_calculator_unfer_v2_modified import df_GR_final, file_names, ax
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use("Qt5Agg") #backend changes the plotting style
import matplotlib.dates as mdates
from operator import itemgetter
from collections import defaultdict
import statsmodels.api as sm
from sklearn import linear_model
from itertools import cycle
import warnings
from scipy.optimize import OptimizeWarning
import time

#supressing warnings for curve_fit to avoid crowding of terminal!!
warnings.simplefilter("ignore",OptimizeWarning) 
warnings.simplefilter("ignore",RuntimeWarning)

'''
This code assumes dmps data.
- 1st column (skipping the first 0 value):              time in days
- 2nd column (skipping the first 0 value):              total concentration in current timestamp (row)
- 1st row (skipping first two 0 values):                diameters in meters, ~3nm-1000nm
- 3rd column onwards till the end (under diameters):    concentrations, dN/dlog(dp)
'''

################# DATA FORMATTING #################
folder = r"./dmps Nesrine/" #folder where data files are stored, should be in the same directory as this code
file_names = file_names #copy paths from the file "GR_calculator_unfer_v2_modified"

#let's define two useful functions
def combine_data(): 
    '''
    Loads dmps data for given days.
    Returns one dataframe with all given data.
    '''
    dfs = [] #list of dataframes
    
    #load all given data files and save them in a list
    for i, file_name in enumerate(file_names):
        df = pd.DataFrame(pd.read_csv(folder + file_name,sep='\s+',engine='python'))
        
        #different dmps data files have slightly different diameter values although they represent the same diameter
        #name all other columns with the labels of the first one
        if i == 0:
            diameter_labels = df.columns
            
        df.rename(columns=dict(zip(df.columns, diameter_labels)), inplace=True)
        dfs.append(df)
    
    combined_data = pd.concat(dfs,axis=0,ignore_index=True) #combine dataframes
    return combined_data
def avg_filter(dataframe,resolution: int):
    '''
    Smoothens data in dataframe with average filter and given resolution (minutes), 
    i.e. takes averages in a window without overlapping ranges.
    Discards blocks of time with incomplete timestamps.
    Returns smoothened dataframe and new time in days for that dataframe.
    '''

    dataframe.index = dataframe.index.round('10min') #change timestamps to be exactly 10min intervals

    #if average is taken of less than 3 datapoints, neglect that datapoint
    full_time_range = pd.date_range(start = dataframe.index.min(), end = dataframe.index.max(), freq='10min')
    missing_timestamps = full_time_range.difference(dataframe.index) #missing timestamps
    blocks = pd.date_range(start = dataframe.index.min(), end=  dataframe.index.max(), freq=f'{resolution}min') #blocks of resolution
    
    dataframe = dataframe.resample(f'{resolution}min').mean() #change resolution and take average of values
    dataframe = dataframe.shift(1, freq=f'{int(resolution/2)}min') #set new timestamps to be in the middle of the new resolution

    #find irrelevant timestamps
    irrelevant_ts = []
    for timestamp in missing_timestamps:
        for i in range(len(blocks) - 1):
            if blocks[i] <= timestamp < blocks[i + 1]: #check if timestamp is in this 30min block
                irrelevant_ts.append(blocks[i] + pd.Timedelta(minutes=15))  #add 15 minutes to center the block

    #remove irrelevant timestamps
    for dp in dataframe.columns:
        for ts in irrelevant_ts:
            dataframe.loc[ts,dp] = np.nan #set nan value for irrelevant datapoints
            dataframe = dataframe.dropna() #remove these rows   

    return dataframe

df = combine_data()
df.rename(columns=dict(zip(df.columns[[0,1]], ["time (d)", "total number concentration (N)"])), inplace=True) #rename first two columns
df = df.drop(['total number concentration (N)'], axis=1) #drop total N concentrations from the dataframe as they're not needed

#days to UTC times
df["time (d)"] = df["time (d)"] - df["time (d)"][0] #substract first day
df["time (d)"] = pd.Series(mdates.num2date(df["time (d)"])).dt.tz_localize(None)

#set new UTC timestamps as indices
df.rename(columns={'time (d)': 'time (UTC)'}, inplace=True)
df['time (UTC)']=pd.to_datetime(df['time (UTC)'], format="%Y-%m-%d %H:%M:%S")
df.index=df['time (UTC)']
df = df.drop(['time (UTC)'], axis=1)

df.columns = pd.to_numeric(df.columns) * 10**9 #set numerical diameters as column headers, units from m to nm
df = avg_filter(df,resolution=30) #filtering

#with this we can check the format
#df.to_csv('./data_filtered.csv', sep=',', header=True, index=True, na_rep='nan')

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
def flatten_list(list):
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
def cal_mape(x,y,popt):
    x = np.array(x)
    y = np.array(y)
    absolute_error = np.abs(linear(x, *popt) - y)
    mape = np.mean(absolute_error / y) * 100
    return mape
def cal_derivative(dataframe):
    '''
    Calculates 1st derivatives between neighbouring datapoints. 
    Takes in dataframe and wanted range of time in days (in case of smaller dataframes).
    Returns dataframe with derivatives.
    '''
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns) 
    
    #convert time to days 
    time_days = mdates.date2num(dataframe.index)
    
    for i in dataframe.columns: 
        N = dataframe[i] #concentration
        time = time_days * 24 * 60 * 60 #change days to seconds
        dNdt = np.diff(N)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = dNdt #add calculated derivatives to dataframe
    return  df_derivatives

#define mathematical functions for fitting
def gaussian(x,a,x0,sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def logistic(x,L,x0,k): 
    return L / (1 + np.exp(-k*(x-x0))) 
def linear(x,k,b):
    return k*x + b
def logarithmic(x):
    return np.log(x)

#################### METHODS #######################

def find_modes(dataframe,df_deriv,threshold_deriv: float):
    '''
    Finds modes with derivative threshold.
    Takes a dataframe with concentrations and time (UTC), 
    dataframe with time derivatives of the concentrations
    and the wanted derivative threshold.
    Returns dataframe with found modes.
    '''
    #initialize variables
    df_modes = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns) 
    start_time = None 
    end_time = None

    #define a threshold that determines what is a high concentration
    threshold = df_deriv > threshold_deriv #checks which values surpass the threshold
    # #if range is not long enough to cover the whole horizontal more, dont 
    # for column in threshold:
    #     if threshold[column][:2].all():
    #         threshold.loc[:,column] = False

    #threshold.iloc[0] = False
    start_points = threshold & (~threshold.shift(1,fill_value=False)) #find start points (type: df)

    #shift start points one timestamp earlier
    start_points = start_points.shift(-1, fill_value=False)

    #iterate over diameter channels
    for diam in df_deriv.columns:
        #iterate over timestamps for each diameter channel
        for timestamp in df_deriv.index: 
            #starting point found for a horizontal "mode"
            if start_points.loc[timestamp,diam] == True: 
                start_time = timestamp
                
                #if there is already a defined horizontal "mode" at start time, avoid overlapping
                if not np.isnan(df_modes.loc[start_time,diam]):
                    continue
                
                #find end time after local maximum concentration 
                subset_end = dataframe.index[-1]
                df_subset = dataframe.loc[start_time:subset_end,diam] #subset from mode start to (mode start + 140mins) unless mode start is near the end of a range
                
                #find local concentration maxima
                df_subset_left = df_subset.shift(-1)
                df_subset_right = df_subset.shift(1)
                df_subset_maxima = (df_subset_left < df_subset) & (df_subset > df_subset_right)

                max_time_diff = timedelta(minutes=150) #max time difference between peaks to be concidered the same peak
                
                #make df with maxima
                df_only_maxima = df_subset_maxima[df_subset_maxima.values]
                
                #check if peaks are nearby and choose higher one
                for i in range(len(df_only_maxima)):
                    try:
                        max_time1 = df_only_maxima.index[i]
                        max_time2 = df_only_maxima.index[i+1]
                        max_conc1 = df_subset.loc[max_time1]
                        max_conc2 = df_subset.loc[max_time2]
                        
                        diff_mins = max_time2-max_time1
                        
                        if diff_mins <= max_time_diff and max_conc1 > max_conc2:
                            df_subset_maxima.loc[max_time2] = False
                            #df_only_maxima.values[i+1] = False
                        elif diff_mins <= max_time_diff and max_conc1 < max_conc2:
                            df_subset_maxima.loc[max_time1] = False
                            #df_only_maxima.values[i] = False
                    except IndexError:
                        break
                          
                #if no True values left dont bother finding horizontal mode
                if not df_subset_maxima.any():
                    break

                #choose closest maximum after start time
                for i in range(len(df_subset_maxima.values)):
                    if df_subset_maxima.index[i] > start_time and df_subset_maxima.values[i] == True:
                        closest_maximum = df_subset.values[i]
                        break
                
                start_conc = dataframe.loc[timestamp,diam]
                end_conc = ((closest_maximum + start_conc ) / 2) * 0.8  #(N_max + N_start)/2 * 0.8
                max_conc_time = df_subset.index[df_subset == closest_maximum].tolist()[0] #rough estimate of maximum concentration time 
                max_conc_time_i = dataframe.index.get_loc(max_conc_time) #index of max_conc_time
                
                #new ending limit to find end concentration time, same as max time difference for peaks
                try:
                    #new_subset_end = dataframe.index[max_conc_time_i + (max_conc_time_i - dataframe.index.get_loc(start_time))] 
                    new_subset_end = dataframe.index[max_conc_time_i + 5] 
                except IndexError:
                    new_subset_end = dataframe.index[-1] #limits to range aroung GRs
                
                end_conc_i = closest(dataframe.loc[max_conc_time:new_subset_end,diam],end_conc) #index of end point after peak
                end_time = dataframe.index[end_conc_i+max_conc_time_i] #end time found!
                
                #attempt to filter peaks that are not at least 1000 cm⁻³ higher than their min value
                min_conc = min(dataframe.loc[max_conc_time:end_time,diam])
                if closest_maximum - min_conc < 700:
                    continue

                #skip modes with same start or end time
                if not start_time == end_time:
                    #when a start/end time has been found fill df with concentrations between start and end time
                    subset = dataframe[diam].loc[start_time:end_time]
                    
                    #dont save modes that consist of less than 4 points
                    if len(subset.values) > 3:
                        df_modes.loc[subset.index,diam] = subset #fill dataframe
                
                # if diam == 35.640105:
                #     breakpoint()
                    
                #restart initial values
                start_time = None 
                end_time = None 
                
            #keep iterating over timestamps for starting points of horizontal "modes"
            else:
                continue
            
    #df_modes.to_csv('./find_modes.csv', sep=',', header=True, index=True, na_rep='nan')
    return df_modes, threshold_deriv
def find_ranges(): 
    '''
    Finds ranges around growth rates from previously calculated mode fitting data.
    Returns a list of dataframes with the wanted ranges and their growth rates.
    '''
    
    #pidennä ikkunaa kun loivempi??
    df_GR_values = df_GR_final #open calculated values in Gabi's code
    threshold = 0 #GR [nm/h], 0 = all growth rates

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
            start_time = start_time - timedelta(hours=4) #5 hours
            end_time = end_time + timedelta(hours=4)
            
            start_diam = start_diam / 1.5 #factor of 1.5
            end_diam = end_diam * 1.5

            #make a df with wanted range and add them to the list
            df_mfit_con = df[(df.index >= start_time) & (df.index <= end_time)]
            df_ranges.append(df_mfit_con[df_mfit_con.columns[(df_mfit_con.columns >= start_diam) & (df_mfit_con.columns <= end_diam)]])
            growth_rates.append(growth_rate) #save also the growth rates

    return df_ranges, growth_rates
df_ranges, growth_rates = find_ranges()

def maximum_concentration(df_modes): 
    '''
    Calculates the maximum concentration.
    Takes in dataframe from wanted area in the PSD.
    Returns:
    max_conc_time =         list of maximum concentration times (UTC)
    max_conc_diameter =     list of maximum concentration diameters (nm)
    max_conc =              list of maximum concentrations in corresponding datapoints
    maxcon_x_days =         list of time in days
    fitting_params =        list of gaussian fit parameters and more:
                            [[start time of mode UTC, end time of mode UTC,
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
    
    #times to days
    t_d = mdates.date2num(df_modes.index)

    #gaussian fit to every mode
    for dp_index in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for t_index in range(len(df_modes.index)):
            time = t_d[t_index]
            concentration = df_modes.iloc[t_index,dp_index] #find concentration values from the dataframe i.e range (y)
            
            if np.isnan(concentration) and len(y) > 2: #gaussian fit when all values of one mode have been added to the y list
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
                    popt,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma],bounds=((0,0,-np.inf),(np.max(y),np.inf,np.inf)))
                    if ((popt[1]>=x.max()) | (popt[1]<=x.min())): #checking that the peak is within time range
                        pass
                        #print("Peak outside range. Skipping.")
                    else:
                        max_conc_time = np.append(max_conc_time,popt[1]+x_min) #make a list of times with the max concentration time in each diameter
                        max_conc_diameter = np.append(max_conc_diameter,float(df_modes.columns[dp_index])) #make list of diameters with max concentrations
                        max_conc = np.append(max_conc,popt[0]+y_min) #maximum concentrations
                        
                        #create dfs with start and end points
                        df_mode_start = pd.DataFrame(np.nan, index=df_modes.index, columns=df_modes.columns)
                        df_mode_end = pd.DataFrame(np.nan, index=df_modes.index, columns=df_modes.columns)
                        
                        start_time_days = x[0]+x_min
                        end_time_days = x[-1]+x_min
                        
                        start_time_UTC = mdates.num2date(start_time_days-0.25/24).replace(tzinfo=None)
                        start_time_UTC = pd.Series(start_time_UTC).dt.round('30min')[0] #rounded to the nearest 30min increment
                        start_time_UTC += timedelta(minutes=15) #shift 15mins forward
                        end_time_UTC = mdates.num2date(end_time_days-0.25/24).replace(tzinfo=None)
                        end_time_UTC = pd.Series(end_time_UTC).dt.round('30min')[0]
                        end_time_UTC += timedelta(minutes=15)
                        
                        df_mode_start.loc[start_time_UTC,df_modes.columns[dp_index]] = y[0]+y_min #replace nan value with concentration value at start/end point
                        df_mode_end.loc[end_time_UTC,df_modes.columns[dp_index]] = y[-1]+y_min
                        
                        dfs_mode_start.append(df_mode_start) #add to list of dfs
                        dfs_mode_end.append(df_mode_end)
                        
                        #gaussian fit parameters with start and end times and minimum concentration of horizontal mode
                        fitting_params.append([start_time_UTC, end_time_UTC, popt[0]+y_min, popt[1]+x_min, popt[2]])                               
                except:
                    pass
                    #print("Diverges. Skipping.")
                                    
                x = [] #reset
                y = [] 
            elif not np.isnan(concentration): #separates mode values
                x = np.append(x,time)
                y = np.append(y,concentration)

    #convert time to UTC
    max_conc_time = [dt.replace(tzinfo=None) for dt in mdates.num2date(max_conc_time)]
    max_conc_time = np.array(max_conc_time)
    
    return max_conc_time, max_conc_diameter, max_conc, fitting_params, dfs_mode_start, dfs_mode_end
def appearance_time(df_modes):
    '''max_conc_time
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
    t_d = mdates.date2num(df_modes.index)

    #logistic fit to every mode
    for dp_index in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for t_index in range(len(df_modes.index)):
            time = t_d[t_index]
            concentration = df_modes.iloc[t_index,dp_index] #find concentration values from the dataframe (y)    

            if np.isnan(concentration) and len(y) > 2: #logistic fit when all values of one mode have been added to the y list 
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
                    popt,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma],bounds=((0,0,-np.inf),(np.max(y),np.inf,np.inf)))
                    if ((popt[1]>=x.max()) | (popt[1]<=x.min())): #checking that the peak is within time range
                        pass
                        #print("Peak outside range. Skipping.")
                    else:
                        max_conc_time = popt[1] #from gaussian fit

                        #limit x and y to values between start time of mode and maximum concentration time in mode
                        max_conc_index = closest(x, max_conc_time)
                        x_sliced = x[:max_conc_index+1]
                        y_sliced = y[:max_conc_index+1]     
                        
                        #logistic fit
                        if len(y_sliced) != 0:
                            #initial guess for parameters
                            L = popt[0] #maximum value of gaussian fit (concentration)
                            x0 = np.nanmean(x_sliced) #midpoint x value (appearance time)
                            k = 1.0 #growth rate

                            try:
                                popt,pcov = curve_fit(logistic,x_sliced,y_sliced,p0=[L,x0,k],bounds=((popt[0]*0.999,0,-np.inf),(popt[0],np.inf,np.inf)))
                                if ((popt[1]>=x_sliced.max()) | (popt[1]<=x_sliced.min())): #checking that the peak is within time range   
                                    pass
                                    #print("Peak outside range. Skipping.")
                                else:
                                    appear_time = np.append(appear_time,popt[1]+x_min) #make a list of times with the appearance time in each diameter
                                    appear_diameter = np.append(appear_diameter,float(df_modes.columns[dp_index])) 
                                    mid_conc.append((popt[0]+y_min)/2) #appearance time concentration (~50% maximum concentration), L/2, y_min to preserve shape of graph
                                    
                                    #convert time to UTC
                                    start_time_days = x[0]+x_min
                                    end_time_days = x[-1]+x_min
                                    
                                    start_time_UTC = mdates.num2date(start_time_days-0.25/24).replace(tzinfo=None)
                                    start_time_UTC = pd.Series(start_time_UTC).dt.round('30min')[0] #rounded to the nearest 30min increment
                                    start_time_UTC += timedelta(minutes=15) #shift 15mins forward
                                    end_time_UTC = mdates.num2date(end_time_days-0.25/24).replace(tzinfo=None)
                                    end_time_UTC = pd.Series(end_time_UTC).dt.round('30min')[0]
                                    end_time_UTC += timedelta(minutes=15)

                                    #logistic fit parameters with time range and minimum concentration of horizontal mode
                                    fitting_params.append([start_time_UTC, end_time_UTC, popt[0]+y_min, popt[1]+x_min, popt[2]])                                 
                            except:
                                pass
                                #print("Logistic diverges. Skipping.")
                except:
                    pass
                    #print("Gaussian diverges. Skipping.")
                
                x = [] #reset
                y = [] 
            elif not np.isnan(concentration): #separates mode values
                x = np.append(x,time)
                y = np.append(y,concentration)

    #convert time to UTC
    appear_time = [dt.replace(tzinfo=None) for dt in mdates.num2date(appear_time)]
    appear_time = np.array(appear_time)

    return appear_time, appear_diameter, mid_conc, fitting_params
def init_ranges(choose_range_i):
    '''
    Goes through all ranges and calculates the points for maximum concentration
    and appearance time methods along with other useful information.
    x = time, y = diameter, z = concentration
    '''
    maxcon_xyz = [] #maximum concentration
    maxcon_x, maxcon_y, maxcon_z = [], [], []
    mc_fitting_params = [] 
    
    appear_xyz = [] #appearance time
    appear_x, appear_y, appear_z = [], [], []
    at_fitting_params = []
    
    GRs_mc, GRs_at = [], [] #grs of ranges
    dfs_mode_start, dfs_mode_end = [], [] #start/end of mode
    range_indices_mc, range_indices_at = [], [] #ranges of points

    start_time = time.time()
    df_ranges, growth_rates = find_ranges()
    print("Ranges found! (1/4) "+"(%s seconds)" % (time.time() - start_time))
    
    #filter ranges if a specific range index is chosen
    if choose_range_i is not None:
        df_ranges = [df_ranges[choose_range_i]]
        growth_rates = [growth_rates[choose_range_i]]
        print("**************************")
        print("Mode fitting growth rates:",growth_rates)

    start_time = time.time()
    for i,(df_range,growth_rate) in enumerate(zip(df_ranges,growth_rates)): #go through every range around GRs
        #calculate derivative and define modes
        df_deriv = cal_derivative(df_range) 
        df_modes, threshold_deriv = find_modes(df_range,df_deriv,threshold_deriv=0.09) #threshold!!
        
        mc_x, mc_y, mc_z, mc_params, dfs_start, dfs_end = maximum_concentration(df_modes)
        at_x, at_y, at_z, at_params = appearance_time(df_modes)
        
        #add to lists
        if len(mc_x) > 0:
            maxcon_x.extend(mc_x) #maximum concentration
            maxcon_y.extend(mc_y)
            maxcon_z.extend(mc_z)
            mc_fitting_params.extend(mc_params)
            GRs_mc.append(growth_rate) #growth rates
            range_indices_mc.extend([i] * len(mc_x))
            
            dfs_mode_start.extend(dfs_start)
            dfs_mode_end.extend(dfs_end)

        if len(at_x) > 0:
            appear_x.extend(at_x) #appearance time
            appear_y.extend(at_y)
            appear_z.extend(at_z)
            at_fitting_params.extend(at_params)
            GRs_at.append(growth_rate)
            range_indices_at.extend([i] * len(at_x))
    print("Fitting done! (2/4) "+"(%s seconds)" % (time.time() - start_time))
    
    #combine to same lists
    maxcon_xyz = [maxcon_x,maxcon_y,maxcon_z,range_indices_mc] #add also which range it came from
    appear_xyz = [appear_x,appear_y,appear_z,range_indices_at]
    
    return maxcon_xyz, appear_xyz, df_ranges, mc_fitting_params, at_fitting_params, GRs_mc, GRs_at, dfs_mode_start, dfs_mode_end, threshold_deriv

xyz_maxcon, xyz_appear, df_ranges, *others = init_ranges(choose_range_i=None)

def filter_duplicates(xyz,mtd):
    '''
    Filters duplicates of maximum concentration and appearance time 
    methods if points are within mtd (maximum time difference).
    Returns lists of datapoints without duplicated.
    Picks point that is closest to the average distance of 
    dublicate points from one of the ranges.
    '''
    filt_i = [] #list for indices of removed points

    #filter points that are basically the same but found with a different range
    for diam in df.columns.values: #loop through diameter values     
        channel = [[xyz[0][j],xyz[1][j],xyz[2][j]] for j, d in enumerate(xyz[1]) if d == diam] #choose points that are in diam channel
        
        if len(channel) > 1: #if there are several points in one channel, check time difference
            max_time_diff = mtd
            
            #find groups of similar times in this diameter channel and add them to a list
            same_times_list = []
            for elem in channel: #loop through points in diam channel
                time = elem[0]
                same_times = [point[0] for point in channel if abs(point[0]-time) < timedelta(days=max_time_diff)] #list of times close to current time
                
                if same_times not in same_times_list and len(same_times) > 1: #length must be more than 1
                    same_times_list.append(same_times)

            #iterate over all groups of similar times
            for same_time in same_times_list:      
                #REMOVE DUPLICATES
                x, y, z, ranges = xyz

                #identify elements and their indices to extract
                mask_x = np.isin(x, same_time)
                extracted_indices = [i for i,j in enumerate(mask_x) if j == True]    

                #extract elements for calculations
                x = np.array(x)
                ranges = np.array(ranges)
                extracted_times = x[mask_x]
                extracted_ranges = ranges[mask_x]

                #find out which range of the two dots are closer to their average time 
                #take mean time of all dots and compare it to the mean time of the lines
                average_time = pd.to_datetime(pd.Series(extracted_times)).mean()
                df_GR_values = df_GR_final #use calculated values in Gabi's code

                #loop through the modefitting lines (ranges) where removed points where found in
                avg_line_location = []
                for index in extracted_ranges:
                    t_start = df_GR_values['start'][index]
                    t_end = df_GR_values['end'][index]
                    d_start = df_GR_values['d_initial'][index]
                    d_end = df_GR_values['d_final'][index]
                    
                    t_mean = t_start + (t_end-t_start)/2
                    d_mean = (d_start+d_end)/2
                    avg_line_location.append((t_mean,d_mean))

                #calculate which range (index) is closest to the dots
                closest_line_i = closest(avg_line_location,(average_time,diam))
                
                #define the indices of the points to be removed
                removed_indices = np.delete(extracted_indices,closest_line_i)

                #save them to a list
                for ix in removed_indices:
                    filt_i.append(ix)
               
    #filter duplicates
    xyz[0] = np.delete(xyz[0],filt_i)
    xyz[1] = np.delete(xyz[1],filt_i)
    xyz[2] = np.delete(xyz[2],filt_i)
    xyz[3] = np.delete(xyz[3],filt_i)
    
    return xyz, filt_i

xyz_maxcon, filt_i_mc = filter_duplicates(xyz_maxcon,mtd=90/(60*24)) 
xyz_appear, filt_i_at = filter_duplicates(xyz_appear,mtd=90/(60*24))
#mtd = maximum time difference between similar stars (in days)

################## GROWTH RATES ####################

def find_dots(times,diams):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with wanted times and diameters for plotting growth rates.
    
    Parameters:
    times (list): List of time values in days.
    diams (list): List of corresponding diameter values.
    
    Returns:
    list: Combined lists of nearby datapoints.
    '''
    #convert time to days
    times = mdates.date2num(times)
    
    #combine to the same list and sort data by diameter
    data_sorted = np.array(sorted(zip(times, diams), key=itemgetter(1,0))) #[[time1,diam1],[time2,diam2]...]

    data_pairs = []
    base_max_time_diff = 150/(60*24) #max time difference in days = 150mins = 2,5h
    higher_max_time_diff = 180/(60*24) #180mins = 3h
    
    #iterate through each datapoint to find suitable pairs of mode fitting datapoints
    for i, datapoint in enumerate(data_sorted):
        time0, diam0 = datapoint #current datapoint
        
        #iterate through diameter channels after current datapoint and look for the nearest datapoint in that channel
        for ii in range(1,3): #one step represents one diam channel, allows one channel in between
            #print(i,ii)
            #diameter for current iteration
            diam_channel = df.columns.values[df.columns.values >= diam0][ii]
            
            #search for datapoints in this diameter channel
            channel_points = [point for point in data_sorted if point[1] == diam_channel]
            if not channel_points: #skip if no datapoints in current channel
                continue
            
            #closest datapoint
            nearby_datapoint = tuple(min(channel_points, key=lambda point: abs(point[0] - time0)))
            time_diff = abs(nearby_datapoint[0] - time0)
            
            max_time_diff = base_max_time_diff #MAKE THIS BETTER LATER
    
            #check diameter difference
            if time_diff <= max_time_diff: 
                data_pairs.append([(datapoint[0],datapoint[1]),nearby_datapoint]) #add nearby datapoint pairs to list
                combined = combine_connected_pairs(data_pairs) #combine overlapping pairs to make lines
                combined = [sorted(sublist, key=lambda x: x[1]) for sublist in combined] #make sure datapoints in every line are sorted by diameter
                
                #make a linear fit to check mape for line with new datapoint
                iii, current_line = [(i,line) for i,line in enumerate(combined) if nearby_datapoint in line][0]
                
                if len(current_line) <= 2: #pointless to analyze mape with less than 3 datapoints
                    continue #proceed to next line
                elif len(current_line) >= 3:
                    x = [datapoint[1] for datapoint in current_line] #diams
                    y = [datapoint[0] for datapoint in current_line] #times
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                else:#THIS ISNT BEING USED CURRENTLY
                    #do fit to previous datapoints and calculate mape with the new datapoint included
                    x = [datapoint[1] for datapoint in current_line[:-1]] #diams (last datapoint excluded)
                    y = [datapoint[0] for datapoint in current_line[:-1]] #times (last datapoint excluded)
                    popt, pcov = curve_fit(linear, x, y)

                    #add newest datapoint back to x and y
                    x = [datapoint[1] for datapoint in current_line] #diams
                    y = [datapoint[0] for datapoint in current_line] #times
                    mape = cal_mape(x,y,popt) #!!!
                    
                    #check mape of the whole line also
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)


                mape_threshold = 7*len(current_line)**(-1) #15*x^(-1) #UUHM??
                #mape_threshold = 0.1*len(current_line)**(-1) #0.01*x^(-1)
                #if mape_threshold > 0.025:
                #    mape_threshold = 0.025
            
                #if len(current_line) == 11:
                #    print(mape)
                #print("threshold:",mape_threshold)    
                
                #print("mape:",mape, "threshold:",mape_threshold)
                if mape > mape_threshold:
                    #calculate mape without first datapoint of line to see if mape is smaller
                    x = [datapoint[1] for datapoint in current_line[1:]]
                    y = [datapoint[0] for datapoint in current_line[1:]] #times
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    
                    if mape > mape_threshold: #if mape is still too big
                        #delete recently added datapoint from both lists
                        combined = [[point for point in component if point != nearby_datapoint] for component in combined] 
                        data_pairs = [pair for pair in data_pairs if not any(np.array_equal(nearby_datapoint, point) for point in pair)]

                        x = [datapoint[1] for datapoint in current_line[:-1]]
                        y = [datapoint[0] for datapoint in current_line[:-1]] #times
                        popt, pcov = curve_fit(linear, x, y)
                        mape = cal_mape(x,y,popt) #update mape value 
                    else:
                        #remove the first datapoint of the line
                        first_datapoint = current_line[0]
                        combined[iii] = current_line[1:]
                        #remove also data pair with this datapoint
                        data_pairs = [pair for pair in data_pairs if not any(np.array_equal(first_datapoint, point) for point in pair)]
                
                '''
                #try splitting line into two parts from the middle to lower mape
                #does it when all lines haven't been found yet
                if len(combined[iii]) >= 8: #at least 8 datapoints needed
                    middle_index = int(len(combined[iii])/2) #rounding down with int()
                    line_1st_half = combined[iii][:middle_index]
                    #uneven numbers include one more element in the 2nd half
                    line_2nd_half = combined[iii][-middle_index:] if len(combined[iii])%2 == 0 else combined[iii][-(middle_index+1):]
                    
                    #calculate if mape lowered significantly
                    x = [datapoint[1] for datapoint in line_1st_half]
                    y = [datapoint[0] for datapoint in line_1st_half] #times
                    popt, pcov = curve_fit(linear, x, y)
                    mape1 = cal_mape(x,y,popt)
                    
                    x = [datapoint[1] for datapoint in line_2nd_half]
                    y = [datapoint[0] for datapoint in line_2nd_half] #times
                    popt, pcov = curve_fit(linear, x, y)
                    mape2 = cal_mape(x,y,popt)
                    
                    #relative change in mape
                    rel_diff1 = ((mape1-mape)/mape) * 100
                    rel_diff2 = ((mape2-mape)/mape) * 100
                    
                    #if mape improves (decreases) by 40% split line in two
                    if rel_diff1 <= -40 or rel_diff2 <= -40:
                        #remove the second half of current line and add it as its own line
                        combined[iii] = line_1st_half
                        combined.append(line_2nd_half)
                        
                        #remove from data_pairs also
                        connecting_pair = [line_1st_half[-1],line_2nd_half[0]]
                        data_pairs = [pair for pair in data_pairs if not np.array_equal(connecting_pair, pair)]
                    '''
                break
            else: #keep looking for next datapoint until end of points if the next one isnt suitable
                continue
    
    combined = [sorted(component, key=lambda x: x[1]) for component in combined] # Sort each individual component by diameter
    
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
                x = [datapoint[1] for datapoint in line] #diams
                y = [datapoint[0] for datapoint in line] #time
                popt, pcov = curve_fit(linear, x, y)
                mape = cal_mape(x,y,popt)
                GR = 1/(popt[0]*24) #growth rate

                #mape is not the same in different scales so here the threshold must be stricter
                #maximum mape 1% and GR is not bigger than +-15nm/h
                if mape <= 100 and abs(GR) <= 400:
                    filtered_datapoints.append(line)
                    break
                else:
                    break
                #else:
                #    line = line[:-i] #exclude last elements one by one
                #    i += 1
                    #print("deleted",line[-1])
                    #removed_points.append(line[-1])    #add removed datapoint to another list

    return filtered_datapoints  
def init_find():
    start_time = time.time()
    #find consequtive datapoints
    mc_data = find_dots(times= xyz_maxcon[0],diams= xyz_maxcon[1]) #maximum concentration
    at_data = find_dots(times= xyz_appear[0],diams= xyz_appear[1]) #appearance time
    
    #filter series of datapoints that are too short or with high deviation
    mc_filtered = filter_dots(mc_data)
    at_filtered = filter_dots(at_data)
    
    #extract times and diameters
    time_mc = [[seg[0] for seg in mc_segment] for mc_segment in mc_filtered]
    diam_mc = [[seg[1] for seg in mc_segment] for mc_segment in mc_filtered]
    time_at = [[seg[0] for seg in at_segment] for at_segment in at_filtered]
    diam_at = [[seg[1] for seg in at_segment] for at_segment in at_filtered]
    print("Dots found! (3/4) "+"(%s seconds)" % (time.time() - start_time))
    
    return time_mc, diam_mc, time_at, diam_at
    
#################### PLOTTING ######################

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

    #linear fit for comparison
    #lr = linear_model.LinearRegression().fit(diam, time) #x,y

    #robust fit
    rlm_HuberT = sm.RLM(time, sm.add_constant(diam), M=sm.robust.norms.HuberT()) #statsmodel robust linear model
    rlm_results = rlm_HuberT.fit()
    
    #predict data of estimated models
    t_rlm = rlm_results.predict(sm.add_constant(diam_linear))
    t_params = rlm_results.params

    if len(t_params) > 1: #sometimes lines with datapoints that all have the same diameter, REMOVE WHEN FILTERING IS FIXED
        #change days to UTC
        time_UTC_rlm = [dt.replace(tzinfo=None) for dt in mdates.num2date(t_rlm)]

        #growth rate annotation
        gr = 1/(t_params[1]*24) #unit to nm/h from time in days
        
        plt.plot(time_UTC_rlm, diam_linear,color=color,linewidth=2)
            
        midpoint_idx = len(diam_linear) // 2 #growth rate value
        midpoint_time = time_UTC_rlm[midpoint_idx]
        midpoint_value = diam[midpoint_idx]
        plt.annotate(f'{gr:.2f}', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
        
        #print("Statsmodel robus linear model results: \n",rlm_results.summary())
        #print("\nparameters: ",rlm_results.params)
        #print(help(sm.RLM.fit))
def linear_fit(time,diam):
    '''
    y = time in days
    x = diameters in nm
    Flipped as the error is in time.
    '''
    #linear least square fits
    #popt, pcov = curve_fit(linear, np.log(diam), time) #logarthmic diam
    popt, pcov = curve_fit(linear, diam, time) #linear diam
    gr = 1/(popt[0]*24) #unit to nm/h from time in days
    time_fit = popt[0]*np.array(diam) + popt[1]
    
    #convert days to UTC
    time_UTC = [dt.replace(tzinfo=None) for dt in mdates.num2date(time_fit)]

    #plotting
    plt.plot(time_UTC,diam,lw=3) #line
    
    midpoint_idx = len(time_fit) // 2 #growth rate value
    midpoint_time = time_UTC[midpoint_idx]
    midpoint_value = diam[midpoint_idx]
    plt.annotate(f'{gr:.2f} nm/h', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, fontweight='bold')

def plot_PSD(dataframe,draw_rectangles=False):
    '''
    Plots dots for maximum concentration and 
    appearance time methods.
    Additionally does adjustments to PSD.
    '''
    st = time.time()
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
        robust_fit(time_seg_mc, diam_seg_mc,"white")
        robust_fit(time_seg_at, diam_seg_at,"green")
    
    #range rectangles
    if draw_rectangles == True:
        for df_range in df_ranges:
            start_time = df_range.index[0]
            end_time = df_range.index[-1]
            start_diam = df_range.columns[0]
            end_diam = df_range.columns[-1]
            
            #convert to matplotlib date representation
            start_time = mdates.date2num(start_time)
            end_time = mdates.date2num(end_time)
            
            width = abs(start_time-end_time)
            height = abs(start_diam-end_diam)
            
            rect = patches.Rectangle((start_time, start_diam), width, height, linewidth=0.7, linestyle='--', edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    
    #adjustments to plot
    plt.legend(fontsize=9,fancybox=False,framealpha=0.9)
    for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
        legend_handle.set_markeredgewidth(0.5)
        legend_handle.set_markeredgecolor("black")
    
    plt.xlim(dataframe.index[0],dataframe.index[-1])
    plt.ylim(dataframe.columns[0],dataframe.columns[-1])
    plt.ylabel("diameter (nm)",fontsize=14) #add y-axis label
    plt.xlabel("time",fontsize=14) #add y-axis label
    ax.set_title(f'growth rate unit: [nm/h]', loc='right', fontsize=8) 
    
    print("Plotting done! (4/4) "+"(%s seconds)" % (time.time() - st))
plot_PSD(df,draw_rectangles=False)

def plot_channel(dataframe,diameter_list,choose_GR,draw_range_edges,choose_range_i):
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
    xyz_maxcon, xyz_appear, df_ranges, fitting_parameters_gaus, fitting_parameters_logi, \
        GRs_mc, GRs_at, dfs_mode_start, dfs_mode_end, threshold_deriv = init_ranges(choose_range_i=choose_range_i) 
    
    '''filter excess'''
    xyz_maxcon, filt_i_mc = filter_duplicates(xyz_maxcon,mtd=90/(60*24)) 
    xyz_appear, filt_i_at = filter_duplicates(xyz_appear,mtd=90/(60*24))
    #mtd = maximum time difference between similar stars (in days)
    
    #filter also other variables
    fitting_parameters_gaus = [j for i, j in enumerate(fitting_parameters_gaus) if i not in filt_i_mc]
    fitting_parameters_logi = [j for i, j in enumerate(fitting_parameters_logi) if i not in filt_i_at]
    GRs_mc = [j for i, j in enumerate(GRs_mc) if i not in filt_i_mc]
    GRs_at = [j for i, j in enumerate(GRs_at) if i not in filt_i_at]
    dfs_mode_start = [j for i, j in enumerate(dfs_mode_start) if i not in filt_i_mc]
    dfs_mode_end = [j for i, j in enumerate(dfs_mode_end) if i not in filt_i_mc]
    

    '''1.5 (if a specific range is wanted) choose range that is plotted and modify datasets accodingly'''
    if choose_GR != None: 
        gr_indices = [i for i, gr in enumerate(GRs_mc) if gr == choose_GR] #maximum concentration 
        xyz_maxcon[0] = [xyz_maxcon[0][i] for i in gr_indices]
        xyz_maxcon[1] = [xyz_maxcon[1][i] for i in gr_indices]
        dfs_mode_start = [dfs_mode_start[i] for i in gr_indices]
        dfs_mode_end = [dfs_mode_end[i] for i in gr_indices]
        xyz_maxcon[2] = [xyz_maxcon[2][i] for i in gr_indices]
        fitting_parameters_gaus = [fitting_parameters_gaus[i] for i in gr_indices]

        gr_indices = [i for i, gr in enumerate(GRs_at) if gr == choose_GR] #appearance time
        xyz_appear[0] = [xyz_appear[0][i] for i in gr_indices]
        xyz_appear[1] = [xyz_appear[1][i] for i in gr_indices]
        xyz_appear[2] = [xyz_appear[2][i] for i in gr_indices]
        fitting_parameters_logi = [fitting_parameters_logi[i] for i in gr_indices]

    '''2 define lists and their shapes'''
    mode_edges = []             #[(diameter,start_time (UTC),end_time (UTC)), ...]
    range_edges = []            #[(diameter,start_time (UTC),end_time (UTC)), ...]
    xy_maxcon =  []             #[(max con diameter, max con time (UTC), max con), ...]
    fitting_params_gaus = []    #[(max con diameter, mode start time UTC, mode end time UTC, *params...), ...]
    fitting_params_logi = []    #[(appearance time diameter, mode start time UTC, mode end time UTC, *params...), ...]
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
    ax1 = np.atleast_2d(ax1) #to avoid problems with plotting only one channel
    lines_and_labels = set() #later use for legends
    
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
        #gaussian
        for params in fitting_params_gaus:
            diam, start_time, end_time, a, mu, sigma = params

            for time_UTC in mode_times_UTC:
                if diam == diameter_list[row_num] and time_UTC[1][0] == start_time and time_UTC[1][-1] == end_time: #check that plotting happens in the right mode
                    #convert time to days
                    time_days = mdates.date2num(time_UTC[1])
                    
                    line2, = ax1[row_num,0].plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
                    lines_and_labels.add((line2,"gaussian fit"))
                    ax2.plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
        
        #logistic
        for params in fitting_params_logi:
            diam, start_time, end_time, L, x0, k = params

            for time_UTC in mode_times_UTC:     
                if diam == diameter_list[row_num] and time_UTC[1][0] == start_time and time_UTC[1][-1] == end_time: #check that plotting happens in the right mode
                    #convert time to days
                    time_days = mdates.date2num(time_UTC[1])
                    
                    line3, = ax1[row_num,0].plot(time_UTC[1], logistic(time_days,L,x0,k), '--', color="gold",lw=1.2)
                    lines_and_labels.add((line3,"logistic fit"))
                    ax2.plot(time_UTC[1], logistic(time_days,L,x0,k), '--', color="gold",lw=1.2)

        #maximum concentration
        for i in xy_maxcon:
            diam, x_maxcon, y_maxcon = i
            if diam == diameter_list[row_num]:
                line4, = ax1[row_num,0].plot(x_maxcon, y_maxcon, '*', color="white", ms=5, mew=0.6,alpha=0.8)
                lines_and_labels.add((line4,"maximum concentration"))
                ax2.plot(x_maxcon, y_maxcon, '*', color="white", ms=5, mew=0.6,alpha=0.8)

        #appearance time
        for i in appearances:
            diam, time, conc = i
            if diam == diameter_list[row_num]:
                line5, = ax1[row_num,0].plot(time, conc, '*', color="green", ms=5, mew=0.6,alpha=0.8)
                lines_and_labels.add((line5,"appearance time"))
                ax2.plot(time, conc, '*', color="green", ms=5, mew=0.6,alpha=0.8)
        
        ax1[row_num,0].set_xlim(dataframe.index[0],dataframe.index[-1])
        ax1[row_num,0].set_facecolor("lightgray")
        #ax1[row_num,0].xaxis.set_tick_params(rotation=30)
        ax1[row_num,0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ax1[row_num,0].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        #if row_num == 0:
        #    ax1[row_num,0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    #PLOTS ON THE RIGHT
    df_1st_derivatives = cal_derivative(df)

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
        lines_and_labels.add((line6,f"threshold = {str(threshold_deriv)}"))
        
        #start and end points of modes
        found_ranges = []
        for edges in mode_edges:
            diam, start, end = edges
            if diam == diameter_list[row_num] and edges not in found_ranges:
                line7 = ax1[row_num,1].axvspan(start, end, alpha=0.18, color='darkorange')
                lines_and_labels.add((line7,"mode edges"))
                found_ranges.append(edges) #plot the same range once              
        
        if draw_range_edges == True:
            #start and end points of ranges
            found_ranges = []
            for edges in range_edges:
                diam, start, end = edges
                if diam == diameter_list[row_num] and edges not in found_ranges:
                    line8 = ax1[row_num,1].axvspan(start, end, alpha=0.2, color='gray')
                    lines_and_labels.add((line8,"range"))
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
        #if row_num == 0:
        #    ax1[row_num,1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        

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
    #unzip the valid entries into separate lists for the legend
    if len(lines_and_labels) <= 2:
        return print("This diameter channel(s) has no fits!\n**************************")


    #filter duplicates
    lines_and_labels = {entry[1]: entry for entry in lines_and_labels}
    lines_and_labels = set(lines_and_labels.values())

    #print(lines_and_labels)
    #left plots
    lines_and_labels1 = [elem for elem in lines_and_labels if elem[1] in ["gaussian fit","logistic fit","maximum concentration","appearance time","mode edges"]]
    valid_lines, valid_labels = zip(*lines_and_labels1)
    legend_1 = ax1[0, 0].legend(valid_lines, valid_labels, fancybox=False, framealpha=0.9, fontsize=4, loc="upper right")
    
    #right plots
    if draw_range_edges == True:
        lines_and_labels2 = [elem for elem in lines_and_labels if elem[1] in ["maximum concentration","appearance time","mode edges","range",f"threshold = {str(threshold_deriv)}"]]
    else:
        lines_and_labels2 = [elem for elem in lines_and_labels if elem[1] in ["maximum concentration","appearance time","mode edges",f"threshold = {str(threshold_deriv)}"]]
        
    valid_lines, valid_labels = zip(*lines_and_labels2)
    legend_2 = ax1[0, 1].legend(valid_lines, valid_labels, fancybox=False, framealpha=0.9, fontsize=4, loc="upper right")
    legend_1.remove() #to have the legend on top of graph lines
    legend_11 = ax2.add_artist(legend_1)
    
    
    #set black edges to star markers in the legend
    #indices of right legend handles
    white_star11_i = [i for i,line_label in enumerate(lines_and_labels1) if line_label[1] == "maximum concentration"]
    green_star11_i = [i for i,line_label in enumerate(lines_and_labels1) if line_label[1] == "appearance time"]
    white_star2_i = [i for i,line_label in enumerate(lines_and_labels2) if line_label[1] == "maximum concentration"]
    green_star2_i = [i for i,line_label in enumerate(lines_and_labels2) if line_label[1] == "appearance time"]
    #legend handles
    white_star11 = [legend_11.legend_handles[i] for i in white_star11_i]
    green_star11 = [legend_11.legend_handles[i] for i in green_star11_i]
    white_star2 = [legend_2.legend_handles[i] for i in white_star2_i]
    green_star2 = [legend_2.legend_handles[i] for i in green_star2_i]
    #setting black edges
    stars = [white_star11,green_star11,white_star2,green_star2]
    for star in stars:
        if star:
            star[0].set_markeredgewidth(0.4)
            star[0].set_markeredgecolor("black")

    
    #plot line when day changes
    new_day = None
    for i in dataframe.index:
        day = i.strftime("%d")  
        if day != new_day:
            i = i - timedelta(minutes=15) #shift back 15mins due to resolution change
            for row_num in range(len(y_list)):
                ax1[row_num,0].axvline(x=i, color='black', linestyle='-', lw=0.8)
                ax1[row_num,1].axvline(x=i, color='black', linestyle='-', lw=0.8)
        new_day = day
    
    fig.tight_layout()
    
    print("Drawing diameter channel(s):",diameter_list)
plot_channel(df,diameter_list=[df.columns[14],df.columns[15]],choose_GR=None,draw_range_edges=True,choose_range_i=None)
plt.show()

####################################################
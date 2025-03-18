from GR_calculator_unfer_v2_modified import file_names, ax
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from scipy.optimize import curve_fit
from matplotlib import use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from operator import itemgetter
from collections import defaultdict
import statsmodels.api as sm
from sklearn import linear_model
from warnings import simplefilter
from scipy.optimize import OptimizeWarning
from time import time

#supressing warnings for curve_fit to avoid crowding of terminal!!
simplefilter("ignore",OptimizeWarning) 
simplefilter("ignore",RuntimeWarning)

#backend changes the interface of plotting
use("Qt5Agg") 

'''
This code assumes dmps data.
- 1st column (skipping the first 0 value):              time in days
- 2nd column (skipping the first 0 value):              total concentration in current timestamp (row)
- 1st row (skipping first two 0 values):                diameters in meters, ~3nm-1000nm
- 3rd column onwards till the end (under diameters):    concentrations, dN/dlog(dp)
'''

init_plot_channel = True #True to plot channels
channel_indices = [9] #Indices of diameter channels, 1=small

## parameters ##

#find_modes
maximum_time_difference = 2.5 #hours (between two peaks)
minimum_conc_difference = 700 #cm^(-3) (between starting conc and peak conc)
derivative_threshold = 330 #cm^(-3)/h (starting points of horizontal modes)

#find_dots

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
    [ d( dN/dlog(Dp) )/dt ] = cm⁻³/h
    '''
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns) 
    
    #convert time to days 
    time_days = mdates.date2num(dataframe.index)
    
    for i in dataframe.columns: 
        N = dataframe[i] #concentration
        time = time_days * 24 #change days to hours
        dNdt = np.diff(N)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = dNdt #add calculated derivatives to dataframe
    return df_derivatives

#define mathematical functions for fitting
def gaussian(x,a,x0,sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def logistic(x,L,x0,k): 
    return L / (1 + np.exp(-k*(x-x0))) 
def linear(x,k,b):
    return k*x + b

#################### METHODS #######################

def find_modes(dataframe,df_deriv,mtd,mcd,threshold_deriv):
    '''
    Finds modes with derivative threshold.
    Takes a dataframe with concentrations and time (UTC), 
    dataframe with time derivatives of the concentrations
    and the wanted derivative threshold.
    Returns dataframe with found modes.
    mtd = maximum time difference between peaks to be considered the same horizontal "mode"
    mcd = minimum concentration difference between starting concentration and peak concentration
    '''
    #initialize variables
    #df_modes = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns) 
    df_modes = pd.DataFrame()

    #define a threshold that determines what is a high concentration
    threshold = df_deriv > threshold_deriv #checks which values surpass the threshold

    #threshold.iloc[0] = False
    start_points = threshold & (~threshold.shift(1,fill_value=False)) #find start points (type: df)

    #shift start points one timestamp earlier
    start_points = start_points.shift(-1, fill_value=False)
    
    #find local concentration maxima
    df_left = dataframe.shift(-1)
    df_right = dataframe.shift(1)
    df_maxima = (df_left < dataframe) & (dataframe > df_right)

    max_time_diff = timedelta(hours=mtd) #max time difference between peaks to be concidered the same peak

    #iterate over diameter channels
    for diam in df_deriv.columns:
        #subset of start points
        start_points_only = start_points[diam][start_points[diam].values]
        start_times = start_points_only.index
        end_time = None
        
        #loop through start times
        for start_time in start_times:
  
            try:
                if start_time < end_time:
                    continue
            except TypeError:
                pass
            
            
            #find end time after local maximum concentration 
            #subset_end = dataframe.index[-1]  
            try:
                subset_end = start_time + timedelta(hours=24) #one day ahead
            except IndexError:
                subset_end = dataframe.index[-1]  
                
            df_subset = dataframe.loc[start_time:subset_end,diam]
            df_subset_maxima = df_maxima.loc[start_time:subset_end,diam]
            
            #make df with maxima
            df_subset_only_maxima = df_subset_maxima[df_subset_maxima.values]
            
            #check if peaks are nearby and choose higher one
            for i in range(len(df_subset_only_maxima)):
                try:
                    max_time1 = df_subset_only_maxima.index[i]
                    max_time2 = df_subset_only_maxima.index[i+1]
                    max_conc1 = dataframe.loc[max_time1,diam]
                    max_conc2 = dataframe.loc[max_time2,diam]
                    
                    diff_mins = max_time2-max_time1
                    
                    if diff_mins <= max_time_diff and max_conc1 > max_conc2:
                        df_maxima.loc[max_time2,diam] = False
                        df_subset_maxima.loc[max_time2] = False
                    elif diff_mins <= max_time_diff and max_conc1 < max_conc2:
                        df_maxima.loc[max_time1,diam] = False
                        df_subset_maxima.loc[max_time1] = False
                except IndexError:
                    break
            
            #if no True values left dont bother finding horizontal mode
            if not df_subset_maxima.any():
                continue

            #choose closest maximum after start time
            for i in range(len(df_subset_maxima.values)):
                if df_subset_maxima.index[i] > start_time and df_subset_maxima.values[i] == True:
                    closest_maximum = df_subset.values[i]
                    break
            
            start_conc = dataframe.loc[start_time,diam]
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
            min_conc = min(dataframe.loc[start_time:max_conc_time,diam])
            if closest_maximum - min_conc < mcd:
                continue

            #skip modes with same start or end time
            if not start_time == end_time:
                #when a start/end time has been found fill df with concentrations between start and end time
                subset = dataframe[diam].loc[start_time:end_time]
                
                average_conc = sum(subset.values) / len(subset)
                
                #dont save modes that consist of less than 4 points
                if len(subset.values) > 3 and average_conc > start_conc:
                    #df_modes.loc[subset.index,diam] = subset #fill dataframe
                    new_row = pd.DataFrame({"start_time": [start_time], "end_time": [end_time], "diameter": [diam]})
                    
                    df_modes = pd.concat([df_modes,new_row],ignore_index=True)
                    
                    # if diam == 7.5626058:
                    #     breakpoint()
       
    #df_modes.to_csv('./find_modes.csv', sep=',', header=True, index=True, na_rep='nan')
    return df_modes, threshold_deriv
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

    #extract values from dataframe
    mode_values = df_modes.values
    start_times = mode_values[:,0]
    end_times = mode_values[:,1]
    mode_diams = mode_values[:,2]
    
    #gaussian fit to every horizontal area of growth
    for i in range(len(start_times)):
        start_time = start_times[i]
        end_time = end_times[i]
        diam = mode_diams[i]
        
        #find values from the dataframe
        subset = df.loc[start_time:end_time,diam]
        x = subset.index #time
        x = mdates.date2num(x) #time in days
        y = subset.values #concentration
    
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
                #save results to lists
                max_conc_time = np.append(max_conc_time,popt[1]+x_min) #CHANGE TO JUST APPEND???
                max_conc_diameter = np.append(max_conc_diameter,diam)
                max_conc = np.append(max_conc,popt[0]+y_min)
                
                #gaussian fit parameters with start and end times and minimum concentration of horizontal mode
                fitting_params.append([start_time, end_time, popt[0]+y_min, popt[1]+x_min, popt[2]])                               
        except:
            pass
            #print("Diverges. Skipping.")

    #convert time to UTC
    max_conc_time = [dt.replace(tzinfo=None) for dt in mdates.num2date(max_conc_time)]
    max_conc_time = np.array(max_conc_time)
    
    return max_conc_time, max_conc_diameter, max_conc, fitting_params, start_times, end_times, mode_diams
def appearance_time(df_modes):
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
    
    #extract values from dataframe
    mode_values = df_modes.values
    start_times = mode_values[:,0]
    end_times = mode_values[:,1]
    mode_diams = mode_values[:,2]
    
    #logistic fit to every horizontal area of growth
    for i in range(len(start_times)):
        start_time = start_times[i]
        end_time = end_times[i]
        diam = mode_diams[i]
        
        #find values from the dataframe
        subset = df.loc[start_time:end_time,diam]
        x = subset.index #time
        x = mdates.date2num(x) #time in days
        y = subset.values #concentration
    
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

                    try: #logistic fit
                        popt,pcov = curve_fit(logistic,x_sliced,y_sliced,p0=[L,x0,k],bounds=((popt[0]*0.999,0,-np.inf),(popt[0],np.inf,np.inf)))
                        if ((popt[1]>=x_sliced.max()) | (popt[1]<=x_sliced.min())): #checking that the peak is within time range   
                            pass
                            #print("Peak outside range. Skipping.")
                        else:
                            appear_time = np.append(appear_time,popt[1]+x_min) #make a list of times with the appearance time in each diameter
                            appear_diameter = np.append(appear_diameter,diam) 
                            mid_conc.append((popt[0]+y_min)/2) #appearance time concentration (~50% maximum concentration), L/2, y_min to preserve shape of graph

                            #logistic fit parameters with time range and minimum concentration of horizontal mode
                            fitting_params.append([start_time, end_time, popt[0]+y_min, popt[1]+x_min, popt[2]])                                 
                    except:
                        pass
                        #print("Logistic diverges. Skipping.")                            
        except:
            pass
            #print("Diverges. Skipping.")

    #convert time to UTC
    appear_time = [dt.replace(tzinfo=None) for dt in mdates.num2date(appear_time)]
    appear_time = np.array(appear_time)

    return appear_time, appear_diameter, mid_conc, fitting_params
def init_methods(dataframe,mtd,mcd,threshold_deriv):
    '''
    Goes through all ranges and calculates the points for maximum concentration
    and appearance time methods along with other useful information.
    x = time, y = diameter, z = concentration
    '''
    maxcon_xyz = [] #maximum concentration
    appear_xyz = [] #appearance time

    start_time = time()
    #calculate derivative and define modes
    df_deriv = cal_derivative(dataframe) 
    df_modes, threshold_deriv = find_modes(dataframe,df_deriv,mtd,mcd,threshold_deriv) #threshold!!
    
    #methods
    mc_x, mc_y, mc_z, mc_params, mode_starts, mode_ends, mode_diams = maximum_concentration(df_modes)
    at_x, at_y, at_z, at_params = appearance_time(df_modes)
    print("Fitting done! (2/4) "+"(%s seconds)" % (time() - start_time))
    
    #combine to same lists
    maxcon_xyz = [mc_x,mc_y,mc_z]
    appear_xyz = [at_x,at_y,at_z]
    
    return maxcon_xyz, appear_xyz, mc_params, at_params, mode_starts, mode_ends, mode_diams, threshold_deriv
xyz_maxcon, xyz_appear, *others = init_methods(df,mtd=maximum_time_difference,mcd=minimum_conc_difference,threshold_deriv=derivative_threshold)

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
    start_time = time()
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
    print("Dots found! (3/4) "+"(%s seconds)" % (time() - start_time))
    
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

def plot_PSD(dataframe):
    '''
    Plots dots for maximum concentration and 
    appearance time methods.
    Additionally does adjustments to PSD.
    '''
    st = time()
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
    
    print("Plotting done! (4/4) "+"(%s seconds)" % (time() - st))
plot_PSD(df)

def plot_channel(dataframe,diameter_list_i,mtd,mcd,threshold_deriv):
    '''
    Plots chosen diameter channels over UTC time, with thresholds and gaussian fit.
    ax[0,0] = whole channel over time, with ranges      ax[0,1] = derivative of concentrations
    ax[n,0] = n amount of channels                      ax[n,1] = derivative of concentrations
                                                ...
    Inputs dataframe with data and diameters (numerical).
    '''   
    
    '''1 assemble all datasets'''
    xyz_maxcon, xyz_appear, fitting_parameters_gaus, fitting_parameters_logi, \
        mode_starts, mode_ends, mode_diams, threshold_deriv = init_methods(dataframe,mtd,mcd,threshold_deriv)

    '''2 define lists and their shapes'''
    mode_edges = []             #[(diameter,start_time (UTC),end_time (UTC)), ...]
    xy_maxcon =  []             #[(max con diameter, max con time (UTC), max con), ...]
    fitting_params_gaus = []    #[(max con diameter, mode start time UTC, mode end time UTC, *params...), ...]
    fitting_params_logi = []    #[(appearance time diameter, mode start time UTC, mode end time UTC, *params...), ...]
    mode_times = []         #[(diameter, time UTC), ...]
    appearances = []            #[(diameter, time (UTC), concentration), ...] 

    '''3 find data in datasets with chosen diameters'''
    diameter_list = [df.columns[i] for i in diameter_list_i]
    
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
        
        #MODE EDGES
        indices = [i for i, a in enumerate(mode_diams) if a == diam]
        mode_edges_diam = [(mode_diams[b],mode_starts[b],mode_ends[b]) for b in indices]
        [mode_edges.append(i) for i in mode_edges_diam]
        
        #MODE TIMES
        for diam, start, end in mode_edges_diam:
            subset = df.loc[start:end,diam]
            mode_t = subset.index
            mode_times.append((diam,mode_t))

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

        #gaussian fit 
        for params in fitting_params_gaus:
            diam, start_time, end_time, a, mu, sigma = params

            for time_UTC in mode_times:
                if diam == diameter_list[row_num] and time_UTC[1][0] == start_time and time_UTC[1][-1] == end_time: #check that plotting happens in the right mode
                    #convert time to days
                    time_days = mdates.date2num(time_UTC[1])
                    
                    line2, = ax1[row_num,0].plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
                    lines_and_labels.add((line2,"gaussian fit"))
                    ax2.plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
        
        #logistic fit
        for params in fitting_params_logi:
            diam, start_time, end_time, L, x0, k = params

            for time_UTC in mode_times:     
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
        
        #start and end points of modes
        found_ranges = []
        for edges in mode_edges:
            diam, start, end = edges
            if diam == diameter_list[row_num] and edges not in found_ranges:
                line7 = ax1[row_num,1].axvspan(start, end, alpha=0.18, color='darkorange')
                lines_and_labels.add((line7,"mode edges"))
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
        ax1[row_num,1].set_ylim(-50,threshold_deriv*1.8)
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
    lines_and_labels2 = [elem for elem in lines_and_labels if elem[1] in ["maximum concentration","appearance time","mode edges",f"threshold = {str(threshold_deriv)} cm⁻³/h"]]
        
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

if init_plot_channel:
    plot_channel(df,diameter_list_i=channel_indices,mtd=maximum_time_difference,mcd=minimum_conc_difference,threshold_deriv=derivative_threshold)

plt.show()
####################################################
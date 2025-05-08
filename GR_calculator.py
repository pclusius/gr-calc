from GR_calculator_unfer_v2_mod import file_names, ax
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
import aerosol.functions as af #janne's aerosol functions
import history.peak_fitting as pf
import json 
import xarray as xr

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

## parameters ##

#find_modes
maximum_peak_difference = 2 #hours (time between two peaks in smoothed data (window 3))
derivative_threshold = 200 #cm^(-3)/h (starting points of horizontal peak areas, determines what is a high concentration) (NOTICE: concentration diff is half of this between timesteps as the resolution is 30min)

#find_dots
show_mae = False #show mae values of lines instead of growth rates, unit hours
maximum_time_difference_dots = 2.5 #hours (between current and nearby point)
mae_threshold_factor = 2 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
gr_precentage_error_threshold = 50 #% (precentage error of growth rates when adding new points to gr lines)

#channel plotting
init_plot_channel = False #True to plot channels
channel_indices = [5] #Indices of diameter channels, 1=small
show_start_times_and_maxima = True #True to show all possible start times of peak areas (black arrow) and maximas associated

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
def cal_mape(x,y,popt):
    x = np.array(x)
    y = np.array(y)
    y_predicted = linear(x, *popt)
    absolute_error = np.abs(y - y_predicted)
    mape = np.mean(absolute_error / y) * 100
    return mape
def cal_mae(x,y,popt):
    """
    Calculates mean absolute error (MAE).
    Result in hours.
    """
    x = np.array(x)
    y = np.array(y)
    y_predicted = linear(x, *popt)
    mean_absolute_error = np.mean(np.abs(y - y_predicted)) * 24
    return mean_absolute_error
def cal_mase(x,y,popt):
    """
    Problems:
    - dividing by deviation allows for lines that are long in time to have more deviation
    - error is smaller the bigger the MAD is...
    """
    x = np.array(x)
    y = np.array(y)
    y_predicted = linear(x, *popt)
    mean_absolute_error = np.mean(np.abs(y - y_predicted))
    mean_absolute_deviation = np.mean(np.abs(y - np.mean(y)))
    mase = np.mean(mean_absolute_error/mean_absolute_deviation)
    return mase   
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
def average_filter(dataframe,window):         
    '''
    Smoothens data in dataframe with averafe filter and given window.
    Returns smoothened dataframe.
    '''
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

#################### METHODS #######################
''' jannen koodista
#change time from dates to days
df.index = mdates.date2num(df.index)
fit_results = pf.fit_multimodes(df)

#back to dates
df.index = [dt.replace(tzinfo=None) for dt in mdates.num2date(df.index)]

#write json data to a file
with open('fit_results.json', 'w') as output_file:
	json.dump(fit_results, output_file, indent=2)

#load the results
with open("fit_results.json") as file:
    fits = json.load(file)

#making a dataframe from json file
rows_list = []
for diam_channel in fits:
    dp = diam_channel['diameter']
    peak_ts = diam_channel['peak_timestamps']
    
    for i, gaussians in enumerate(diam_channel['gaussians']):
        mean = gaussians['mean']
        sigma = gaussians['sigma']
        amplitude = gaussians['amplitude']

        dict_row = {'diameter':dp,'amplitude':amplitude,'peak_time':peak_ts[i],'sigma':sigma} #diam unit nm
        rows_list.append(dict_row)

df_fits = pd.DataFrame(rows_list)  

#diameters to index, peak_time days to datetime objects
df_fits['peak_time'] = [dt.replace(tzinfo=None) for dt in mdates.num2date(df_fits['peak_time'])]
df_fits.index=df_fits['diameter']
df_fits = df_fits.drop(['diameter'], axis=1)
'''

def find_modes(dataframe,df_deriv,mpd,threshold_deriv):
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
    df_modes = pd.DataFrame()
    start_times_list = []
    maxima_list = []

    #define a boolean dataframe where the derivative threshold has been surpassed
    df_surpassed = df_deriv > threshold_deriv

    #find start points and shift them one timestamp earlier
    df_start_points = df_surpassed & (~df_surpassed.shift(1,fill_value=False)) 
    df_start_points = df_start_points.shift(-1, fill_value=False)

    #find local concentration maxima
    df_left = dataframe.shift(-1)
    df_right = dataframe.shift(1)
    df_maxima = (df_left < dataframe) & (dataframe > df_right)

    max_peak_diff = timedelta(hours=mpd) #max time difference between peaks to be considered the same peak

    #iterate over diameter channels
    for diam in df_deriv.columns:
        
        #define start times and end time
        df_start_points_only = df_start_points[diam][df_start_points[diam].values]
        start_times = df_start_points_only.index
        end_time = None
        
        #save for use in channel plotting
        start_concs = []
        for start_time in start_times:
            start_conc = dataframe.loc[start_time,diam]
            start_concs.append(start_conc)
        start_times_list.append((diam,start_times,start_concs)) 
        
        #loop through start times
        for start_time in start_times:
            try:
                if start_time < end_time:
                    continue
            except TypeError:
                pass
            
            #find end time after local maximum concentration 
            try:
                subset_end = start_time + timedelta(hours=10) #10 hours ahead
            except IndexError:
                subset_end = dataframe.index[-1]  
                
            df_subset = dataframe.loc[start_time:subset_end,diam]
            df_subset_maxima = df_maxima.loc[start_time:subset_end,diam].copy()

            #check if channel has any peaks
            if not df_subset_maxima.any():
                continue
            
            #make df with maxima only
            df_subset_only_maxima = df_subset_maxima[df_subset_maxima.values].copy()

            #check if peaks are nearby and choose higher one
            for i in range(len(df_subset_only_maxima)):
                try:
                    max_time1 = df_subset_only_maxima.index[i]
                    max_time2 = df_subset_only_maxima.index[i+1]
                    max_conc1 = dataframe.loc[max_time1,diam]
                    max_conc2 = dataframe.loc[max_time2,diam]
                    
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
            maximum_concs = []
            maxima_times = df_subset_only_maxima.index
            for max_time in maxima_times:
                max_conc = dataframe.loc[max_time,diam]
                maximum_concs.append(max_conc)
            maxima_list.append((diam,maxima_times,maximum_concs)) 

            #choose closest maximum after start time
            for i in range(len(df_subset_maxima.values)):
                if df_subset_maxima.index[i] > start_time and df_subset_maxima.values[i] == True:
                    closest_maximum = df_subset.values[i]
                    break
            else: #if no maxima after start time skip this start time
                break
            
            #define concentration threshold for ending point
            min_conc = np.nanmin(dataframe[diam].values) #global minimum
            end_conc = (closest_maximum + min_conc) * 0.5 #(N_max + N_min)/2 

            #estimate time for maximum concentration
            max_conc_time = df_subset.index[df_subset == closest_maximum].tolist()[0] #rough estimate of maximum concentration time 
            max_conc_time_i = dataframe.index.get_loc(max_conc_time) #index of max_conc_time
            
            #define subset from this time to end
            df_subset_maxcon = dataframe.loc[max_conc_time:subset_end,diam] 
            
            #iterate over concentrations after maximum to find ending time
            for i, (time,conc) in enumerate(df_subset_maxcon.items()):

                #check for another maximum along the way
                if i != 0 and df_subset_maxima.loc[time]:
                    step_before_other_peak = df_subset_maxcon.index[i-1]
                    end_conc = min(df_subset_maxcon.loc[max_conc_time:step_before_other_peak]) #end point after peak
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
            subset = dataframe[diam].loc[start_time:end_time]
            if len(subset.values) > 3:
                #df_modes.loc[subset.index,diam] = subset #fill dataframe
                new_row = pd.DataFrame({"start_time": [start_time], "end_time": [end_time], "diameter": [diam]})
                df_modes = pd.concat([df_modes,new_row],ignore_index=True)
                    
    df_modes.to_csv('./find_modes.csv', sep=',', header=True, index=True, na_rep='nan')
    return df_modes, threshold_deriv, start_times_list, maxima_list
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
    mode_edges = []

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

                #save gaussian fit parameters and mode edges for channel plotting 
                fitting_params.append([diam,popt[0], popt[1]+x_min, popt[2]]) #[diam,a,mu,sigma]
                mode_edges.append([diam,start_time,end_time]) #[diameter,start time, end time]
                                           
        except:
            pass
            #print("Diverges. Skipping.")

    #convert time to UTC
    max_conc_time = [dt.replace(tzinfo=None) for dt in mdates.num2date(max_conc_time)]
    max_conc_time = np.array(max_conc_time)
    
    return max_conc_time, max_conc_diameter, max_conc, fitting_params, mode_edges
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
    mode_edges = []
    
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
                
                #logistic fit for more than 2 datapoints
                if len(y_sliced) > 2: 
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
                            mid_conc.append(((popt[0]+y_min)+y_min)/2) #appearance time concentration (~50% maximum concentration), L/2

                            #save logistic fit parameters and mode edges for channel plotting 
                            fitting_params.append([diam, y_min, popt[0], popt[1]+x_min, popt[2]]) #[diam,y min for scale,L,x0,k]
                            
                            start_time_logi = mdates.num2date(x_sliced[0]+x_min).replace(tzinfo=None) #in days
                            end_time_logi = mdates.num2date(x_sliced[-1]+x_min).replace(tzinfo=None)
                            mode_edges.append([diam,start_time_logi,end_time_logi]) #[diameter,start time, end time]
                    except:
                        pass
                        #print("Logistic diverges. Skipping.")                            
        except:
            pass
            #print("Diverges. Skipping.")

    #convert time to UTC
    appear_time = [dt.replace(tzinfo=None) for dt in mdates.num2date(appear_time)]
    appear_time = np.array(appear_time)

    return appear_time, appear_diameter, mid_conc, fitting_params, mode_edges
def init_methods(dataframe,mpd,threshold_deriv):
    '''
    Goes through all ranges and calculates the points for maximum concentration
    and appearance time methods along with other useful information.
    x = time, y = diameter, z = concentration
    '''
    maxcon_xyz = [] #maximum concentration
    appear_xyz = [] #appearance time

    start_time = time()
    #smoothen data, calculate derivative and define modes
    df_filtered = average_filter(dataframe,window=3)
    df_deriv = cal_derivative(df_filtered) 
    df_modes, threshold_deriv, start_times_list, maxima_list = find_modes(df_filtered,df_deriv,mpd,threshold_deriv)
    
    #methods
    mc_x, mc_y, mc_z, mc_params, mode_edges_gaussian = maximum_concentration(df_modes)
    at_x, at_y, at_z, at_params, mode_edges_logistic = appearance_time(df_modes)
    print("Fitting done! (2/4) "+"(%s seconds)" % (time() - start_time))
    
    #combine to same lists
    maxcon_xyz = [mc_x,mc_y,mc_z]
    appear_xyz = [at_x,at_y,at_z]
    
    return maxcon_xyz, appear_xyz, mc_params, at_params, mode_edges_gaussian, mode_edges_logistic, threshold_deriv, start_times_list, maxima_list
xyz_maxcon, xyz_appear, *others = init_methods(df,mpd=maximum_peak_difference,threshold_deriv=derivative_threshold)

################## GROWTH RATES ####################

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
    return ([point[1] for point in line[exclude_start:len(line)-exclude_end]],  #x values
            [point[0] for point in line[exclude_start:len(line)-exclude_end]])  #y values
    
def find_dots(times,diams,mtd,a,gret):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with times and diameters for plotting growth rates.
    
    gret = growth rate error threshold for filtering bigger changes in gr when adding new points to lines
    mtd = maximum time difference between current point and nearby point to add to line
    mae = mean average error
    '''
    #convert time to days
    times = mdates.date2num(times)
    
    #combine to the same list and sort data by diameter
    data_sorted = np.array(sorted(zip(times, diams), key=itemgetter(1,0))) #[[time1,diam1],[time2,diam2]...]

    #init
    unfinished_lines = []
    finalized_lines = []
    df_maes = pd.DataFrame()   
    
    #iterate over each datapoint to find suitable pairs of mode fitting datapoints
    for datapoint in data_sorted:
        time0, diam0 = datapoint
        datapoint = tuple(datapoint)
        
        #iterate over diameter channels after current datapoint and look for the nearest datapoint
        for ii in range(1,3): #allows one channel in between
            diam_channel = df.columns.values[df.columns.values >= diam0][ii] #diam in current bin
            diam_diff = diam_channel-diam0
            channel_points = [point for point in data_sorted if point[1] == diam_channel]
            
            #skip if no datapoints in current channel
            if not channel_points: 
                continue
            
            #allowed time difference depends on growth rate and diam difference between last and new point
            low_time_limit = time0-mtd/24 #days
            high_time_limit = time0+mtd/24
            
            closest_channel_points = [point for point in channel_points if point[0] >= low_time_limit and point[0] <= high_time_limit]
            
            #skip if no datapoints in current channel
            if not closest_channel_points: 
                continue
            
            #find nearest new point 
            nearby_datapoint = tuple(min(channel_points, key=lambda point: abs(point[0] - time0)))
            time1 = nearby_datapoint[0]
            
            if points_in_existing_line(unfinished_lines,datapoint):
                #find index of that line
                for line in unfinished_lines:
                    if datapoint in line and len(line) > 2:
                        #calculate growth rate
                        x, y = extract_data(line) #x=diams,y=times
                        popt, pcov = curve_fit(linear, x, y)
                        GR = 1/(popt[0]) #nm/days
                        
                        min_time = min(mdates.date2num(df.index))
                        b = 1.5/24 #1.5 hours
                        low_time_limit = 1/(abs(GR) * 1.5) * diam_diff - b + time0 #days
                        high_time_limit = 1.5/abs(GR) * diam_diff + b + time0
                        
                        #minimize MAE when choosing the new point
                        maes = []
                        for point in closest_channel_points:
                            line_with_new_point = line + [point]
                            x, y = extract_data(line_with_new_point) #x=diams,y=time
                            popt, pcov = curve_fit(linear, x, y)
                            mae = cal_mae(x,y,popt)
                            maes.append(mae)

                        min_mae_i = maes.index(min(maes))
                        nearby_datapoint = tuple(closest_channel_points[min_mae_i])
                        break

            time_diff = abs(nearby_datapoint[0] - time0)

            
            if time1 >= low_time_limit and time1 <= high_time_limit: 
            #if time_diff <= max_time_diff: 
                ### add new point to a line ###
                if not points_in_existing_line(unfinished_lines,datapoint,nearby_datapoint):
                    unfinished_lines.append([datapoint,nearby_datapoint])
                    
                elif points_in_existing_line(unfinished_lines,datapoint,nearby_datapoint):
                    #find index of that line
                    for line in unfinished_lines:
                        if datapoint in line:
                            print(line)
                            line.append(nearby_datapoint)
                            break
                
                #make sure datapoints in every line are sorted by diameter
                unfinished_lines = [list(set(line)) for line in unfinished_lines]
                unfinished_lines = [sorted(sublist, key=lambda x: x[1]) for sublist in unfinished_lines] 


                ### make a linear fit to check mae for line with new datapoint ###
                iii, current_line = [(i,line) for i,line in enumerate(unfinished_lines) if nearby_datapoint in line][0]
                
                #define variables for linear fit
                x, y = extract_data(current_line) #x=diams,y=times
                x_last_excluded, y_last_excluded = extract_data(current_line,exclude_end=1)
                x_first_excluded, y_first_excluded = extract_data(current_line,exclude_start=1)
                
                if len(current_line) <= 2:
                    break #proceed to next datapoint
                elif len(current_line) == 3:
                    popt, pcov = curve_fit(linear, x, y)
                    mae = cal_mae(x,y,popt)
                else:
                    #fit to line excluding last datapoint, but calculate mae with full line
                    popt_last_excluded, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)
                    GR_last_excluded = 1/(popt_last_excluded[0]*24) #nm/h
                    
                    #fit to full line
                    popt, pcov = curve_fit(linear, x, y)
                    mae = cal_mae(x,y,popt)
                    GR = 1/(popt[0]*24)
                    
                    gr_abs_precentage_error = abs(GR-GR_last_excluded) / abs(GR_last_excluded) * 100

                
                ### check mae and gr error thresholds ###
                mae_threshold = a*len(current_line)**(-1) #a*x^(-1)
                gr_error_threshold = gret
                
                if mae > mae_threshold:
                    #calculate mae without the first and then last datapoint 
                    popt, pcov = curve_fit(linear, x_first_excluded, y_first_excluded)
                    mae_no_first = cal_mae(x_first_excluded,y_first_excluded,popt) 
                    
                    popt, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)
                    mae_no_last = cal_mae(x_last_excluded,y_last_excluded,popt)
                    
                    #remove last or first point based on mae comparison
                    if mae_no_last <= mae_no_first and len(current_line) > 4:
                        unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                        finalized_lines.append(current_line[:-1])
                        unfinished_lines.append([datapoint,nearby_datapoint]) #new line starts with end of previous one
                    else:
                        unfinished_lines[iii] = current_line[1:] #another chance for shorter lines
                
                elif len(current_line) > 3 and gr_abs_precentage_error > gr_error_threshold:
                    #remove last point if threshold is exceeded
                    if len(current_line) > 4:
                        #delete line with recently added datapoint from unfinished lines
                        unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                        finalized_lines.append(current_line[:-1])
                        unfinished_lines.append([datapoint,nearby_datapoint])
                    else:
                        unfinished_lines[iii] = current_line[1:]

                break
            else: #keep looking for next datapoint until end of points if the next one isnt suitable
                continue
    
    #add rest of the lines to finalized lines and by diameter
    finalized_lines.extend(unfinished_lines)
    finalized_lines = [sorted(line, key=lambda x: x[1]) for line in finalized_lines] 
    
    '''
    #try splitting line into two parts from the middle to lower mae
    for i, finalized_line in enumerate(finalized_lines):
        if len(finalized_line) >= 7: #at least 7 datapoints needed  
            middle_index = len(finalized_line)//2
            line_1st_half = finalized_line[:middle_index+1] #overlap +1
            line_2nd_half = finalized_line[middle_index:]
            
            #calculate if mae lowered in both halves
            #whole line
            x = [datapoint[1] for datapoint in finalized_line]
            y = [datapoint[0] for datapoint in finalized_line] #diams
            popt, pcov = curve_fit(linear, x, y)
            mae = cal_mae(x,y,popt)
            
            #1st half
            x = [datapoint[1] for datapoint in line_1st_half]
            y = [datapoint[0] for datapoint in line_1st_half] #diams
            popt, pcov = curve_fit(linear, x, y)
            mae1 = cal_mae(x,y,popt)
            
            #2nd half
            x = [datapoint[1] for datapoint in line_2nd_half]
            y = [datapoint[0] for datapoint in line_2nd_half]
            popt, pcov = curve_fit(linear, x, y)
            mae2 = cal_mae(x,y,popt)
            
            if mae1 < mae and mae2 <= mae:
                #remove the second half of current line and add it as its own line to finalized lines
                finalized_lines[i] = line_1st_half
                finalized_lines.append(line_2nd_half)  
    '''
    #calculate maes to show on plot
    for finalized_line in finalized_lines:
        x, y = extract_data(finalized_line) #x=diams,y=times
        popt, pcov = curve_fit(linear, x, y)
        mae = cal_mae(x,y,popt)
        
        new_row = pd.DataFrame({"length": [len(x)], "mae": [mae]})
        df_maes = pd.concat([df_maes,new_row],ignore_index=True)
        
    df_maes.to_csv('./df_maes.csv', sep=',', header=True, index=True, na_rep='nan')
    
    return finalized_lines

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
                if mape <= 100 and abs(GR) <= 15000:
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
def init_find(mtd,a,gret):
    start_time = time()
    #find consequtive datapoints
    mc_data = find_dots(times=xyz_maxcon[0],diams=xyz_maxcon[1],mtd=mtd,a=a,gret=gret) #maximum concentration
    at_data = find_dots(times=xyz_appear[0],diams=xyz_appear[1],mtd=mtd,a=a,gret=gret) #appearance time
    
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

def robust_fit(time,diam,color,show_mae):
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
        
        if show_mae:
            #mae annotation
            y = np.array(time)
            y_predicted = np.array(t_rlm)
            mae = np.mean(np.abs(y_predicted - y)) * 24
            plt.annotate(f'{mae:.2f}', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
            ax.set_title(f'mean absolute error (MAE) unit: [hours]', loc='right', fontsize=8) 
        else:
            #growth rate annotation
            plt.annotate(f'{gr:.2f}', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
            ax.set_title(f'growth rate unit: [nm/h]', loc='right', fontsize=8) 

        
        #print("Statsmodel robus linear model results: \n",rlm_results.summary())
        #print("\nparameters: ",rlm_results.params)
        #print(help(sm.RLM.fit))
def linear_fit(time,diam,color,show_mae):
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
    #plt.plot(time_UTC,diam,lw=3) #line
    plt.plot(time_UTC, diam,color=color,linewidth=2)
    
    midpoint_idx = len(time_fit) // 2 #growth rate value
    midpoint_time = time_UTC[midpoint_idx]
    midpoint_value = diam[midpoint_idx]

    if show_mae:
        #mape annotation
        y = np.array(time)
        y_predicted = np.array(time_fit)
        mae = np.mean(np.abs(y_predicted - y)) * 24
        
        plt.annotate(f'{mae:.2f}', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
        ax.set_title(f'mean absolute error (MAE) unit: [hours]', loc='right', fontsize=8) 
    else:
        #growth rate annotation
        plt.annotate(f'{gr:.2f}', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
        ax.set_title(f'growth rate unit: [nm/h]', loc='right', fontsize=8) 

def plot_PSD(dataframe,show_mae,mtd,a,gret):
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
    plt.plot(xyz_maxcon[0], xyz_maxcon[1], '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6,label='maximum concentration') 
    plt.plot(xyz_appear[0], xyz_appear[1], '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6,label='appearance time')
    
    # #janne's max con
    # plt.plot(df_fits['peak_time'].values, df_fits.index, '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6,label='maximum concentration') 
    
    time_mc, diam_mc, time_at, diam_at = init_find(mtd,a,gret)
    
    #growth rates (and maes)
    for time_seg_mc, diam_seg_mc, time_seg_at, diam_seg_at in zip(time_mc,diam_mc,time_at,diam_at):
        linear_fit(time_seg_mc, diam_seg_mc,"white",show_mae) #maximum concentration
        linear_fit(time_seg_at, diam_seg_at,"green",show_mae) #appearance time
        #robust_fit(time_seg_mc, diam_seg_mc,"white",show_mae)
        #robust_fit(time_seg_at, diam_seg_at,"green",show_mae)

    #adjustments to plot
    plt.legend(fontsize=9,fancybox=False,framealpha=0.9)
    for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
        legend_handle.set_markeredgewidth(0.5)
        legend_handle.set_markeredgecolor("black")
    
    plt.xlim(dataframe.index[0],dataframe.index[-1])
    plt.ylim(dataframe.columns[0],dataframe.columns[-1])
    plt.ylabel("diameter (nm)",fontsize=14) #add y-axis label
    plt.xlabel("time",fontsize=14) #add y-axis label
    
    print("Plotting done! (4/4) "+"(%s seconds)" % (time() - st))
plot_PSD(df,show_mae,maximum_time_difference_dots,mae_threshold_factor,gr_precentage_error_threshold)

def plot_channel(dataframe,diameter_list_i,mpd,threshold_deriv,show_start_times_and_maxima):
    '''
    Plots chosen diameter channels over UTC time, with thresholds and gaussian fit.
    ax[0,0] = whole channel over time, with ranges      ax[0,1] = derivative of concentrations
    ax[n,0] = n amount of channels                      ax[n,1] = derivative of concentrations
                                                ...
    Inputs dataframe with data and diameters (numerical).
    
    start_times = [(diameter, [start time 1, start time 2...], ...)]
    '''   
    
    '''1 assemble all datasets'''
    xyz_maxcon, xyz_appear, fitting_parameters_gaus, \
        fitting_parameters_logi, mode_edges_gaussian, mode_edges_logistic, \
        threshold_deriv, start_times_list, maxima_list = init_methods(dataframe,mpd,threshold_deriv)

    '''2 define lists and their shapes'''
    xy_maxcon =  []             #[(max con diameter, max con time (UTC), max con), ...]
    fitting_params_gaus = []    #[(max con diameter, *params...), ...]
    fitting_params_logi = []    #[(appearance time diameter, y min for scale, *params...), ...]
    appearances = []            #[(diameter, time (UTC), concentration), ...] 
    mode_edges_gaus = []
    mode_edges_logi = []

    '''3 find data in datasets with chosen diameters'''
    diameter_list = [df.columns[i] for i in diameter_list_i]
    
    for diam in diameter_list:
        #MAXIMUM CONCENTRATION & TIME
        indices = [i for i, a in enumerate(xyz_maxcon[1]) if a == diam] #indices of datapoints with wanted diameter
        xy_maxcons = [(xyz_maxcon[1][b],xyz_maxcon[0][b],xyz_maxcon[2][b]) for b in indices]
        [xy_maxcon.append(i) for i in xy_maxcons]

        #FITTING PARAMETERS
        [fitting_params_gaus.append(params) for params in fitting_parameters_gaus if params[0] == diam]
        [fitting_params_logi.append(params) for params in fitting_parameters_logi if params[0] == diam]

        #APPEARANCE TIME & CONCENTRATION
        indices = [i for i, a in enumerate(xyz_appear[1]) if a == diam]
        appearance = [(xyz_appear[1][b],xyz_appear[0][b],xyz_appear[2][b]) for b in indices]
        [appearances.append(i) for i in appearance]
        
        #MODE EDGES
        [mode_edges_gaus.append(mode) for mode in mode_edges_gaussian if mode[0] == diam]
        [mode_edges_logi.append(mode) for mode in mode_edges_logistic if mode[0] == diam]


    '''4 plotting'''
    fig, ax1 = plt.subplots(len(diameter_list),2,figsize=(9, 4.7), dpi=300)
    fig.subplots_adjust(wspace=0.38, hspace=0.29) #adjust spaces between subplots
    ax1 = np.atleast_2d(ax1) #to avoid problems with plotting only one channel
    lines_and_labels = set() #later use for legends
    
    #parameters
    #define x and y for the whole channel
    x = dataframe.index #time
    y_list = [dataframe[diam] for diam in diameter_list] #concentrations
    
    #also smoothed concentrations
    df_filtered = average_filter(dataframe,window=3)
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
            
        #all possible start times of modes
        if show_start_times_and_maxima:
            for start, maximum in zip(start_times_list,maxima_list):
                start_diam, start_times, start_concs = start
                max_diam, max_times, max_concs = maximum
                
                if start_diam == diameter_list[row_num]:
                    #start points
                    #ax1[row_num,0].vlines(x=start_times,ymin=0,ymax=100, color='black', linestyle='-', lw=0.8)
                    starts = ax1[row_num,0].scatter(start_times, start_concs, s=5, c='black',marker='>', alpha=0.6, zorder=10)
                    lines_and_labels.add((starts,"possible peak area starting points"))
                        
                if max_diam == diameter_list[row_num]:  
                    #maximas
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
        '''
        
        #gaussian fit 
        for params, edges in zip(fitting_params_gaus,mode_edges_gaus):
            diam_params, a, mu, sigma = params
            diam_gaus, start_time, end_time = edges 
            
            mode_times_UTC = dataframe.loc[start_time:end_time,diam].index
            mode_times = mdates.date2num(mode_times_UTC) #time to days
            mode_concs = dataframe.loc[start_time:end_time,diam].values
            
            if diam_params == diameter_list[row_num] and diam_gaus == diameter_list[row_num]: #check that plotting happens in the right channel
                line2, = ax1[row_num,0].plot(mode_times_UTC, gaussian(mode_times,a,mu,sigma)+min(mode_concs), '--', color="mediumturquoise",lw=1.2)
                lines_and_labels.add((line2,"gaussian fit"))
                #ax2.plot(mode_times_UTC, gaussian(mode_times,a,mu,sigma), '--', color="mediumturquoise",lw=1.2)
        
        #logistic fit
        for params, edges_logi in zip(fitting_params_logi,mode_edges_logi):
            diam_params, y_min, L, x0, k = params
            diam_logi, start_time_logi, end_time_logi = edges_logi

            mode_times_UTC = dataframe.loc[start_time_logi:end_time_logi,diam].index
            mode_times = mdates.date2num(mode_times_UTC) #time to days
     
            if diam_params == diameter_list[row_num] and diam_logi == diameter_list[row_num]:
                line3, = ax1[row_num,0].plot(mode_times_UTC, logistic(mode_times,L,x0,k)+y_min, '--', color="gold",lw=1.2)
                lines_and_labels.add((line3,"logistic fit"))
                #ax2.plot(mode_times_UTC, logistic(mode_times,L,x0,k), '--', color="gold",lw=1.2)

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
        
        ax1[row_num,0].set_xlim(dataframe.index[0],dataframe.index[-1])
        ax1[row_num,0].set_facecolor("lightgray")
        #ax1[row_num,0].xaxis.set_tick_params(rotation=30)
        ax1[row_num,0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ax1[row_num,0].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        #if row_num == 0:
        #    ax1[row_num,0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    #PLOTS ON THE RIGHT
    df_1st_derivatives = cal_derivative(dataframe)

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
        for edges in mode_edges_gaus:
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
                ax1[row_num,1].plot(x_maxcon, y_maxcon, '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6)  
        
        #appearance time
        for i in appearances:
            diam, time, conc = i
            conc = conc*0 #to place the start lower where y = 0
            if diam == diameter_list[row_num]:
                ax1[row_num,1].plot(time, conc, '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6)
                #ax2.plot(time, conc, '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6)


        
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
    lines_and_labels2 = [elem for elem in lines_and_labels if elem[1] in ["maximum concentration","appearance time","mode edges",f"threshold = {str(threshold_deriv)} cm⁻³/h"]] 
    valid_lines, valid_labels = zip(*lines_and_labels2)
    legend_2 = ax1[0, 1].legend(valid_lines, valid_labels, fancybox=False, framealpha=0.9, fontsize=4, loc="upper right")


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
    
    #fig.tight_layout()
    print("Drawing diameter channel(s):",diameter_list)

if init_plot_channel:
    plot_channel(df,diameter_list_i=channel_indices,mpd=maximum_peak_difference,threshold_deriv=derivative_threshold,show_start_times_and_maxima=show_start_times_and_maxima)

plt.show()
####################################################
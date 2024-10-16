from GR_calculator_unfer_v2_modified import * 
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from operator import itemgetter
from statistics import mean
from itertools import chain
import time

'''
This code assumes dmps data.
- 1st column (skipping the first 0 value):              time in days
- 2nd column (skipping the first 0 value):              total concentration in current timestamp (row)
- 1st row (skipping first two 0 values):                diameters in meters, ~3nm-1000nm
- 3rd column onwards till the end (under diameters):    concentrations, dN/dlog(dp)
'''

################ DATA FORMATTING ################
folder = r"./dmps Nesrine/" #folder where data files are stored, should be in the same directory as this code
#dm160612.sum
#dm160401.sum
#dm160402.sum
#dm160403.sum
#output_modefit_2016_06_5median_6nm_fix_xmin.csv

file_names = file_names #copy paths from the file "GR_calculator_unfer_v2_modified"

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
        
        #different dmps data files have slightly different diameter values although the represent the same diameter
        #name all other columns with the labels of the first one
        if test == True:
            diameter_labels = df.columns
            test = False
        df.rename(columns=dict(zip(df.columns, diameter_labels)), inplace=True)

        dfs.append(df) #add dataframes to a list of dataframes
    
    combined_data = pd.concat(dfs,axis=0,ignore_index=True) #combine dataframes
    return combined_data
df = combine_data() #create dataframe with combined data

df.rename(columns=dict(zip(df.columns[[0,1]], ["time (d)", "total number concentration (N)"])), inplace=True) #rename first two columns
df = df.drop(['total number concentration (N)'], axis=1) #drop total N concentrations from the dataframe as they're not needed
time_d = df['time (d)'].values.astype(float) #save time as days before changing them to UTC

#change days into UTC (assuming time is in "days from start of measurement"):
def df_days_into_UTC(dataframe,time): #for a dataframe
    '''
    Changes days into UTC datetime.
    Start date is from the first given dmps data file.
    Returns dataframe with datetimes.
    '''
    time_steps = dataframe["time (d)"] - time[0] #calculate timesteps between every timestamp
    start_date_str = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00" #define start date
    start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S") #change to date time
    dataframe["time (d)"] = [start_datetime + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
    return dataframe
def list_days_into_UTC(list,time): #for a list
    time_steps = list - time[0]
    start_date_str = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:15:00" #WITH RESOLUTION OF 30mins!
    start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    try: #converting timesteps to datetime
        list = [start_datetime + timedelta(days=i) for i in time_steps] 
    except: #in case the list has just one element
        list = start_datetime + timedelta(days=time_steps)

    return list
df = df_days_into_UTC(df,time_d)

#set new UTC timestamps as indices
df.rename(columns={'time (d)': 'time (UTC)'}, inplace=True)
df['time (UTC)']=pd.to_datetime(df['time (UTC)'], format="%Y-%m-%d %H:%M:%S")
df.index=df['time (UTC)']
df = df.drop(['time (UTC)'], axis=1)

#set numerical diameters as column headers
df.columns = pd.to_numeric(df.columns) * 10**9 #change units from m to nm

#filtering
def median_filter(dataframe,window):         
    '''
    Smoothens data in dataframe with median filter and given window.
    Returns smoothened dataframe.
    '''
    
    for i in dataframe.columns:
        dataframe[i] = dataframe[i].rolling(window=window, center=True).median()
    dataframe.dropna(inplace=True)
    return dataframe 
#median_filter(df,window=5)

def avg_filter(dataframe,resolution):
    '''
    Smoothens data in dataframe with average filter and given resolution (minutes), 
    i.e. takes averages in a window without overlapping ranges.
    Discards blocks of time with incomplete timestamps.
    Returns smoothened dataframe and new time in days for that dataframe.
    '''

    dataframe.index = dataframe.index.round('10T') #change timestamps to be exactly 10min intervals

    #if average is taken of less than 3 datapoints, neglect that datapoint
    full_time_range = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq='10T')
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
df, time_d_res = avg_filter(df,resolution=30)

#with this we can check the format
df.to_csv('./data_format_nofilter.csv', sep=',', header=True, index=True, na_rep='nan')

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

#derivatives
def cal_1st_derivative(dataframe,time_range):
    '''
    Calculates 1st derivatives between neighbouring datapoints. 
    Takes in dataframe and wanted range of time (in case of smaller dataframes).
    Returns dataframe with derivatives that has been smoothened with median filter (window 5).
    '''
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns)
    
    for i in dataframe.columns: 
        #N = np.log10(dataframe[i]) #log10 of concentration ###LOG FROM CONC###
        #N.replace([-np.inf,np.inf], 0, inplace=True) #replace all inf and -inf values with 0 ###LOG FROM CONC###
        N = dataframe[i] #concentration ###NO LOG FROM CONC###
        time = time_range * 24 * 60 * 60 #change days to seconds
        dNdt = np.diff(N)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = dNdt #add calculated derivatives to dataframe
    return  df_derivatives
#df_1st_derivatives = cal_1st_derivative(df,time_range=time_d_res)

#with this we can check the format
cal_1st_derivative(df,time_range=time_d_res).to_csv('./1st_derivatives_filterafter.csv', sep=',', header=True, index=True, na_rep='nan')

def cal_2nd_derivative(dataframe,time_range):
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns)
    
    #calculate derivatives for each diameter
    for i in dataframe.columns: 
        dNdt = dataframe[i] #1st derivative
        time = time_range * 24 * 60 * 60 #change days to seconds
        second_dNdt = np.diff(dNdt)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = second_dNdt
    return df_derivatives
#df_2nd_derivatives = cal_2nd_derivative(df,time_range=time_d_res)

#minimum and maximum
def cal_min_max(df_1st_deriv,df_2nd_deriv): #returns [df_max, df_min]
    df_1stderiv_zeros = find_zero(df_1st_deriv) #dataframe with stationary points of the 1st derivative
    #find max and min values
    df_max = df[(df_1stderiv_zeros == 0) & (df_2nd_deriv < 0)] #stationary points exist, 2nd derivative is negative
    df_min = df[(df_1stderiv_zeros == 0) & (df_2nd_deriv > 0)] #stationary points exist, 2nd derivative is positive
    return [df_max,df_min]
#df_max = cal_min_max(df_1st_derivatives,df_2nd_derivatives)[0]

#finding modes with derivative threshold
def find_modes(dataframe,df_deriv,threshold_deriv,threshold_diff): #finds modes, derivative threshold
    df_mode_ranges = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns)

    #threshold that determines what is a high concentration
    threshold = abs(df_deriv) > threshold_deriv #checks which values surpass the threshold ###NO LOG FROM CONC###
    #threshold = abs(df_deriv) > 0.000009 #checks which values surpass the threshold ###LOG FROM CONC###
    
    #threshold = (df_deriv > df_deriv.shift(1))

    #determine start and end points
    start_points = threshold & (~threshold.shift(1,fill_value=False))
    #end_points = threshold & (~threshold.shift(-1,fill_value=False))

    #finding values within start and end points
    start_times_temp = []
    start_time = None 
    end_time = None
    conc_diff = threshold_diff #concentration difference between the start and end times

    for diam in df_deriv.columns:
        
        for timestamp in df_deriv.index: #identify pairs of start and end times
            if start_time != None: #when a start time has been found find list of values between start/end times
                subset = dataframe[diam].loc[start_time:end_time]
                df_mode_ranges.loc[subset.index,diam] = subset #fill dataframe with mode ranges
                
                #restart initial values
                start_times_temp = []
                start_time = None 
                end_time = None                     
                                  
            elif start_points.loc[timestamp,diam] == True and df_deriv.loc[timestamp,diam] > 0: #checks also if the starting point derivative is neg or not
                start_time = timestamp
                start_conc = df_deriv.loc[timestamp,diam]

                #finding end time after local maximum concentration 
                try:
                    subset_end_i = dataframe.index.get_loc(start_time) + 14
                    subset_end = dataframe.index[subset_end_i] #time after 140mins from starting point
                except IndexError:
                    subset_end = dataframe.index[-1]
                
                df_subset = dataframe.loc[start_time:subset_end,diam] #subset from mode start to (mode start + 140mins) unless mode start is near the end of a range
                end_conc = ( max(df_subset.values) + start_conc ) / 2  #(N_max + N_start)/2
                max_conc_time = df_subset.index[df_subset == max(df_subset.values)].tolist()[0] #rough estimate of maximum concentration time 
                max_conc_time_i = dataframe.index.get_loc(max_conc_time) #index of max_conc_time
                
                try:
                    new_subset_end = dataframe.index[max_conc_time_i + (max_conc_time_i - dataframe.index.get_loc(start_time))] #new ending limit to find end concentration time, same distance from max conc time as start time is
                except IndexError:
                    new_subset_end = dataframe.index[-1] #NOW IT LIMITS IT TO THE RANGE
                
                end_conc_i = closest(dataframe.loc[max_conc_time:new_subset_end,diam],end_conc) #index of end point
                end_time = dataframe.index[end_conc_i+max_conc_time_i] 

                if start_time == end_time: #skip modes with same start/end time
                    start_time = None
                    end_time = None
                else:
                    start_times_temp.append(start_time)
                

                #print("DIAM:",diam,"max conc time",max_conc_time,"subset end (finding end conc)",new_subset_end," :: ","STARTTIME",start_time,"ENDTIME",end_time)

            else:
                continue
    
    #with this we can check the format
    #find_modes(df,df_1st_derivatives,threshold_deriv=0.03,threshold_diff=2000)[0].to_csv('./modes.csv', sep=',', header=True, index=True, na_rep='nan')

    return df_mode_ranges
"""
def find_modes_2nd_deriv(dataframe):
    df_mode_ranges = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns)

    #threshold that determines what is a high dN/dt
    threshold = abs(df_2nd_derivatives) > 3 #checks which values surpass the threshold

    #determine start and end points
    start_points = threshold & (~threshold.shift(1,fill_value=False))
    end_points = threshold & (~threshold.shift(-1,fill_value=False))

    #finding values within a mode
    start_time = None #initial values
    end_time = None

    for k in dataframe.columns:
        for l in dataframe.index[1:]: #identify pairs of start and end times, taking the derivative removes the first row
            if start_time != None and end_time != None: #when start and end times have their values find list of values between them
                subset = dataframe[k].loc[start_time:end_time]
                #fill dataframe with mode ranges
                df_mode_ranges.loc[subset.index, k] = subset

                start_time = None 
                end_time = None
            elif start_points.loc[l,k] == True:
                start_time = l
            elif end_points.loc[l,k] == True and start_time != None:
                end_time = l
            else:
                continue
    return df_mode_ranges

def find_modes_globalthreshold(dataframe, factor): #factor between 0-1, scales threshold
    df_mode_ranges = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns)

    #threshold that determines what is a high dN/dlogdp, global maximum of the specific diameter * factor
    threshold = dataframe.max() * factor
    high_values = dataframe > threshold #checks which values surpass the threshold

    #determine start and end points
    start_points = high_values & (~high_values.shift(1,fill_value=False))
    end_points = high_values & (~high_values.shift(-1,fill_value=False))

    #finding values within a mode
    start_time = None #initial values
    end_time = None

    for k in dataframe.columns:
        for l in dataframe.index: #identify pairs of start and end times
            if start_time != None and end_time != None: #when start and end times have their values find list of values between them
                subset = dataframe[k].loc[start_time:end_time]
                #fill dataframe with mode ranges
                df_mode_ranges.loc[subset.index, k] = subset

                start_time = None 
                end_time = None
            elif start_points.loc[l,k] == True:
                start_time = l
            elif end_points.loc[l,k] == True and start_time != None:
                end_time = l
            else:
                continue
    return df_mode_ranges
"""

#define functions for fitting
def gaussian(x,a,x0,sigma): #from Janne's code
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
def logistic(x,L,x0,k): #from Janne's code
        return L * (1 + np.exp(-k*(x-x0)))**(-1) 

def find_ranges():
    df_GR_values = pd.DataFrame(pd.read_csv("./Gr_final.csv",sep=',',engine='python')) #open calculated values in Gabi's code
    threshold = 0 #GR [nm/h]

    df_ranges = []

    for i in df_GR_values.index: #go through every fitted growth rate
        growth_rate = df_GR_values.loc[i,"GR"]
        if abs(growth_rate) >= threshold: #find maximum concentration if growth rate is higher than threshold value
            row = df_GR_values.loc[i,:]
            
            #start and end times/diameters of fitted lines
            start_time = row["start"] 
            end_time = row["end"]
            start_diam = row["d_initial"]
            end_diam = row["d_final"]

            #make the ranges bigger with given parameters
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S") - timedelta(hours=5) #5 hours
            end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") + timedelta(hours=5)
            
            if growth_rate < 0: #if growth rate is negative
                start_diam = start_diam * 1.5 #factor of 1.5
                end_diam = end_diam / 1.5
            else:
                start_diam = start_diam / 1.5
                end_diam = end_diam * 1.5

            #make a df with wanted range and add them to the list
            df_mfit_con = df[(df.index >= start_time) & (df.index <= end_time)]
            df_ranges.append(df_mfit_con[df_mfit_con.columns[(df_mfit_con.columns >= start_diam) & (df_mfit_con.columns <= end_diam)]])

    return df_ranges

def max_con(): #returns [times, diameters]
    df_1st_derivatives = cal_1st_derivative(df,time_range=time_d_res) #calculate derivative
    df_modes = find_modes(df,df_1st_derivatives,threshold_deriv=0.03,threshold_diff=10000)

    #gaussian fit to every mode
    max_conc_time = []
    max_conc_diameter = []

    for i in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for j in range(len(df_modes.index)):
            concentration = df_modes.iloc[j,i] #find concentration values from the dataframe (y)
            time = time_d_res[j] - np.min(time_d_res)

            if np.isnan(concentration) and len(y) != 0: #gaussian fit when all values of one mode have been added to the y list
                mu=np.mean(x) #parameters
                sigma = np.std(x)
                a = np.max(y)

                try: #gaussian fit
                    params,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma])
                    if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                        print("Peak outside range. Skipping.")
                    else:
                        max_conc_time = np.append(max_conc_time,params[1] + np.min(time_d_res)) #make a list of times with the max concentration time in each diameter
                        max_conc_diameter = np.append(max_conc_diameter,float(df_modes.columns[i])) #make list of diameters with max concentrations

                        #plot start and end time
                        #plt.plot(start_times[j], start_diams[i], '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
                        #plt.plot(end_times[j], end_diams[i], '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)
                        
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

    max_conc_time = list_days_into_UTC(max_conc_time,time_d_res) #change days to UTC in the list of max concentration times
    
    return max_conc_time, max_conc_diameter
def maxcon_modefit(): #find ranges around growth rates, use these points to calculate the maximum concentration
    #create lists for results
    max_conc_time = []
    max_conc_diameter = []
    max_conc = []
    fitting_params = []

    dfs_mode_start = []
    dfs_mode_end = [] 
    df_modes = []

    df_ranges = find_ranges()

    for df_range in df_ranges: #go through every range around GRs

        time_range = df.index.intersection(df_range.index) #find matching indices
        indices = [df.index.get_loc(row) for row in time_range]
        time_d_range = time_d_res[indices] #define time in days again depending on chosen range
        

        df_range_deriv = cal_1st_derivative(df_range,time_range=time_d_range) #calculate derivative
        threshold_deriv = 0.05 #choose derivative threshold
        df_mode = find_modes(df_range,df_range_deriv,threshold_deriv=threshold_deriv,threshold_diff=5000) #find modes
        df_modes.append(df_mode) #add dfs to list

        #gaussian fit to every mode
        for dp in range(len(df_mode.columns)):
            x = [] #time
            y = [] #concentration
            for t in range(len(df_mode.index)):
                concentration = df_mode.iloc[t,dp] - np.min(df_mode.iloc[:,dp]) #find concentration values from the dataframe (y)
                time = time_d_range[t] - np.min(time_d_range)

                if np.isnan(concentration) and len(y) != 0: #gaussian fit when all values of one mode have been added to the y list
                    mu=np.mean(x) #parameters
                    sigma = np.std(x)
                    a = np.max(y)

                    try: #gaussian fit
                        params,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma])
                        if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                            print("Peak outside range. Skipping.")
                        else:
                            max_conc_time = np.append(max_conc_time,params[1] + np.min(time_d_range)) #make a list of times with the max concentration time in each diameter
                            max_conc_diameter = np.append(max_conc_diameter,float(df_mode.columns[dp])) #make list of diameters with max concentrations
                            max_conc = np.append(max_conc,params[0] + np.min(df_mode.iloc[:,dp])) #maximum concentrations
                            
                            #create dfs with start and end points
                            df_mode_start = pd.DataFrame(np.nan, index=df_range.index, columns=df_range.columns)
                            df_mode_end = pd.DataFrame(np.nan, index=df_range.index, columns=df_range.columns)
                            
                            start_time_days = x[0] + np.min(time_d_range) #define start/end time (days)
                            end_time_days = x[-1] + np.min(time_d_range)
                            
                            start_time_UTC = pd.Series(list_days_into_UTC(start_time_days-0.25/24,time_d_res)).round('30T')[0] #UTC, rounded to the nearest 30min increment
                            end_time_UTC = pd.Series(list_days_into_UTC(end_time_days-0.25/24,time_d_res)).round('30T')[0]
                            start_time_UTC += timedelta(minutes=15) #makes it easier to round
                            end_time_UTC += timedelta(minutes=15)
                            
                            df_mode_start.loc[start_time_UTC,df_mode.columns[dp]] = y[0] #replace nan value with concentration value at start/end point
                            df_mode_end.loc[end_time_UTC,df_mode.columns[dp]] = y[-1]
                            
                            dfs_mode_start.append(df_mode_start) #add to list of dfs
                            dfs_mode_end.append(df_mode_end)
                            
                            fitting_params.append([start_time_days, end_time_days, params[0] + np.min(df_mode.iloc[:,dp]), params[1]+np.min(time_d_range), params[2]]) #start/end times of modes (days) + gaussian fit parameters
                            
                            #plot start and end time
                            #plt.plot(start_times[n], start_diams[m], '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
                            #plt.plot(end_times[n], end_diams[m], '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)                                 
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

    max_conc_time = list_days_into_UTC(max_conc_time,time_d_res) #change days to UTC in the list of max concentration times

    return max_conc_time, max_conc_diameter, df_modes, dfs_mode_start, dfs_mode_end, max_conc, fitting_params, threshold_deriv

def appearance_time(): #appearance time (modified from Janne's code)
    #create lists for results
    appear_time = []
    appear_diameter = []

    df_1st_derivatives = cal_1st_derivative(df,time_range=time_d_res) #calculate derivative
    df_modes = find_modes(df,df_1st_derivatives,threshold_deriv=0.03,threshold_diff=10000) #find modes

    #logistic fit to every mode
    for dp in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for t in range(len(df_modes.index)):
            concentration = df_modes.iloc[t,dp] - np.min(df_modes.iloc[:,dp]) #find concentration values from the dataframe (y)
            time = time_d_res[t] - np.min(time_d_res)

            if np.isnan(concentration) and len(y) != 0: #logistic fit when all values of one mode have been added to the y list

                #gaussian fit to get maximum concentration values
                mu=np.mean(x) #parameters
                sigma = np.std(x)
                a = np.max(y)

                try: 
                    params,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma])
                    if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                        print("Peak outside range. Skipping.")
                    else:
                        max_conc = params[0]
    
                except:
                    print("Diverges. Skipping.")

                
                #limit x and y to values between start time of mode and maximum concentration time in mode
                max_conc_index = closest(y, max_conc)
                x = x[:max_conc_index]
                y = y[:max_conc_index]
                
                #logistic fit
                if len(y) != 0:
                    L = np.max(y) #maximum value (concentration)
                    x0 = np.nanmean(x) #midpoint x value
                    k = 1.0 #growth rate

                    try: 
                        params,pcov = curve_fit(logistic,x,y,p0=[L,x0,k])
                        if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                            print("Peak outside range. Skipping.")
                        else:
                            appear_time = np.append(appear_time,params[1] + np.min(time_d_res)) #make a list of times with the appearance time in each diameter
                            appear_diameter = np.append(appear_diameter,float(df_modes.columns[dp])) 
                            
                    except:
                        print("Diverges. Skipping.")
                else:
                        print("NO Y VALUES IN THIS ROW/MAXIMUM IS THE FIRST VALUE OR THIS RANGE")
                
                x = [] #reset
                y = [] 
            elif not np.isnan(concentration): #separates mode values
                x = np.append(x,time)
                y = np.append(y,concentration)
            elif np.isnan(concentration): #skips nan values
                x = [] #reset
                y = []

    appear_time = list_days_into_UTC(appear_time,time_d_res) #change days to UTC in the list of max concentration times

    return appear_time, appear_diameter
def appearance_time_ranges(): #appearance time (modified from Janne's code)
    #create lists for results
    appear_time = []
    appear_diameter = []
    mid_conc = []
    fitting_params = []

    df_ranges = find_ranges()

    for df_range in df_ranges: #go through every range around GRs
        
        time_range = df.index.intersection(df_range.index) #find matching indices
        indices = [df.index.get_loc(row) for row in time_range]
        time_d_range = time_d_res[indices] #define time in days again depending on chosen range
        
        df_range_deriv = cal_1st_derivative(df_range,time_range=time_d_range) #calculate derivative
        df_mode = find_modes(df_range,df_range_deriv,threshold_deriv=0.05,threshold_diff=5000) #find modes

        #logistic fit to every mode
        for dp in range(len(df_mode.columns)):
            x = [] #time
            y = [] #concentration
            for t in range(len(df_mode.index)):
                concentration = df_mode.iloc[t,dp] - np.min(df_mode.iloc[:,dp]) #find concentration values from the dataframe (y)
                time = time_d_range[t] - np.min(time_d_range)

                if np.isnan(concentration) and len(y) != 0: #logistic fit when all values of one mode have been added to the y list
                    
                    #gaussian fit to get maximum concentration values
                    mu=np.mean(x) #parameters
                    sigma = np.std(x)
                    a = np.max(y)

                    try: 
                        params,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma])
                        if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                            print("Peak outside range. Skipping.")
                        else:
                            max_conc = params[0]

                            #limit x and y to values between start time of mode and maximum concentration time in mode
                            max_conc_index = closest(y, max_conc)
                            x_sliced = x[:max_conc_index]
                            y_sliced = y[:max_conc_index]
                        
                            
                            #logistic fit
                            if len(y_sliced) != 0:
                                L = np.max(y_sliced) #maximum value (concentration)
                                x0 = np.nanmean(x_sliced) #midpoint x value
                                k = 1.0 #growth rate

                                try: 
                                    params,pcov = curve_fit(logistic,x_sliced,y_sliced,p0=[L,x0,k])
                                    if ((params[1]>=x_sliced.max()) | (params[1]<=x_sliced.min())): #checking that the peak is within time range
                                        print("Peak outside range. Skipping.")
                                    else:
                                        appear_time = np.append(appear_time,params[1] + np.min(time_d_range)) #make a list of times with the appearance time in each diameter
                                        appear_diameter = np.append(appear_diameter,float(df_mode.columns[dp])) 
                                        
                                        mid_conc_index = closest(x_sliced, params[1]) #find closest value in x to the calculated parameter for appearance time
                                        mid_conc.append(y_sliced[mid_conc_index] + np.min(df_mode.iloc[:,dp])) #appearance time concentration (~50% maximum concentration) 

                                        fitting_params.append([x[0]+ np.min(time_d_range),x[-1]+ np.min(time_d_range),params[0]+ np.min(df_mode.iloc[:,dp]),params[1]+np.min(time_d_range),params[2]]) #logistic fit parameters with time range
                                        

                                        #plot start and end time
                                        #plt.plot(start_times[n], start_diams[m], '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
                                        #plt.plot(end_times[n], end_diams[m], '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)
                                        
                                except:
                                    print("Diverges. Skipping.")
                        
                            else:
                                print("NO Y VALUES IN THIS ROW")
        
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

    appear_time = list_days_into_UTC(appear_time,time_d_res) #change days to UTC in the list of max concentration times

    return appear_time, appear_diameter, mid_conc, fitting_params


#################### PLOTTING ######################


def plot_PSD(dataframe):
    #plot line when day changes
    new_day = None
    for i in df.index:
        day = i.strftime("%d")  
        if day != new_day:
            plt.axvline(x=i, color='black', linestyle='-', linewidth=0.5)
        new_day = day
    
    #x_max_con,y_max_con = max_con()
    #x_appearance, y_appearance = appearance_time()
    x_maxcon,y_maxcon,*others = maxcon_modefit()
    x_appearance_ranges, y_appearance_ranges,*others = appearance_time_ranges()

    #plt.plot(x_max_con, y_max_con, '*', alpha=0.5, color='white', ms=5) #without using mode fitting
    #plt.plot(x_appearance, y_appearance, '*', alpha=0.5, color='green', ms=5) #without using mode fitting 
    plt.plot(x_maxcon,y_maxcon, '*', alpha=0.5, color='white', ms=5,label='maximum concentration') #using mode fitting 
    plt.plot(x_appearance_ranges, y_appearance_ranges, '*', alpha=0.5, color='green', ms=5,label='appearance time') #using mode fitting 
    
    plt.legend(fontsize=6,fancybox=False,framealpha=0.9)
    for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
        legend_handle.set_markeredgewidth(0.5)
        legend_handle.set_markeredgecolor("black")
                    
    plt.yscale('log')
    plt.xlim(dataframe.index[2],dataframe.index[-3])
plot_PSD(df)


def plot_channel(dataframe,diameter_list,draw_range_edges):
    '''
    Plots chosen diameter channels over UTC time, with thresholds and gaussian fit.
    ax[0,0] = whole channel over time, with ranges      ax[0,1] = derivative of concentrations
    ax[n,0] = n amount of channels                      ax[n,1] = derivative of concentrations
                                                ...
    Inputs dataframe with data and diameters (numerical).
    range_edges = True, draws the edges of ranges around growth rates
    Assumes chosen channel has modes that have been found with the maximum concentration method!!!
    '''   

    x_maxcon,y_maxcon,df_modes, dfs_mode_start, dfs_mode_end, max_conc, fitting_parameters_gaus, threshold_deriv = maxcon_modefit()
    appear_time, appear_diameter, mid_conc,fitting_parameters_logi = appearance_time_ranges()

    mode_edges = []             #[(diameter,start_time (UTC),end_time (UTC)), ...]
    range_edges = []            #[(diameter,start_time (UTC),end_time (UTC)), ...]
    xy_maxcon =  []             #[(max con diameter, max con time (UTC), max con), ...]
    fitting_params_gaus = []    #[(max con diameter, mode start time (days), mode end time (days), *params...), ...]
    fitting_params_logi = []    #[(appearance time diameter, mode start time (days), mode end time (days), *params...), ...]
    mode_times_UTC = []         #[(diameter, time UTC), ...]
    appearances = []            #[(diameter, time (UTC), concentration), ...] 

    #finding data with chosen diameters
    for diam in diameter_list:
        #MAXIMUM CONCENTRATION & TIME
        indices = [i for i, a in enumerate(y_maxcon) if a == diam]
        xy_maxcons = [(y_maxcon[b],x_maxcon[b],max_conc[b]) for b in indices]
        [xy_maxcon.append(i) for i in xy_maxcons]

        #FITTING PARAMETERS FOR GAUSSIAN FIT
        indices = [i for i, a in enumerate(y_maxcon) if a == diam]
        fittings = [(y_maxcon[b],fitting_parameters_gaus[b][0],fitting_parameters_gaus[b][1],fitting_parameters_gaus[b][2],fitting_parameters_gaus[b][3],fitting_parameters_gaus[b][4]) for b in indices]
        [fitting_params_gaus.append(i) for i in fittings]

        #FITTING PARAMETERS FOR LOGISTIC FIT
        indices = [i for i, a in enumerate(appear_diameter) if a == diam]
        fittings = [(appear_diameter[b],fitting_parameters_logi[b][0],fitting_parameters_logi[b][1],fitting_parameters_logi[b][2],fitting_parameters_logi[b][3],fitting_parameters_logi[b][4]) for b in indices]
        [fitting_params_logi.append(i) for i in fittings]

        #APPEARANCE TIME & CONCENTRATION
        indices = [i for i, a in enumerate(appear_diameter) if a == diam]
        appearance = [(appear_diameter[b],appear_time[b],mid_conc[b]) for b in indices]
        [appearances.append(i) for i in appearance]
        
        for df_mode_start,df_mode_end in zip(dfs_mode_start,dfs_mode_end):
            try:
                #START/END TIME & TIME RANGE OF MODE
                start_time = df_mode_start[diam].notna().idxmax() #finding the start/end value in df
                print("start time!!",start_time)
                print("df mode start:", df_mode_start)
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
                print("Channel not in this range.")

    #find unique values
    xy_maxcon = list(dict.fromkeys(xy_maxcon))


    #####   PLOTTING   #####
    fig, ax1 = plt.subplots(len(diameter_list),2,figsize=(20, 14), dpi=90)

    #define x and y for the whole channel
    x = dataframe.index #time
    y_list = [] #concentrations
    for diam in diameter_list:
        y = dataframe[diam]
        y_list.append(y)


    #concentration
    row_num = 0 #keeps track of which row of figure we are plotting in
    for y in y_list:
        ax1[row_num,0].set_title(f'diameter: ≈{diameter_list[row_num]:.2f} nm', loc='left', fontsize=10) #diameter titles

        #left axis (normal scale)
        color1 = "royalblue"
        ax1[row_num,0].set_ylabel("dN/dlogDp [cm^(-3)]", color=color1)
        ax1[row_num,0].plot(x, y, color=color1)
        ax1[row_num,0].tick_params(axis='y', labelcolor=color1)

        #start and end points of modes
        found_ranges = []
        for i in mode_edges:
            diam, start, end = i
            if diam == diameter_list[row_num] and i not in found_ranges:
                line1 = ax1[row_num,0].axvline(x=start, color='black', linestyle='--', linewidth=1)
                ax1[row_num,0].axvline(x=end, color='black', linestyle='--', linewidth=1)
                ax1[row_num,0].annotate(f'S', (start, float(np.max(y))), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=8, fontweight='bold',color='black')
                ax1[row_num,0].annotate(f'E', (end, float(np.max(y))), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=8, fontweight='bold',color='black')
                found_ranges.append(i) #plot one range once
        

        #right axis (logarithmic scale)
        ax2 = ax1[row_num,0].twinx()
        color2 = "green"
        ax2.set_ylabel("log10(dN/dlogDp) [cm^(-3)]", color=color2) 
        ax2.plot(x, y, color=color2)
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
            
            start_time_UTC = pd.Series(list_days_into_UTC(start_time-0.25/24,time_d_res)).round('30T')[0] #change days into UTC
            end_time_UTC = pd.Series(list_days_into_UTC(end_time-0.25/24,time_d_res)).round('30T')[0] #RESOLUTION IS 30MINS HERE
            start_time_UTC += timedelta(minutes=15) #makes it easier to round
            end_time_UTC += timedelta(minutes=15)

            for time_UTC,time_days in zip(mode_times_UTC,mode_times_days):
                if diam == diameter_list[row_num] and time_UTC[1][0] == start_time_UTC and time_UTC[1][-1] == end_time_UTC: #check that plotting happens in the right mode
                    line2, = ax1[row_num,0].plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="orange")
                    ax2.plot(time_UTC[1], gaussian(time_days,a,mu,sigma), '--', color="orange")
        
        #logistic
        for params in fitting_params_logi:
            diam1, start_time, end_time, L, x0, k = params
           
            start_time_UTC = pd.Series(list_days_into_UTC(start_time-0.25/24,time_d_res)).round('30T')[0] #change days into UTC
            end_time_UTC = pd.Series(list_days_into_UTC(end_time-0.25/24,time_d_res)).round('30T')[0]
            start_time_UTC += timedelta(minutes=15) #makes it easier to round
            end_time_UTC += timedelta(minutes=15)

            for time_UTC,time_days in zip(mode_times_UTC,mode_times_days):     
                if diam1 == diameter_list[row_num] and time_UTC[1][0] == start_time_UTC and time_UTC[1][-1] == end_time_UTC: #check that plotting happens in the right mode
                    line3, = ax1[row_num,0].plot(time_UTC[1], logistic(time_days,L,x0,k), '--', color="yellow")
                    ax2.plot(time_UTC[1], logistic(time_days,L,x0,k), '--', color="yellow")

        #maximum concentration
        for i in xy_maxcon:
            diam, x_maxcon, y_maxcon = i
            if diam == diameter_list[row_num]:
                line4, = ax1[row_num,0].plot(x_maxcon, y_maxcon, '*', color="white", ms=8,alpha=0.8)
                ax2.plot(x_maxcon, y_maxcon, '*', color="white", ms=8,alpha=0.8)

        #appearance time
        for i in appearances:
            diam, time, conc = i
            if diam == diameter_list[row_num]:
                line5, = ax1[row_num,0].plot(time, conc, '*', color="green", ms=8,alpha=0.8)
                ax2.plot(time, conc, '*', color="green", ms=8,alpha=0.8)
        
        ax1[row_num,0].set_xlim(dataframe.index[0],dataframe.index[-1])
        ax1[row_num,0].set_facecolor("lightgray")

        row_num += 1


    #DERIVATIVE
    df_1st_derivatives = cal_1st_derivative(df,time_range=time_d_res)

    x = df_1st_derivatives.index #time
    y_list = [] #concentrations
    for diam in diameter_list:
        y = df_1st_derivatives[diam]
        y_list.append(y)

    row_num = 0 #keeps track of which row of figure we are plotting in
    for y in y_list:
        #left axis
        color1 = "royalblue"
        ax1[row_num,1].set_ylabel("d(dN/dlogDp)dt [cm^(-3)*s^(-1)]", color=color1)
        ax1[row_num,1].plot(x, y, color=color1)
        ax1[row_num,1].tick_params(axis='y', labelcolor=color1)
        line6 = ax1[row_num,1].axhline(y=threshold_deriv, color="royalblue", linestyle='--', linewidth=1)
        ax1[row_num,1].axhline(y=-threshold_deriv, color="royalblue", linestyle='--', linewidth=1)
        
        #start and end points of modes
        found_ranges = []
        for i in mode_edges:
            diam, start, end = i
            if diam == diameter_list[row_num] and i not in found_ranges:
                line7 = ax1[row_num,1].axvline(x=start, color='black', linestyle='--', linewidth=1)
                ax1[row_num,1].axvline(x=end, color='black', linestyle='--', linewidth=1)
                ax1[row_num,1].annotate(f'S', (start, float(np.max(y))), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=8, fontweight='bold',color='black')
                ax1[row_num,1].annotate(f'E', (end, float(np.max(y))), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=8, fontweight='bold',color='black')
                found_ranges.append(i) #plot one range once
        
        if draw_range_edges == True:
            #start and end points of ranges
            cmap = plt.get_cmap('Dark2') #colors for each pair of start and end
            found_ranges = []
            num = 0
            for edges in range_edges:
                diam, start, end = edges
                if diam == diameter_list[row_num] and edges not in found_ranges:
                    color = cmap(num)
                    if num == 0:
                        line8 = ax1[row_num,1].axvline(x=start,  color=color, linestyle='--', linewidth=1)
                    else:
                        ax1[row_num,1].axvline(x=start,  color=color, linestyle='--', linewidth=1)
                    ax1[row_num,1].axvline(x=end, color=color, linestyle='--', linewidth=1)
                    ax1[row_num,1].annotate(f'S', (start, float(np.max(y))), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=8, fontweight='bold', color=color)
                    ax1[row_num,1].annotate(f'E', (end, float(np.max(y))), textcoords="offset points", xytext=(0, 0), ha='center', fontsize=8, fontweight='bold',color=color)
                    found_ranges.append(edges) #plot one range once
                    num += 0.125

        #maximum concentration
        for i in xy_maxcon:
            diam, x_maxcon, y_maxcon = i
            y_maxcon = y_maxcon*0 #to place the start lower where y = 0
            if diam == diameter_list[row_num]:
                ax1[row_num,1].plot(x_maxcon, y_maxcon, '*', color="white",ms=8,alpha=0.8)
        
        #appearance time
        for i in appearances:
            diam, time, conc = i
            conc = conc*0 #to place the start lower where y = 0
            if diam == diameter_list[row_num]:
                ax1[row_num,1].plot(time, conc, '*', color="green", ms=8,alpha=0.8)
                ax2.plot(time, conc, '*', color="green", ms=8,alpha=0.8)

        ax1[row_num,1].set_xlim(dataframe.index[0],dataframe.index[-1])
        ax1[row_num,1].set_facecolor("lightgray")

        """ Logarithmic derivate
        #right axis
        ax2 = ax1[row_num,1].twinx()
        color2 = "green"
        ax2.set_ylabel("log10(d(dN/dlogDp)dt)", color=color2) 
        ax2.plot(x, y, color=color2,zorder=1)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        ax2.axhline(y=0.03, color='green', linestyle='--', linewidth=1, label="threshold = 0.03",zorder=1)
        ax2.axhline(y=-0.03, color='green', linestyle='--', linewidth=1,zorder=1)
        """

        row_num += 1

    #common titles
    ax1[0,0].set_title("Concentration") 
    ax1[0,1].set_title("Derivative")
    ax1[2,0].set_xlabel("Time (UTC)")
    ax1[2,1].set_xlabel("Time (UTC)")
    
    #common legends
    ax1[1,0].legend([line1,line2,line3,line4,line5],["mode edges","gaussian fit","logistic fit","maximum concentration","appearance time"],fontsize=10,fancybox=False,framealpha=0.9,loc='center right', bbox_to_anchor=(-0.13, 0.5))
    if draw_range_edges == True:
        ax1[1,1].legend([line6,line7,line8],[f"threshold = {str(threshold_deriv)}","mode edges","range edges"],fontsize=10,fancybox=False,framealpha=0.9,loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax1[1,1].legend([line6,line7],[f"threshold = {str(threshold_deriv)}","mode edges"],fontsize=10,fancybox=False,framealpha=0.9,loc='center left', bbox_to_anchor=(1, 0.5))
    
    #set black edges to star markers in the legend
    white_star = ax1[1,0].get_legend().legend_handles[3]
    green_star = ax1[1,0].get_legend().legend_handles[4]
    white_star.set_markeredgewidth(0.5)
    green_star.set_markeredgewidth(0.5)
    white_star.set_markeredgecolor("black")
    green_star.set_markeredgecolor("black")

    #ax1[0,1].xaxis.set_major_locator(plt.MaxNLocator(8))
    #ax1[1,1].xaxis.set_major_locator(plt.MaxNLocator(8))
    #ax1[2,1].xaxis.set_major_locator(plt.MaxNLocator(8))

    fig.tight_layout()
plot_channel(df,[9.0794158,15.816562000000001,47.609805],draw_range_edges=False)

plt.show()

####################################################
#INSTEAD OF THIS A MEDIAN FILTER WOULD FIX IT TOO
#filter lonely datapoints (if 1 or 2 consecutive datapoints are lonely)
#df_filtered = pd.DataFrame(np.nan, index=df.index, columns=df.columns) #dataframe with the same size as df and values of 0 initially
#lonely_datapoints = (df.shift(1) == 0) & (df.shift(-1) == 0) | (df.shift(2) == 0) & (df.shift(-1) == 0) | (df.shift(1) == 0) & (df.shift(-2) == 0)
#df_filtered = df[~lonely_datapoints]

#previous ideas
"""

def slopes(dataframe):
    df_slopes = pd.DataFrame()
    for j in dataframe.columns:
        #slope of data with a window of 5 (matching the median filter window) centered
        df_slopes[j] = (dataframe[j].shift(-2) - dataframe[j].shift(2)) / 5
    df_slopes = df_slopes.dropna() #drop nan values
    return df_slopes


#lets look at the concentration dataframe one column=diameter at a time
for k in df.columns:
    #checking if a datapoint has zeros around it or only two values, returns boolean #filters lonely datapoints
    lone_datapoints_df = (df[k].shift(1) == 0) & (df[k].shift(-1) == 0) | (df[k].shift(2) == 0) & (df[k].shift(-1) == 0) | (df[k].shift(1) == 0) & (df[k].shift(-2) == 0)
    lone_datapoints_df_slopes = (df_slopes[k].shift(1) == 0) & (df_slopes[k].shift(-1) == 0) | (df_slopes[k].shift(2) == 0) & (df_slopes[k].shift(-1) == 0) | (df_slopes[k].shift(1) == 0) & (df_slopes[k].shift(-2) == 0)
    
    #remove the global min value from datapoints, additionally
    #k_reduced = df[~lone_datapoints][k] - df_min[k].min() - df_max[k].max()*0.05 PROBABLY USELESS
    
    #lets set a value that determines what concentration will be used to find the start and end of a mode
    N_limit = df_max[k].max() * 0.05  #global maximum value in one diameter * 0.05
     
    for l in df[~lone_datapoints_df][k]: #go through values in one column of filtered datapoints

        #find the closest value to N_limit FINDS JUST ONE VALUEE!!! :(
        closest_value = df.loc[(df['A'] - target_value).abs().idxmin(), 'A']


        timestamp = df[~lone_datapoints_df].loc[df[~lone_datapoints_df][k] == l].index[0] #finds the timestap of current datapoint 
        #if the slopes corresponding to the same timestamp as concentration values are positive add them as starting points
        if df_slopes[timestamp][k] >= 0 and :
            df_start[k] = l
        else: #if theyre negative add them as ending points



#filter lonely datapoints with 1 or two consecutive value not equal to 0
#df_slopes_filtered = pd.DataFrame()
for k in df_slopes.columns:
    #checking if a datapoint has zeros around it or only two slope values, returns boolean
    lone_datapoints = (df_slopes[k].shift(1) == 0) & (df_slopes[k].shift(-1) == 0) | (df_slopes[k].shift(2) == 0) & (df_slopes[k].shift(-1) == 0) | (df_slopes[k].shift(1) == 0) & (df_slopes[k].shift(-2) == 0)

    #filters away lone datapoints
    #df_slopes_filtered[f"filtered_{k}"] = df_slopes[~lone_datapoints][k]
    for l in df_slopes[~lone_datapoints][k]: #go through slope values in one column
        if l == 0: #skip values of 0
            continue
        if l >= 80: #trying if slopes of >80 is a big enough speed
            if df_slopes[~lone_datapoints][k] for a in range() : #checking if the slope is the first value of a mode
                df_start
        elif <= -80:
        
        else:
            continue
"""


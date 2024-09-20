from GR_calculator_unfer_v2_modified import * 
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

def median_filter(dataframe,window): #smoothens data                  
    '''
    Smoothens data in dataframe with median filter and given window.
    Returns smoothened dataframe.
    '''
    
    for i in dataframe.columns:
        dataframe[i] = dataframe[i].rolling(window=window, center=True).median()
    dataframe.dropna(inplace=True)
    return dataframe 
df.rename(columns=dict(zip(df.columns[[0,1]], ["time (d)", "total number concentration (N)"])), inplace=True) #rename first two columns
df = df.drop(['total number concentration (N)'], axis=1) #drop total N concentrations from the dataframe as they're not needed
time_d = df['time (d)'].values.astype(float) #save time as days before changing them to UTC

#change days into UTC (assuming time is in "days from start of measurement"):
def df_days_into_UTC(dataframe): #for a dataframe
    '''
    Changes days into UTC datetime.
    Start date is from the first given dmps data file.
    Returns dataframe with datetimes.
    '''
    time_steps = dataframe["time (d)"] - time_d[0] #calculate timesteps between every timestamp
    start_date_str = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00" #define start date
    start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S") #change to date time
    dataframe["time (d)"] = [start_datetime + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
    return dataframe
def list_days_into_UTC(list): #for a list
    time_steps = list - time_d[0]
    start_date_str = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00"
    start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    list = [start_datetime + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
    return list
df = df_days_into_UTC(df)

#set new UTC timestamps as indices
df.rename(columns={'time (d)': 'Timestamp (UTC)'}, inplace=True)
df['Timestamp (UTC)']=pd.to_datetime(df['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S")
df.index=df['Timestamp (UTC)']
df = df.drop(['Timestamp (UTC)'], axis=1)

#set numerical diameters as column headers
df.columns = pd.to_numeric(df.columns) * 10**9 #change units from m to nm

#with this we can check the format
#df.to_csv('./data_format_filter.csv', sep=',', header=True, index=True, na_rep='nan')
df.to_csv('./data_format_nofilter.csv', sep=',', header=True, index=True, na_rep='nan')

#NO RESAMPLING UNLIKE IN GABIS CODE
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
    return  median_filter(df_derivatives,window=5) #filter after derivating
#df_1st_derivatives = cal_1st_derivative(df,time_range=time_d)

#with this we can check the format
#cal_1st_derivative(df).to_csv('./1st_derivatives_nofilter.csv', sep=',', header=True, index=True, na_rep='nan')
#cal_1st_derivative(df).to_csv('./1st_derivatives_filterbefore.csv', sep=',', header=True, index=True, na_rep='nan')
cal_1st_derivative(df,time_range=time_d).to_csv('./1st_derivatives_filterafter.csv', sep=',', header=True, index=True, na_rep='nan')
#cal_1st_derivative(df).to_csv('./1st_derivatives_logconc_nofilter.csv', sep=',', header=True, index=True, na_rep='nan') 
#cal_1st_derivative(df).to_csv('./1st_derivatives_logconc_filterbefore.csv', sep=',', header=True, index=True, na_rep='nan')
#cal_1st_derivative(df).to_csv('./1st_derivatives_logconc_filterafter.csv', sep=',', header=True, index=True, na_rep='nan')

def cal_2nd_derivative(dataframe):
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns)
    
    #calculate derivatives for each diameter
    for i in dataframe.columns: 
        dNdt = dataframe[i] #1st derivative
        time = time_d * 60 * 60 #change days to seconds
        second_dNdt = np.diff(dNdt)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = second_dNdt
    return df_derivatives
df_2nd_derivatives = cal_2nd_derivative(df)

#minimum and maximum
def cal_min_max(df_1st_deriv,df_2nd_deriv): #returns [df_max, df_min]
    df_1stderiv_zeros = find_zero(df_1st_deriv) #dataframe with stationary points of the 1st derivative
    #find max and min values
    df_max = df[(df_1stderiv_zeros == 0) & (df_2nd_deriv < 0)] #stationary points exist, 2nd derivative is negative
    df_min = df[(df_1stderiv_zeros == 0) & (df_2nd_deriv > 0)] #stationary points exist, 2nd derivative is positive
    return [df_max,df_min]
#df_max = cal_min_max(df_1st_derivatives,df_2nd_derivatives)[0]

#finding modes with derivative threshold
def find_modes_1st_deriv(dataframe,df_deriv,threshold_deriv,threshold_diff): #finds modes, derivative threshold
    df_mode_ranges = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns)
    start_time_list = []
    start_diam_list = []
    end_time_list = []
    end_diam_list = []

    #threshold that determines what is a high concentration
    threshold = abs(df_deriv) > threshold_deriv #checks which values surpass the threshold ###NO LOG FROM CONC###
    #threshold = abs(df_deriv) > 0.000009 #checks which values surpass the threshold ###LOG FROM CONC###
    
    #threshold = (df_deriv > df_deriv.shift(1))

    #determine start and end points
    start_points = threshold & (~threshold.shift(1,fill_value=False))
    end_points = threshold & (~threshold.shift(-1,fill_value=False))

    #start_points.to_csv('./START.csv', sep=',', header=True, index=True, na_rep='nan')
    #end_points.to_csv('./END.csv', sep=',', header=True, index=True, na_rep='nan')

    #finding values within a mode
    start_times_temp = [] #initial values
    start_time = None 
    end_time = None
    max_con_diff = threshold_diff #maximum concentration difference between the start and end times

    for k in df_deriv.columns:
        for l in df_deriv.index: #identify pairs of start and end times
            if start_time != None and end_time != None and abs(dataframe.loc[start_time,k]-dataframe.loc[end_time,k]) <= max_con_diff: #when start and end times have their values find list of values between them
                subset = dataframe[k].loc[start_time:end_time]
                #fill dataframe with mode ranges
                df_mode_ranges.loc[subset.index, k] = subset

                
                #save start and end times with their diameters in lists
                start_time_list.append(start_time)
                start_diam_list.append(k)
                end_time_list.append(end_time)
                end_diam_list.append(k)
                
                #restart initial values
                start_time = None 
                end_time = None
                start_times_temp = []
            
            #choose a previous start time if the difference of concentration values is too big
            elif start_time != None and end_time != None and abs(dataframe.loc[start_time,k]-dataframe.loc[end_time,k]) > max_con_diff:
                for m in reversed(start_times_temp):
                    start_time = m
                    if abs(dataframe.loc[start_time,k]-dataframe.loc[end_time,k]) <= max_con_diff:
                        break
                                  
            elif start_points.loc[l,k] == True and df_deriv.loc[l,k] > 0: #checks also if the starting point derivative is neg or not
                start_time = l
                start_times_temp.append(start_time)
            elif end_points.loc[l,k] == True and start_time != None and df_deriv.loc[l,k] < 0:
                end_time = l
            else:
                continue
    
    #plot start and end points
    #plt.plot(start_time_list, start_diam_list, '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
    #plt.plot(end_time_list, end_diam_list, '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)
    
    #with this we can check the format
    #find_modes_1st_deriv(df,df_1st_derivatives,threshold_deriv=0.03,threshold_diff=2000)[0].to_csv('./modes.csv', sep=',', header=True, index=True, na_rep='nan')

    return df_mode_ranges, start_time_list, start_diam_list, end_time_list, end_diam_list
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
    threshold = 5 #GR [nm/h]

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
    df_1st_derivatives = cal_1st_derivative(df,time_range=time_d) #calculate derivative
    df_modes, start_times, start_diams, end_times, end_diams = find_modes_1st_deriv(df,df_1st_derivatives,threshold_deriv=0.03,threshold_diff=10000)

    #gaussian fit to every mode
    max_conc_time = []
    max_conc_diameter = []

    for i in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for j in range(len(df_modes.index)):
            concentration = df_modes.iloc[j,i] #find concentration values from the dataframe (y)
            time = time_d[j] - np.min(time_d)

            if np.isnan(concentration) and len(y) != 0: #gaussian fit when all values of one mode have been added to the y list
                mu=np.mean(x) #parameters
                sigma = np.std(x)
                a = np.max(y)

                try: #gaussian fit
                    params,pcov = curve_fit(gaussian,x,y,p0=[a,mu,sigma])
                    if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                        print("Peak outside range. Skipping.")
                    else:
                        max_conc_time = np.append(max_conc_time,params[1] + np.min(time_d)) #make a list of times with the max concentration time in each diameter
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

    max_conc_time = list_days_into_UTC(max_conc_time) #change days to UTC in the list of max concentration times
    
    return max_conc_time, max_conc_diameter
def maxcon_modefit(): #find ranges around growth rates, use these points to calculate the maximum concentration
    #create lists for results
    max_conc_time = []
    max_conc_diameter = []
    dfs_range_start = []
    dfs_range_end = [] 
    df_modes = []

    df_ranges = find_ranges()

    for df_range in df_ranges: #go through every range around GRs

        time_range = df.index.intersection(df_range.index) #find matching indices
        indices = [df.index.get_loc(row) for row in time_range]
        time_d_range = time_d[indices] #define time in days again depending on chosen range
        
        df_range_deriv = cal_1st_derivative(df_range,time_range=time_d_range) #calculate derivative
        df_mode = find_modes_1st_deriv(df_range,df_range_deriv,threshold_deriv=0.03,threshold_diff=5000)[0] #find modes
        df_modes.append(df_mode) #add dfs to list

        #gaussian fit to every mode
        for m in range(len(df_mode.columns)):
            x = [] #time
            y = [] #concentration
            for n in range(len(df_mode.index)):
                concentration = df_mode.iloc[n,m] #find concentration values from the dataframe (y)
                time = time_d_range[n] - np.min(time_d_range)

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
                            max_conc_diameter = np.append(max_conc_diameter,float(df_mode.columns[m])) #make list of diameters with max concentrations

                            #create dfs with start and end points
                            df_range_start = pd.DataFrame(np.nan, index=df_range.index, columns=df_range.columns)
                            df_range_end = pd.DataFrame(np.nan, index=df_range.index, columns=df_range.columns)
                            
                            start_time = list_days_into_UTC([x[0] + np.min(time_d_range)]) #define start/end time
                            end_time = list_days_into_UTC([x[-1] + np.min(time_d_range)])

                            df_range_start.loc[start_time[0],df_mode.columns[m]] = y[0] #replace nan value with concentration value at start/end point
                            df_range_end.loc[end_time[0],df_mode.columns[m]] = y[-1]

                            dfs_range_start.append(df_range_start) #add to list of dfs
                            dfs_range_end.append(df_range_end)
 

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

    max_conc_time = list_days_into_UTC(max_conc_time) #change days to UTC in the list of max concentration times
    
    return max_conc_time, max_conc_diameter, df_modes, dfs_range_start, dfs_range_end

def appearance_time(): #appearance time (modified from Janne's code)
    #create lists for results
    appear_time = []
    appear_diameter = []

    df_1st_derivatives = cal_1st_derivative(df,time_range=time_d) #calculate derivative
    df_modes = find_modes_1st_deriv(df,df_1st_derivatives,threshold_deriv=0.03,threshold_diff=10000)[0] #find modes

    #logistic fit to every mode
    for dp in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for t in range(len(df_modes.index)):
            concentration = df_modes.iloc[t,dp] - np.min(df_modes.iloc[:,dp]) #find concentration values from the dataframe (y)
            time = time_d[t] - np.min(time_d)

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
                            appear_time = np.append(appear_time,params[1] + np.min(time_d)) #make a list of times with the appearance time in each diameter
                            appear_diameter = np.append(appear_diameter,float(df_modes.columns[dp])) 


                            #plot start and end time
                            #plt.plot(start_times[n], start_diams[m], '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
                            #plt.plot(end_times[n], end_diams[m], '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)
                            
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

    appear_time = list_days_into_UTC(appear_time) #change days to UTC in the list of max concentration times

    return appear_time, appear_diameter
def appearance_time_ranges(): #appearance time (modified from Janne's code)
    #create lists for results
    appear_time = []
    appear_diameter = []

    df_ranges = find_ranges()
    #max_conc_times, max_conc_diameters = maxcon_modefit() #timestamps and diameters of maximum concentrations in ranges

    for df_range in df_ranges: #go through every range around GRs
        
        time_range = df.index.intersection(df_range.index) #find matching indices
        indices = [df.index.get_loc(row) for row in time_range]
        time_d_range = time_d[indices] #define time in days again depending on chosen range
        
        df_range_deriv = cal_1st_derivative(df_range,time_range=time_d_range) #calculate derivative
        df_modes = find_modes_1st_deriv(df_range,df_range_deriv,threshold_deriv=0.03,threshold_diff=5000)[0] #find modes

        #logistic fit to every mode
        for dp in range(len(df_modes.columns)):
            x = [] #time
            y = [] #concentration
            for t in range(len(df_modes.index)):
                concentration = df_modes.iloc[t,dp] - np.min(df_modes.iloc[:,dp]) #find concentration values from the dataframe (y)
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
                                if df_modes.columns[dp] == 9.0794158:
                                    print("HERE1 y: ",y+ np.min(df_modes.iloc[:,dp]))
                            else:
                                appear_time = np.append(appear_time,params[1] + np.min(time_d_range)) #make a list of times with the appearance time in each diameter
                                appear_diameter = np.append(appear_diameter,float(df_modes.columns[dp])) 
                                

                                #plot start and end time
                                #plt.plot(start_times[n], start_diams[m], '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
                                #plt.plot(end_times[n], end_diams[m], '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)
                                
                        except:
                            print("Diverges. Skipping.")
                            if df_modes.columns[dp] == 9.0794158:
                                print("HERE2 y: ",y+ np.min(df_modes.iloc[:,dp]))
                
                    else:
                        print("NO Y VALUES IN THIS ROW")
                        if df_modes.columns[dp] == 9.0794158:
                            print("HERE3 y: ",y+ np.min(df_modes.iloc[:,dp]))

                    
                    x = [] #reset
                    y = [] 
                elif not np.isnan(concentration): #separates mode values
                    x = np.append(x,time)
                    y = np.append(y,concentration)
                elif np.isnan(concentration): #skips nan values
                    x = [] #reset
                    y = []

    appear_time = list_days_into_UTC(appear_time) #change days to UTC in the list of max concentration times

    return appear_time, appear_diameter

#################### PLOTTING ######################

def plot_channel(dataframe,diameter_list):
    '''
    Plots chosen diameter channels over UTC time, with thresholds and gaussian fit.
    ax[0,0] = whole channel over time, with ranges      ax[0,1] = derivative of concentrations
    ax[n,0] = n amount of channels                      ax[n,1] = derivative of concentrations
                                                ...
    Inputs dataframe with data and diameters (numerical).
    '''   
    """
    #calculate derivatives of data in ranges
    df_ranges = find_ranges()
    df_ranges_deriv = []
    df_modes = []
    for df_range in df_ranges:
        time_range = dataframe.index.intersection(df_range.index) #find matching indices
        indices = [dataframe.index.get_loc(row) for row in time_range]
        time_d_range = time_d[indices] #define time in days again depending on chosen range

        df_range_deriv = cal_1st_derivative(df_range,time_range=time_d_range) #calculate derivative
        df_mode = find_modes_1st_deriv(df_range,df_range_deriv,threshold_deriv=0.03,threshold_diff=5000)[0] #find modes
        
        df_ranges_deriv.append(df_range_deriv) #add to lists
        df_modes.append(df_mode)
    """
    x_maxcon,y_maxcon,df_modes, dfs_range_start, dfs_range_end = maxcon_modefit()

    edges = [] #[(diameter,start_time,end_time),...], edges[0]=diameter / edges[1]=start time / edges[2]=end time
    
    #finding start and end times with chosen diameters
    for diam in diameter_list:
        for i,df_range_start,df_range_end in zip(range(len(df_modes)),dfs_range_start,dfs_range_end):
            try:
                #channel = df_modes[i][diam]
                start_time = df_range_start[diam].notna().idxmax()
                end_time = df_range_end[diam].notna().idxmax()

                edges.append((diam,start_time,end_time))

            except KeyError:
                print("Channel not in this range.")
        

    
    ##plotting##
    fig, ax1 = plt.subplots(len(diameter_list),2,figsize=(14, 15), dpi=90)

    #concentration
    #WHOLE CHANNEL
    x = dataframe.index #time
    y_list = [] #concentrations
    for diam in diameter_list:
        y = dataframe[diam]
        y_list.append(y)

    row_num = 0 #keeps track of which row of figure we are plotting in
    for y in y_list:
        ax1[row_num,0].set_title(f'diameter: ≈{diameter_list[row_num]:.2f} nm', loc='left', fontsize=10) #diameter titles

        #left axis
        color1 = "blue"
        ax1[row_num,0].set_ylabel("dN/dlogDp", color=color1)
        ax1[row_num,0].plot(x, y, color=color1)
        ax1[row_num,0].tick_params(axis='y', labelcolor=color1)

        #start and end points of ranges
        ax1[row_num,0].plot(edges[:][1],(edges[:][0]),color="red") #start ???
        ax1[row_num,0].plot(edges[:][2],(edges[:][0]),color="red") #end

        #right axis
        ax2 = ax1[row_num,0].twinx()
        color2 = "green"
        ax2.set_ylabel("log10(dN/dlogDp)", color=color2) 
        ax2.plot(x, y, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        ax1[row_num,0].set_xlim(dataframe.index[0],dataframe.index[-1])

        row_num += 1
    
    #RANGES
    


    #derivative
    #WHOLE CHANNEL
    df_1st_derivatives = cal_1st_derivative(df,time_range=time_d)

    x = df_1st_derivatives.index #time
    y_list = [] #concentrations
    for diam in diameter_list:
        y = df_1st_derivatives[diam]
        y_list.append(y)

    row_num = 0 #keeps track of which row of figure we are plotting in
    for y in y_list:
        #left axis
        color1 = "blue"
        ax1[row_num,1].set_ylabel("d(dN/dlogDp)dt", color=color1)
        ax1[row_num,1].plot(x, y, color=color1)
        ax1[row_num,1].tick_params(axis='y', labelcolor=color1)

        #right axis
        ax2 = ax1[row_num,1].twinx()
        color2 = "green"
        ax2.set_ylabel("log10(d(dN/dlogDp)dt)", color=color2) 
        ax2.plot(x, y, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        ax1[row_num,1].axhline(y=0.03, color='blue', linestyle='--', linewidth=1,label="threshold = 0.03")
        ax1[row_num,1].axhline(y=-0.03, color='blue', linestyle='--', linewidth=1)
        ax2.axhline(y=0.03, color='green', linestyle='--', linewidth=1, label="threshold = 0.03")
        ax2.axhline(y=-0.03, color='green', linestyle='--', linewidth=1)

        ax1[row_num,1].legend(fontsize=10,fancybox=False,framealpha=0.9)
        ax1[row_num,1].set_xlim(df_1st_derivatives.index[0],df_1st_derivatives.index[-1])

        row_num += 1


    #RANGES
    
    ##

    #common titles
    ax1[0,0].set_title("Concentration") 
    ax1[0,1].set_title("Derivative")

    
    fig.tight_layout()
    #plt.show()
    
plot_channel(df,[9.0794158,10.924603,26.678849999999997])



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
    x_maxcon,y_maxcon,list1,list2,list3 = maxcon_modefit()
    x_appearance_ranges, y_appearance_ranges = appearance_time_ranges()

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
    plt.show()
#plot_PSD(df)

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


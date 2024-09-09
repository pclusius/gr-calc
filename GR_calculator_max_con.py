from GR_calculator_unfer_v2_modified import * 
#import GR_calculator_unfer_v2_modified 
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#ASSUMES DMPS DATA

################ DATA FORMATTING ################
folder = r"./dmps Nesrine/" #folder where data files are stored, should be in the same directory as this code
#dm160612.sum

paths = paths #copy user-given paths from the file "GR_calculator_unfer_v2_modified"

#load data for as many days as given
def combine_data():
    dfs = []
    test = True
    #load all given data files and save them a list
    for i in paths:
        df = pd.DataFrame(pd.read_csv(folder + i,sep='\s+',engine='python'))
        #make sure all columns have the same diameter values, name all other columns with the labels of the first one
        if test == True:
            diameter_labels = df.columns
            test = False
        df.rename(columns=dict(zip(df.columns, diameter_labels)), inplace=True)
        dfs.append(df) #add dataframe to list
    #combine datasets
    combined_data = pd.concat(dfs,axis=0,ignore_index=True)
    return combined_data
df = combine_data() #defining dataframe with combined data

def median_filter(dataframe,window): #filter data                     
    for i in dataframe.columns:
        dataframe[i] = dataframe[i].rolling(window=window, center=True).median()
    dataframe.dropna(inplace=True)
    return dataframe
#median_filter(df,window=5) #filter before derivating 

df.rename(columns=dict(zip(df.columns[[0,1]], ["time (d)", "total number concentration (N)"])), inplace=True) #rename first two columns
df = df.drop(['total number concentration (N)'], axis=1) #drop total N concentrations from the dataframe as they're not needed
time_d = df['time (d)'].values.astype(float) #save time as days before changing them to UTC


#change days into UTC (assuming time is in "days from start of measurement") for a dataframe
def df_days_into_UTC(dataframe):
    time_steps = dataframe["time (d)"] - time_d[0]
    start_date_measurement = f"20{paths[0][2:4]}-{paths[0][4:6]}-{paths[0][6:8]} 00:00:00"
    start_date = datetime.strptime(start_date_measurement, "%Y-%m-%d %H:%M:%S")
    dataframe["time (d)"] = [start_date + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
def list_days_into_UTC(list): #same for a list
    time_steps = list - time_d[0]
    start_date_measurement = f"20{paths[0][2:4]}-{paths[0][4:6]}-{paths[0][6:8]} 00:00:00"
    start_date = datetime.strptime(start_date_measurement, "%Y-%m-%d %H:%M:%S")
    list = [start_date + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
    return list
df_days_into_UTC(df)

#set new UTC timestamps as indices, titles start from "total number concentration (N)"
df.rename(columns={'time (d)': 'Timestamp (UTC)'}, inplace=True)
df['Timestamp (UTC)']=pd.to_datetime(df['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S")
df.index=df['Timestamp (UTC)']
df = df.drop(['Timestamp (UTC)'], axis=1)

df.columns = pd.to_numeric(df.columns) * 10**9 #change units of diameters from m to nm

#with this we can check the format
#df.to_csv('./data_format_filter.csv', sep=',', header=True, index=True, na_rep='nan')
#df.to_csv('./data_format_nofilter.csv', sep=',', header=True, index=True, na_rep='nan')

#NO RESAMPLING UNLIKE IN GABIS CODE
####################################################

"""
def local_max_min(dataframe):
    df_max = pd.DataFrame(0, index=dataframe.index, columns=dataframe.columns) #dataframes with the same size as df and values of 0 initially
    df_min = pd.DataFrame(0, index=dataframe.index, columns=dataframe.columns)
    for i in dataframe.columns:
        #comparing to datapoints 2 before and 2 after (same window as median filter 5), returns boolean
        local_max = (dataframe[i] > dataframe[i].shift(2)) & (dataframe[i] > dataframe[i].shift(-2))
        local_min = (dataframe[i] < dataframe[i].shift(2)) & (dataframe[i] < dataframe[i].shift(-2))
        
        #filling new dataframes with maxima and minima, True = df[i] and False = nan
        df_max[i] = dataframe[i].where(local_max, np.nan)
        df_min[i] = dataframe[i].where(local_min, np.nan)
    return df_max,df_min
def slopes(dataframe):
    df_slopes = pd.DataFrame()
    for j in dataframe.columns:
        #slope of data with a window of 5 (matching the median filter window) centered
        df_slopes[j] = (dataframe[j].shift(-2) - dataframe[j].shift(2)) / 5
    df_slopes = df_slopes.dropna() #drop nan values
    return df_slopes
"""

def cal_1st_derivative(dataframe): #returns filtered df_derivatives
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns)
    
    #calculate derivatives for each diameter
    for i in dataframe.columns: 
        #N = np.log10(dataframe[i]) #log10 of concentration ###LOG FROM CONC###
        #N.replace([-np.inf,np.inf], 0, inplace=True) #replace all inf and -inf values with 0 ###LOG FROM CONC###
        N = dataframe[i] #concentration ###NO LOG FROM CONC###
        time = time_d * 24 * 60 * 60 #change days to seconds
        dNdt = np.diff(N)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = dNdt

    return  median_filter(df_derivatives,window=5) #filter after derivating
    #return  df_derivatives
df_1st_derivatives = cal_1st_derivative(df)
#with this we can check the format
#cal_1st_derivative(df).to_csv('./1st_derivatives_nofilter.csv', sep=',', header=True, index=True, na_rep='nan')
#cal_1st_derivative(df).to_csv('./1st_derivatives_filterbefore.csv', sep=',', header=True, index=True, na_rep='nan')
#cal_1st_derivative(df).to_csv('./1st_derivatives_filterafter.csv', sep=',', header=True, index=True, na_rep='nan')
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

def find_zero(dataframe):
    df_zero_deriv = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns)
    
    for i in dataframe.columns:
        #if value is zero replace nan with zero
        zero_value = dataframe[i] == 0
        df_zero_deriv[i] = dataframe[i].where(zero_value, np.nan)
    return df_zero_deriv

def cal_min_max(df_1st_deriv,df_2nd_deriv): #returns [df_max, df_min]
    df_1stderiv_zeros = find_zero(df_1st_deriv) #dataframe with stationary points of the 1st derivative
    #find max and min values
    df_max = df[(df_1stderiv_zeros == 0) & (df_2nd_deriv < 0)] #stationary points exist, 2nd derivative is negative
    df_min = df[(df_1stderiv_zeros == 0) & (df_2nd_deriv > 0)] #stationary points exist, 2nd derivative is positive
    return [df_max,df_min]
df_max = cal_min_max(df_1st_derivatives,df_2nd_derivatives)[0]

def find_modes_1st_deriv(dataframe):
    df_mode_ranges = pd.DataFrame(np.nan, index=dataframe.index, columns=dataframe.columns)
    start_time_list = []
    start_diam_list = []
    end_time_list = []
    end_diam_list = []

    #threshold that determines what is a high concentration
    threshold = abs(df_1st_derivatives) > 0.03 #checks which values surpass the threshold ###NO LOG FROM CONC###
    #threshold = abs(df_1st_derivatives) > 0.000009 #checks which values surpass the threshold ###LOG FROM CONC###
    
    #threshold = (df_1st_derivatives > df_1st_derivatives.shift(1))

    #determine start and end points
    start_points = threshold & (~threshold.shift(1,fill_value=False))
    end_points = threshold & (~threshold.shift(-1,fill_value=False))

    #finding values within a mode
    start_time = None #initial values
    end_time = None

    for k in df_1st_derivatives.columns:
        for l in df_1st_derivatives.index: #identify pairs of start and end times
            if start_time != None and end_time != None: #when start and end times have their values find list of values between them
                subset = dataframe[k].loc[start_time:end_time]
                #fill dataframe with mode ranges
                df_mode_ranges.loc[subset.index, k] = subset

                
                #save start and end times with their diameters in lists
                start_time_list.append(start_time)
                start_diam_list.append(k)
                end_time_list.append(end_time)
                end_diam_list.append(k)

                #plt.plot(start_time_list, start_diam_list, '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
                #plt.plot(end_time_list, end_diam_list, '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)
                

                start_time = None 
                end_time = None
            elif start_points.loc[l,k] == True:
                start_time = l
            elif end_points.loc[l,k] == True and start_time != None:
                end_time = l
            else:
                continue
    return df_mode_ranges, start_time_list, start_diam_list, end_time_list, end_diam_list

#with this we can check the format
find_modes_1st_deriv(df)[0].to_csv('./modes.csv', sep=',', header=True, index=True, na_rep='nan')

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
    
def gaus(x,a,x0,sigma): #defining gaussian curve, from Janne's code
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
def max_con(dataframe): #returns [times, diameters]
    df_modes = find_modes_1st_deriv(df)[0]
    #start_times = find_modes_1st_deriv(df)[1]
    #start_diams = find_modes_1st_deriv(df)[2]
    #end_times = find_modes_1st_deriv(df)[3]
    #end_diams = find_modes_1st_deriv(df)[4]

    #gaussian fit to every mode
    max_conc_time = []
    max_conc_diameter = []

    for m in range(len(df_modes.columns)):
        x = [] #time
        y = [] #concentration
        for n in range(len(df_modes.index)):
            concentration = df_modes.iloc[n,m] #find concentration values from the dataframe (y)
            time = time_d[n] - np.min(time_d)

            if np.isnan(concentration) and len(y) != 0: #gaussian fit when all values of one mode have been added to the y list
                mu=np.mean(x) #parameters
                sigma = np.std(x)
                a = np.max(y)
                #print("FML") DOESNT PRINT FML???
                try: #gaussian fit
                    params,pcov = curve_fit(gaus,x,y,p0=[a,mu,sigma])
                    if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                        print("Peak outside range. Skipping.")
                    else:
                        max_conc_time = np.append(max_conc_time,params[1] + np.min(time_d)) #make a list of times with the max concentration time in each diameter
                        max_conc_diameter = np.append(max_conc_diameter,float(df_modes.columns[m])) #make list of diameters with max concentrations


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
    
    return max_conc_time, max_conc_diameter

def closest(list, number): #find closes element to a given value in list, returns index
    value = []
    for i in list:
        value.append(abs(number-i))
    return value.index(min(value))

def max_con_modefit(): #in ranges where mode fitting has succeeded, use these points to calculate the maximum concentration
    df_GR_values = pd.DataFrame(pd.read_csv("./Gr_final.csv",sep=',',engine='python')) #open calculated values in Gabi's code
    my_GR = df_GR_values.iloc[12,:]  #choose interesting growth rate, now defined in row 14 (14-2=12)

    start_time = my_GR["start"] #take values from 
    end_time = my_GR["end"]
    start_diam = my_GR["d_initial"]
    end_diam = my_GR["d_final"]

    #find GR area within the concentration values
    #modefit_con_time = df.index[(df.index > start_time) & (df.index < end_time)]
    #modefit_con_diam = df.columns[(df.columns > start_diam) & (df.columns < end_diam)]
    df_mfit_con = df[(df.index >= start_time) & (df.index <= end_time)]
    df_mfit_con = df_mfit_con[df_mfit_con.columns[(df_mfit_con.columns >= start_diam) & (df_mfit_con.columns <= end_diam)]]
    print("HERE1",len(df_mfit_con.columns))
    #if no values fall into the given range pick nearby two values
    if len(df_mfit_con.columns) == 0:
        df_mfit_con.columns = [df.columns[closest(df.columns,start_diam)],df.columns[closest(df.columns,end_diam)]]??
    
    print("HERE2",df_mfit_con)

max_con_modefit()
        
def plot(dataframe):
    #plot line when day changes
    new_day = None
    for i in df.index:
        day = i.strftime("%d")  
        if day != new_day:
            plt.axvline(x=i, color='black', linestyle='-', linewidth=0.5)
        new_day = day
    plt.plot(max_con(df)[0], max_con(df)[1], '*', alpha=0.5, color='white', ms=5)
    plt.yscale('log')
    plt.xlim(dataframe.index[2],dataframe.index[-3])
    plt.show()

#plot(df)



####################################################
#INSTEAD OF THIS A MEDIAN FILTER WOULD FIX IT TOO
#filter lonely datapoints (if 1 or 2 consecutive datapoints are lonely)
#df_filtered = pd.DataFrame(np.nan, index=df.index, columns=df.columns) #dataframe with the same size as df and values of 0 initially
#lonely_datapoints = (df.shift(1) == 0) & (df.shift(-1) == 0) | (df.shift(2) == 0) & (df.shift(-1) == 0) | (df.shift(1) == 0) & (df.shift(-2) == 0)
#df_filtered = df[~lonely_datapoints]

#previous ideas
"""
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


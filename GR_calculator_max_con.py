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
#folder = r"./dmps Nesrine/" #folder where data files are stored, should be in the same directory as this code
#dm160612.sum
#output_modefit_2016_06_5median_6nm_fix_xmin.csv

file_names = file_names #copy paths from the file "GR_calculator_unfer_v2_modified"
#df = pd.DataFrame(pd.read_csv(paths,sep='\s+',engine='python'))


#load data for as many days as given
def combine_data():
    dfs = []
    test = True
    #load all given data files and save them a list
    for i in file_names:
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
    start_date_measurement = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00"
    start_date = datetime.strptime(start_date_measurement, "%Y-%m-%d %H:%M:%S")
    dataframe["time (d)"] = [start_date + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
def list_days_into_UTC(list): #same for a list
    time_steps = list - time_d[0]
    start_date_measurement = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00"
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
df.to_csv('./data_format_filter.csv', sep=',', header=True, index=True, na_rep='nan')
#df.to_csv('./data_format_nofilter.csv', sep=',', header=True, index=True, na_rep='nan')

#NO RESAMPLING UNLIKE IN GABIS CODE
####################################################

"""
def slopes(dataframe):
    df_slopes = pd.DataFrame()
    for j in dataframe.columns:
        #slope of data with a window of 5 (matching the median filter window) centered
        df_slopes[j] = (dataframe[j].shift(-2) - dataframe[j].shift(2)) / 5
    df_slopes = df_slopes.dropna() #drop nan values
    return df_slopes
"""

def cal_1st_derivative(dataframe,time_days): #returns filtered df_derivatives
    df_derivatives = pd.DataFrame(np.nan, index=dataframe.index[1:], columns=dataframe.columns)
    
    #calculate derivatives for each diameter
    for i in dataframe.columns: 
        #N = np.log10(dataframe[i]) #log10 of concentration ###LOG FROM CONC###
        #N.replace([-np.inf,np.inf], 0, inplace=True) #replace all inf and -inf values with 0 ###LOG FROM CONC###
        N = dataframe[i] #concentration ###NO LOG FROM CONC###
        time = time_days * 24 * 60 * 60 #change days to seconds
        dNdt = np.diff(N)/np.diff(time) #derivative
        df_derivatives.loc[:, i] = dNdt

    return  median_filter(df_derivatives,window=5) #filter after derivating
    #return  df_derivatives
df_1st_derivatives = cal_1st_derivative(df,time_days=time_d)
#with this we can check the format
#cal_1st_derivative(df).to_csv('./1st_derivatives_nofilter.csv', sep=',', header=True, index=True, na_rep='nan')
#cal_1st_derivative(df).to_csv('./1st_derivatives_filterbefore.csv', sep=',', header=True, index=True, na_rep='nan')
cal_1st_derivative(df,time_days=time_d).to_csv('./1st_derivatives_filterafter.csv', sep=',', header=True, index=True, na_rep='nan')
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
                #start_time_list.append(start_time)
                #start_diam_list.append(k)
                #end_time_list.append(end_time)
                #end_diam_list.append(k)
                
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
    
    return df_mode_ranges, start_time_list, start_diam_list, end_time_list, end_diam_list

#with this we can check the format
#find_modes_1st_deriv(df,df_1st_derivatives,threshold_deriv=0.03,threshold_diff=2000)[0].to_csv('./modes.csv', sep=',', header=True, index=True, na_rep='nan')

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

def closest(list, number): #find closes element to a given value in list, returns index
    value = []
    for i in list:
        value.append(abs(number-i))
    return value.index(min(value))

#functions 
def gaus(x,a,x0,sigma): #from Janne's code
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
def logi(x,L,x0,k): #from Janne's code
        return L * (1 + np.exp(-k*(x-x0)))**(-1) 

def max_con(): #returns [times, diameters]
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
                    params,pcov = curve_fit(gaus,x,y,p0=[a,mu,sigma])
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

def maxcon_modefit(): #in ranges where mode fitting has succeeded, use these points to calculate the maximum concentration
    #create lists for results
    max_conc_time = []
    max_conc_diameter = []
    df_ranges = find_ranges()

    for df_range in df_ranges: #go through every range around GRs

        time_range = df.index.intersection(df_range.index) #find matching indices
        indices = [df.index.get_loc(row) for row in time_range]
        time_d_range = time_d[indices] #define time in days again depending on chosen range
        
        df_range_deriv = cal_1st_derivative(df_range,time_days=time_d_range) #calculate derivative
        df_modes = find_modes_1st_deriv(df_range,df_range_deriv,threshold_deriv=0.03,threshold_diff=7000)[0] #find modes

        #gaussian fit to every mode
        for m in range(len(df_modes.columns)):
            x = [] #time
            y = [] #concentration
            for n in range(len(df_modes.index)):
                concentration = df_modes.iloc[n,m] #find concentration values from the dataframe (y)
                time = time_d_range[n] - np.min(time_d_range)

                if np.isnan(concentration) and len(y) != 0: #gaussian fit when all values of one mode have been added to the y list
                    mu=np.mean(x) #parameters
                    sigma = np.std(x)
                    a = np.max(y)

                    try: #gaussian fit
                        params,pcov = curve_fit(gaus,x,y,p0=[a,mu,sigma])
                        if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                            print("Peak outside range. Skipping.")
                        else:
                            max_conc_time = np.append(max_conc_time,params[1] + np.min(time_d_range)) #make a list of times with the max concentration time in each diameter
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

def appearance_time(self,event): #appearance time (modified from Janne's code)
    #create lists for results
    appear_time = []
    appear_diameter = []
    
    df_ranges = find_ranges()
    max_conc_times, max_conc_diameters = maxcon_modefit() #timestamps and diameters of maximum concentrations in ranges

    for df_range in df_ranges: #go through every range around GRs
        max_concs = []

        time_range = df.index.intersection(df_range.index) #find matching indices
        indices = [df.index.get_loc(row) for row in time_range]
        time_d_range = time_d[indices] #define time in days again depending on chosen range
        
        df_range_deriv = cal_1st_derivative(df_range,time_days=time_d_range) #calculate derivative
        df_modes = find_modes_1st_deriv(df_range,df_range_deriv,threshold_deriv=0.03,threshold_diff=7000)[0] #find modes

        #logistic fit to every mode
        for dp in range(len(df_modes.columns)):
            x = [] #time
            y = [] #concentration
            for t in range(len(df_modes.index)):
                concentration = df_modes.iloc[t,dp] - np.min(df_modes.iloc[:,dp]) #find concentration values from the dataframe (y)
                time = time_d_range[t] - np.min(time_d_range)

                if np.isnan(concentration) and len(y) != 0: #logistic fit when all values of one mode have been added to the y list

                    #gaussian fit first to find maximum concentrations 
                    mu=np.mean(x) #parameters
                    sigma = np.std(x)
                    a = np.max(y)

                    try: #gaussian fit
                        params,pcov = curve_fit(gaus,x,y,p0=[a,mu,sigma])
                        if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                            print("Peak outside range. Skipping.")
                        else:
                            max_concs = np.append(max_concs,params[1] + np.min(time_d_range)) #make a list of times with the max concentration time in each diameter WAHT TO PUT HEREE


                            #plot start and end time
                            #plt.plot(start_times[n], start_diams[m], '>', alpha=0.5, color="green", ms=3, mec="white", mew=0.3)
                            #plt.plot(end_times[n], end_diams[m], '<', alpha=0.3, color="red", ms=3, mec="white", mew=0.3)
                            
                    except:
                        print("Diverges. Skipping.")

                    
                    #logistic fit
                    L = df_modes.loc[max_conc_times,max_conc_diameters] #maximum value (concentration)
                    x0 = np.nanmean(x) #midpoint x value
                    k = 1.0 #growth rate

                    try: 
                        params,pcov = curve_fit(logi,x,y,p0=[L,x0,k])
                        if ((params[1]>=x.max()) | (params[1]<=x.min())): #checking that the peak is within time range
                            print("Peak outside range. Skipping.")
                        else:
                            appear_time = np.append(appear_time,params[1] + np.min(time_d_range)) #make a list of times with the appearance time in each diameter
                            appear_diameter = np.append(appear_diameter,float(df_modes.columns[dp])) 


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
    return appear_time, appear_diameter

def plot(dataframe):
    #plot line when day changes
    new_day = None
    for i in df.index:
        day = i.strftime("%d")  
        if day != new_day:
            plt.axvline(x=i, color='black', linestyle='-', linewidth=0.5)
        new_day = day
    
    #plt.plot(max_con()[0], max_con()[1], '*', alpha=0.5, color='white', ms=5,label='maximum concentration') #without using mode fitting
    plt.plot(maxcon_modefit()[0],maxcon_modefit()[1], '*', alpha=0.5, color='white', ms=5,label='maximum concentration') #using mode fitting 
    
    plt.legend(fontsize=6,fancybox=False,framealpha=0.9)
    for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
        legend_handle.set_markeredgewidth(0.5)
        legend_handle.set_markeredgecolor("black")
                    
    #plt.legend(fontsize=6,fancybox=False,facecolor="blue",framealpha=0.6)
    plt.yscale('log')
    plt.xlim(dataframe.index[2],dataframe.index[-3])
    plt.show()

plot(df)



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


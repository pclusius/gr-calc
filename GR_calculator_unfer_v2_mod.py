# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:54:10 2024

@author: unfer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
import matplotlib ###
matplotlib.use("Qt5Agg") ###backend changes the plotting style
from datetime import timedelta
from datetime import datetime
from scipy.optimize import curve_fit, OptimizeWarning 
import warnings
from scipy import stats
from os import listdir ###
import json ###
from operator import itemgetter ###
from collections import defaultdict ###
import time ###

#parameters
show_mape = False #show mape values of lines instead of growth rates, unit %

###################################################################
#paths = "./dmps Nesrine/dm160401.sum" ###path to data file
folder = "./dmps Nesrine/" #folder with data files
#dados1= pd.read_csv(paths,sep='\s+',engine='python') ###steps to '\s+'
#df1 = pd.DataFrame(dados1)
#dm160612.sum

#file_names = ["dm160612.sum"]
file_names = ["dm160410.sum","dm160411.sum","dm160412.sum"]
#file_names = ["dm160410.sum","dm160411.sum"]
#file_names = ["dm160426.sum","dm160427.sum","dm160428.sum"]
#modefit_names = ["output_modefit_2016_06_12.csv"]
modefit_names = ["output_modefit_2016_04_10.csv","output_modefit_2016_04_11.csv","output_modefit_2016_04_12.csv"]
#modefit_names = ["output_modefit_2016_04_10.csv","output_modefit_2016_04_11.csv"]
#modefit_names = ["output_modefit_2016_04_26.csv","output_modefit_2016_04_27.csv","output_modefit_2016_04_28.csv"]

#set epoch
start_date_str = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00"
matplotlib.dates.set_epoch(start_date_str)

### load data for n days: ###
def combine_data(files,separation):
    dfs = []
    test = True
    #load all given data files and save them a list
    for i in files:
        #choose the right path
        if separation == '\s+':
            df = pd.DataFrame(pd.read_csv(folder + i,sep=separation,engine='python'))
        elif separation == ',':
            df = pd.DataFrame(pd.read_csv(i,sep=separation,engine='python'))

        #make sure all columns have the same diameter values, name all other columns with the labels of the first one
        if test == True:
            diameter_labels = df.columns
            test = False
        df.rename(columns=dict(zip(df.columns, diameter_labels)), inplace=True)
        dfs.append(df) #add dataframe to list
    #combine datasets
    combined_data = pd.concat(dfs,axis=0,ignore_index=True)
    return combined_data
df1 = combine_data(file_names,separation='\s+')

###----

diameters = df1.columns[2:].astype(float)*10**9 ### save diameters as floats before replacing them with new column names / units from m to nm
df1.columns = ["time (d)", "total number concentration (N)"] + [f"dN/dlogDp_{i}" for i in range(1,df1.shape[1]-1)] ###rename columns
time_d = df1.iloc[:,0].astype(float) ###ADDED save time as days before changing them to UTC 

### assuming time is in "days from start of measurement"
def days_into_UTC():
    time_steps = df1["time (d)"] - time_d[0]
    start_date_measurement = f"20{file_names[0][2:4]}-{file_names[0][4:6]}-{file_names[0][6:8]} 00:00:00"
    start_date = datetime.strptime(start_date_measurement, "%Y-%m-%d %H:%M:%S")
    df1["time (d)"] = [start_date + timedelta(days=i) for i in time_steps] #converting timesteps to datetime
days_into_UTC() ###

df1.rename(columns={'time (d)': 'Timestamp (UTC)'}, inplace=True) ###CHANGE '# "datetime (UTC)"' -> 'time (d)'
df1['Timestamp (UTC)']=pd.to_datetime(df1['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S")
df1.index=df1['Timestamp (UTC)']
df1 = df1.drop(['Timestamp (UTC)'], axis=1)

df1[(df1<0)] = 0                     ## Treat negative data as zero
df1[(df1.sum(axis=1)==0)] = np.nan   ## Discart a whole zero size distribution
df1 = df1.dropna()

#df1 = df1.resample('10Min').mean().interpolate(method='time', limit_area='inside', limit=6) ###CHANGE 5Min -> 10Min CHECK FROM HERE UP ###commented away

N_tot = list(df1["total number concentration (N)"]) ### save total number concentrations to a list
df1 = df1.drop(['total number concentration (N)'], axis=1) ### drop N_tot from the dataframe
#df1['N_tot'] = (df1.sum(axis=1)*0.0265) ### commented away
#df1[df1['N_tot']>df1['N_tot'].quantile(0.999)] = np.nan  ### commented away
#df1 = df1.dropna() ### commented away
#df1 = df1.drop(['N_tot'], axis=1) ### commented away
df1.columns = pd.to_numeric(diameters) ### rename columns back to diameters

###
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
 

    return dataframe
df1 = avg_filter(df1,resolution=30)
###

#with this we can check the format
df1.to_csv('./combined_data.csv', sep=',', header=True, index=True, na_rep='nan')

##############################################################
###LOADING A JSON FILE

with open("10_12042016_mode_fits.json") as file:
    mode_fits = json.load(file)

#processing modefitting data
#making a dataframe from json file
rows_list = []
for timestamp in mode_fits:
    ts = timestamp['time']
    peak_diams = timestamp['peak_diams']
    
    for i, gaussians in enumerate(timestamp['gaussians']):
        mean = gaussians['mean']
        sigma = gaussians['sigma']
        amplitude = gaussians['amplitude']

        dict_row = {'timestamp':ts,'amplitude':amplitude,'peak_diameter':peak_diams[i]*10**9,'sigma':sigma} #diam unit m to nm
        rows_list.append(dict_row)

df_modefits = pd.DataFrame(rows_list)  

#timestamps to index, timestamp strings to datetime objects
df_modefits['timestamp']=pd.to_datetime(df_modefits['timestamp'], format="%Y-%m-%d %H:%M:%S")
df_modefits.index=df_modefits['timestamp']
df_modefits = df_modefits.drop(['timestamp'], axis=1)

#print('df_modefits',df_modefits) ###

##################################################

###
#mitä pidempi suora on tai mitä enemmän pisteitä löytyy sitä enemmän vaihtelua mape sallii myöhemmin
#riippuen datapisteiden määrästä mape muuttuu, mape N:n funktiona, alussa mape isompi, myöhemmin pienempi
#YHDISTÄ KAIKKI MOODIT ENSIN YHTEEN
#ETSI AINA AIKASTEPPI KERRALLAAN +-10nm VÄLILLÄ SEURAAVA PISTE, MOODIEN VÄLISSÄ +- VOI OLLA LAAJEMPI (15nm?)
def timestamp_indexing(timestamps):
    '''
    Creates list of indices corresponding to given timestamp list.
    Needed when there are gaps in time lists.
    '''
    start_time = timestamps[0]
    time_interval = 30 #minutes
    indices = []

    for time in timestamps:
        mins_since_start = abs((time - start_time).total_seconds() / 60) #for each timestamp
        index = int(mins_since_start / time_interval) #divide by interval
        indices.append(index)
    
    return np.array(indices)
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
    y_predicted = linear(x, *popt)
    absolute_error = np.abs(y - y_predicted)
    mape = np.mean(absolute_error / y) * 100
    return mape
def cal_mase(x,y,popt):
    y_predicted = linear(x, *popt)
    mean_absolute_error = np.mean(np.abs(y - y_predicted))
    mean_absolute_deviation = np.mean(np.abs(y - np.mean(y)))
    mase = np.mean(mean_absolute_error/mean_absolute_deviation)
    return mase   

def combine_segments_ver2(df):
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with wanted times and diameters for plotting growth rates.
    ''' 
    #extract times and diameters from df
    times = df.index
    diams = df['peak_diameter']
    
    #combine to the same list and sort data by diameter
    data_sorted = np.array(sorted(zip(times, diams), key=itemgetter(0,1))) #[[time1,diam1],[time2,diam2]...]
    
    #init
    unfinished_lines = []
    finalized_lines = []
    df_mapes = pd.DataFrame()
    
    max_time_diff = 90 #90mins = 1,5h
    
    #iterate through each datapoint to find suitable pairs of mode fitting datapoints
    for i, datapoint in enumerate(data_sorted):
        time0, diam0 = datapoint #current datapoint
        
        #diam difference in channels changes in a logarithmic scale
        nextnext_diam = df1.columns[closest(df1.columns,diam0)+2] #diameter in the channel one after
        max_diam_diff = abs(df1.columns[closest(df1.columns,diam0)]-nextnext_diam) #max one diameter channel empty in between

        #iterate through timestamps after current datapoint and look for the nearest datapoint in timestamp
        timesteps = int(max_time_diff / 30)
        
        for ii in range(1,timesteps+1): #one step represents 30mins: [30mins, 90mins]
            #timestamp for current iteration
            timestamp = time0 + timedelta(minutes=30)*ii
            #print(i,ii,"time",time0)
            #search for datapoints in this timestamp (segment)
            ts_points = [point for point in data_sorted if point[0] == timestamp]  
            if not ts_points: #skip if no datapoints in current timestamp
                continue
            
            #closest datapoint
            nearby_datapoint = tuple(min(ts_points, key=lambda point: abs(point[1] - diam0)))
            diam_diff = abs(nearby_datapoint[1] - diam0)
            
            #check diameter difference
            if diam_diff <= max_diam_diff: 
                
                #if datapoint is not already in a line
                if not any(tuple(datapoint) in line for line in unfinished_lines):
                    unfinished_lines.append([tuple(datapoint),nearby_datapoint])
                
                #datapoint is already in a line
                elif any(tuple(datapoint) in line for line in unfinished_lines):
                    #find index of that line
                    for line_i, line in enumerate(unfinished_lines):
                        if tuple(datapoint) in line:
                            #add nearby datapoint to that line
                            unfinished_lines[line_i].append(nearby_datapoint)

                #make sure datapoints in every line are sorted by time
                unfinished_lines = [list(set(line)) for line in unfinished_lines]
                unfinished_lines = [sorted(sublist, key=lambda x: x[0]) for sublist in unfinished_lines] 
                
                #make a linear fit to check mape for line with new datapoint
                iii, current_line = [(i,line) for i,line in enumerate(unfinished_lines) if nearby_datapoint in line][0]
                
                if len(current_line) <= 2: #pointless to analyze mape with less than 3 datapoints
                    continue #proceed to next line
                elif len(current_line) >= 3:
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line]) #times
                    y = [datapoint[1] for datapoint in current_line] #diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                else:#THIS ISNT BEING USED CURRENTLY
                    #do fit to previous datapoints and calculate mape with the new datapoint included
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line[:-1]]) #times (last datapoint excluded)
                    y = [datapoint[1] for datapoint in current_line[:-1]] #diams (last datapoint excluded)
                    popt, pcov = curve_fit(linear, x, y)

                    #print("timestamps:",[datapoint[0] for datapoint in current_line[:-1]],"x:",x,"y:",y)
                    #add newest datapoint back to x and y
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line])
                    y = [datapoint[1] for datapoint in current_line]
                    mape = cal_mape(x,y,popt) #!!!
                    
                    #check mape of the whole line also
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    
                    #print("time0_index:",time0_index,"y_fit:","new x:",x,"new y:",y,"mape:",mape)
                
                #mape threshold linearly gets stricter until 5 datapoints
                #with 5 or more points threshold is 3% and with 3 points 5%
                #mape_threshold = -1*len(line)+8 if len(line) < 5 else 3.2
                mape_threshold = 15*len(current_line)**(-1) #15*x^(-1)
                # if mape_threshold > 3:
                #     mape_threshold = 3
                    
                new_row = pd.DataFrame({"length": [len(current_line)], "mape": [mape]})
                df_mapes = pd.concat([df_mapes,new_row],ignore_index=True)
                
                if mape > mape_threshold:
                    #calculate mape without first datapoint of line to see if mape is smaller
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line[1:]])
                    y = [datapoint[1] for datapoint in current_line[1:]] #diams
                    
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    if mape > mape_threshold: #if mape is still too big
                        #delete recently added datapoint from unfinished lines
                        unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                        
                        #allow lines to start from the same point as others end in
                        #save current line to another list and add the ending point as the start for a new line
                        finalized_lines.append(current_line[:-1])
                        unfinished_lines.append([tuple(datapoint),nearby_datapoint])

                        x = timestamp_indexing([datapoint[0] for datapoint in current_line[:-1]])
                        y = [datapoint[1] for datapoint in current_line[:-1]] #diams
                        popt, pcov = curve_fit(linear, x, y)
                        mape = cal_mape(x,y,popt) #update mape value 
                    else:
                        #remove current line from unfinished lines
                        unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                        
                        #add current line without the first element to finalized lines
                        finalized_lines.append(current_line[1:])    
                
                '''
                #BASICALLY LIMITS LONGEST LINE TO 8 DATAPOINTS!!!
                #try splitting line into two parts from the middle to lower mape
                #does it when all lines haven't been found yet 
                if len(combined[iii]) >= 8: #at least 8 datapoints needed
                    
                    middle_index = len(combined[iii])//2
                    line_1st_half = combined[iii][:middle_index]
                    #uneven numbers include one more element in the 2nd half
                    line_2nd_half = combined[iii][middle_index:]
                    
                    #print(combined[iii][0],combined[iii][-1])
                    #print(line_1st_half[0],line_1st_half[-1])
                    #print(line_2nd_half[0],line_2nd_half[-1],"\n")
                    #calculate if mape lowered significantly
                    x = timestamp_indexing([datapoint[0] for datapoint in line_1st_half])
                    y = [datapoint[1] for datapoint in line_1st_half] #diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape1 = cal_mape(x,y,popt)
                    
                    x = timestamp_indexing([datapoint[0] for datapoint in line_2nd_half])
                    y = [datapoint[1] for datapoint in line_2nd_half]
                    popt, pcov = curve_fit(linear, x, y)
                    mape2 = cal_mape(x,y,popt)
                    
                    #print([datapoint[0] for datapoint in line_1st_half])
                    #print(x,y)
                    #relative change in mape
                    rel_diff1 = ((mape1-mape)/mape) * 100
                    rel_diff2 = ((mape2-mape)/mape) * 100
                    
                    #print(mape1,mape2, mape)
                    
                    #print("relative differences:",rel_diff1,rel_diff2)
                    
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
    
    df_mapes.to_csv('./df_mapes_modefitting.csv', sep=',', header=True, index=True, na_rep='nan')
    
    #add rest of the lines to finalized lines and by timestamp
    finalized_lines.extend(unfinished_lines)
    finalized_lines = [sorted(line, key=lambda x: x[0]) for line in finalized_lines] 
    
    return finalized_lines

###################################################

def linear(x,k,b): ###
    return k*x + b

###
def filter_segments_ver2(combined):
    '''
    Filter datapoints of lines that are too short or
    with too big of an error.
    '''
    
    #filter lines shorter than 4
    combined = [subpoints for subpoints in combined if len(subpoints) >= 4]
    
    filtered_lines = []
    for line in combined: 
        while True:
            try:
                x = timestamp_indexing([datapoint[0] for datapoint in line])
                y = [datapoint[1] for datapoint in line] #diams
                
                popt, pcov = curve_fit(linear, x, y)
                absolute_error = np.abs(linear(x, *popt) - y)
                mape = np.mean(absolute_error / y) * 100
                GR = popt[0] * 2

                #maximum error 10% and GR is not bigger than +-10nm/h
                if mape <= 10 and abs(GR) <= 1000:
                    filtered_lines.append(line)
                    break
                else:
                    break

            except:
                print("Linear fit diverges.")
    
    return filtered_lines
###

########################

###
def process_data_ver2(df):
    start_time = time.time()
    comb_segs = combine_segments_ver2(df)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Combined segments done! (1/2)')
    filter_segs = filter_segments_ver2(comb_segs)
    print('Filtering done! (2/2)')    
 
    return filter_segs
###


###############################################################################


###
print('\n'+'*********** Processing mode fitting data'+'\n')
filter_segs = process_data_ver2(df_modefits) #data from Janne's code
###


###############################################################################


def plot2_ver2(filter_segs,show_mape):
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200) ### figsize=(12, 3), dpi=300 -> figsize=(12, 5), dpi=200

    #colormap
    x = df1.index[0:]
    y = df1.columns.astype(float)
    plt.pcolormesh(x, y, df1[0:].T, cmap='RdYlBu_r', zorder=0, norm=colors.LogNorm(vmin=1e1, vmax=1e4))
    ax.set_yscale('log')
    cbar = plt.colorbar(orientation='vertical', shrink=0.8, extend="max", pad=0.04)
    cbar.set_label('dN/dlogDp', size=14)
    cbar.ax.tick_params(labelsize=12)
    
    #modefit points (as black stars)
    time = df_modefits.index
    diam = df_modefits['peak_diameter']
    plt.plot(time,diam,'*', alpha=0.5, color='black', markersize=5, label='mode fitting')
    
    #growth rates
    '''
    y = time in days
    x = diameters in nm
    Flipped as the error is in time.
    '''    
    #linear least square fits
    time = [[seg[0] for seg in segment] for segment in filter_segs]
    diam = [[seg[1] for seg in segment] for segment in filter_segs]
    
    df_GR_final = pd.DataFrame(columns=['start','end','d_initial','d_final','GR'])
    for i in range(len(time)):
        x = timestamp_indexing(time[i]) #time
        params, pcov = curve_fit(linear, x, diam[i])
        gr = params[0]*2 #unit to nm/h from nm/0.5h (due to how x is defined)
        diam_fit = params[0]*x + params[1]
        plt.plot(time[i],diam_fit,lw=3) #line
        
        midpoint_idx = len(time[i]) // 2 #growth rate value
        midpoint_time = time[i][midpoint_idx]
        midpoint_value = diam_fit[midpoint_idx]

        if show_mape:
            #mape annotation
            y = np.array(diam[i])
            y_predicted = np.array(diam_fit)
            absolute_error = np.abs(y_predicted - y)
            mape = np.mean(absolute_error / y) * 100
            
            plt.annotate(f'{mape:.2f}', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7, fontweight='bold')
        else:
            plt.annotate(f'{gr:.2f}', (midpoint_time, midpoint_value), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7, fontweight='bold')
        
        #save values to df for later use
        df_GR_final.loc[i] = [time[i][0],time[i][-1],diam[i][0],diam[i][-1],gr]
    
    return ax, df_GR_final

#ax = plot2(df_GR_final) ### 
ax, df_GR_final = plot2_ver2(filter_segs,show_mape=show_mape) ### 
df_GR_final.to_csv('./Gr_final.csv', sep=',', header=True, index=True, na_rep='nan')




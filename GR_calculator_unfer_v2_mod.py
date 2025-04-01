# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:54:10 2024
Modified on ???

@author: unfer
@modified: bouhlal
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
import matplotlib.dates as mdates ###

## parameters ##
show_mape = False #show mape values of lines instead of growth rates, unit %
mape_threshold_factor = 15 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
gr_error_threshold = 1.5 #nm/h (error of growth rates when adding new points to gr lines)

###################################################################
#paths = "./dmps Nesrine/dm160401.sum" ###path to data file
folder = "./dmps Nesrine/" #folder with data files
#dados1= pd.read_csv(paths,sep='\s+',engine='python') ###steps to '\s+'
#df1 = pd.DataFrame(dados1)

#file_names = ["dm160612.sum"]
file_names = ["dm160410.sum","dm160411.sum","dm160412.sum"]
#file_names = ["dm160426.sum","dm160427.sum","dm160428.sum"]

#modefit_names = ["output_modefit_2016_06_12.csv"]
modefit_names = ["output_modefit_2016_04_10.csv","output_modefit_2016_04_11.csv","output_modefit_2016_04_12.csv"]
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

#days to UTC times
df1["time (d)"] = df1["time (d)"] - df1["time (d)"][0] #substract first day ###
df1["time (d)"] = pd.Series(mdates.num2date(df1["time (d)"])).dt.tz_localize(None) ###

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
def linear(x,k,b):
    return k*x + b
###

def combine_segments_ver2(df,a,gr_error_threshold): ###
    '''
    Finds nearby datapoints based on time and diameter constraints.
    Fits linear curve to test if datapoints are close enough.
    Returns lists with wanted times and diameters for plotting growth rates.
    
    gret = growth rate error threshold for filtering bigger changes in gr when adding new points to lines
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
    
    max_time_diff = 90 #90mins = 1,5h between current and nearby point
    
    #iterate through each datapoint to find suitable pairs of mode fitting datapoints
    for i, datapoint in enumerate(data_sorted):
        time0, diam0 = datapoint #current datapoint
        
        #diam difference in channels changes in a logarithmic scale
        nextnext_diam = df1.columns[closest(df1.columns,diam0)+2] #diameter in the channel one after
        max_diam_diff = abs(df1.columns[closest(df1.columns,diam0)]-nextnext_diam) #max two diameter channels empty in between

        #iterate through timestamps after current datapoint and look for the nearest datapoint in timestamp
        timesteps = int(max_time_diff / 30)
        
        for ii in range(1,timesteps+1): #one step represents 30mins: [30mins, 90mins]
            timestamp = time0 + timedelta(minutes=30)*ii

            #search for datapoints in this timestamp (segment)
            ts_points = [point for point in data_sorted if point[0] == timestamp]  
            if not ts_points: #skip if no datapoints in current timestamp
                continue
            
            #closest datapoint next in list
            nearby_datapoint = tuple(min(ts_points, key=lambda point: abs(point[1] - diam0)))
            diam_diff = abs(nearby_datapoint[1] - diam0)
            
            #check diameter difference
            if diam_diff <= max_diam_diff: 
                #print("current datapoint",datapoint,"nearby datapoint",nearby_datapoint)
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
                
                
                ##check the length of current line##
                if len(current_line) <= 2: #pointless to analyze mape with less than 3 datapoints
                    break #proceed to next line
                elif len(current_line) == 3:
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line]) #times
                    y = [datapoint[1] for datapoint in current_line] #diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                else:
                    # #calculate mape of last 4 points
                    # x_last4 = timestamp_indexing([datapoint[0] for datapoint in current_line[-4:]])
                    # y_last4 = [datapoint[1] for datapoint in current_line[-4:]]
                    # popt_last4, pcov = curve_fit(linear, x_last4, y_last4)
                    # mape_last4 = cal_mape(x_last4,y_last4,popt_last4)

                    #do fit to previous datapoints and calculate mape with the new datapoint included
                    x_last_excluded = timestamp_indexing([datapoint[0] for datapoint in current_line[:-1]]) #times
                    y_last_excluded = [datapoint[1] for datapoint in current_line[:-1]] #diams
                    popt_last_excluded, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)

                    #calculate mape with all points but popt of line without the last point
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line])
                    y = [datapoint[1] for datapoint in current_line]
                    mape_last_excluded = cal_mape(x,y,popt_last_excluded)
                    GR_last_excluded = popt_last_excluded[0] * 2
                    
                    #check mape of the whole line also
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    GR = popt[0] * 2
            
                    mape_absolute_error = abs(mape_last_excluded-mape)
                    gr_abs_error = abs(GR-GR_last_excluded)

                
                #printing
                # for time,diam in current_line:
                #         print("time:",time,"diameter:",diam)
                            
                # #print("length of line:",len(current_line))
                # print(f"mape threshold:",mape_threshold,"mape:",mape)
                # if len(current_line) > 3:
                #     print("gr_abs_error:",gr_abs_error)
                
                #thresholds    
                mape_threshold = a*len(current_line)**(-1) #a*x^(-1)
                gr_error_threshold = gr_error_threshold
            
                #check mape and gr error threshold
                if mape > mape_threshold:
                    #mape without first datapoint of line
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line[1:]])
                    y = [datapoint[1] for datapoint in current_line[1:]] #diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape_no_first = cal_mape(x,y,popt)
                    
                    #without last datapoint
                    x = timestamp_indexing([datapoint[0] for datapoint in current_line[:-1]])
                    y = [datapoint[1] for datapoint in current_line[:-1]] #diams
                    popt, pcov = curve_fit(linear, x, y)
                    mape_no_last = cal_mape(x,y,popt)

                    print("mape_no_first",mape_no_first,"mape_no_last",mape_no_last)
                    
                    #if mape is smaller without the last datapoint remove last datapoint
                    #if the line is shorter than 4 give another chance for shorter lines to start anew
                    if mape_no_last <= mape_no_first and len(current_line) > 4:
                        unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                        
                        #allow lines to start from the same point as others end in
                        #save current line to another list and add the ending point as the start for a new line
                        finalized_lines.append(current_line[:-1])
                        unfinished_lines.append([tuple(datapoint),nearby_datapoint])

                    else:
                        #remove first point in current line from unfinished lines
                        unfinished_lines[iii] = current_line[1:]
                
                elif len(current_line) > 4 and gr_abs_error > gr_error_threshold:
                    #delete line with recently added datapoint from unfinished lines
                    unfinished_lines = [line for line in unfinished_lines if nearby_datapoint not in line]
                    
                    #allow lines to start from the same point as others end in
                    #save current line to another list and add the ending point as the start for a new line
                    finalized_lines.append(current_line[:-1])
                    unfinished_lines.append([tuple(datapoint),nearby_datapoint])
                    
                #give another chance for short lines
                elif len(current_line) == 4 and gr_abs_error > gr_error_threshold:
                    #remove first point in current line from unfinished lines
                    unfinished_lines[iii] = current_line[1:]
                print()
                break
            else: #keep looking for next datapoint until end of points if the next one isnt suitable
                continue
    
    #add rest of the lines to finalized lines and by timestamp
    finalized_lines.extend(unfinished_lines)
    finalized_lines = [sorted(line, key=lambda x: x[0]) for line in finalized_lines] 
    
    #try splitting line into two parts from the middle to lower mape
    for i, finalized_line in enumerate(finalized_lines):
        if len(finalized_line) >= 7: #at least 7 datapoints needed  
            middle_index = len(finalized_line)//2
            line_1st_half = finalized_line[:middle_index+1] #overlap +1
            line_2nd_half = finalized_line[middle_index:]
            
            #calculate if mape lowered in both halves
            #whole line
            x = timestamp_indexing([datapoint[0] for datapoint in finalized_line])
            y = [datapoint[1] for datapoint in finalized_line] #diams
            popt, pcov = curve_fit(linear, x, y)
            mape = cal_mape(x,y,popt)
            
            #1st half
            x = timestamp_indexing([datapoint[0] for datapoint in line_1st_half])
            y = [datapoint[1] for datapoint in line_1st_half] #diams
            popt, pcov = curve_fit(linear, x, y)
            mape1 = cal_mape(x,y,popt)
            
            #2nd half
            x = timestamp_indexing([datapoint[0] for datapoint in line_2nd_half])
            y = [datapoint[1] for datapoint in line_2nd_half]
            popt, pcov = curve_fit(linear, x, y)
            mape2 = cal_mape(x,y,popt)
            
            if mape1 < mape and mape2 <= mape:
                #remove the second half of current line and add it as its own line to finalized lines
                finalized_lines[i] = line_1st_half
                finalized_lines.append(line_2nd_half)  
            
            # #relative change in mape
            # rel_diff1 = ((mape1-mape)/mape) * 100
            # rel_diff2 = ((mape2-mape)/mape) * 100
            
            # print("rel_diff1",rel_diff1,"rel_diff2",rel_diff2)
            # print("mape",mape,"mape1",mape1,"mape2",mape2)
            # print()
            
            # #if mape improves (decreases) by 40% split line in two
            # if rel_diff1 <= -40 or rel_diff2 <= -40:
            #     #remove the second half of current line and add it as its own line to finalized lines
            #     finalized_lines[i] = line_1st_half
            #     finalized_lines.append(line_2nd_half)  
    
    #calculate mapes to show on plot
    for finalized_line in finalized_lines:
        x = timestamp_indexing([datapoint[0] for datapoint in finalized_line])
        y = [datapoint[1] for datapoint in finalized_line] #diams
        popt, pcov = curve_fit(linear, x, y)
        mape = cal_mape(x,y,popt)
        
        new_row = pd.DataFrame({"length": [len(x)], "mape": [mape]})
        df_mapes = pd.concat([df_mapes,new_row],ignore_index=True)
        
    df_mapes.to_csv('./df_mapes_modefitting.csv', sep=',', header=True, index=True, na_rep='nan')
    
    return finalized_lines

def filter_segments_ver2(combined): ###
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

def process_data_ver2(df,a,gr_error_threshold):
    start_time = time.time()
    comb_segs = combine_segments_ver2(df,a=a,gr_error_threshold=gr_error_threshold)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Combined segments done! (1/2)')
    filter_segs = filter_segments_ver2(comb_segs)
    print('Filtering done! (2/2)')    
 
    return filter_segs
print('\n'+'*********** Processing mode fitting data'+'\n')
filter_segs = process_data_ver2(df_modefits,mape_threshold_factor,gr_error_threshold) #data from Janne's code
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
    plt.plot(time,diam,'.', alpha=0.8, color='black', mec='black', mew=0.4, ms=6, label='mode fitting')
    
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
        plt.plot(time[i],diam_fit,lw=2) #line
        
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


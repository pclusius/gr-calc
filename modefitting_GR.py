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
import time ###
import matplotlib.dates as mdates ###

#choose dates
start_date = "2016-04-10"
end_date = "2016-04-12"

## parameters ##
show_mape = False #show mape values of lines instead of growth rates, unit %
mape_threshold_factor = 15 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
gr_error_threshold = 60 #% (precentage error of growth rates when adding new points to gr lines)

###################################################################

#load data and convert time columns to timestamps
df1 = pd.DataFrame(pd.read_csv('smeardata_20250506.csv',sep=',',engine='python'))
df1['timestamp'] = pd.to_datetime(df1[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
df1 = df1.set_index('timestamp')
df1 = df1.drop(['Year','Month','Day','Hour','Minute','Second'], axis=1)

#drop bins with no data
df1 = df1.dropna(axis=1, how='all')

#select wanted time period and shift times by 15minutes forward
df1_selected = df1.loc[start_date:end_date].shift(periods=15, freq='Min')

#put diameter bins in order
sorted_columns = sorted(df1_selected.columns, key=lambda column: (int(column.split('e')[-1]) , int(column[10:13])))
df1_selected = df1_selected[sorted_columns]

#replace arbitrary column names by diameter float values
diameter_ints = []
for column_str in sorted_columns:
    number = column_str[10:13]
    decimal_pos = int(column_str[-1])
    column_float = float(number[:decimal_pos] + '.' + number[decimal_pos:])
    diameter_ints.append(column_float)
diameter_ints[-1] = 1000.0 #set last bin as 1000
df1_selected.columns = diameter_ints

#with this we can check the format
df1_selected.to_csv('./combined_data.csv', sep=',', header=True, index=True, na_rep='nan')

#set epoch
matplotlib.dates.set_epoch(start_date)

##############################################################
###LOADING A JSON FILE

with open("fit_results.json") as file:
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

        dict_row = {'timestamp':ts,'amplitude':amplitude,'peak_diameter':peak_diams[i],'sigma':sigma} #diam unit m to nm
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
    return (timestamp_indexing([point[0] for point in line[exclude_start:len(line)-exclude_end]]),  #x values
            [point[1] for point in line[exclude_start:len(line)-exclude_end]])  #y values
###

def find_growth(df,a,gr_error_threshold):
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
    
    max_time_diff = 60 #mins = 1h between current and nearby point
    
    #iterate through each datapoint to find suitable pairs of mode fitting datapoints
    for i, datapoint in enumerate(data_sorted):
        time0, diam0 = datapoint
        datapoint = tuple(datapoint)
        
        #diam difference in channels changes in a logarithmic scale
        #nextnext_diam = df1.columns[closest(df1.columns,diam0)+2] #diameter in the channel one after
        #max_diam_diff = abs(df1.columns[closest(df1.columns,diam0)]-nextnext_diam) #max two diameter channels empty in between

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
            
            max_diam_diff = 10*ii #nm*timestep
            
            #check diameter difference
            if diam_diff <= max_diam_diff: 
                ### add new point to a line ###
                if not points_in_existing_line(unfinished_lines,datapoint,nearby_datapoint):
                    unfinished_lines.append([datapoint,nearby_datapoint])

                elif points_in_existing_line(unfinished_lines,datapoint,nearby_datapoint):
                    #find index of that line
                    for line_i, line in enumerate(unfinished_lines):
                        if datapoint in line:
                            #add nearby datapoint to that line
                            unfinished_lines[line_i].append(nearby_datapoint)

                #make sure datapoints in every line are sorted by time
                unfinished_lines = [list(set(line)) for line in unfinished_lines]
                unfinished_lines = [sorted(sublist, key=lambda x: x[0]) for sublist in unfinished_lines] 
                
                
                ### make a linear fit to check mape for line with new datapoint ###
                iii, current_line = [(i,line) for i,line in enumerate(unfinished_lines) if nearby_datapoint in line][0]
                
                #define variables for linear fit
                x, y = extract_data(current_line) #x=times,y=diams
                x_last_excluded, y_last_excluded = extract_data(current_line,exclude_end=1)
                x_first_excluded, y_first_excluded = extract_data(current_line,exclude_start=1)
                
                ##check the length of current line##
                if len(current_line) <= 2: #pointless to analyze mape with less than 3 datapoints
                    break #proceed to next line
                elif len(current_line) == 3:
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                else:
                    # #calculate mape of last 4 points
                    # x_last4 = timestamp_indexing([datapoint[0] for datapoint in current_line[-4:]])
                    # y_last4 = [datapoint[1] for datapoint in current_line[-4:]]
                    # popt_last4, pcov = curve_fit(linear, x_last4, y_last4)
                    # mape_last4 = cal_mape(x_last4,y_last4,popt_last4)

                    #fit to line excluding last datapoint, but calculate mape with full line
                    popt_last_excluded, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)
                    GR_last_excluded = popt_last_excluded[0] * 2
                    
                    #fit to full line
                    popt, pcov = curve_fit(linear, x, y)
                    mape = cal_mape(x,y,popt)
                    GR = popt[0] * 2

                    gr_abs_precentage_error = abs(GR-GR_last_excluded) / abs(GR_last_excluded) * 100
                    
                    
                ### check mae and gr error thresholds ###    
                mape_threshold = a*len(current_line)**(-1) #a*x^(-1)
                gr_error_threshold = gr_error_threshold

                if mape > mape_threshold:
                    #calculate mae without the first and then last datapoint 
                    popt, pcov = curve_fit(linear, x_first_excluded, y_first_excluded)
                    mape_no_first = cal_mape(x_first_excluded,y_first_excluded,popt)

                    popt, pcov = curve_fit(linear, x_last_excluded, y_last_excluded)
                    mape_no_last = cal_mape(x_last_excluded,y_last_excluded,popt)

                    #remove last or first point based on mae comparison
                    if mape_no_last <= mape_no_first and len(current_line) > 4:
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
    
    #add rest of the lines to finalized lines and by timestamp
    finalized_lines.extend(unfinished_lines)
    finalized_lines = [sorted(line, key=lambda x: x[0]) for line in finalized_lines] 
    
    # #try splitting line into two parts from the middle to lower mape
    # for i, finalized_line in enumerate(finalized_lines):
    #     if len(finalized_line) >= 7: #at least 7 datapoints needed  
    #         middle_index = len(finalized_line)//2
    #         line_1st_half = finalized_line[:middle_index+1] #overlap +1
    #         line_2nd_half = finalized_line[middle_index:]
            
    #         #calculate if mape lowered in both halves
    #         #whole line
    #         x = timestamp_indexing([datapoint[0] for datapoint in finalized_line])
    #         y = [datapoint[1] for datapoint in finalized_line] #diams
    #         popt, pcov = curve_fit(linear, x, y)
    #         mape = cal_mape(x,y,popt)
            
    #         #1st half
    #         x = timestamp_indexing([datapoint[0] for datapoint in line_1st_half])
    #         y = [datapoint[1] for datapoint in line_1st_half] #diams
    #         popt, pcov = curve_fit(linear, x, y)
    #         mape1 = cal_mape(x,y,popt)
            
    #         #2nd half
    #         x = timestamp_indexing([datapoint[0] for datapoint in line_2nd_half])
    #         y = [datapoint[1] for datapoint in line_2nd_half]
    #         popt, pcov = curve_fit(linear, x, y)
    #         mape2 = cal_mape(x,y,popt)
            
    #         if mape1 < mape and mape2 <= mape:
    #             #remove the second half of current line and add it as its own line to finalized lines
    #             finalized_lines[i] = line_1st_half
    #             finalized_lines.append(line_2nd_half)  
            
    #         # #relative change in mape
    #         # rel_diff1 = ((mape1-mape)/mape) * 100
    #         # rel_diff2 = ((mape2-mape)/mape) * 100
            
    #         # print("rel_diff1",rel_diff1,"rel_diff2",rel_diff2)
    #         # print("mape",mape,"mape1",mape1,"mape2",mape2)
    #         # print()
            
    #         # #if mape improves (decreases) by 40% split line in two
    #         # if rel_diff1 <= -40 or rel_diff2 <= -40:
    #         #     #remove the second half of current line and add it as its own line to finalized lines
    #         #     finalized_lines[i] = line_1st_half
    #         #     finalized_lines.append(line_2nd_half)  
    
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

def filter_lines(combined):
    '''
    Filter datapoints of lines that are too short or
    with too big of an error.
    '''
    
    #filter lines shorter than 4
    filtered_lines = [subpoints for subpoints in combined if len(subpoints) >= 4]
    
    #filter lines with too high of a growth rate
    #???
    
    return filtered_lines

def process_data(df,a,gr_error_threshold):
    start_time = time.time()
    growth_lines = find_growth(df,a=a,gr_error_threshold=gr_error_threshold)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Periods of growth found! (1/2)')
    filtered_lines = filter_lines(growth_lines)
    print('Filtering done! (2/2)')    
 
    return filtered_lines
print('\n'+'*********** Processing mode fitting data'+'\n')
filter_segs = process_data(df_modefits,mape_threshold_factor,gr_error_threshold) #data from Janne's code
###

###############################################################################

def plot2_ver2(filter_segs,show_mape):
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200) ### figsize=(12, 3), dpi=300 -> figsize=(12, 5), dpi=200

    #colormap
    x = df1_selected.index[0:]
    y = df1_selected.columns.astype(float)
    plt.pcolormesh(x, y, df1_selected[0:].T, cmap='RdYlBu_r', zorder=0, norm=colors.LogNorm(vmin=1e1, vmax=1e4))
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


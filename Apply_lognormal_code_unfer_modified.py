# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:07:31 2024

@author: unfer
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta ###CHANGE added timedelta
from lognormal_fit_multiprocessing_unfer_modified import Main as logf ###CHANGE lognormal_fit_multiprocessing_unfer -> lognormal_fit_multiprocessing_unfer_modified
from multiprocessing import Process

path1 = r"./dmps Nesrine/dm160428.sum"                   ###CHANGE ./CerroMirador_SMPS_INV&CORR_Apr19toMar23.csv -> file name changed  #put here the data file name
dados1 = pd.read_csv(path1, sep='\s+',engine='python')      ###CHANGE separate with "\s+" instead of ";"
df1 = pd.DataFrame(dados1)
diameters_str = list(df1.columns[2:]) ### save diameters before replacing them with new column names
df1.columns = ["time (d)", "total number concentration (N)"] + [f"dN/dlogDp_{i}" for i in range(1,df1.shape[1]-1)] ###rename columns
time_d = df1.iloc[:,0].astype(float) ###ADDED save time as days before changing them to UTC 
 
### assuming time is in "days from start of measurement"
def days_into_UTC():
    time_steps = df1["time (d)"] - time_d[0]
    start_date_measurement = f"20{path1[-10:-8]}-{path1[-8:-6]}-{path1[-6:-4]} 00:00:00"
    start_date = datetime.strptime(start_date_measurement, "%Y-%m-%d %H:%M:%S") 

    #converting timesteps to datetime
    df1["time (d)"] = [start_date + timedelta(days=i) for i in time_steps]    

days_into_UTC() ### 
df1.rename(columns={'time (d)': 'Timestamp (UTC)'}, inplace=True)    ###CHANGE '# "datetime (UTC)"' -> 'time (d)'
df1['Timestamp (UTC)'] = pd.to_datetime(df1['Timestamp (UTC)'], format="%Y-%m-%d %H:%M:%S")
df1.index = df1['Timestamp (UTC)']
df1 = df1.drop(['Timestamp (UTC)'], axis=1)

df1[(df1 < 0)] = 0  ## Treat negative data as zero
df1[(df1.sum(axis=1) == 0)] = np.nan  ## Discard a whole zero size distribution
df1 = df1.dropna()

#interpolation
#df1 = df1.resample('10Min').mean().interpolate(method='time', limit_area='inside', limit=6) ###CHANGE 5Min -> 10Min ###commented away
N_tot = list(df1["total number concentration (N)"]) ### save total number concentrations to a list
df1 = df1.drop(['total number concentration (N)'], axis=1) ### drop N_tot from the dataframe
df1 = df1.dropna()
df1.columns = pd.to_numeric(diameters_str) ### df1.columns -> diameters_str
df1.columns = df1.columns * 10 ** 9 ### change units from m to nm

### quantile
df1['N_tot'] = (df1.sum(axis=1)*0.0265)
#df1[df1['N_tot']>df1['N_tot'].quantile(0.999)] = np.nan ### commented away
df1 = df1.dropna()
df1 = df1.drop(['N_tot'], axis=1)

### limit diameter
df1[df1.columns[df1.columns < 6]] = np.nan ### diameters <6nm to nan values
df1.dropna(axis=1, inplace=True) ### drop columns with diameters <6nm

### median filter
#for i in df1.columns:
#    df1[i] = df1[i].rolling(window=5, center=True).median() ### window of 5 datapoints i.e. 2 neighbouring value
#df1.dropna(inplace=True)

###
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

    return dataframe
df1 = avg_filter(df1,resolution=30)
###

#df1 = df1.resample('30min').mean() ### change resolution to 30mins from 10mins, take average of values
#df1 = df1.shift(1, freq='15min') #set new timestamps to be in the middle of the new resolution

"""
### grouping in case of data over several months..
month_year_groups = df1.groupby([df1.index.year, df1.index.month])
month_year_dataframes = [group for _, group in month_year_groups]

month_year_labels = [(group_name[0], f"{group_name[1]:02d}") for group_name, _ in month_year_groups]


#####################################################################


def lognormal(): 
    for z, dataframe in enumerate(month_year_dataframes):
        year, month = month_year_labels[z]
        main_instance = logf(dataframe, year, month)
"""

#ADDED DAY AS WELL
year_month_day_groups = df1.groupby([df1.index.year, df1.index.month, df1.index.day])
year_month_day_dataframes = [group for _, group in year_month_day_groups]

year_month_day_labels = [(group_name[0], f"{group_name[1]:02d}", f"{group_name[2]:02d}") for group_name, _ in year_month_day_groups]

#####################################################################

def lognormal(): 
    for z, dataframe in enumerate(year_month_day_dataframes):
        year, month, day = year_month_day_labels[z]
        main_instance = logf(dataframe, year, month, day)

if __name__ == "__main__":    ###multiprocessing works differently on WIN/MAC, not needed on LINUX?
    process = Process(target=lognormal)
    process.start()
    process.join()
        

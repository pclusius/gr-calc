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

path1 = r"./dmps Nesrine/dm160402.sum"                   ###CHANGE ./CerroMirador_SMPS_INV&CORR_Apr19toMar23.csv -> file name changed  #put here the data file name
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
df1 = df1.resample('10Min').mean().interpolate(method='time', limit_area='inside', limit=6) ###CHANGE 5Min -> 10Min ###commented away
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
for i in df1.columns:
    #df1[i] = df1[i].rolling(window=3, center=True).median() ### window of 3 datapoints i.e. 1 neighbouring value
    df1[i] = df1[i].rolling(window=5, center=True).median() ### window of 5 datapoints i.e. 2 neighbouring value
df1.dropna(inplace=True)


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
        

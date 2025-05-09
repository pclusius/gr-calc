#created: 9.5.2025
#author: Nesrine Bouhlal

# automatic growth rate calculator #

'''
assumptions:
- datasets in json files (AVAA), separated by commas
- time in days & diameters in X*e^Y format 
  (e.g. "HYY_DMPS.d112e2" where diameter is 11.2nm)
- 
'''

import pandas as pd
from matplotlib import use
from matplotlib.dates import set_epoch

def main():
    ## DATASET ##
    file_name = "smeardata_20250506.csv" #in the same folder as the code
    start_date = "2016-04-10"
    end_date = "2016-04-12"
    
    ## PARAMETERS ##
    # maximum concentration and appearance time #
    #find_modes
    maximum_peak_difference = 2 #hours (time between two peaks in smoothed data (window 3))
    derivative_threshold = 200 #cm^(-3)/h (starting points of horizontal peak areas, determines what is a high concentration) 
                               #(NOTICE: concentration diff is half of this between timesteps as the resolution is 30min)

    #find_dots
    show_mae = False #show mae values of lines instead of growth rates, unit hours
    maximum_time_difference_dots = 2.5 #hours (between current and nearby point)
    mae_threshold_factor = 2 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_precentage_error_threshold = 50 #% (precentage error of growth rates when adding new points to gr lines)

    #channel plotting
    init_plot_channel = False #True to plot channels
    channel_indices = [5] #Indices of diameter channels, 1=small
    show_start_times_and_maxima = True #True to show all possible start times of peak areas (black arrow) and maximas associated
    
    # mode fitting #
    show_mape = False #show mape values of lines instead of growth rates, unit %
    mape_threshold_factor = 15 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_error_threshold = 50 #% (precentage error of growth rates when adding new points to gr lines)
    
    ##############################################################################################
    
    ## LOAD DATA ##
    df = load_data(file_name,start_date,end_date)

    ## CONFIGURATIONS ##
    use("Qt5Agg") #backend changes the UI for plotting
    set_epoch(start_date) #set epoch

    ## CALLING FUNCTIONS ##
    import modefitting_peaks
    import modefitting_GR
    import maxcon_appeartime
    
    df_modefit_peaks = modefitting_peaks.find_peaks(df)
    modefitting_GR?? = modefitting_GR.??
    
    
    ## PLOTTING ##
    
def load_data(file_name,start_date,end_date):
    #load data and convert time columns to timestamps
    df = pd.DataFrame(pd.read_csv(file_name,sep=',',engine='python'))
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    df = df.set_index('timestamp')
    df = df.drop(['Year','Month','Day','Hour','Minute','Second'], axis=1)

    #drop bins with no data
    df = df.dropna(axis=1, how='all')

    #select wanted time period and shift times by 15minutes forward
    df = df.loc[start_date:end_date].shift(periods=15, freq='Min')

    #put diameter bins in order
    sorted_columns = sorted(df.columns, key=lambda column: (int(column.split('e')[-1]) , int(column[10:13])))
    df = df[sorted_columns]

    #replace arbitrary column names by diameter float values
    diameter_ints = []
    for column_str in sorted_columns:
        number = column_str[10:13]
        decimal_pos = int(column_str[-1])
        column_float = float(number[:decimal_pos] + '.' + number[decimal_pos:])
        diameter_ints.append(column_float)
    diameter_ints[-1] = 1000.0 #set last bin as 1000
    df.columns = diameter_ints
    return df

if __name__ == "__main__":
    main()
#created: 9.5.2025
#author: Nesrine Bouhlal

# automatic growth rate calculator #

import pandas as pd
import numpy as np
from matplotlib import use, colors
from matplotlib.dates import set_epoch, num2date, date2num
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from warnings import simplefilter
from datetime import timedelta
from time import time
import statsmodels.api as sm

'''
assumptions:
- datasets in json files (AVAA), separated by commas
- time in days & diameters in X*e^Y format 
  (e.g. "HYY_DMPS.d112e2" where diameter is 11.2nm)
- 
'''

def main():
    ## DATASET ##
    file_name = "smeardata_20250506.csv" #in the same folder as the code
    start_date = "2016-04-10"
    end_date = "2016-04-12"
    
    ## PARAMETERS ##
    # mode fitting #
    fit_multimodes = False #choose True if the fit_results -file does not yet exist for your time period (this process takes a few minutes)
    mape_threshold_factor = 15 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_error_threshold = 50 #% (precentage error of growth rates when adding new points to gr lines)
    
    # maximum concentration and appearance time #
    #find_modes
    maximum_peak_difference = 2 #hours (time between two peaks in smoothed data (window 3))
    derivative_threshold = 200 #cm^(-3)/h (starting points of horizontal peak areas, determines what is a high concentration) 
                               #(NOTICE: concentration diff is half of this between timesteps as the resolution is 30min)

    #find_dots
    maximum_time_difference_dots = 2.5 #hours (between current and nearby point)
    mae_threshold_factor = 2 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_precentage_error_threshold = 50 #% (precentage error of growth rates when adding new points to gr lines)

    #channel plotting
    init_plot_channel = True #True to plot channels
    channel_indices = [19] #Indices of diameter channels, 1=small
    show_start_times_and_maxima = True #True to show all possible start times of peak areas (black arrow) and maximas associated
    
    
    ##############################################################################################

    ## LOAD DATA ##
    df = load_data(file_name,start_date,end_date)

    ## CONFIGURATIONS ##
    use("Qt5Agg") #backend changes the UI for plotting
    simplefilter("ignore",OptimizeWarning) #supress warnings for curve_fit to avoid crowding of terminal!!
    simplefilter("ignore",RuntimeWarning)
    set_epoch(start_date) #set epoch

    ## CALLING FUNCTIONS ##
    import modefitting_peaks
    import modefitting_GR
    import maxcon_appeartime
    
    st = time() #progress 
    
    print('\n'+'******** Processing mode fitting data'+'\n')
    df_modefit_peaks = modefitting_peaks.find_peaks(df,fit_multimodes)
    print("Peaks found! (1/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    modefit_gr_points = modefitting_GR.cal_modefit_growth(df_modefit_peaks,a=mape_threshold_factor,gret=gr_error_threshold)
    print("Growth periods found! (2/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    print('\n'+'******** Processing maximum concentration and appearance time data'+'\n')
    maxcon_peaks, apptime_peaks, *_ = maxcon_appeartime.init_methods(df,mpd=maximum_peak_difference,derivative_threshold=derivative_threshold)
    print("Peaks found! (3/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    maxcon_gr_points, apptime_gr_points = maxcon_appeartime.init_find(df,maxcon_peaks,apptime_peaks,mtd=maximum_time_difference_dots,a=mae_threshold_factor,gret=gr_precentage_error_threshold)
    print("Growth periods found! (4/5) "+"(%s seconds)" % (time() - st))
    st = time()

    
    ## PLOTTING ##
    plot_results(df,df_modefit_peaks,modefit_gr_points,maxcon_peaks,maxcon_gr_points,apptime_peaks,apptime_gr_points)
    if init_plot_channel:
        maxcon_appeartime.plot_channel(df,channel_indices,maximum_peak_difference,derivative_threshold,show_start_times_and_maxima)
    plt.show()
    print("Plotting done! (5/5) "+"(%s seconds)" % (time() - st))

##################################################################################################

 
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
def plot_results(df_data,df_modefits,modefit_gr_points,maxcon_peaks,maxcon_gr_points,apptime_peaks,apptime_gr_points):
    def robust_fit(x,y):
        """
        Robust linear regression using statsmodels with HuberT weighting.
        """
        #linear fit for comparison
        #lr = linear_model.LinearRegression().fit(diam, time) #x,y

        x_linear = np.linspace(np.min(x), np.max(x),num=len(y))
        rlm_HuberT = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()) #statsmodel robust linear model
        rlm_results = rlm_HuberT.fit()
        
        #predict data of estimated models
        y_rlm = rlm_results.predict(sm.add_constant(x_linear))
        y_params = rlm_results.params

        #print("Statsmodel robus linear model results: \n",rlm_results.summary())
        #print("\nparameters: ",rlm_results.params)
        #print(help(sm.RLM.fit))
        
        return x_linear, y_rlm, y_params
    def gr_annotation(gr,time,diam):
        midpoint_idx = len(time) // 2 
        plt.annotate(f'{gr:.2f}', (time[midpoint_idx], diam[midpoint_idx]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
        ax.set_title(f'growth rate unit: [nm/h]', loc='right', fontsize=8) 
    
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200) ### figsize=(12, 3), dpi=300 -> figsize=(12, 5), dpi=200

    #original data
    x, y = df_data.index[0:], df_data.columns
    plt.pcolormesh(x, y, df_data[0:].T, cmap='RdYlBu_r', zorder=0, norm=colors.LogNorm(vmin=1e1, vmax=1e4))
    ax.set_yscale('log')
    cbar = plt.colorbar(orientation='vertical', shrink=0.8, extend="max", pad=0.04)
    cbar.set_label('dN/dlogDp', size=14)
    cbar.ax.tick_params(labelsize=12)
    
    #plot line when day changes
    interval = int((24 * 60)/ 30) #assuming 30min resolution
    for i in df_data.index[::interval]:
        plt.axvline(x=i-timedelta(minutes=15), color='black', linestyle='-', lw=1)
    
    
    #MODE FITTING
    #black points
    plt.plot(df_modefits.index,df_modefits['peak_diameter'],'.', alpha=0.8, color='black', mec='black', mew=0.4, ms=6, label='mode fitting')

    #growth rates
    for line in modefit_gr_points:
        time, diam = zip(*line) #UTC, nm
        time_fit, diam_fit, params = robust_fit(date2num(time), diam)
        plt.plot(time_fit, diam_fit,lw=2) #line

        #growth rate annotation
        gr = params[1]/24 #nm/h
        gr_annotation(gr,time,diam_fit)
    
    
    #MAXIMUM CONCENTRATION & APPEARANCE TIME
    #white and green points
    plt.plot(maxcon_peaks[0], maxcon_peaks[1], '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6,label='maximum concentration') 
    plt.plot(apptime_peaks[0], apptime_peaks[1], '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6,label='appearance time')
    
    #growth rates
    for points, color in [(maxcon_gr_points,"white"),(apptime_gr_points,"green")]:
        for line in points:
            time, diam = zip(*line) #days, nm
            diam_fit, time_fit, params = robust_fit(diam, time)
            
            #change days to UTC and plot line
            time_UTC = [dt.replace(tzinfo=None) for dt in num2date(time_fit)]
            plt.plot(time_fit,diam_fit,color=color,lw=2)
            
            #growth rate annotation
            gr = 1/(params[1]*24) #nm/h
            gr_annotation(gr,time_UTC,diam_fit)


    #adjustments to plot
    plt.legend(fontsize=9,fancybox=False,framealpha=0.9)
    for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
        legend_handle.set_markeredgewidth(0.5)
        legend_handle.set_markeredgecolor("black")
    
    plt.xlim(df_data.index[0],df_data.index[-1])
    plt.ylim(df_data.columns[0],df_data.columns[-1])
    plt.ylabel("diameter (nm)",fontsize=14) #add y-axis label
    plt.xlabel("time",fontsize=14) #add y-axis label

if __name__ == "__main__":
    main()
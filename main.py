# created: 9.5.2025
# author: Nesrine Bouhlal

# automatic growth rate calculator #

import pandas as pd
import numpy as np
from matplotlib import use, colors
from matplotlib.dates import set_epoch, num2date, date2num
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeWarning
from sklearn.linear_model import RANSACRegressor
from warnings import simplefilter
from datetime import timedelta, datetime
from time import time
import statsmodels.api as sm
import xarray as xr
from adjustText import adjust_text

'''
assumptions:
- datasets in json files (AVAA), separated by commas
- time in days & diameters in X*e^Y format 
  (e.g. "HYY_DMPS.d112e2" where diameter is 11.2nm)

abbreviations:
MF = mode fitting, MC = maximum concentration, AT = appearance time
'''

def main():
    ## DATASET ##
    file_name = "Vavihill.nc" #in the same folder as the code
    start_date = "2008-07-10"
    end_date = "2008-07-12"
    
    ## PARAMETERS ##
    # mode fitting #
    fit_multimodes = False #choose True if the fit_results -file does not yet exist for your time period (this process takes a few minutes)
    mape_threshold_factor = 15 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_error_threshold = 50 #% (precentage error of growth rates when adding new points to gr lines)
    
    
    # maximum concentration and appearance time #
    #find_peak_areas
    maximum_peak_difference = 2 #hours (time between two peaks in smoothed data (window 3))
    derivative_threshold = 200 #cm^(-3)/h (starting points of horizontal peak areas, determines what is a high concentration) 
                               #(NOTICE: concentration diff is half of this between timesteps as the resolution is 30min)

    #find_growth
    mae_threshold_factor = 2 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_precentage_error_threshold = 70 #% (precentage error of growth rates when adding new points to gr lines)
    maximum_diameter_channel = 60 #nm (highest diameter channel where growth lines are extended)
    maximum_growth_start_channel = 30 #nm (highest diameter channel where growth lines are allowed to start)

    #channel plotting
    init_plot_channel = False #True to plot channels
    channel_indices = [18,19,20] #Indices of diameter channels, 1=small
    show_start_times_and_maxima = True #True to show all possible start times of peak areas (black arrow) and maximas associated
    
    
    ##############################################################################################

    ## LOAD DATA ##
    df = load_NC_data(file_name,start_date,end_date)
    print(df)
    
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
    df_MF_peaks = modefitting_peaks.find_peaks(df,fit_multimodes)
    print("Peaks found! (1/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    MF_gr_points = modefitting_GR.cal_modefit_growth(df_MF_peaks,a=mape_threshold_factor,gret=gr_error_threshold)
    print("Growth periods found! (2/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    print('\n'+'******** Processing maximum concentration and appearance time data'+'\n')
    df_MC, df_AT, incomplete_MC_xyz, incomplete_AT_xyz, *_ = maxcon_appeartime.init_methods(df,mpd=maximum_peak_difference,mdc=maximum_diameter_channel,derivative_threshold=derivative_threshold)
    print("Peaks found! (3/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    MC_gr_points, AT_gr_points = maxcon_appeartime.init_find(df,df_MC,df_AT,mgsc=maximum_growth_start_channel,a=mae_threshold_factor,gret=gr_precentage_error_threshold)
    print("Growth periods found! (4/5) "+"(%s seconds)" % (time() - st))
    st = time()

    
    ## PLOTTING ##
    plot_results(file_name,df,df_MF_peaks,MF_gr_points,df_MC,incomplete_MC_xyz,MC_gr_points,df_AT,incomplete_AT_xyz,AT_gr_points)
    if init_plot_channel:
        maxcon_appeartime.plot_channel(df,channel_indices,maximum_peak_difference,maximum_diameter_channel,derivative_threshold,show_start_times_and_maxima)
    plt.show()
    print("Plotting done! (5/5) "+"(%s seconds)" % (time() - st))

##################################################################################################

 
def load_AVAA_data(file_name,start_date,end_date):
    #load data and convert time columns to timestamps
    df = pd.DataFrame(pd.read_csv(file_name,sep=',',engine='python'))
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    df = df.set_index('timestamp')
    df = df.drop(['Year','Month','Day','Hour','Minute','Second'], axis=1)

    #drop bins with no data
    df = df.dropna(axis=1, how='all')
    
    #print time period
    first_time = df.index[0]
    last_time = df.index[-1] 
    print("start of period:",first_time.strftime("%Y-%m-%d %H:%M"),"\nend of period:",last_time.strftime("%Y-%m-%d %H:%M"))

    #select wanted time period
    df = df.loc[start_date:end_date]
    
    #round data to nearest 30min and shift times by 15minutes forward
    df.index = df.index.round("30min")
    df = df.shift(periods=15, freq='Min')
    #df.index = df.index.floor("min")

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
def load_NC_data(file_name,start_date,end_date):
    ds = xr.open_dataset(file_name,engine='netcdf4')
    data_mean = ds['PNSD']
    
    #print time period
    first_time = ds['time'].values[0].astype('M8[s]').astype(datetime)
    last_time = ds['time'].values[-1].astype('M8[s]').astype(datetime)  
    print("start of period:",first_time.strftime("%Y-%m-%d %H:%M"),"\nend of period:",last_time.strftime("%Y-%m-%d %H:%M"))
    
    #select time period
    data_subset_mean = data_mean.sel(time=slice(start_date, end_date))

    x = data_subset_mean['time'].values
    y = data_subset_mean['bin'].values * 10e8
    z_mean = data_subset_mean.values
    
    #assemble data to dataframe
    df = pd.DataFrame(data=z_mean,index=x,columns=y)
    
    #drop bins with no data
    df = df.dropna(axis=1, how='all')
    
    return df
    
def plot_results(file_name,df_data,df_MF_peaks,MF_gr_points,df_MC,incomplete_MC_xyz,MC_gr_points,df_AT,incomplete_AT_xyz,AT_gr_points):
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
    def RANSAC(x,y):
        X = np.array(x)  # Replace 'points' with your data
        Y = np.array(y)

        # Reshape for RANSAC
        X = X.reshape(-1, 1)

        # Initialize RANSAC
        ransac = RANSACRegressor(residual_threshold=5.0)  # Adjust threshold based on data
        ransac.fit(X, Y)

        # Get inliers and outliers
        inliers = ransac.inlier_mask_
        outliers = ~ransac.inlier_mask_
        
        # parameters
        params = ransac.estimator_.coef_

        # Line Parameters
        line_X = np.linspace(min(X), max(X), 100).reshape(-1, 1)
        line_Y = ransac.predict(line_X)
        
        return line_X, line_Y, params
    def gr_annotation(gr,t,d):
        midpoint_idx = len(t) // 2 
        annotation = plt.annotate(f'{gr:.2f}', (t[midpoint_idx], d[midpoint_idx]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
        ax.set_title(f'growth rate unit: [nm/h]', loc='right', fontsize=8) 
        
        return annotation   
    def detect_growth(MF_gr_points,MC_gr_points,AT_gr_points):
        
        MF_growth, MC_growth, AT_growth = [],[],[]

        for MF_line in MF_gr_points:
            
            #mode fitting and maximum concentration
            for MC_line in MC_gr_points:
                MF_t, MF_d = zip(*MF_line)
                MC_t, MC_d = zip(*MC_line)
                
                if any(t >= min(date2num(MF_t)) for t in MC_t) and any(t <= max(date2num(MF_t)) for t in MC_t) and \
                   any(d >= min(MF_d) for d in MC_d) and any(d <= max(MF_d) for d in MC_d):
                    MF_growth.append(MF_line)
                    if MC_line not in MC_growth:
                        MC_growth.append(MC_line)
                    break

            #mode fitting and appearance time
            for AT_line in AT_gr_points:
                MF_t, MF_d = zip(*MF_line)
                AT_t, AT_d = zip(*AT_line)
                
                if any(t >= min(date2num(MF_t)) for t in AT_t) and any(t <= max(date2num(MF_t)) for t in AT_t) and \
                   any(d >= min(MF_d) for d in AT_d) and any(d <= max(MF_d) for d in AT_d):
                    MF_growth.append(MF_line)
                    if AT_line not in AT_growth:
                        AT_growth.append(AT_line)
                    break
        
        return MF_growth, MC_growth, AT_growth

        
    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)

    #original data
    x, y = df_data.index[0:], df_data.columns
    plt.pcolormesh(x, y, df_data[0:].T, cmap='RdYlBu_r', zorder=0, norm=colors.LogNorm(vmin=1e1, vmax=5e4))
    ax.set_yscale('log')
    cbar = plt.colorbar(orientation='vertical', shrink=0.8, extend="max", pad=0.04)
    cbar.set_label('dN/dlogDp', size=14)
    cbar.ax.tick_params(labelsize=12)
    
    #plot line when day changes
    interval = int((24 * 60)/ 30) #assuming 30min resolution
    for i in df_data.index[::interval]:
        plt.axvline(x=i-timedelta(minutes=15), color='black', linestyle='-', lw=1)
    
    
    MF_gr_points, MC_gr_points, AT_gr_points = detect_growth(MF_gr_points,MC_gr_points,AT_gr_points)

    #MODE FITTING
    #asseble all diams to a set for faster performance and filter peaks that arent in a line
    all_diameters = set(d for line in MF_gr_points for d in list(zip(*line))[1])
    df_MF_peaks = df_MF_peaks[df_MF_peaks['peak_diameter'].isin(all_diameters)]
    
    #black points 
    plt.plot(df_MF_peaks.index,df_MF_peaks['peak_diameter'],'.', alpha=0.8, color='black', mec='black', mew=0.4, ms=6, label='mode fitting')

    #growth rates    
    MF_texts = []
    for line in MF_gr_points:
        t, d = zip(*line) #UTC, nm
        #t_fit, d_fit, params = robust_fit(date2num(t), d)
        t_fit, d_fit, params = RANSAC(date2num(t), d)
        plt.plot(t_fit, d_fit,color='black',alpha=0.9,lw=2) #line

        #growth rate annotation
        gr = params[0]/24 #nm/h
        MF_text = gr_annotation(gr,t,d_fit)
        MF_texts.append(MF_text)
    
    
    #MAXIMUM CONCENTRATION & APPEARANCE TIME
    #filter points not in a line
    all_times = set(d for line in MC_gr_points for d in list(zip(*line))[0])
    all_times = [num2date(dt).replace(tzinfo=None) for dt in all_times]
    df_MC = df_MC[df_MC['timestamp'].isin(all_times)]
    
    all_times = set(d for line in AT_gr_points for d in list(zip(*line))[0])
    all_times = [num2date(dt).replace(tzinfo=None) for dt in all_times]
    df_AT = df_AT[df_AT['timestamp'].isin(all_times)]
    
    #white and green points
    plt.plot(df_MC['timestamp'], df_MC['peak_diameter'], '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6,label='maximum concentration') 
    plt.plot(df_AT['timestamp'], df_AT['diameter'], '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6,label='appearance time')
    
    #growth rates
    MC_texts, AT_texts = [], []
    
    for points, incomplete_points, texts, color in [(MC_gr_points,incomplete_MC_xyz, MC_texts,"white"),(AT_gr_points,incomplete_AT_xyz, AT_texts,"green")]:
        for line in points:
            t, d = zip(*line) #days, nm
            d_fit, t_fit, params = robust_fit(d, t)
            
            #change days to UTC and plot line
            t_UTC = [dt.replace(tzinfo=None) for dt in num2date(t)]
            t_fit_UTC = [dt.replace(tzinfo=None) for dt in num2date(t_fit)]
            
            #growth rate annotation
            gr = 1/(params[1]*24) #nm/h
            text = gr_annotation(gr,t_fit_UTC,d_fit)
            texts.append(text)
            
            #plot poorly defined lines as dotted lines and well defined as full lines
            for t in t_UTC:
                if t in incomplete_points[0]:
                    plt.plot(t_fit,d_fit,ls="dashed",color=color,lw=1.5)
                    break
            else:
                plt.plot(t_fit,d_fit,color=color,lw=2)
    
    # annotations = MF_texts + MC_texts + AT_texts
    # adjust_text(annotations, arrowprops=dict(arrowstyle="->", color='gray'))

    #adjustments to plot
    plt.legend(fontsize=9,fancybox=False,framealpha=0.9)
    for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
        legend_handle.set_markeredgewidth(0.5)
        legend_handle.set_markeredgecolor("black")
    
    plt.xlim(df_data.index[0],df_data.index[-1])
    plt.ylim(df_data.columns[0],df_data.columns[-1])
    plt.ylabel("diameter (nm)",fontsize=14) #add y-axis label
    plt.xlabel("time",fontsize=14) #add y-axis label
    plt.title(f"{file_name}")
    

if __name__ == "__main__":
    main()
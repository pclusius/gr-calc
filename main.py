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
from highlight_text import fig_text, ax_text

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
    file_name = "Vavihill.nc" #in the same folder as this code
    start_date = "2008-05-10"
    end_date = "2008-05-12"
    
    ## PARAMETERS ##
    # mode fitting #
    fit_multimodes = False #choose True if the fit_results -file does not yet exist for your time period (this process takes a few minutes)
    mape_threshold_factor = 15 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_error_threshold = 60 #% (precentage error of growth rates when adding new points to gr lines)
    show_mape = False 
    
    
    # maximum concentration and appearance time #
    #find_peak_areasc
    maximum_peak_difference = 2 #hours (time between two peaks in smoothed data (window 3))
    derivative_threshold = 200 #cm^(-3)/h (starting points of horizontal peak areas, determines what is a high concentration) 
                               #(NOTICE: concentration diff is half of this between timesteps as the resolution is 30min)

    #find_growth
    mae_threshold_factor = 1 #a*x^(-1) (constant a that determines mean average error thresholds for different line lengths)
    gr_precentage_error_threshold = 60 #% (precentage error of growth rates when adding new points to gr lines)
    maximum_diameter_channel = 60 #nm (highest diameter channel where growth lines are extended)
    maximum_growth_start_channel = 40 #nm (highest diameter channel where growth lines are allowed to start)

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
    df_MF_peaks = modefitting_peaks.find_peaks(df,file_name,fit_multimodes)
    print("Peaks found! (1/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    MF_gr_points, df_mapes = modefitting_GR.cal_modefit_growth(df_MF_peaks,a=mape_threshold_factor,gret=gr_error_threshold)
    print("Growth periods found! (2/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    print('\n'+'******** Processing maximum concentration and appearance time data'+'\n')
    df_MC, df_AT, incomplete_MC_xyz, incomplete_AT_xyz, mc_area_edges, at_area_edges, *_ = maxcon_appeartime.init_methods(df,mpd=maximum_peak_difference,mdc=maximum_diameter_channel,derivative_threshold=derivative_threshold)
    print("Peaks found! (3/5) "+"(%s seconds)" % (time() - st))
    st = time()
    
    MC_gr_points, AT_gr_points = maxcon_appeartime.init_find(df,df_MC,df_AT,mgsc=maximum_growth_start_channel,a=mae_threshold_factor,gret=gr_precentage_error_threshold)
    print("Growth periods found! (4/5) "+"(%s seconds)" % (time() - st))
    st = time()

    
    ## PLOTTING ##
    plot_results(file_name,df,df_MF_peaks,MF_gr_points,df_mapes,df_MC,incomplete_MC_xyz,MC_gr_points,df_AT,incomplete_AT_xyz,AT_gr_points,mc_area_edges,at_area_edges,show_mape)
    if init_plot_channel:
        maxcon_appeartime.plot_channel(df,channel_indices,maximum_peak_difference,maximum_diameter_channel,derivative_threshold,show_start_times_and_maxima)
    print("Plotting done! (5/5) "+"(%s seconds)" % (time() - st))
    plt.show()

##################################################################################################

 
def load_AVAA_data(file_name,start_date,end_date):
    '''
    Loads data downloaded from AVAA platforms (SmartSMEAR).
    '''
    
    #load data and convert time columns to timestamps
    df = pd.DataFrame(pd.read_csv(file_name,sep=',',engine='python'))
    df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    df = df.set_index('timestamp')
    df = df.drop(['Year','Month','Day','Hour','Minute','Second'], axis=1)

    #drop bins with no data
    df = df.dropna(axis=1, how='all')
    
    #print time period
    print("start of period:",df.index[0].strftime("%Y-%m-%d %H:%M"))
    print("end of period:",df.index[-1].strftime("%Y-%m-%d %H:%M"))
    
    #select wanted time period
    df = df.loc[start_date:end_date]
    
    #round data to nearest 30min and shift times by 15minutes forward
    original_timestamps = df.index.copy()
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

    #check for duplicate timestamps
    if len(df.index) != len(set(df.index)):
        print("Original timestamps:", list(original_timestamps))
        print("ERROR: Multiple identical timestamps detected! Please make sure the data has been sampled evenly.")
        raise SystemExit
    
    return df
def load_NC_data(file_name,start_date,end_date):
    '''
    Loads data with nc format. 
    '''
    
    ds = xr.open_dataset(file_name,engine='netcdf4')
    data_mean = ds['PNSD']
    
    #print time period
    first_time = ds['time'].values[0].astype('M8[s]').astype(datetime)
    last_time = ds['time'].values[-1].astype('M8[s]').astype(datetime)  
    print("start of period:",first_time.strftime("%Y-%m-%d %H:%M"),"\nend of period:",last_time.strftime("%Y-%m-%d %H:%M"))
    
    print(start_date)
    #select time period
    #include half day before and after chosen time period
    # start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(hours=12)
    # end_date = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(hours=12)
    data_subset_mean = data_mean.sel(time=slice(start_date, end_date))
    

    x = data_subset_mean['time'].values
    y = data_subset_mean['bin'].values * 10e8
    z_mean = data_subset_mean.values
    
    #assemble data to dataframe
    df = pd.DataFrame(data=z_mean,index=x,columns=y)
    
    #drop bins with no data
    df = df.dropna(axis=1, how='all')
    
    #check for duplicate timestamps
    if len(df.index) != len(set(df.index)):
        print("ERROR: Multiple identical timestamps detected! Please make sure the data has been sampled evenly.")
        raise SystemExit
    
    return df
    
def plot_results(file_name,df_data,df_MF_peaks,MF_gr_points,df_mapes,df_MC,incomplete_MC_xyz,MC_gr_points,df_AT,incomplete_AT_xyz,AT_gr_points,mc_area_edges,at_area_edges,show_mape):
    def gr_annotation(gr,t,d):
        midpoint_idx = len(t) // 2 
        annotation = plt.annotate(f'{gr:.2f}', (t[midpoint_idx], d[midpoint_idx]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
        ax.set_title(f'growth rate unit: [nm/h]', loc='right', fontsize=8) 
        
        return annotation   
        
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
    
    #growth rates
    plot_all_lines = True
    if plot_all_lines:
        all_diameters_MF, all_times_MC, all_times_AT = [], [], []

        #SAVE ALL DIAMS AND TIMES
        all_diameters_MF = set(d for line in MF_gr_points.values() for d in list(zip(*line['points']))[1])
        all_times_MC = set(d for line in MC_gr_points.values() for d in list(zip(*line['points']))[0])
        all_times_MC = [num2date(dt).replace(tzinfo=None) for dt in all_times_MC]
        all_times_AT = set(d for line in AT_gr_points.values() for d in list(zip(*line['points']))[0])
        all_times_AT = [num2date(dt).replace(tzinfo=None) for dt in all_times_AT]

        #GROWTH RATES
        #mode fitting
        for i,line in enumerate(MF_gr_points.values()):
            t, d = zip(*line['points']) #UTC, nm
            t_fit, d_fit = zip(*line['fitted points'])
            plt.plot(t_fit, d_fit,color='black',alpha=0.9,lw=2) #line

            #annotation
            if show_mape: #mape
                mape = df_mapes['mape'][i]
                midpoint_idx = len(t) // 2 
                plt.annotate(f'{mape:.2f}', (t[midpoint_idx], d[midpoint_idx]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
                ax.set_title(f'mean absolute error (MAE) unit: [hours]', loc='right', fontsize=8)
            else: #growth rate
                gr = line['growth rate'] #nm/h
                MF_text = gr_annotation(gr,t,d_fit)

        #maximum concentration and appearance time
        for points, incomplete_points, color in [(MC_gr_points.values(),incomplete_MC_xyz,"white"),(AT_gr_points.values(),incomplete_AT_xyz,"green")]:
            for line in points:
                t, d = zip(*line['points']) #days, nm
                t_fit, d_fit = zip(*line['fitted points'])
                
                #change days to UTC and plot line
                t_UTC = [dt.replace(tzinfo=None) for dt in num2date(t)]
                t_fit_UTC = [dt.replace(tzinfo=None) for dt in num2date(t_fit)]
                
                #growth rate annotation
                gr = line['growth rate'] #nm/h
                text = gr_annotation(gr,t_fit_UTC,d_fit)
                
                #plot poorly defined lines as dotted lines and well defined as full lines
                incomplete_times = [point[0] for point in incomplete_points]
                
                for t in t_UTC:
                    if t in incomplete_times:
                        plt.plot(t_fit,d_fit,ls="dashed",color=color,lw=1.5,zorder=10)
                        break
                else:
                    plt.plot(t_fit,d_fit,color=color,lw=2,zorder=10)
    else:
        import growth_events
        events = growth_events.detect_events(MF_gr_points,MC_gr_points,AT_gr_points,mc_area_edges,at_area_edges)
        event_grs = growth_events.event_growth_rates(events)

        all_diameters_MF, all_times_MC, all_times_AT = [], [], []
        
        for event in events.values():
            for line in event:
                
                #MODE FITTING
                if line['method'] == 'MF':
                    t, d = zip(*line['points']) #UTC, nm
                    t_fit, d_fit = zip(*line['fitted points'])
                    plt.plot(t_fit, d_fit,color='black',alpha=0.9,lw=2,zorder=5) #line
                    all_diameters_MF.extend(d)
                    
                    #annotation
                    if show_mape: #mape
                        mape = df_mapes['mape'][i]
                        midpoint_idx = len(t) // 2 
                        plt.annotate(f'{mape:.2f}', (t[midpoint_idx], d[midpoint_idx]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
                        ax.set_title(f'mean absolute error (MAE) unit: [hours]', loc='right', fontsize=8)
                    else: #growth rate
                        gr = line['growth rate'] #nm/h
                        MF_text = gr_annotation(gr,t,d_fit)
                
                #MAXIMUM CONCENTRATION
                if line['method'] == 'MC':
                    t, d = zip(*line['points']) #days, nm
                    t_fit, d_fit = zip(*line['fitted points'])
                    
                    #change days to UTC and plot line
                    t_UTC = [dt.replace(tzinfo=None) for dt in num2date(t)]
                    t_fit_UTC = [dt.replace(tzinfo=None) for dt in num2date(t_fit)]
                    all_times_MC.extend(t_UTC)
                    
                    #plot poorly defined lines as dotted lines and well defined as full lines
                    incomplete_times_MC = [point[0] for point in incomplete_MC_xyz]
                    
                    for t in t_UTC:
                        if t in incomplete_times_MC:
                            plt.plot(t_fit,d_fit,ls="dashed",color='white',lw=1.5,zorder=10)
                            break
                    else:
                        plt.plot(t_fit,d_fit,color='white',lw=2,zorder=10)
                    
                    #growth rate annotation
                    gr = line['growth rate'] #nm/h
                    text = gr_annotation(gr,t_fit_UTC,d_fit)

                
                #APPEARANCE TIME
                if line['method'] == 'AT':
                    t, d = zip(*line['points']) #days, nm
                    t_fit, d_fit = zip(*line['fitted points'])
                    
                    #change days to UTC and plot line
                    t_UTC = [dt.replace(tzinfo=None) for dt in num2date(t)]
                    t_fit_UTC = [dt.replace(tzinfo=None) for dt in num2date(t_fit)]
                    all_times_AT.extend(t_UTC)
                    
                    #plot poorly defined lines as dotted lines and well defined as full lines
                    incomplete_times_AT = [point[0] for point in incomplete_AT_xyz]
                    
                    for t in t_UTC:
                        if t in incomplete_times_AT:
                            plt.plot(t_fit,d_fit,ls="dashed",color='green',lw=1.5,zorder=10)
                            break
                    else:
                        plt.plot(t_fit,d_fit,color='green',lw=2,zorder=10)
                    
                    #growth rate annotation
                    gr = line['growth rate'] #nm/h
                    text = gr_annotation(gr,t_fit_UTC,d_fit)
        # print(event_grs)
        # for i in event_grs.index:
        #     avg_gr = event_grs.loc[i]['avg growth rate']
        #     min_gr = event_grs.loc[i]['min growth rate']
        #     max_gr = event_grs.loc[i]['max growth rate']
        #     mid_loc = event_grs.loc[i]['mid location']
            
        #     if max_gr < 0:
        #         text = f'{avg_gr:.2f} ({min_gr:.2f}– {max_gr:.2f})'
        #     else:
        #         text = f'{avg_gr:.2f} ({min_gr:.2f}–{max_gr:.2f})'
            
        #     ax.text(
        #         x=mid_loc[0],  # position on x-axis
        #         y=mid_loc[1],  # position on y-axis
        #         s=text,
        #         fontsize=7,
        #         bbox=dict(edgecolor="none", facecolor="white", alpha=0.8, pad=1),
        #         zorder=20,
        #         ha='center', 
        #         va='center'
        #     )
            
            
        
    #filter points not in a line
    df_MF_peaks = df_MF_peaks[df_MF_peaks['peak_diameter'].isin(all_diameters_MF)]
    df_MC = df_MC[df_MC['timestamp'].isin(all_times_MC)]
    df_AT = df_AT[df_AT['timestamp'].isin(all_times_AT)]
    
    #plot points
    plt.plot(df_MF_peaks.index,df_MF_peaks['peak_diameter'],'.', alpha=0.8, color='black', mec='black', mew=0.4, ms=6, label='mode fitting',zorder=0)
    plt.plot(df_MC['timestamp'], df_MC['peak_diameter'], '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6,label='maximum concentration',zorder=0) 
    plt.plot(df_AT['timestamp'], df_AT['diameter'], '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6,label='appearance time',zorder=0)

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
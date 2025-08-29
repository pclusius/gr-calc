# created: 9.5.2025
# author: Nesrine Bouhlal

# automatic growth rate calculator #
import numpy as np
import pandas as pd
import sys, os, json
import matplotlib.pyplot as plt
from matplotlib import use, colors
from matplotlib.dates import set_epoch, num2date
from scipy.optimize import OptimizeWarning
from warnings import simplefilter
from datetime import timedelta, datetime
from time import time
from xarray import open_dataset
import argparse
from pdb import set_trace as bp

default_file_name = "Beijing.nc" #in the same folder as this code, unless -d or --data flag was pointing to another directory
default_start_date = "2004-09-20" #YYYY-MM-DD HH:MM:SS (time of day is optional)
default_end_date = "2004-09-22"
default_fitting = "new"

'''
abbreviations:
MF = mode fitting, MC = maximum concentration, AT = appearance time
'''

## Create an ArgumentParser object
parser = argparse.ArgumentParser(
                description='Program to calculate growth rates from smps/dmps etc. data',
                )


## RESULTS ##
result_config = {
    'plot_all_points': False, #plots all points for all methods XXX
    'plot_all_lines': False, #plots all lines XXX
    # 'plot_all_events': True, #plots all events
    'plot_final_events': True, #plots the final results XXX
    # 'print_final_event_info': False, #prints info about each final event on screen
    'save_final_event_info': False, #saves info about each final event in a file  XXX
    'print_ts_info': False, #prints info about each timestamp of an event
    'save_ts_info': False, #saves info about timestamps in each event in a file
    'plot_event_info': False, #plots estimated growth rate and range for each event (white box) XXX
    'plot_DT': False, #plots disappearance times (NOT FINISHED!)
    'ens_number': None #Inital value, gets updated, don't change!)
}

#Graph size and colorscale can be changed from the show_results function!
plot_config = {
    'figsize':(14, 5),
    'vmin':1e1,
    'vmax':5e3,
    'cmap':'RdYlBu_r'
    # 'vmin':1e20,
    # 'vmax':5e20,
    # 'cmap':'Greys'
}

## DATASET ##

## PARAMETERS ##
# mode fitting #
# fit_multimodes = False #True if the fit_results -file does not yet exist for your time period
mape_threshold_factor = 15 #a*x^(-1) (constant 'a' that determines mean average error thresholds for different line lengths)
gr_error_threshold_MF = 60 #% (precentage error of growth rates when adding new points to lines)

# maximum concentration and appearance time #
#peak areas
maximum_peak_difference = 2 #hours (max time between two peaks in smoothed data (window 3))
derivative_threshold = 200 #cm^(-3)/h (starts of horizontal peak areas, determines the appearance of a possible event)
#(NOTICE: concentration diff is half of this between timesteps as the resolution is 30min)

#find_growth
mae_threshold_factor = 1 #a*x^(-1) (constant 'a' that determines mean average error thresholds for different line lengths)
gr_error_threshold_MCAT = 60 #% (precentage error of growth rates when adding new points to lines)
maximum_diameter_channel = 60 #nm (highest diameter channel where growth lines are extended)
maximum_growth_start_channel = 40 #nm (highest diameter channel where growth lines are allowed to start)

#channel plotting (maximum concentration and appearance time)
channel_indices = [] #Indices of diameter channels (1=small), empty list ([]) if no channels plotted
show_start_times_and_maxima = True #True to show all possible start times of peak areas (black arrow)
#and maximas associated (small black dot)

bufferhours = 4 # hours to buffer before and after



##############################################################################################
def main(ens_number=None):

    ## LOAD DATA ##

    df,df_plot = load_NC_data(file_name,start_date,end_date)
    if args.auto_limits:
        plot_config['vmin'] = max(1,   np.min(df_plot))
        plot_config['vmax'] = max(100, np.max(df_plot))

    #print(df)
    ## CONFIGURATIONS ##
    use("Qt5Agg") #backend changes the UI for plotting
    simplefilter("ignore",OptimizeWarning) #supress warnings for curve_fit to avoid crowding of terminal!!
    simplefilter("ignore",RuntimeWarning)
    if ens_number is None or ens_number == 0:
        set_epoch(start_date) #set epoch

    ## CALLING FUNCTIONS ##
    if args.ensemble>1:
        import modefitting_peaks_ensemble
    else:
        import modefitting_peaks
    import modefitting_GR
    import maxcon_appeartime

    def log_step(message, start_time, step_num, total_steps=4):
        print(f"{message} ({step_num}/{total_steps}) ({time() - start_time:.2f} seconds)")
        return time()
    print('\n'+'******** Processing mode fitting data'+'\n')
    st = time() #progress

    # Step 1: Find mode fitting peaks
    if args.ensemble>1:
        df_MF_peaks = modefitting_peaks_ensemble.find_peaks(df,file_name,start_date,
                                                n_samples=args.samples,ensemble_size=args.ensemble,
                                                method = args.ensemble_method, ens_number = ens_number
                                            )
    else:
        df_MF_peaks = modefitting_peaks.find_peaks(df,file_name,start_date,
                                                n_samples=args.samples
                                            )
    st = log_step("Peaks found!", st, 1)

    # Step 2: Find periods of growth
    MF_gr_points = modefitting_GR.find_growth(df_MF_peaks,a=mape_threshold_factor,gret=gr_error_threshold_MF)
    st = log_step("Growth periods found!", st, 2)

    # Step 3: Find maximum concentration peaks and appearance times
    print('\n'+'******** Processing maximum concentration and appearance time data'+'\n')
    df_MC, df_AT, df_DT, incomplete_MC, incomplete_AT, mc_area_edges, *_ = maxcon_appeartime.init_methods(
        df,mpd=maximum_peak_difference,mdc=maximum_diameter_channel,derivative_threshold=derivative_threshold)
    st = log_step("Peaks found!", st, 3)

    # Step 4: Find their growth periods
    MC_gr_points, AT_gr_points = maxcon_appeartime.init_find(
        df,df_MC,df_AT,mgsc=maximum_growth_start_channel,a=mae_threshold_factor,gret=gr_error_threshold_MCAT)
    st = log_step("Growth periods found!", st, 4)

    # Step 5: Results
    ts_info = show_results(file_name,start_date,df,df_plot,df_MF_peaks,MF_gr_points,df_MC,incomplete_MC,MC_gr_points,
                 df_AT,incomplete_AT,AT_gr_points,df_DT,mc_area_edges,result_config,maximum_growth_start_channel)
    if channel_indices:
        maxcon_appeartime.plot_channel(df_plot,channel_indices,maximum_peak_difference,
                                       maximum_diameter_channel,derivative_threshold,show_start_times_and_maxima)
    plt.savefig(f'{args.figname}.png')

    if args.close_figs:
        plt.close()
    elif args.ensemble_method==0:
        plt.ion()
        plt.show()
    else:
        plt.close()

    return ts_info

##################################################################################################

def load_AVAA_data(file_name,start_date,end_date):
    '''
    Loads data downloaded from AVAA platforms (SmartSMEAR).
    In case tar-data is not available.
    Time in days & diameters in X*e^Y format
    (e.g. "HYY_DMPS.d112e2" where diameter is 11.2nm)
    '''

    def process_df(dataframe):
        #round data to nearest 30min and shift times by 15minutes forward
        original_timestamps = dataframe.index.copy()
        dataframe.index = dataframe.index.round("30min")
        dataframe = dataframe.shift(periods=15, freq='Min')

        #put diameter bins in order
        sorted_columns = sorted(dataframe.columns, key=lambda column: (int(column.split('e')[-1]) , int(column[10:13])))
        dataframe = dataframe[sorted_columns]

        #replace arbitrary column names by diameter float values
        diameter_ints = []
        for column_str in sorted_columns:
            number = column_str[10:13]
            decimal_pos = int(column_str[-1])
            column_float = float(number[:decimal_pos] + '.' + number[decimal_pos:])
            diameter_ints.append(column_float)
        diameter_ints[-1] = 1000.0 #set last bin as 1000
        dataframe.columns = diameter_ints

        return dataframe, original_timestamps

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
    df_plot = df.loc[start_date:end_date]

    #include half day before and after for gr calculations
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") - timedelta(hours=bufferhours)
        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") + timedelta(hours=bufferhours)
    except ValueError:
        start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(hours=bufferhours)
        end_date = datetime.strptime(f'{end_date} 23:59:59', "%Y-%m-%d %H:%M:%S") + timedelta(hours=bufferhours)

    df = df.loc[start_date:end_date]

    df, *_ = process_df(df)
    df_plot, original_timestamps = process_df(df_plot)

    #check for duplicate timestamps
    if len(df.index) != len(set(df.index)):
        print("Original timestamps:", [str(stamp) for stamp in original_timestamps])
        print("ERROR: Multiple identical timestamps detected! Please make sure the data has been sampled evenly.")
        raise SystemExit

    return df, df_plot


def load_NC_data(file_name,start_date,end_date):
    '''
    Loads data with nc format.
    '''
    def assemble_df(subset):
        x = subset['time'].values
        y = subset['bin'].values * 10e8
        z_mean = subset.values
        dataframe = pd.DataFrame(data=z_mean,index=x,columns=y)
        dataframe = dataframe.dropna(axis=1, how='all') #drop bins with no data
        return dataframe

    ds = open_dataset(file_name,engine='netcdf4')
    data = ds['PNSD']

    #print time period
    first_time = ds['time'].values[0].astype('M8[s]').astype(datetime)
    last_time = ds['time'].values[-1].astype('M8[s]').astype(datetime)
    print("start of period in the datafile:",first_time.strftime("%Y-%m-%d %H:%M"))
    print("end of period in the datafile:",last_time.strftime("%Y-%m-%d %H:%M"))
    earlier = first_time>(pd.to_datetime(start_date)-pd.Timedelta(f'{bufferhours}h'))
    later = last_time<(pd.to_datetime(start_date)+pd.Timedelta(f'{bufferhours}h'))
    if any([earlier,later]):
        print(f'The selected data does not contain the whole period.')
        exit('Check the dates above and use them. You can also use --start and --duration')

    #select time period
    data_plotting = data.sel(time=slice(start_date, end_date)) #plotting

    #include half day before and after for gr calculations
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") - timedelta(hours=bufferhours)
        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") + timedelta(hours=bufferhours)
    except ValueError:
        start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(hours=bufferhours)
        end_date = datetime.strptime(f'{end_date} 23:59:59', "%Y-%m-%d %H:%M:%S") + timedelta(hours=bufferhours)

    data_subset = data.sel(time=slice(start_date, end_date))

    df_plot = assemble_df(data_plotting) #df for plotting
    df = assemble_df(data_subset) #df for gr calculations

    #check for duplicate timestamps
    if len(df.index) != len(set(df.index)):
        print("ERROR: Multiple identical timestamps detected! Please make sure the data has been sampled evenly.")
        raise SystemExit

    return df, df_plot


def show_results(file_name,start_date,df_data,df_plot,df_MF_peaks,MF_gr_points,df_MC,incomplete_MC,MC_gr_points,
                 df_AT,incomplete_AT,AT_gr_points,df_DT,mc_area_edges,result_config,mgsc):

    import growth_events

    if args.ensemble_method==1:
        dfdf = None

    all_events, final_events = growth_events.init_events(df_data,df_plot,MF_gr_points,MC_gr_points,AT_gr_points,mc_area_edges,mgsc)

    ts_info = growth_events.timestamp_info(all_events)

    if any([result_config['plot_all_points'],result_config['plot_all_lines'],
            result_config['plot_final_events'],result_config['plot_DT']]):

        fig, ax = plt.subplots(figsize=plot_config['figsize'], dpi=200)

        #colormap
        x, y = df_plot.index[0:], df_plot.columns
        plt.pcolormesh(x, y, df_plot[0:].T, cmap=plot_config['cmap'], zorder=0,
            norm=colors.LogNorm(vmin=plot_config['vmin'], vmax=plot_config['vmax']))
        ax.set_yscale('log')
        cbar = plt.colorbar(orientation='vertical', shrink=0.8, extend="max", pad=0.04)
        cbar.set_label('dN/dlogDp', size=14)
        cbar.ax.tick_params(labelsize=12)
        if plot_config['vmin']>1e10:
            cbar.ax.set_visible(False)
        #line when day changes
        start_time = df_plot.index[0]
        end_time = df_plot.index[-1]
        first_midnight = (start_time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        for day in pd.date_range(start=first_midnight, end=end_time, freq='D'):
            plt.axvline(x=day, color='black', linestyle='-', lw=1)


        #GROWTH RATES
        #helper functions for plotting
        def plot_line(t_fit, d_fit, gr, color, linestyle='solid',zorder=10, alpha=1.0):
            plt.plot(t_fit, d_fit, color=color, lw=2, ls=linestyle, zorder=zorder, alpha=alpha)
            mid_idx = len(t_fit) // 2
            # XXX numerot jätetty pois plotista
            plt.annotate(f'{gr:.2f}', (t_fit[mid_idx], d_fit[mid_idx]),
                        textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7)
        def to_utc(times):
            return [dt.replace(tzinfo=None) for dt in num2date(times)]

        all_diams_MF, all_times_MC, all_times_AT = [], [], []

        #all lines
        if True: # XXX Mun oikopolku
            if result_config['plot_all_lines']:
                method_config = {
                    'MF': (MF_gr_points.values(), 'black', all_diams_MF, None),
                    # XXX poistettu muut kuin MF
                    'MC': (MC_gr_points.values(), 'white', all_times_MC, incomplete_MC),
                    'AT': (AT_gr_points.values(), 'green', all_times_AT, incomplete_AT)
                }

                for method, (lines, color, collector, incomplete_points) in method_config.items():
                    for line in lines:
                        t, d = zip(*line['points'])
                        t_fit, d_fit = zip(*line['fitted points'])
                        gr = line['growth rate'] #nm/h
                        # bp()
                        if method == 'MF':
                            # XXX kommentoi jos vain pisteet, ei viivoja
                            alpha = 0.1 if args.ensemble > 1 else 1
                            plot_line(t_fit, d_fit, gr, color, alpha=alpha)
                            collector.extend(d) #save for plotting points
                        else: #MC or AT
                            #check for lines at the edge of data period
                            incomplete_times = [pt[0] for pt in incomplete_points]
                            linestyle = 'dashed' if any(t in incomplete_times for t in to_utc(t_fit)) else 'solid'
                            plot_line(t_fit, d_fit, gr, color, linestyle)
                            collector.extend(to_utc(t))
            else:
                # if result_config['plot_all_events']: #all events
                #     events = all_events
                #     print('Plotting all events...')
                if result_config['plot_final_events']: #final events
                    events = final_events
                    print('Plotting final events...')

                    # if any([result_config['plot_all_events'],result_config['plot_final_events']]):
                if result_config['plot_final_events']:

                    if not events: #handle no events
                        mid_ts = len(df_plot.index)//2
                        mid_dp = len(df_plot.columns)//2
                        ax.text(df_plot.index[mid_ts],df_plot.columns[mid_dp],'No events found!', ha='center', va='center',size=8)

                    incomplete_config = {
                        'MC': ('white', all_times_MC, incomplete_MC),
                        'AT': ('green', all_times_AT, incomplete_AT)
                    }

                    for event in events.values():
                        for line in event['lines']:
                            t, d = zip(*line['points'])
                            t_fit, d_fit = zip(*line['fitted points'])
                            gr = line['growth rate'] #nm/h

                            #MODE FITTING
                            if line['method'] == 'MF':
                                plot_line(t_fit, d_fit, gr, 'black', zorder=5,alpha=1.00)
                                all_diams_MF.extend(d)

                            #MAXIMUM CONCENTRATION & APPEARANCE TIME
                            elif line['method'] in incomplete_config:
                                color, collector, incomplete_points = incomplete_config[line['method']]
                                incomplete_times = [pt[0] for pt in incomplete_points]
                                linestyle = 'dashed' if any(t in incomplete_times for t in to_utc(t_fit)) else 'solid'
                                plot_line(t_fit, d_fit, gr, color, linestyle)
                                collector.extend(to_utc(t))

            #event information (white box)
            if result_config['plot_event_info'] and not result_config['plot_final_events']:
                print('ERROR: Cannot plot event information as no events are plotted at the moment!')
                raise SystemExit
            elif result_config['plot_event_info']:
                for event_label in events:
                    info = events[event_label]

                    #skip if only one line
                    if info['min growth rate'] == info['max growth rate']:
                        continue

                    #text formatting
                    sign = ' ' if info['max growth rate'] < 0 else ''
                    text = f"{info['avg growth rate']:.2f}nm/h ({info['min growth rate']:.2f}–{sign}{info['max growth rate']:.2f})"

                    ax.text(
                        x=info['mid location'][0],  # position on x-axis
                        y=info['mid location'][1],  # position on y-axis
                        s=text,
                        fontsize=7,
                        bbox=dict(edgecolor="none", facecolor="white", alpha=0.8, pad=1),
                        zorder=20,
                        ha='center',
                        va='center'
                    )

            #filter points not in a line
            if not result_config['plot_all_points']:
                df_MF_peaks = df_MF_peaks[df_MF_peaks['peak_diameter'].isin(all_diams_MF)]
                df_MC = df_MC[df_MC['timestamp'].isin(all_times_MC)]
                df_AT = df_AT[df_AT['timestamp'].isin(all_times_AT)]

            #plot points
            # XXX kommentoi jos pisteitä ei tarvita
            plt.plot(df_MF_peaks.index,df_MF_peaks['peak_diameter'],'.', alpha=0.8, color='black', mec='black', mew=0.4, ms=6, label='mode fitting')

            plt.plot(df_MC['timestamp'], df_MC['peak_diameter'], '.', alpha=0.8, color='white',mec='black',mew=0.4, ms=6,label='maximum concentration')
            plt.plot(df_AT['timestamp'], df_AT['diameter'], '.', alpha=0.8, color='green',mec='black',mew=0.4, ms=6,label='appearance time')
            if result_config['plot_DT']:
                plt.plot(df_DT['timestamp'], df_DT['diameter'], '.', alpha=0.8, color='blue',mec='black',mew=0.4, ms=6,label='disappearance time',zorder=0)


        #adjustments to plot
        plt.legend(fontsize=9,fancybox=False,framealpha=0.9)
        for legend_handle in ax.get_legend().legend_handles: #change marker edges in the legend to be black
            legend_handle.set_markeredgewidth(0.5)
            legend_handle.set_markeredgecolor("black")

        plt.xlim(df_plot.index[0],df_plot.index[-1])
        plt.ylim(df_data.columns[0],df_data.columns[-1])
        plt.ylabel("diameter (nm)",fontsize=14) #add y-axis label
        plt.xlabel("time",fontsize=14) #add y-axis label
        ax.set_title(f'growth rate unit: [nm/h]', loc='right', fontsize=8)
        # plt.title(f"file name: {file_name}, fitting method: {args.fitting}")



    # PRINTING #
    # if result_config['print_final_event_info']:
    #     print('\n'+'*'*70)
    #     print(f'Found {len(final_events)} growth events:')
    #
    #     for i, event in enumerate(final_events.values(),start=1):
    #         print(f'\n*Event{i}*')
    #
    #         for line in event['lines']:
    #             start_point = line['fitted points'][0]
    #             end_point = line['fitted points'][-1]
    #             gr = line['growth rate']
    #             method = line['method']
    #
    #             #change time from days to dates
    #             start = (num2date(start_point[0]).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M'),round(start_point[1],2))
    #             end = (num2date(end_point[0]).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M'),round(end_point[1],2))
    #
    #             print(f"{start} → {end} | {gr:.2f}nm/h | {method}")
    #
    #         #estimated event growth rate
    #         info = final_events[f'event{i}']
    #
    #         print(f"Estimated event growth rate: {info['avg growth rate']:.2f} ({info['min growth rate']:.2f}-{info['max growth rate']:.2f}) nm/h")
    #         print(f"MAFE: {info['MAFE']:.3f}")
    #         if event['num of lines'] > 2:
    #             print('AFEs:')
    #             for AFE in info['respective AFEs']:
    #                 print(f'{AFE[0]}: {AFE[1]:.3f}')
    #
    # if result_config['print_ts_info']:
    #     print('\n'+'*'*70)
    #     print('Growth lines in each timestamp.')
    #
    #     for event_label,stamps in ts_info.items():
    #         print(f'\n{event_label}:')
    #
    #         for ts_label,ts in stamps.items():
    #             print(f'{ts_label}:')
    #
    #             for line in ts['lines']:
    #                 start_point = line['fitted points'][0]
    #                 end_point = line['fitted points'][-1]
    #                 gr = line['growth rate']
    #                 method = line['method']
    #
    #                 #change time from days to dates
    #                 start = (num2date(start_point[0]).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M'),round(start_point[1],2))
    #                 end = (num2date(end_point[0]).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M'),round(end_point[1],2))
    #
    #                 print(f"{start} → {end} | {gr:.2f}nm/h | {method} ")
    #
    #             if not ts['lines']:
    #                 print('No lines in this timestamp!')
    #
    #             if not any(val is None for val in [ts['avg growth rate'], ts['min growth rate'], ts['max growth rate']]):
    #                 print(f"Estimated growth rate for timestamp: {ts['avg growth rate']:.2f} ({ts['min growth rate']:.2f}-{ts['max growth rate']:.2f}) nm/h")
    #

    # SAVING #
    if result_config['save_final_event_info']:
        #change days to dates (type: str)
        # for event in final_events.values():
        #     for line in event['lines']:
        #         bp()
        #         for key in ['points','fitted points']:
        #             line[key] = [(num2date(point[0]).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S'),point[1])
        #                          for point in line[key]]

        with open(f'{file_name[0:3]}{start_date[2:4]}{start_date[5:7]}{start_date[-2:]}_final_events.json', 'w') as output_file:
            json.dump(final_events, output_file, indent=2)

    if result_config['save_ts_info']:
        #change days to dates (type: str)
        for event in ts_info.values():
            for ts in event.values():
                for line in ts['lines']:
                    line['fitted points'] = [(num2date(point[0]).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S'),point[1])
                                    for point in line['fitted points']]

        with open(f'{file_name[0:3]}{start_date[2:4]}{start_date[5:7]}{start_date[-2:]}_ts_info.json', 'w') as output_file:
            json.dump(ts_info, output_file, indent=2)

    # bp()
    # if False:
    def intp(df):
        X=df.iloc[:,:2].resample('5min').interpolate()
        Y=df.iloc[:,2].resample('5min').ffill()
        G=df.iloc[:,3].resample('5min').interpolate()
        return X.join([Y,G], how='left')

    if args.ensemble_method == 1:
        # pass
        first = True
        dfdf = None
        # bp()
        for key in MF_gr_points:

            # bp()
            # for points in MF_gr_points[key]['fitted points']:
            # bp()
            datefloats = np.array(MF_gr_points[key]['fitted points']).T[0]
            linediams = np.array(MF_gr_points[key]['fitted points']).T[1]
            datetimes = [num2date(i) for i in datefloats]
            gr = MF_gr_points[key]['growth rate']
            if first:
                dfdf =intp( pd.DataFrame(index=datetimes, data={
                    'date_floats':datefloats,
                    'linediams':linediams,
                    'line':[key]*len(datefloats),
                    'gr':[gr]*len(datefloats)
                    }))
                # bp()

                first = False
            else:
                dfdf_0 = pd.DataFrame(index=datetimes, data={
                    'date_floats':datefloats,
                    'linediams':linediams,
                    'line':[key]*len(datefloats),
                    'gr':[gr]*len(datefloats)
                    })
                dfdf =pd.concat([dfdf, intp(dfdf_0)])

        # bp()
        return ts_info,df_MF_peaks,dfdf,MF_gr_points,df_plot
    else:
        return ts_info,df_MF_peaks

if __name__ == "__main__":


    ## Add arguments
    parser.add_argument('--file', '-f', default=default_file_name, type=str, help='input netCDF file name')
    parser.add_argument('--start','-s', default=default_start_date, type=str, help='Start date: YYYY-MM-DD HH:MM:SS (time of day is optional)')
    parser.add_argument('--end','-e' , default=default_end_date, type=str, help='End date: YYYY-MM-DD HH:MM:SS (time of day is optional)')
    parser.add_argument('--figname','-n' , default='figure', type=str, help='name for plot file (if saved)')
    parser.add_argument('--samples', default=10000, type=int, help='n samples')
    parser.add_argument('--ensemble',default=1, type=int, help='how many ensembles to use. 1=no ensemble')
    parser.add_argument('--ensemble_method',default=0, type=int, help='0 = find average points before growth identification, 1 = identify all ensemble members')
    parser.add_argument('--close_figs',action='store_true', help='close figures (only save them)')
    parser.add_argument('--no_basefig',action='store_true', help="don't plot heatmap, only lines ant points")
    parser.add_argument('--auto_limits',action='store_true', help="take heatmap vmin and vmax from min and max of data")
    parser.add_argument('--plot_all_points', action='store_true',help = 'plots all points for all methods XXX')
    parser.add_argument('--plot_all_lines', action='store_true', help = 'plots all lines XXX')
    # parser.add_argument('--plot_all_events', action='store_true' help = 'plots all events')
    parser.add_argument('--plot_final_events', action='store_true', help = 'plots the final results XXX')
    # parser.add_argument('--print_final_event_info', action='store_true',help = 'prints info about each final event on screen')
    parser.add_argument('--save_final_event_info', action='store_true',help = 'saves info about each final event in a file  XXX')
    parser.add_argument('--print_ts_info', action='store_true',help = 'prints info about each timestamp of an event')
    parser.add_argument('--save_ts_info', action='store_true',help = 'saves info about timestamps in each event in a file')
    parser.add_argument('--plot_event_info', action='store_true',help = 'plots estimated growth rate and range for each event (white box) XXX')
    # parser.add_argument('--plot_DT', action='store_true',help = 'plots disappearance times (NOT FINISHED!)')
    parser.add_argument('--duration','-d', default=0, type=int,help = 'Lenght of days to consider, starting from --start. if zero, --end is used (default 0)')

    ## Parse arguments
    args = parser.parse_args()
    for key in ['plot_all_points','plot_all_lines','plot_final_events','save_final_event_info',
                'print_ts_info','save_ts_info','plot_event_info']:
        exec(f"result_config['{key}'] = True if args.{key} else result_config['{key}']")

    file_name = args.file
    start_date = args.start
    end_date = args.end
    duration = args.duration

    if duration>0:
        end_date = (pd.to_datetime(start_date)+pd.Timedelta(f'{duration-1}D')).strftime('%Y-%m-%d')
    # if args.fitting == 'old':
    #     new_fitting_functions = False
    # elif args.fitting == 'new':
    #     new_fitting_functions = True
    if args.no_basefig:
        plot_config['vmin'] = 1e20
        plot_config['vmax'] = 5e20
        plot_config['cmap'] = 'Greys'


    ts_info,df_MF_peaks = main()

    import print_event
    fout = open(f'{file_name[0:3]}{start_date[2:4]}{start_date[5:7]}{start_date[-2:]}_event_Summary.txt', 'w')
    # for j in range(1,len(ts_info)+1):
    #     print('event:', j, print_event.print_event(ts_info, j))
    print('# timestamp           MF  mean_diam  gr          AT  mean_diam  gr          MC  mean_diam  gr          ')
    fout.write('# timestamp           MF  mean_diam  gr          AT  mean_diam  gr          MC  mean_diam  gr          \n')
    for j in range(1,len(ts_info)+1):
        Dic = print_event.print_event(ts_info,j)
        # print()
        print(f'# event{j}')
        fout.write(f'# event{j}'+'\n')
        for k in Dic.keys():
            print(k+'   '+Dic[k])
            fout.write(k+'   '+Dic[k]+'\n')
    fout.close()
    #
    # # bp()
    # if args.ensemble==1 or args.ensemble_method==0:
    #     ts_info,df_MF_peaks = main()
    # elif args.ensemble>1 and args.ensemble_method==1:
    #     for ens in range(args.ensemble):
    #         # result_config['ens_number'] = ens
    #         if ens==0:
    #             ts_info_0,df_MF_peaks_0,dfdf_0, _,plotdata = main(ens)
    #             # retu = main(ens)
    #             # bp()
    #         # os.system('rm *.json')
    #         else:
    #             ts_info_0,df_MF_peaks_0,dfdf, _,_ = main(ens)
    #             # df_MF_peaks_0 = pd.concat([df_MF_peaks_0,df_MF_peaks]).sort_index()
    #             dfdf_0 = pd.concat([dfdf_0,dfdf]).sort_index()
    #             # linedf_0 = pd.concat([linedf_0,linedf]).sort_index()
    #         # bp()
    #     plt.figure()
    #     plt.scatter(dfdf_0.index, dfdf_0['linediams'],s=1)
    #     plt.gca().set_yscale('log')
    #     plt.gca().set_ylim(3,1000)
    #     plt.gca().set_xlim(0,1)
    #
    #
    #
    #
    #     time = plotdata.index
    #     dia = plotdata.columns
    #     plt.pcolormesh(time, dia, plotdata.iloc[:,:].T, norm='log')
    #     plt.gca().set_yscale('log')
    #
    #     from sklearn.cluster import AgglomerativeClustering
    #     def clus(dist,df):
    #         clusteringDist = AgglomerativeClustering(distance_threshold=dist, n_clusters=None, linkage='ward' ).fit(df.iloc[:,:2])
    #         n_clust = clusteringDist.labels_.max()+1
    #         return clusteringDist, n_clust
    #
    #     def scatterplot(df):
    #         # plt.figure()
    #         for i_c in range(n_clust):
    #             x,y = df.index[clusteringDist.labels_==i_c], df['linediams'].iloc[clusteringDist.labels_==i_c]
    #             plt.scatter(x,y,s=3, alpha=1)
    #
    #     dflog = dfdf_0.copy()
    #     dflog['linediams']=dflog['linediams'].apply(lambda x: np.log10(x))
    #     dist=0.7
    #     clusteringDist, n_clust = clus(dist,dflog)
    #
    #     for i in range(n_clust):
    #         msk = clusteringDist.labels_==i
    #         time = dfdf_0.loc[msk].index.mean()
    #         gr = dfdf_0.iloc[msk,3].mean()
    #         dia = dfdf_0.iloc[msk,1].mean()
    #         if np.logical_and(time>pd.to_datetime(start_date, utc=True),time<pd.to_datetime(end_date,utc=True)+pd.Timedelta('1D')):
    #             plt.scatter(time, dia, c='k', s=4)
    #             plt.text(time,dia,f'{gr:.2f}', fontsize=15)
    #         # x,y = dfdf_0.index[clusteringDist.labels_==i], dfdf_0['linediams'].iloc[clusteringDist.labels_==i]
    #         # plt.plot(x,y, lw=1)
    #
    #     scatterplot(dfdf_0)
    #
    #     plt.ion()
    #     plt.show()





















        # clusteringDist = AgglomerativeClustering(distance_threshold=10, n_clusters=None, linkage='ward' ).fit(dfdf_0.iloc[:,:2])
        # n_clust = clusteringDist.labels_.max()+1
        #
        #
        # dist=300
        # clusteringDist, n_clust = clus(dist,dfdf_0)
        # dist=0.7
        # clusteringDist, n_clust = clus(dist,dflog)
        # ppp(dflog)
        # import piecewise_regression
    # plt.figure()
    # plt.ylim(3,1000);plt.xlim(-0.01,0.99); plt.gca().set_yscale('log')
        #
        #     times = [(x[i]-x[0]).seconds/3600 for i in range(len(x))]
        #     Y = y.values
        #     pw_fit = piecewise_regression.Fit(times, Y,n_breakpoints=1)
        #     print(f'Cluster {i_c}')
        #     if pw_fit.get_results()['converged']:
        #         print(f'alpha1: {pw_fit.get_results()["estimates"]['alpha1']['estimate']}')
        #         pw_fit.plot_fit(color="red", linewidth=2)
        #         pw_fit.plot_data(s=4)
        #     else:
        #         pw_fit = piecewise_regression.Fit(times, Y,n_breakpoints=2)
        #         if pw_fit.get_results()['converged']:
        #             print(f'alpha1: {pw_fit.get_results()["estimates"]['alpha1']['estimate']}')
        #             print(f'alpha2: {pw_fit.get_results()["estimates"]['alpha2']['estimate']}')
        #             pw_fit.plot_fit(color="red", linewidth=2)
        #             pw_fit.plot_data(s=4)
        #         else:
        #             pw_fit = piecewise_regression.Fit(times, Y,n_breakpoints=3)
        #             if pw_fit.get_results()['converged']:
        #                 print(f'alpha1: {pw_fit.get_results()["estimates"]['alpha1']['estimate']}')
        #                 print(f'alpha2: {pw_fit.get_results()["estimates"]['alpha2']['estimate']}')
        #                 print(f'alpha3: {pw_fit.get_results()["estimates"]['alpha3']['estimate']}')
        #                 pw_fit.plot_fit(color="red", linewidth=2)
        #                 pw_fit.plot_data(s=4)

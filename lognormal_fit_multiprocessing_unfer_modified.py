# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:00:35 2023

@author: unfer
"""
import math
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed

######################################################################################################
class Main():
    def __init__(self, dataframe, year, month):
        self.v = False
        print("init Main")
    
        self.df1 = pd.DataFrame(dataframe)
        self.df1.columns = pd.to_numeric(self.df1.columns)
        self.pnsd_diam = self.df1.columns.values
        self.pnsd_logdiam = np.log(self.pnsd_diam)

        print(month,year)
     
        #############################################################################################################
        #############################################################################################################

        final_list = []
        start_time = datetime.now()
        size = self.df1.shape[0]
        
        print("no. of tasks, ", size)
        
        results = Parallel(n_jobs=-2, backend='multiprocessing', verbose=3)(
            delayed(self.lognormal_fit)(i) for i in range(0,size))
        
        for i in results:
            final_list.append(i[1])
        fl, df = self.composeDF(final_list)

        ########## ADDITION ###########
        def filter_m1(dataframe):     ###filters rows with values of A lower than variable min_A
            min_A12 = 1000 ### limit for m1_A & m2_A
            #min_A3 = 280 ### limit for m3_A
            #min_A4 = 190 ### limit for m4_A

            dataframe.loc[dataframe["m1_A"]<min_A12,"m1_A"] = np.nan
            dataframe.loc[dataframe["m1_A"].isna(),"m1_d"] = np.nan
            #dataframe.loc[dataframe["m2_A"]<min_A12,"m2_A"] = np.nan
            #dataframe.loc[dataframe["m2_A"].isna(),"m2_d"] = np.nan
            #dataframe.loc[dataframe["m3_A"]<min_A3,"m3_A"] = np.nan
            #dataframe.loc[dataframe["m3_A"].isna(),"m3_d"] = np.nan
            #dataframe.loc[dataframe["m4_A"]<min_A4,"m4_A"] = np.nan
            #dataframe.loc[dataframe["m4_A"].isna(),"m4_d"] = np.nan
        filter_m1(df)
        ###############################
        
        print(df)
        df.to_csv('./output_modefit_'+str(year)+'_'+str(month)+'.csv', sep=',', header=True, index=True, na_rep='nan')
 
        print('')
        print('********* FINISHED *********')
        print("Total time taken:", datetime.now() - start_time)
        print('****************************')
        print('')

    def composeDF(self,final_list):
        final_dataframe = pd.DataFrame(final_list)
        final_dataframe.columns = ['i','Timestamp (UTC)','m1_A','m1_d','m1_s','m2_A','m2_d','m2_s','m3_A','m3_d','m3_s','m4_A','m4_d','m4_s','R2','PE', 'flag']

        final_dataframe = final_dataframe.sort_values(by='Timestamp (UTC)', ascending=True)
        final_dataframe.index = final_dataframe['Timestamp (UTC)']
        final_dataframe = final_dataframe.drop(['i','Timestamp (UTC)'], axis=1)

        return final_list, final_dataframe
    
    def converter_data(self,data):
        try:
            data_formatada = datetime.strptime(data, "%Y-%m-%d %H:%M:%S.%f%z").strftime("%Y-%m-%d %H:%M:%S")
            return data_formatada
        except ValueError:
            data_formatada = datetime.strptime(data, "%Y-%m-%d %H:%M:%S%z").strftime("%Y-%m-%d %H:%M:%S")
            return data_formatada
    
    def lognormal_fit(self, i):
        self.idx = self.df1.index[i].strftime('%Y-%m-%d %H:%M:%S')
        self.pnsd_row = self.df1.iloc[i]
        self.pnsd_row_np = self.pnsd_row.to_numpy()
     

        dominant_mode1,peak_d1, A_fit1, diam_fit1, sigma_fit1, dominant_mode2,peak_d2,A_fit2, diam_fit2, sigma_fit2, dominant_mode3,peak_h3,A_fit3, diam_fit3, sigma_fit3, dominant_mode4, A_fit4, diam_fit4, sigma_fit4, r2, percent_error, flag = self.find_modes(i)
        #plot_fit(df1, i, A_fit1, diam_fit1, sigma_fit1, A_fit2, diam_fit2, sigma_fit2, A_fit3, diam_fit3, sigma_fit3,
                 #A_fit4, diam_fit4, sigma_fit4, r2, percent_error, '1st fit', 'No')

        dominant_mode1, A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, dominant_mode2, A_fit2_opt, diam_fit2_opt, sigma_fit2_opt, A_fit3_opt, diam_fit3_opt, sigma_fit3_opt, A_fit4_opt, diam_fit4_opt, sigma_fit4_opt, r2_opt, pe_opt, dN_total_fit = self.Optimizing_4modes(
            dominant_mode1, peak_d1, A_fit1, diam_fit1, sigma_fit1, dominant_mode2, peak_d2, A_fit2, diam_fit2,
            sigma_fit2, dominant_mode3, peak_h3, A_fit3, diam_fit3, sigma_fit3, dominant_mode4, A_fit4, diam_fit4,
            sigma_fit4, flag)
        #plot_fit(df1, i, A_fit1_opt_4m, diam_fit1_opt_4m, sigma_fit1_opt_4m, A_fit2_opt_4m, diam_fit2_opt_4m,
                 #sigma_fit2_opt_4m, A_fit3_opt_4m, diam_fit3_opt_4m, sigma_fit3_opt_4m, A_fit4_opt_4m, diam_fit4_opt_4m,
                 #sigma_fit4_opt_4m, r2_opt_4m, pe_4m, 'Optimization', 'No')

        if r2 >= r2_opt:
            if r2 == r2_opt and percent_error >= pe_opt:
                A_fit1_final, diam_fit1_final, sigma_fit1_final, A_fit2_final, diam_fit2_final, sigma_fit2_final, A_fit3_final, diam_fit3_final, sigma_fit3_final, A_fit4_final, diam_fit4_final, sigma_fit4_final, r2_final, pe_final = A_fit4_opt, diam_fit4_opt, sigma_fit4_opt, A_fit3_opt, diam_fit3_opt, sigma_fit3_opt, A_fit2_opt, diam_fit2_opt, sigma_fit2_opt, A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, r2_opt, pe_opt
                save = 'yes'
                fit = 'Optimization'
                #plot_fit(self.df1, i, A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, A_fit2_opt, diam_fit2_opt,
                         #sigma_fit2_opt, A_fit3_opt, diam_fit3_opt, sigma_fit3_opt, A_fit4_opt,
                         #diam_fit4_opt, sigma_fit4_opt, r2_opt, pe_opt, fit, save)
                if self.v:
                    print('Optimization is better!')
            else:
                A_fit1_final, diam_fit1_final, sigma_fit1_final, A_fit2_final, diam_fit2_final, sigma_fit2_final, A_fit3_final, diam_fit3_final, sigma_fit3_final, A_fit4_final, diam_fit4_final, sigma_fit4_final, r2_final, pe_final = A_fit1, diam_fit1, sigma_fit1, A_fit2, diam_fit2, sigma_fit2, A_fit3, diam_fit3, sigma_fit3, A_fit4, diam_fit4, sigma_fit4, r2, percent_error
                save = 'yes'
                fit = '1st fit'
                #plot_fit(self.df1, i, A_fit1, diam_fit1, sigma_fit1, A_fit2, diam_fit2, sigma_fit2, A_fit3, diam_fit3,
                         #sigma_fit3, A_fit4, diam_fit4, sigma_fit4, r2, percent_error, fit, save)
                if self.v:
                    print('1st guess is better!')

        else:
            A_fit1_final, diam_fit1_final, sigma_fit1_final, A_fit2_final, diam_fit2_final, sigma_fit2_final, A_fit3_final, diam_fit3_final, sigma_fit3_final, A_fit4_final, diam_fit4_final, sigma_fit4_final, r2_final, pe_final = A_fit4_opt, diam_fit4_opt, sigma_fit4_opt, A_fit3_opt, diam_fit3_opt, sigma_fit3_opt, A_fit2_opt, diam_fit2_opt, sigma_fit2_opt, A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, r2_opt, pe_opt
            save = 'yes'
            fit = 'Optimization'
            #plot_fit(self.df1, i, A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, A_fit2_opt, diam_fit2_opt,
                     #sigma_fit2_opt, A_fit3_opt, diam_fit3_opt, sigma_fit3_opt, A_fit4_opt,
                     #diam_fit4_opt, sigma_fit4_opt, r2_opt, pe_opt, fit, save)
            if self.v:
                print('Optimization is better!')

        return (i, [i, self.idx, round(A_fit1_final,2), diam_fit1_final, round(sigma_fit1_final,3), round(A_fit2_final,2), diam_fit2_final, round(sigma_fit2_final,3),
                                       round(A_fit3_final,2), diam_fit3_final, round(sigma_fit3_final,3), round(A_fit4_final,2), diam_fit4_final, round(sigma_fit4_final,3), r2_final, pe_final, flag])

    def find_diam_idx(self, range_min, range_max, diam_values):
        min_idx = np.argmax(diam_values >= range_min)
        max_idx = np.argmin(diam_values <= range_max)
        return min_idx, max_idx + 1

    def find_modes(self, i):
        diam_values = self.pnsd_diam
        flag = np.nan

        dict_modes = {1: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                      2: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                      3: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                      4: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}

        ## MODE 1 ##  ### d = 3-19nm
        series = self.pnsd_row_np[0:10] ### [0:1] -> [5:10] assuming there was a mistake and the original indices should be [0:10]
        peak_h1 = series.max()
        peak_d1 = np.argmax(series)
        A_range1 = np.linspace(0 * peak_h1, 1.2 * peak_h1, 50)
        sigma_range1 = np.linspace(1.1, 1.8, 30)
        x_min = 3 ### 10 -> 3
        x_max = 19
        mode = 'Nucleation'
        factor1 = 30
        dict_modes[1] = [peak_h1, A_range1, sigma_range1, x_min, x_max, mode, factor1]

        ## MODE 2 ##
        series = self.pnsd_row_np[11:17] ### changed indices [12:27] -> [11:17]
        peak_h2 = series.max()
        peak_d2 = self.pnsd_row[13:25].idxmax() ### changed indices [20:49] -> [13:25]
        A_range2 = np.linspace(0 * peak_h2, 1.2 * peak_h2, 50)
        sigma_range2 = np.linspace(1.1, 1.8, 30)
        x_min = 20
        x_max = 49
        mode = 'Aitken1'
        factor2 = 90
        dict_modes[2] = [peak_h2, A_range2, sigma_range2, x_min, x_max, mode, factor2]

        ## MODE 3 ##
        series = self.pnsd_row_np[17:21] ### changed indices [28:39] -> [17:21]
        peak_h3 = series.max()
        peak_d3 = self.pnsd_row[25:36].idxmax() ### changed indices [50:100] -> [25:36]
        A_range3 = np.linspace(0 * peak_h3, 1.2 * peak_h3, 50)
        sigma_range3 = np.linspace(1.1, 1.8, 30)
        x_min = 50
        x_max = 94  
        mode = 'Aitken2'
        factor3 = 150
        dict_modes[3] = [peak_h3, A_range3, sigma_range3, x_min, x_max, mode, factor3]

        ## MODE 4 ##
        series = self.pnsd_row_np[22:30] ### changed indices [40:63] -> [22:30]
        peak_h4 = series.max()
        #peak_d4 = np.argmax(series)
        A_range4 = np.linspace(0 * peak_h4, 1.2* peak_h4, 50)
        sigma_range4 = np.linspace(1.1, 1.8, 30)
        x_min = 100
        x_max = 400
        mode = 'Accumulation'
        factor4 = 300
        dict_modes[4] = [peak_h4, A_range4, sigma_range4, x_min, x_max, mode, factor4]
       
        if self.v:
            print(peak_d2, peak_d3, peak_h2, peak_h3)


        ## Check whether Aitken 2 is too close to Aitken 1; if so, it becomes only Aitken 1
        if peak_d2 > 37 and peak_d3 < 55 and peak_h2 > peak_h3:  
            peak_h3 = 0  # 50 nm to 94 nm
            #peak_d3 = self.pnsd_diam[28:39].idxmax()
            A_range3 = np.linspace(0 * peak_h3, 1.2 * peak_h3, 50)
            sigma_range3 = np.linspace(1.1, 1.8, 30)
            x_min = 50
            x_max = 94
            mode = 'Aitken2'
            factor3 = 150
            dict_modes[3] = [peak_h3, A_range3, sigma_range3, x_min, x_max, mode, factor3]

        dict_modes_sorted = dict(sorted(dict_modes.items(), key=lambda item: self.get_peak_h(item[1]), reverse=True))
        valores_ordenados = list(dict_modes_sorted.values())
        modes_in_order = [value for values in valores_ordenados for value in values]

        A_fit1, sigma_fit1, diam_fit1, dominant_mode1 = self.fit_one_mode(i, modes_in_order[3], modes_in_order[4],
                                                                     modes_in_order[1], modes_in_order[2], diam_values,
                                                                     modes_in_order[5], modes_in_order[6])

        A_fit2, sigma_fit2, diam_fit2, dominant_mode2 = self.fit_two_modes(A_fit1, sigma_fit1, diam_fit1,
                                                                      i, modes_in_order[10], modes_in_order[11],
                                                                      modes_in_order[8], modes_in_order[9], diam_values,
                                                                      modes_in_order[12], modes_in_order[13])

        A_fit3, sigma_fit3, diam_fit3, dominant_mode3 = self.fit_three_modes(A_fit1, sigma_fit1, diam_fit1, A_fit2,
                                                                        sigma_fit2, diam_fit2, i, modes_in_order[17], modes_in_order[18],
                                                                        modes_in_order[15], modes_in_order[16],
                                                                        diam_values, modes_in_order[19], modes_in_order[20])

        A_fit4, sigma_fit4, diam_fit4, dominant_mode4 = self.fit_four_modes(A_fit1, sigma_fit1, diam_fit1, A_fit2,
                                                                       sigma_fit2, diam_fit2, A_fit3, sigma_fit3,
                                                                       diam_fit3, i, modes_in_order[24], modes_in_order[25],
                                                                       modes_in_order[22], modes_in_order[23],
                                                                       diam_values, modes_in_order[26], modes_in_order[27])

        ### Putting in order of diameters for the output ###
        dict_modes_fit = {1: [np.nan, np.nan, np.nan, np.nan],
                          2: [np.nan, np.nan, np.nan, np.nan],
                          3: [np.nan, np.nan, np.nan, np.nan],
                          4: [np.nan, np.nan, np.nan, np.nan]}

        dict_modes_fit[1] = [diam_fit1, A_fit1, sigma_fit1, dominant_mode1]
        dict_modes_fit[2] = [diam_fit2, A_fit2, sigma_fit2, dominant_mode2]
        dict_modes_fit[3] = [diam_fit3, A_fit3, sigma_fit3, dominant_mode3]
        dict_modes_fit[4] = [diam_fit4, A_fit4, sigma_fit4, dominant_mode4]

        dict_modes_sorted_fit = dict(sorted(dict_modes_fit.items(), key=lambda item: self.get_peak_h(item[1]), reverse=False))
        valores_ordenados_fit = list(dict_modes_sorted_fit.values())
        modes_in_order_fit = [value for values in valores_ordenados_fit for value in values]

        A_fit1, diam_fit1, sigma_fit1, dominant_mode1 = modes_in_order_fit[1], modes_in_order_fit[0], \
        modes_in_order_fit[2], modes_in_order_fit[3]
        A_fit2, diam_fit2, sigma_fit2, dominant_mode2 = modes_in_order_fit[5], modes_in_order_fit[4], \
        modes_in_order_fit[6], modes_in_order_fit[7]
        A_fit3, diam_fit3, sigma_fit3, dominant_mode3 = modes_in_order_fit[9], modes_in_order_fit[8], \
        modes_in_order_fit[10], modes_in_order_fit[11]
        A_fit4, diam_fit4, sigma_fit4, dominant_mode4 = modes_in_order_fit[13], modes_in_order_fit[12], \
        modes_in_order_fit[14], modes_in_order_fit[15]

        if abs(np.log10(diam_fit1) - np.log10(diam_fit2)) <= 0.20 and diam_fit1 > 18:
            if self.v:
                print('Adjacent mode too close! Fitting it again...')
            peak_h1 = 0  # 10 nm to 19 nm
            A_range1 = np.linspace(0 * peak_h1, 1.2 * peak_h1, 50)
            sigma_range1 = np.linspace(1.1, 1.8, 30)
            x_min = 3 ### 10 -> 3
            x_max = 19
            mode = 'Nucleation'

            A_fit1, sigma_fit1, diam_fit1, dominant_mode1 = self.fit_four_modes(A_fit2, sigma_fit2, diam_fit2, A_fit3,
                                                                           sigma_fit3, diam_fit3, A_fit4, sigma_fit4,
                                                                           diam_fit4,
                                                                           i, x_min, x_max, A_range1, sigma_range1,
                                                                           diam_values, mode, factor1)

        # !!!
        if diam_fit2 < 25 and diam_fit4 < 110:  
            if self.v:
                print('mode3')
            flag = 2
            ## MODE 3 ##
            peak_h3 = self.pnsd_row_np[17:21].max()  ### [28:39] -> [17:21]
            #peak_d3 = self.pnsd_diam[17:21].idxmax() ### [28:39] -> [17:21]
            A_range3 = np.linspace(0 * peak_h3, 1.2 * peak_h3, 50)
            sigma_range3 = np.linspace(1.1, 1.8, 30)
            x_min = 40  # antes era 30
            x_max = 94
            mode = 'Aitken2'
            dict_modes[3] = [peak_h3, A_range3, sigma_range3, x_min, x_max, mode]
            factor21 = 100

            A_fit3, sigma_fit3, diam_fit3, dominant_mode3 = self.fit_four_modes(A_fit2, sigma_fit2, diam_fit2, A_fit1,
                                                                           sigma_fit1, diam_fit1, A_fit4,
                                                                           sigma_fit4, diam_fit4,
                                                                           i, x_min, x_max, A_range3,
                                                                           sigma_range3, diam_values, mode, factor21)

        if diam_fit3 > 80 and self.pnsd_row_np[20:21].max() < self.pnsd_row_np[22:23].max(): ### [36:39] -> [20:21] & [40:43] -> [22:23]
            if self.v:
                print('here1')
            A_fit3 = 0


        if diam_fit3 > 85 and diam_fit4 < 130 and A_fit3!=0:  
            if self.v:
                print('mode4')
            ## MODE 3 ##
            peak_h3 = self.pnsd_row_np[17:21].max() ### [28:39] -> [17:21]
            #peak_d3 = df1.iloc[i, 17:21].idxmax() ### [i, 28:39] -> [i, 17:21]
            A_range3 = np.linspace(0 * peak_h3, 1.2 * peak_h3, 50)
            sigma_range3 = np.linspace(1.1, 1.8, 30)
            x_min = 50
            x_max = 400
            mode = 'Aitken3'
            factor31 = 500

            A_fit4 = 0
            flag = 4

            A_fit3, sigma_fit3, diam_fit3, dominant_mode3 = self.fit_four_modes(A_fit2, sigma_fit2, diam_fit2, A_fit1,
                                                                           sigma_fit1, diam_fit1, A_fit4, sigma_fit4,
                                                                           diam_fit4, i, x_min, x_max, A_range3, sigma_range3,
                                                                           diam_values, mode, factor31)


        ### Transition
        if self.pnsd_row_np[0:10].sum() > self.pnsd_row_np[11:13].sum(): ### [0:11] -> [0:10] & [12:19] -> [11:13]
            if diam_fit2 < 22:
                if self.v:
                    print('here2')
                flag = 1

                ## MODE 1 ##
                peak_h1 = self.pnsd_row_np[0:16].max() ### [0:27] -> [0:16]
                A_range1 = np.linspace(0 * peak_h1, 1.2* peak_h1, 50)
                sigma_range1 = np.linspace(1.1, 1.8, 30)
                A_fit2 = 0
                x_min = 3 ### 10 -> 3
                x_max = 49
                factor11 = 120
                A_fit1, sigma_fit1, diam_fit1, dominant_mode1 = self.fit_four_modes(0, sigma_fit2, diam_fit2, A_fit3,
                                                                               sigma_fit3, diam_fit3, A_fit4,
                                                                               sigma_fit4, diam_fit4,
                                                                               i, x_min, x_max, A_range1,
                                                                               sigma_range1, diam_values,
                                                                               'Nucleation', factor11)
                diam_fit2 = 40



        ### R2 ###
        diam = self.pnsd_diam
        pnsd = self.pnsd_row_np
        dN_total = round(sum(pnsd), 1)

        lognf_op1 = self.calc_lognf(A_fit1, diam_fit1, sigma_fit1, self.pnsd_logdiam)
        lognf_op2 = self.calc_lognf(A_fit2, diam_fit2, sigma_fit2, self.pnsd_logdiam)
        lognf_op3 = self.calc_lognf(A_fit3, diam_fit3, sigma_fit3, self.pnsd_logdiam)
        lognf_op4 = self.calc_lognf(A_fit4, diam_fit4, sigma_fit4, self.pnsd_logdiam)

        denominador = (sum((pnsd - (lognf_op1 + lognf_op2 + lognf_op3 + lognf_op4)) ** 2))
        r2 = 1 - denominador / (np.var(pnsd) * (len(pnsd) - 1))
        percent_error = (abs(sum(lognf_op1 + lognf_op2 + lognf_op3 + lognf_op4) - dN_total) / dN_total) * 100
        if self.v:
            print("R2=" + str(round(r2, 3)), "| PE=" + str(round(percent_error, 1)) + "%")


        ### Test wether the nucleation mode really exists ###
        ### It runs without the nucleation mode and if the PE is below 10%, then it doenst exist ###
        ### Then it re-fit the Aitken 1 mode considering nucleation mode 0 ###

        if A_fit1 != 0 and A_fit2 > A_fit1 and self.pnsd_row_np[10] > self.pnsd_row_np[0]: 
            if self.v:
                print('here3')
            ## MODE 2 ##
            peak_h2 = self.pnsd_row[11:16].max()  # 20 nm to 49 nm ### [12:27] -> [11:16]
            peak_d2 = self.pnsd_row[11:16].idxmax() ### [12:27] -> [11:16]
            A_range2 = np.linspace(0 * peak_h2, 1.2 * peak_h2, 50)
            sigma_range2 = np.linspace(1.1, 1.8, 30)
            x_min = 3 ### 10 -> 3
            x_max = 49
            mode = 'Aitken1'

            factor21 = 120

            A_fit1_new = 0

            A_fit2_new, sigma_fit2_new, diam_fit2_new, dominant_mode2 = self.fit_four_modes(A_fit1_new, sigma_fit1,
                                                                                       diam_fit1, A_fit3, sigma_fit3,
                                                                                       diam_fit3, A_fit4, sigma_fit4,
                                                                                       diam_fit4,
                                                                                       i, x_min, x_max, A_range2,
                                                                                       sigma_range2, diam_values, mode, factor21)

            ### R2 ###

            lognf_op1 = self.calc_lognf(A_fit1_new, diam_fit1, sigma_fit1, self.pnsd_logdiam)
            lognf_op2 = self.calc_lognf(A_fit2_new, diam_fit2_new, sigma_fit2_new, self.pnsd_logdiam)
            lognf_op3 = self.calc_lognf(A_fit3, diam_fit3, sigma_fit3, self.pnsd_logdiam)
            lognf_op4 = self.calc_lognf(A_fit4, diam_fit4, sigma_fit4, self.pnsd_logdiam)

            denominador = (sum((pnsd - (lognf_op1 + lognf_op2 + lognf_op3 + lognf_op4)) ** 2))
            r2_new = 1 - denominador / (np.var(pnsd) * (len(pnsd) - 1))
            percent_error_new = (abs(sum(lognf_op1 + lognf_op2 + lognf_op3 + lognf_op4) - dN_total) / dN_total) * 100
            if self.v:
                print("R2=" + str(round(r2_new, 3)), "| PE=" + str(round(percent_error_new, 1)) + "%")

            if abs(percent_error_new - percent_error) > 10:
                if self.v:
                    print('here4')
                if percent_error_new < percent_error:
                    return dominant_mode1, peak_d1, A_fit1_new, diam_fit1, sigma_fit1, dominant_mode2, peak_d2, A_fit2_new, diam_fit2_new, sigma_fit2_new, dominant_mode3, peak_h3, A_fit3, diam_fit3, sigma_fit3, dominant_mode4, A_fit4, diam_fit4, sigma_fit4, round(
                        r2_new, 3), round(percent_error_new, 1), flag
                else:
                    return dominant_mode1, peak_d1, A_fit1, diam_fit1, sigma_fit1, dominant_mode2, peak_d2, A_fit2, diam_fit2, sigma_fit2, dominant_mode3, peak_h3, A_fit3, diam_fit3, sigma_fit3, dominant_mode4, A_fit4, diam_fit4, sigma_fit4, round(
                        r2, 3), round(percent_error, 1), flag
            elif r2_new < 0.9 and r2 >= 0.9:
                if self.v:
                    print('here5')
                return dominant_mode1, peak_d1, A_fit1, diam_fit1, sigma_fit1, dominant_mode2, peak_d2, A_fit2, diam_fit2, sigma_fit2, dominant_mode3, peak_h3, A_fit3, diam_fit3, sigma_fit3, dominant_mode4, A_fit4, diam_fit4, sigma_fit4, round(
                    r2, 3), round(percent_error, 1), flag
            else:
                return dominant_mode1, peak_d1, A_fit1_new, diam_fit1, sigma_fit1, dominant_mode2, peak_d2, A_fit2_new, diam_fit2_new, sigma_fit2_new, dominant_mode3, peak_h3, A_fit3, diam_fit3, sigma_fit3, dominant_mode4, A_fit4, diam_fit4, sigma_fit4, round(
                    r2_new, 3), round(percent_error_new, 1), flag
        else:
            return dominant_mode1, peak_d1, A_fit1, diam_fit1, sigma_fit1, dominant_mode2, peak_d2, A_fit2, diam_fit2, sigma_fit2, dominant_mode3, peak_h3, A_fit3, diam_fit3, sigma_fit3, dominant_mode4, A_fit4, diam_fit4, sigma_fit4, round(
                r2, 3), round(percent_error, 1), flag

    def get_peak_h(self,item):
        ### Putting in order of h peak ###
        return item[0] if not np.isnan(item[0]) else 0

    def calc_lognf(self,aF,dF,sigmaF,dlog):
       logsigma = math.log(sigmaF)
       fac = aF / (math.sqrt(2 * np.pi) * logsigma)
       return fac * np.exp(-((np.log(dF) - dlog) ** 2) / (2 * logsigma ** 2))

    def bestfit_for_mode(self,A_range,diam_range,sigma_range,mode,dlog,lognf1=0,lognf2=0,lognf3=0,sqrt_error=1e15):
        A_fit = None 
        sigma_fit = None 
        diam_fit = None 
        for A_i in A_range:
            for sigma_i in sigma_range:
                logs1 = math.log(sigma_i)
                fac = A_i / (math.sqrt(2 * np.pi) * logs1) # np.sqrt(2 * np.pi) = 2.51
                exp_den = (2 * logs1 ** 2)
                for x_i in diam_range:
                    lognf_i = fac * np.exp(-(((math.log(x_i) - dlog) ** 2) / exp_den))
                    sqrt_denominador = sum((mode - lognf_i - lognf1 - lognf2 - lognf3)** 2)

                    if sqrt_denominador < sqrt_error:
                        sqrt_error = sqrt_denominador
                        A_fit = A_i
                        sigma_fit = sigma_i
                        diam_fit = x_i

        return A_fit, sigma_fit, diam_fit, sqrt_error

    def fit_one_mode(self, i, x_min, x_max, A_range, sigma_range, diam_values, dominant_mode, factor, fit_mode=None):

        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam_range = diam_values[min_idx:max_idx]
        diam_range = np.linspace(x_min,x_max,factor)
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        series = self.pnsd_row_np
        mode = series[min_idx:max_idx]

        A_fit, sigma_fit, diam_fit, error = self.bestfit_for_mode(A_range, diam_range, sigma_range, mode, dlog)

        # print()
        r2 = 1 - error / (np.var(series) * (len(series) - 1))
        if self.v:
            print(f"i: ", i, " Mode 1: ", {dominant_mode}, "| A =", round(A_fit, 1), "| diam =", round(diam_fit, 2), "| sigma =",
              round(sigma_fit, 3), "| R2 =", round(r2, 3))
        return A_fit, sigma_fit, diam_fit, error

    def fit_two_modes(self, A_fit1, sigma_fit1, diam_fit1, i, x_min, x_max, A_range, sigma_range, diam_values, dominant_mode, factor,
                      fit_mode=None):
        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam_range = diam_values[min_idx:max_idx]
        diam_range = np.linspace(x_min,x_max,factor)
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        series = self.pnsd_row_np
        mode = series[min_idx:max_idx]

        lognf1 = self.calc_lognf(A_fit1,diam_fit1,sigma_fit1,dlog)

        A_fit, sigma_fit, diam_fit, error = self.bestfit_for_mode(A_range,diam_range,sigma_range,mode,dlog,lognf1)

        r2 = 1 - error / (np.var(series) * (len(series) - 1))
        if self.v:
            print(f"i: ", i, "Mode 2: ", dominant_mode, "| A =", round(A_fit, 1), "| diam =", round(diam_fit, 2), "| sigma =",
              round(sigma_fit, 3), "| R2 =", round(r2, 3))
        return A_fit, sigma_fit, diam_fit, error

    def fit_three_modes(self,A_fit1, sigma_fit1, diam_fit1, A_fit2, sigma_fit2, diam_fit2,
                        i, x_min, x_max, A_range, sigma_range, diam_values, dominant_mode,factor, fit_mode=None):
        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam_range = diam_values[min_idx:max_idx]
        diam_range = np.linspace(x_min,x_max,factor)    
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        series = self.pnsd_row_np
        mode = series[min_idx:max_idx]

        lognf1 = self.calc_lognf(A_fit1,diam_fit1,sigma_fit1,dlog)
        lognf2 = self.calc_lognf(A_fit2,diam_fit2,sigma_fit2,dlog)

        A_fit, sigma_fit, diam_fit, error = self.bestfit_for_mode(A_range,diam_range,sigma_range,mode,dlog,lognf1,lognf2)

        r2 = 1 - error / (np.var(series) * (len(series) - 1))
        if self.v:
            print(f"i: ", i, "Mode 3: ", {dominant_mode}, "| A =", round(A_fit, 1), "| diam =", round(diam_fit, 2), "| sigma =",
              round(sigma_fit, 3), "| R2 =", round(r2, 3))
        return A_fit, sigma_fit, diam_fit, error

    def fit_four_modes(self,A_fit1, sigma_fit1, diam_fit1, A_fit2, sigma_fit2, diam_fit2, A_fit3, sigma_fit3, diam_fit3, i,
                       x_min, x_max, A_range, sigma_range, diam_values, dominant_mode, factor, fit_mode=None):
        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam_range = diam_values[min_idx:max_idx]
        diam_range = np.linspace(x_min,x_max,factor) 
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        series = self.pnsd_row_np
        mode = series[min_idx:max_idx]

        lognf1 = self.calc_lognf(A_fit1,diam_fit1,sigma_fit1,dlog)
        lognf2 = self.calc_lognf(A_fit2,diam_fit2,sigma_fit2,dlog)
        lognf3 = self.calc_lognf(A_fit3,diam_fit3,sigma_fit3,dlog)

        A_fit, sigma_fit, diam_fit, error = self.bestfit_for_mode(A_range,diam_range,sigma_range,mode,dlog,lognf1,lognf2,lognf3)

        r2 = 1 - error / (np.var(series) * (len(series) - 1))
        if self.v:
            print(f"i: ", i, "Mode 4: ", {dominant_mode}, "| A =", round(A_fit, 1), "| diam =", round(diam_fit, 2), "| sigma =",
              round(sigma_fit, 3), "| R2 =", round(r2, 3))

        return A_fit, sigma_fit, diam_fit, error

    def Optimizing_4modes(self,dominant_mode1,peak_d1, A_fit1, diam_fit1, sigma_fit1, dominant_mode2,peak_d2,A_fit2, diam_fit2, sigma_fit2,
                          dominant_mode3, peak_h3,A_fit3, diam_fit3, sigma_fit3,dominant_mode4, A_fit4, diam_fit4, sigma_fit4, flag):

        ### Putting in reverse order of diameters to start backwards ###
        dict_modes_fit = {1: [np.nan, np.nan, np.nan, np.nan],
                          2: [np.nan, np.nan, np.nan, np.nan],
                          3: [np.nan, np.nan, np.nan, np.nan],
                          4: [np.nan, np.nan, np.nan, np.nan]}

        dict_modes_fit[1] = [diam_fit1, A_fit1, sigma_fit1, dominant_mode1]
        dict_modes_fit[2] = [diam_fit2, A_fit2, sigma_fit2, dominant_mode2]
        dict_modes_fit[3] = [diam_fit3, A_fit3, sigma_fit3, dominant_mode3]
        dict_modes_fit[4] = [diam_fit4, A_fit4, sigma_fit4, dominant_mode4]

        dict_modes_sorted_fit = dict(sorted(dict_modes_fit.items(), key=lambda item: self.get_peak_h(item[1]), reverse=True))
        valores_ordenados_fit = list(dict_modes_sorted_fit.values())
        modes_in_order_fit = [value for values in valores_ordenados_fit for value in values]

        A_fit1, diam_fit1, sigma_fit1, dominant_mode1 = modes_in_order_fit[1], modes_in_order_fit[0], modes_in_order_fit[2], modes_in_order_fit[3]  ## Accumulation
        A_fit2, diam_fit2, sigma_fit2, dominant_mode2 = modes_in_order_fit[5], modes_in_order_fit[4], modes_in_order_fit[6], modes_in_order_fit[7]  ## Aitken2
        A_fit3, diam_fit3, sigma_fit3, dominant_mode3 = modes_in_order_fit[9], modes_in_order_fit[8], modes_in_order_fit[10], modes_in_order_fit[11]  ## Aitken1
        A_fit4, diam_fit4, sigma_fit4, dominant_mode4 = modes_in_order_fit[13], modes_in_order_fit[12], modes_in_order_fit[14], modes_in_order_fit[15]  ## Nucleation

        #################################################################
        #################################### Optimizing Accumulation ####
        ## Mode 1 = Acc
        ## Modes 2, 3 and 4 remain fixed
        if self.v:
            print('')
            print('...Optimizing four modes')

        diam_values = self.pnsd_diam

        x_min = 100
        x_max = 400

        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam1 = diam_values[min_idx:max_idx]
        diam1 = np.linspace(diam_fit1*0.9,diam_fit1*1.1,5)
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        mode_op1 = self.pnsd_row_np[min_idx: max_idx]

        A1 = np.linspace(0 * A_fit1, 1.2 * A_fit1, 25)
        
        if sigma_fit1*0.9 >1.1:              
            sigma1 = np.linspace(sigma_fit1*0.9, sigma_fit1*1.1, 10)
        else:
            sigma1 = np.linspace(1.1, sigma_fit1*1.1, 10)
            

        lognf2  = self.calc_lognf(A_fit2, diam_fit2, sigma_fit2, dlog)
        lognf3  = self.calc_lognf(A_fit3, diam_fit3, sigma_fit3, dlog)
        lognf4  = self.calc_lognf(A_fit4, diam_fit4, sigma_fit4, dlog)

        A_fit1_opt, sigma_fit1_opt, diam_fit1_opt, _ = self.bestfit_for_mode(A1,diam1,sigma1,mode_op1,dlog,lognf2,lognf3,lognf4)

        #############################################################
        #################################### Optimizing Aitken 2 ####
        ## Mode 2 = Aitken 2
        ## Modes 1, 3 and 4 remain fixed

        if diam_fit2 < 50:
            if self.v:
                print('here6')
            x_min = 40
            x_max = 94
        else:
            x_min = 50
            x_max = 94

        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam2 = diam_values[min_idx:max_idx]
        diam2 = np.linspace(diam_fit2*0.9,diam_fit2*1.1,5)
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        mode_op2 = self.pnsd_row_np[min_idx:max_idx]

        A2 = np.linspace(0 * A_fit2, 1.2 * A_fit2, 25)
        
        if sigma_fit2*0.9 >1.1:              
            sigma2 = np.linspace(sigma_fit2*0.9, sigma_fit2*1.1, 10)
        else:
            sigma2 = np.linspace(1.1, sigma_fit2*1.1, 10)

        lognf_op1  = self.calc_lognf(A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, dlog)
        lognf3  = self.calc_lognf(A_fit3, diam_fit3, sigma_fit3, dlog)
        lognf4  = self.calc_lognf(A_fit4, diam_fit4, sigma_fit4, dlog)

        A_fit2_opt, sigma_fit2_opt, diam_fit2_opt, _ = self.bestfit_for_mode(A2,diam2,sigma2,mode_op2,dlog,lognf_op1,lognf3,lognf4)

        #############################################################
        #################################### Optimizing Aitken 1 ####
        ## Mode 3 = Aitken 1
        ## Modes 1, 2 and 4 remain fixed

        if A_fit4 == 0:
            if self.v:
                print('here7')
            x_min = 3 ### 10 -> 3
            x_max = 49

        if flag == 1:
            x_min = 30
            x_max = 49
        else:
            x_min = 20
            x_max = 49

        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam3 = diam_values[min_idx:max_idx]
        diam3 = np.linspace(diam_fit3*0.9,diam_fit3*1.1,5)
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        mode_op3 = self.pnsd_row_np[min_idx:max_idx]

        A3 = np.linspace(0 * A_fit3, 1.2 * A_fit3, 25)
        
        if sigma_fit3*0.9 >1.1:              
            sigma3 = np.linspace(sigma_fit3*0.9, sigma_fit3*1.1, 10)
        else:
            sigma3 = np.linspace(1.1, sigma_fit3*1.1, 10)

        lognf_op1   = self.calc_lognf(A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, dlog)
        lognf_op2   = self.calc_lognf(A_fit2_opt, diam_fit2_opt, sigma_fit2_opt, dlog)
        lognf4  = self.calc_lognf(A_fit4, diam_fit4, sigma_fit4, dlog)

        A_fit3_opt, sigma_fit3_opt, diam_fit3_opt, _ = self.bestfit_for_mode(A3,diam3,sigma3,mode_op3,dlog,lognf_op1,lognf_op2,lognf4)

        ###########################################################
        #################################### Optimizing mode 4 ####
        ## Mode 4 = Nucleation
        ## Modes 1, 2 and 3 remain fixed

        if flag == 1:
            x_min = 3 ### 10 -> 3
            x_max = 30
        else:
            x_min = 3 ### 10 -> 3
            x_max = 19

        min_idx, max_idx = self.find_diam_idx(x_min, x_max, diam_values)
        #diam4 = diam_values[min_idx:max_idx]
        
        if diam_fit4*0.9 < 3: ### 10 -> 3
            diam4 = np.linspace(3,diam_fit4*1.1,5) ### 10 -> 3
        else:          
            diam4 = np.linspace(diam_fit4*0.9,diam_fit4*1.1,5)
            
        dlog = self.pnsd_logdiam[min_idx:max_idx]
        mode_op4 = self.pnsd_row_np[min_idx: max_idx]

        A4 = np.linspace(0 * A_fit4, 1.2 * A_fit4, 25)
        
        if sigma_fit4*0.9 >1.1:              
            sigma4 = np.linspace(sigma_fit4*0.9, sigma_fit4*1.1, 10)
        else:
            sigma4 = np.linspace(1.1, sigma_fit4*1.1, 10)

        lognf_op1   = self.calc_lognf(A_fit1_opt, diam_fit1_opt, sigma_fit1_opt, dlog)
        lognf_op2   = self.calc_lognf(A_fit2_opt, diam_fit2_opt, sigma_fit2_opt, dlog)
        lognf_op3   = self.calc_lognf(A_fit3_opt, diam_fit3_opt, sigma_fit3_opt, dlog)

        A_fit4_opt, sigma_fit4_opt, diam_fit4_opt, _ = self.bestfit_for_mode(A4,diam4,sigma4,mode_op4,dlog,lognf_op1,lognf_op2,lognf_op3)

        ###### Statistics ######

        diam = self.pnsd_diam
        pnsd = self.pnsd_row_np

        lognf_op1   = self.calc_lognf(A_fit1_opt, diam_fit1_opt, sigma_fit1_opt,self.pnsd_logdiam)
        lognf_op2   = self.calc_lognf(A_fit2_opt, diam_fit2_opt, sigma_fit2_opt,self.pnsd_logdiam)
        lognf_op3   = self.calc_lognf(A_fit3_opt, diam_fit3_opt, sigma_fit3_opt,self.pnsd_logdiam)
        lognf_op4   = self.calc_lognf(A_fit4_opt, diam_fit4_opt, sigma_fit4_opt,self.pnsd_logdiam)

        denominador = (sum((pnsd - (lognf_op1 + lognf_op2 + lognf_op3 + lognf_op4)) ** 2))
        r2_opt = 1 - denominador / (np.var(pnsd) * (len(pnsd) - 1))

        dN_total = round(sum(pnsd), 1)
        dN_total_fit_opt = round(sum(lognf_op1 + lognf_op2 + lognf_op3 + lognf_op4), 1)
        percent_error_opt = (abs(sum(lognf_op1 + lognf_op2 + lognf_op3 + lognf_op4) - dN_total) / dN_total) * 100
        if self.v:
            print("A1=" + str(round(A_fit1_opt, 1)), "| diam1=" + str(round(diam_fit1_opt, 2)),
                  "| sigma1=" + str(round(sigma_fit1_opt, 3)))
            print("A2=" + str(round(A_fit2_opt, 1)), "| diam2=" + str(round(diam_fit2_opt, 2)),
                  "| sigma2=" + str(round(sigma_fit2_opt, 3)))
            print("A3=" + str(round(A_fit3_opt, 1)), "| diam3=" + str(round(diam_fit3_opt, 2)),
                  "| sigma3=" + str(round(sigma_fit3_opt, 3)))
            print("A4=" + str(round(A_fit4_opt, 1)), "| diam4=" + str(round(diam_fit4_opt, 2)),
                  "| sigma4=" + str(round(sigma_fit4_opt, 3)))

            print("R2=" + str(round(r2_opt, 3)), "| PE=" + str(round(percent_error_opt, 1)) + "%")

        return dominant_mode1, round(A_fit1_opt, 1), round(diam_fit1_opt, 2), round(sigma_fit1_opt,3), dominant_mode2, round(
            A_fit2_opt, 1), round(diam_fit2_opt, 2), round(sigma_fit2_opt, 3), round(A_fit3_opt, 1), round(
            diam_fit3_opt, 2), round(sigma_fit3_opt, 3), round(A_fit4_opt, 1), round(diam_fit4_opt, 2), round(
            sigma_fit4_opt, 3), round(r2_opt, 3), round(percent_error_opt, 2), round(dN_total_fit_opt, 1)

    #############################################################################################################
    #############################################################################################################

    def plot_fit(self, df1, i, A_fit1_final, diam_fit1_final, sigma_fit1_final, A_fit2_final, diam_fit2_final, sigma_fit2_final,
                 A_fit3_final, diam_fit3_final, sigma_fit3_final, A_fit4_final, diam_fit4_final, sigma_fit4_final, r2, percent_error, fit, save):

        def lognormal(A, d, s, x):
            calc = (A / np.sqrt(2 * np.pi) * np.log(s)) * np.exp(-(((np.log(x) - np.log(d)) ** 2) / (2 * np.log(s) ** 2)))
            return calc

        x = pd.to_numeric(df1.columns)
        df1.columns = pd.to_numeric(df1.columns)
        #y_smooth = gaussian_filter1d(df1, sigma=2)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        plt.plot(x, df1.iloc[i], color='black', ls='--', label='PNSD')
        plt.xscale('log')

        res1 = lognormal(A_fit1_final, diam_fit1_final, sigma_fit1_final)
        plt.plot(x, res1, color='blue', label='mode1')
        res2 = lognormal(A_fit2_final, diam_fit2_final, sigma_fit2_final)
        plt.plot(x, res2, color='red', label='mode2')
        res3 = lognormal(A_fit3_final, diam_fit3_final, sigma_fit3_final)
        plt.plot(x, res3, color='green', label='mode3')
        res4 = lognormal(A_fit4_final, diam_fit4_final, sigma_fit4_final)
        plt.plot(x, res4, color='purple', label='mode4')
        plt.plot(x, res1 + res2 + res3 + res4, color='black', lw=3, label='fit')
        ax.legend()

        def format_minor_ticks(x, pos):
            if pos % 2 == 0:
                return str(int(x))[0]
            else:
                return ""

        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10)))
        ax.xaxis.set_minor_formatter(FuncFormatter(format_minor_ticks))
        ax.xaxis.set_tick_params(which='minor', length=3, labelsize=10)
        ax.xaxis.set_tick_params(which='major', length=7, labelsize=12)

        ax.set_xlabel('Diameter ($nm$)', fontsize=14, labelpad=10)
        ax.set_ylabel('dN/dlogDp ($cm^{-3}$)', fontsize=14, labelpad=10)

        ax.yaxis.set_tick_params(which='major', labelsize=12)

        datetime_object = df1.index[i].to_pydatetime()
        name = datetime_object.strftime('%y%m%d_%H%M')

        plt.title(str(i)+ ', '+ str(df1.index[i]) + '\n' + '$R^{2}$ = '+str(r2) + ' | PE = '+ str(percent_error) +' % - '+fit, fontsize=11)

        path = r'C:\Users\unfer\Desktop\To_new_PC\PhD\Data\Punta Arenas\Mode fitting\Figures_modes\New_March22_event\\'

        if save == 'yes':
            #plt.savefig(path + str(i) + '_Modefit_April19_' + name + '.jpeg', dpi=300, pad_inches=0.1,
                        #bbox_inches='tight')
            gabi = 0

        if save == 'No':
            plt.show()


        plt.close()


def main():
    #app = QtWidgets.QApplication(sys.argv)
    obj = Main()
    #sys.exit(app.exec_())
if __name__ == '__main__':
    main()



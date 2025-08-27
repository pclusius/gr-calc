import aerosol.fitting as af #janne's aerosol functions
from aerosol.safe_fit_df import safe_fit_df #Gaurav's aerosol functions
import json
import pandas as pd
from pdb import set_trace as bp
import numpy as np

from sklearn.cluster import KMeans


#####################################################
def find_peaks(df,file,start_date,fit_multimodes=False,new_fitting_functions = False, n_samples=10000,
                ensemble_size=30,method=0,ens_number=None):
    '''
    Finds mode fitting peaks using Janne Lampilahti's
    aerosol.fitting package, and saves them to a json file.
    '''

    file_name = file.split('.')[0]

    # if fit_multimodes:

    #load the results
    fits=[]
    if ens_number is None:
        loopover = range(ensemble_size)
    else:
        loopover = range(ens_number,ens_number+1)
    try:
        for i in loopover:
            with open(f'{file_name[0:3]}{start_date[2:4]}{start_date[5:7]}{start_date[-2:]}_modefit_{i:03d}.json') as file:
                fits.append(json.load(file))
    except FileNotFoundError:
        for i in loopover:
            if new_fitting_functions:
                fit_results = safe_fit_df(df,n_samples=n_samples)
                print('Using new fitting')
                # bp()
            else:
                fit_results = af.fit_multimodes(df)
                print('Using old fitting')
            # XXX NEW functions

            #write json data to a file
            with open(f'{file_name[0:3]}{start_date[2:4]}{start_date[5:7]}{start_date[-2:]}_modefit_{i:03d}.json', 'w') as output_file:
                json.dump(fit_results, output_file, indent=2)
            # print(f"ERROR: No such file or directory '{file_name[0:3]}{start_date[2:4]}{start_date[5:7]}{start_date[-2:]}_modefit.json'")
            # print("Please change: fit_multimodes = True")
            with open(f'{file_name[0:3]}{start_date[2:4]}{start_date[5:7]}{start_date[-2:]}_modefit_{i:03d}.json') as file:
                fits.append(json.load(file))
            # raise SystemExit
    # bp()
    #making a dataframe from json file
    rows_list = []
    if method == 0:
        fits_mean = average_ensemble(fits, ensemble_size)
    else:
        fits_mean = fits[0]

    del fits

    for bfr in fits_mean[0]:
        # bp()
        if new_fitting_functions:
            import numpy as np
            bp()
            timestamp = bfr[0][0]
        else:
            timestamp = bfr
        ts = timestamp['time']
        # peak_diams = timestamp['peak_diams']

        for i, peak_diam in enumerate(timestamp['peak_diams']):
            # mean = gaussians['mean']
            # sigma = gaussians['sigma']
            # amplitude = gaussians['amplitude']

            # dict_row = {'timestamp':ts,'amplitude':amplitude,'peak_diameter':peak_diams[i],'sigma':sigma} #diam unit m to nm
            dict_row = {'timestamp':ts,'peak_diameter':peak_diam} #diam unit m to nm
            rows_list.append(dict_row)

    df_fits = pd.DataFrame(rows_list)

    #timestamps to index, timestamp strings to datetime objects
    try:
        df_fits['timestamp']=pd.to_datetime(df_fits['timestamp'], format="%Y-%m-%d %H:%M:%S")
    except KeyError:
        print("ERROR: Chosen time period does not exist in this dataset!")
        raise SystemExit
    df_fits.index=df_fits['timestamp']
    df_modefits = df_fits.drop(['timestamp'], axis=1)

    return df_modefits


def average_ensemble(fits,n_ens):
    from copy import deepcopy
    from scipy.spatial.distance import cdist
    returnfit = [[]]
    for ts in range(len(fits[0][0])):
        nmodes = []
        peaks = np.array([])
        for i in range(n_ens):
            peakfits = np.array(fits[i][0][ts][0][0]['peak_diams'])
            mask = np.logical_and(peakfits<1e4, peakfits>0)
            nmodes.append(np.sum(mask))
            peaks = np.append(peaks,peakfits[mask])

        vals,counts = np.unique(nmodes, return_counts=True)
        likely_n_modes = vals[np.argmax(counts)]
        kmeans = KMeans(n_clusters=likely_n_modes, random_state=0, n_init="auto").fit(peaks.reshape(-1,1))
        mean_peaks = np.zeros(likely_n_modes)
        # bp()
        for k in range(likely_n_modes):
            x = peaks[kmeans.labels_== k].reshape(-1,1)
            # calculate weights based on the distance to other points
            # XXX note, distances squared
            weights = (1/( cdist(x,x).sum(1) ) )
            weights = weights/weights.sum()
            mean_peaks[k] = np.sum(peaks[kmeans.labels_== k]*weights)

        retdic = {'time':fits[0][0][ts][0][0]['time']}
        retdic['peak_diams'] = list(mean_peaks)
        retdic['number_of_gaussians'] = likely_n_modes
        returnfit[0].append([[retdic]])

    return returnfit

import aerosol.fitting as af #janne's aerosol functions
import json 
import pandas as pd

def find_peaks(df):
    fit_results = af.fit_multimodes(df)

    #write json data to a file
    with open('fit_results.json', 'w') as output_file:
        json.dump(fit_results, output_file, indent=2)

    #load the results
    with open("fit_results.json") as file:
        fits = json.load(file)

    #making a dataframe from json file
    rows_list = []
    for timestamp in fits:
        ts = timestamp['time']
        peak_diams = timestamp['peak_diams']
        
        for i, gaussians in enumerate(timestamp['gaussians']):
            mean = gaussians['mean']
            sigma = gaussians['sigma']
            amplitude = gaussians['amplitude']

            dict_row = {'timestamp':ts,'amplitude':amplitude,'peak_diameter':peak_diams[i]*10**9,'sigma':sigma} #diam unit m to nm
            rows_list.append(dict_row)

    df_fits = pd.DataFrame(rows_list)  

    #timestamps to index, timestamp strings to datetime objects
    df_fits['timestamp']=pd.to_datetime(df_fits['timestamp'], format="%Y-%m-%d %H:%M:%S")
    df_fits.index=df_fits['timestamp']
    df_modefits = df_fits.drop(['timestamp'], axis=1)

    return df_modefits
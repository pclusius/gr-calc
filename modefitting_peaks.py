import aerosol.fitting as af #janne's aerosol functions
import json 
import pandas as pd

def find_peaks(df,file,fit_multimodes=False):
    file_name = file.split('.')[0]
    
    if fit_multimodes:
        fit_results = af.fit_multimodes(df)

        #write json data to a file
        with open(f'{file_name}_modefit.json', 'w') as output_file:
            json.dump(fit_results, output_file, indent=2)

    #load the results
    try:
        with open(f'{file_name}_modefit.json') as file:
            fits = json.load(file)
    except FileNotFoundError:
        print(f"ERROR: No such file or directory '{file_name}_modefit.json'")
        print("Please change: fit_multimodes = True")
        raise SystemExit

    #making a dataframe from json file
    rows_list = []
    for timestamp in fits[0]:
        ts = timestamp['time']
        peak_diams = timestamp['peak_diams']
        
        for i, gaussians in enumerate(timestamp['gaussians']):
            mean = gaussians['mean']
            sigma = gaussians['sigma']
            amplitude = gaussians['amplitude']

            dict_row = {'timestamp':ts,'amplitude':amplitude,'peak_diameter':peak_diams[i],'sigma':sigma} #diam unit m to nm
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
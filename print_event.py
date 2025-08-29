
def print_event(df, number, header=True, line_end=True):
    '''Pretty print for ts_info dictionary/array collection. '''
    import numpy as np

    intstr = '-           -          -'

    data = df[f'event{number}']
    tss = data.keys()
    timeDic = {}
    for ts in tss:
        if len(data[ts]['lines'])==0:
            continue
        x = [intstr,intstr,intstr]
        for k in range(len(data[ts]['lines'])):
            dias = ([td[1] for td in data[ts]['lines'][k]['fitted points']])
            linestr = f'{data[ts]['lines'][k]['method']}  {np.mean(dias):8.3e}  {data[ts]['lines'][k]['growth rate']:8.3e}'
            n_x = len(linestr)
            if data[ts]['lines'][k]['method'] == 'MF':
                location = 0
            elif data[ts]['lines'][k]['method'] == 'AT':
                location = 1
            else:
                location = 2
            x[location] = linestr
        timeDic[ts] = '   '.join(x)

    return timeDic

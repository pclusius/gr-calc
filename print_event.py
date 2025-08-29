def prettyPrint(ts_info, outfile_body):

    fout = open(outfile_body+'_event_Summary.txt', 'w')
    print('# timestamp           MF   mean_diam   gr          AT   mean_diam   gr          MC   mean_diam   gr          ')
    fout.write('# timestamp           MF   mean_diam   gr          AT   mean_diam   gr          MC   mean_diam   gr          \n')
    for j in range(1,len(ts_info)+1):
        Dic = print_event(ts_info,j)
        # print()
        nr_of_lines = 106 - len('# event') - len(f'{j:0d}') - 2
        print(f'# event{j}  '+''.join(['-']*nr_of_lines))
        fout.write(f'# event{j}  '+''.join(['-']*nr_of_lines)+'\n')
        for k in Dic.keys():
            print(k+'   '+Dic[k])
            fout.write(k+'   '+Dic[k]+'\n')
    fout.close()


def print_event(df, number, header=True, line_end=True):
    '''Pretty print for ts_info dictionary/array collection. '''
    import numpy as np

    intstr = '-            -           -'
    data = df[f'event{number}']
    tss = data.keys()
    timeDic = {}
    for ts in tss:
        if len(data[ts]['lines'])==0:
            continue
        x = [intstr,intstr,intstr]
        for k in range(len(data[ts]['lines'])):
            dias = ([td[1] for td in data[ts]['lines'][k]['fitted points']])
            linestr = f'{data[ts]['lines'][k]['method']}  {np.mean(dias):10.3e}  {data[ts]['lines'][k]['growth rate']:10.3e}'
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

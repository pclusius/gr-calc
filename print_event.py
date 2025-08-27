def print_event(df, number, header=True, line_end=True):
    data = df[f'event{number}']
    n_methods = data['num of lines']
    GR_dic = {'MF':None, 'AT':None, 'MC':None}
    GR = [data['lines'][i]['growth rate'] for i in range(n_methods)]
    methods = [data['lines'][i]['method'] for i in range(n_methods)]
    for m,gr in zip(methods, GR):
        GR_dic[m] = gr

    return GR_dic

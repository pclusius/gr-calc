import numpy as np
from copy import deepcopy
from collections import defaultdict
from matplotlib.dates import num2date, date2num
from datetime import timedelta
from pdb import set_trace as bp

################# USEFUL FUNCTIONS ##################
def flatten(xss):
    return [x for xs in xss for x in xs]
def round_half_up(x):
    if x % 1 == 0.5:
        return int(x) + 1
    else:
        return round(x)
def round_up_to_quarter(dt):
    hour = dt.hour
    minute = dt.minute

    if minute < 15:
        new_minute = 15
    elif minute < 45:
        new_minute = 45
    else:
        new_minute = 15
        hour += 1
        dt += timedelta(hours=1)
    if hour == 24:
        dt += timedelta(days=1)
        hour = 0

    return dt.replace(hour=hour, minute=new_minute, second=0, microsecond=0)

def round_down_to_quarter(dt):
    hour = dt.hour
    minute = dt.minute

    if minute > 45:
        new_minute = 45
    elif minute > 15:
        new_minute = 15
    else:
        new_minute = 45
        hour -= 1
        dt -= timedelta(hours=1)
    if hour == -1:
        dt -= timedelta(days=1)
        hour = 23

    return dt.replace(hour=hour, minute=new_minute, second=0, microsecond=0)

################## FORMING EVENTS ###################
def detect_events(df_data,MF_gr_points,MC_gr_points,AT_gr_points,mc_area_edges,mgsc):
    '''Groups lines to the same growth event with multiple conditions.'''

    def group_lines(data):
        '''From AI'''
        # Step 1: Build the graph (connections between line IDs)
        graph = defaultdict(set)
        line_to_id = {}  # Map each unique line to a unique ID
        id_counter = 0

        for lines in data:
            pair_ids = []
            for line in lines:
                # Convert line to a tuple representation for uniqueness
                line_tuple = tuple((k, tuple(v)) if isinstance(v, list) else (k, v) for k, v in line.items())
                if line_tuple not in line_to_id:
                    line_to_id[line_tuple] = id_counter
                    id_counter += 1
                pair_ids.append(line_to_id[line_tuple])

            # Connect all lines in the same pair
            for i in pair_ids:
                for j in pair_ids:
                    if i != j:
                        graph[i].add(j)

        # Ensure all single elements are in the graph, even if unconnected
        for line_id in line_to_id.values():
            if line_id not in graph:
                graph[line_id] = set()

        # Reverse map IDs back to original lines
        id_to_line = {v: {k: (list(val) if isinstance(val, tuple) else val) for k, val in dict(line).items()} for line, v in line_to_id.items()}

        # Step 2: Find connected components
        visited = set()
        groups = []

        def dfs(node, group):
            visited.add(node)
            group.append(id_to_line[node])
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)

        for node in graph:
            if node not in visited:
                group = []
                dfs(node, group)
                groups.append(group)

        # Step 3: Organize into dictionary
        grouped_dict = {f"event{i+1}": group for i, group in enumerate(groups)}
        return grouped_dict


    pairs = []
    #1 mode fitting overlaps
    for MF_line in MF_gr_points.values():
        MF_t_fit, MF_d_fit = zip(*MF_line['fitted points'])
        MF_gr = MF_line['growth rate']
        MF_min_t, MF_max_t = min(MF_t_fit), max(MF_t_fit)
        MF_min_d, MF_max_d = min(MF_d_fit), max(MF_d_fit)

        #modefitting and maximum concentration
        for MC_line in MC_gr_points.values():
            MC_t_fit, MC_d_fit = zip(*MC_line['fitted points'])

            valid_points = [
                (t, d)
                for t, d in zip(MC_t_fit, MC_d_fit)
                if MF_min_t <= t <= MF_max_t  #ensure t is within MF bounds
                and MF_min_d <= d <= MF_max_d #ensure d is within MF bounds
            ]

            if valid_points:
                MF_line['method'] = 'MF'
                MC_line['method'] = 'MC'
                pairs.append([MF_line, MC_line])

        #mode fitting and appearance time
        for AT_line in AT_gr_points.values():
            AT_t_fit, AT_d_fit = zip(*AT_line['fitted points'])
            AT_gr = AT_line['growth rate']

            if AT_gr < 0:
                AT_t_fit = list(reversed(AT_t_fit))
                AT_d_fit = list(reversed(AT_d_fit))

            #b for equation of MF line
            b = MF_d_fit[0] - (MF_gr * MF_t_fit[0] * 24)

            if MF_gr >= 0:
                valid_points = [
                    (t, d)
                    for t, d in zip(AT_t_fit, AT_d_fit)
                    if MF_min_t <= t <= MF_max_t  #ensure t is within MF bounds
                    and d >= MF_gr * t * 24 + b #ensure d is above the MF line
                    and d <= MF_max_d+MF_max_d*0.5  #ensure d is within the top vertical boundary
                ]
            else:
                valid_points = [
                    (t, d)
                    for t, d in zip(AT_t_fit, AT_d_fit)
                    if MF_min_t <= t <= MF_max_t  #ensure t is within MF bounds
                    and d <= MF_gr * t * 24 + b #ensure d is below the MF line
                    and d >= MF_min_d-MF_min_d*0.2  #ensure d is within the bottom vertical boundary
                ]

            #check if there are valid points in the top triangle
            if valid_points:
                AT_line['method'] = 'AT'
                MF_line['method'] = 'MF'
                pairs.append([MF_line, AT_line])

    #black lines with time length of more than 4h and in higher diameter channels
    for MF_line in MF_gr_points.values():
        MF_t, MF_d = zip(*MF_line['points'])
        time_len = max(MF_t)-min(MF_t)
        t_diff_threshold = 4 #hours

        if time_len >= t_diff_threshold/24 and any(d >= mgsc for d in MF_d):
            MF_line['method'] = 'MF'
            pairs.append([MF_line])

    #maximum concentration and appearance time
    for MC_line in MC_gr_points.values():
        for AT_line in AT_gr_points.values():
            MC_t, MC_d = zip(*MC_line['points'])
            AT_t, AT_d = zip(*AT_line['points'])

            mc_areas_with_at_point = [area for area in mc_area_edges for t,d in zip(AT_t,AT_d)
                                      if d == area[0] and t >= date2num(area[1]) and t <= date2num(area[2])]

            #find matching areas
            matching_areas = [area for area in mc_areas_with_at_point for t,d in zip(MC_t,MC_d)
                              if d == area[0] and t >= date2num(area[1]) and t <= date2num(area[2])]

            #at least 2 points have to match
            if len(matching_areas) >= 2:
                MC_line['method'] = 'MC'
                AT_line['method'] = 'AT'

                pairs.append([MC_line, AT_line])

    #group lines to events
    events = group_lines(pairs)


    ######################################
    #2 mean absolute fractional error (MAFE) of white and black lines
    filtered_events = {}

    for event_label, event in events.items():
        growth_rates = np.array([line['growth rate'] for line in event])
        grs_copy = growth_rates.copy()
        remaining_lines_i = list(range(len(growth_rates)))  #track indices of remaining lines

        # print([line['method'] for line in event])
        # print([line['points'] for line in event])

        while True:
            N = len(grs_copy)

            if N == 2:
                abs_error = [np.abs(grs_copy[0]-grs_copy[1])]
                MAFE = np.abs(2 * abs_error[0] / (grs_copy[0]+grs_copy[1]))
            else:
                avg_gr = np.average(grs_copy)
                abs_error = np.abs(grs_copy - avg_gr)
                MAFE = np.abs(2 / N * np.sum(abs_error) / avg_gr)
            # print(grs_copy)
            #check if white lines start from first 5 diameter channels
            all_diams_MC = [d for line in event if line['method'] == 'MC' for d in list(zip(*line['points']))[1]]


            #in case of last two lines with the same method
            remaining_lines = [event[i] for i in remaining_lines_i]
            unique_methods = set([line['method'] for line in remaining_lines])


            if len(remaining_lines) == 2 and len(unique_methods) == 1:
                remaining_lines_i = [] #remove all lines
                break

                # #one is black = remove the one that isnt black
                # elif any(line['method'] == 'MF' for line in remaining_lines):
                #     idx_not_MF = next(i for i, line in enumerate(remaining_lines) if line['method'] != 'MF')
                #     removed = remaining_lines_i.pop(idx_not_MF)
                #     grs_copy = np.delete(grs_copy, idx_not_MF)
                #     break

            if any(diam <= df_data.columns[4] for diam in all_diams_MC):
                #check if thresholds are exceeded
                # print("diams under 6nm")
                # print('MAFE:',MAFE)

                if MAFE <= 1:# and all(error <= 10 for error in abs_error):
                    break


                biggest_abs_error_i = np.argmax(abs_error)

                # print("abs_error",abs_error,"biggest_abs_error_i",biggest_abs_error_i)
                # print("before:",grs_copy)

                removed = remaining_lines_i.pop(biggest_abs_error_i)
                grs_copy = np.delete(grs_copy, biggest_abs_error_i)

                # print("after:",grs_copy)
                # print()

                #stop if only one growth rate remains
                if len(grs_copy) == 1:
                    break

            else:
                #check if thresholds are exceeded
                # print('MAFE:',MAFE)
                #print(event)
                if MAFE <= 3/2:# and all(error <= 10 for error in abs_error):
                    break
                #break
                #index of the growth rate with the largest error
                # biggest_abs_error_i = np.argmax(abs_error)
                # #corres_gr = grs_no_AT_copy[biggest_abs_error_i]
                # #corres_gr_i = list(growth_rates).index(corres_gr) #index in gr list

                # #print(growth_rates, corres_gr, corres_gr_i)
                # #remove the value with the largest error
                # removed_index = remaining_lines_i.pop(biggest_abs_error_i)
                # grs_no_AT_copy = np.delete(grs_no_AT_copy, biggest_abs_error_i)

                #print(event)
                biggest_abs_error_i = np.argmax(abs_error)
                # print("abs_error",abs_error,"biggest_abs_error_i",biggest_abs_error_i)
                # print("before:",grs_copy)

                removed = remaining_lines_i.pop(biggest_abs_error_i)
                grs_copy = np.delete(grs_copy, biggest_abs_error_i)

                # print("after:",grs_copy)
                # print()


                #stop if only one growth rate remains
                if len(grs_copy) == 1:
                    break

        if remaining_lines_i:
            #add to dictionary
            filtered_events[event_label] = [event[i] for i in remaining_lines_i]

    #2.5
    #if a black and white line remain, check that their areas overlap
    for event_label, event in filtered_events.items():
        unique_methods = set([line['method'] for line in event])

        valid_lines_i = []
        if 'MC' in unique_methods and 'MF' in unique_methods and 'AT' not in unique_methods:
            MC_lines = [(i,line['fitted points']) for i,line in enumerate(event) if line['method'] == 'MC']
            MF_lines = [(i,line['fitted points']) for i,line in enumerate(event) if line['method'] == 'MF']

            for MF_line in MF_lines:
                MF_t_fit, MF_d_fit = zip(*MF_line[1])
                MF_min_t, MF_max_t = min(MF_t_fit), max(MF_t_fit)
                MF_min_d, MF_max_d = min(MF_d_fit), max(MF_d_fit)

                #modefitting and maximum concentration
                for MC_line in MC_lines:
                    MC_t_fit, MC_d_fit = zip(*MC_line[1])

                    valid_points = [
                        (t, d)
                        for t, d in zip(MC_t_fit, MC_d_fit)
                        if MF_min_t <= t <= MF_max_t  #ensure t is within MF bounds
                        and MF_min_d <= d <= MF_max_d #ensure d is within MF bounds
                    ]

                    if valid_points:
                        new_lines = {MF_line[0], MC_line[0]} - set(valid_lines_i)
                        valid_lines_i.extend(new_lines)

            #remove lines that don't overlap
            filtered_events[event_label] = [event[i] for i in valid_lines_i]

    return filtered_events
def split_events(filtered_events):
    '''Determine final events.'''

    #split events to smaller bits
    divided_events = {}
    new_events = {}

    count = 1
    for event_label, event in filtered_events.items():
        remaining_lines_i = list(range(len(event)))  #track indices of remaining lines
        MF_lines = [(i,line) for i,line in enumerate(event) if line['method'] == 'MF']
        MC_lines = [(i,line) for i,line in enumerate(event) if line['method'] == 'MC']
        AT_lines = [(i,line) for i,line in enumerate(event) if line['method'] == 'AT']

        for i,MC_line in MC_lines:
            new_event = [MC_line]
            new_event_i = [i]
            MC_t, MC_d = zip(*MC_line['points'])

            #mf line with most time and diameter overlap
            overlapping_points = []
            for ii,MF_line in MF_lines:
                MF_points_overlap = [i for i,MF_point in enumerate(MF_line['fitted points']) if min(MC_t) <= MF_point[0] <= max(MC_t) and min(MC_d) <= MF_point[1] <= max(MC_d)]
                overlap_count = len(MF_points_overlap)

                #and at least 30% (rounded up) of points in either line overlap
                if overlap_count >= round_half_up(len(MC_line['fitted points'])*0.3) or overlap_count >= round_half_up(len(MF_line['fitted points'])*0.3):
                    overlapping_points.append(len(MF_points_overlap))
                else:
                    overlapping_points.append(0)

            if overlapping_points and not all(num == 0 for num in overlapping_points):
                lines = [line for i,line in MF_lines]
                best_MF_line = lines[np.argmax(overlapping_points)]
                new_event.append(best_MF_line)

                idx = [i for i, line in enumerate(event) if line == best_MF_line][0]
                new_event_i.append(idx)
            #WHAT ABOUT WHEN ONLY GREEN AND BLACK???

            #at line with most diameter overlap
            overlapping_points = []
            for ii,AT_line in AT_lines:
                AT_points_overlap = [i for i,AT_point in enumerate(AT_line['fitted points']) if min(MC_d) <= AT_point[1] <= max(MC_d)]
                overlap_count = len(AT_points_overlap)

                #and at least 50% (rounded up) of points in either line overlap
                if overlap_count >= round_half_up(len(MC_line['fitted points'])/2) or overlap_count >= round_half_up(len(AT_line['fitted points'])/2):
                    overlapping_points.append(len(AT_points_overlap))
                else:
                    overlapping_points.append(0)

            if overlapping_points and not all(num == 0 for num in overlapping_points):
                lines = [line for i,line in AT_lines]
                best_AT_line = lines[np.argmax(overlapping_points)]
                new_event.append(best_AT_line)

                idx = [i for i, line in enumerate(event) if line == best_AT_line][0]
                new_event_i.append(idx)


            if len(new_event) > 1:
                new_events[f'event{len(filtered_events.values())+count}'] = new_event #add new event
                remaining_lines_i = [] #remove all remaining lines
                count += 1 #keeps track of event number

        divided_events[event_label] = [event[i] for i in remaining_lines_i]

    divided_events = divided_events | new_events

    return divided_events
def filter_events(events,df_plot,mgsc):
    '''Filter faulty events.'''

    event_indices = [int(e.lstrip('event')) for e in events]

    for event_label, event in events.items():
        methods = [line['method'] for line in event]
        unique_methods = set(methods)
        all_times = [t for line in event for t in list(zip(*line['fitted points']))[0]]
        all_diams = [d for line in event for d in list(zip(*line['fitted points']))[1]]

        #events outside of the colormap
        if all(df_plot.index[0] >= num2date(time).replace(tzinfo=None) or num2date(time).replace(tzinfo=None) >= df_plot.index[-1] for time in all_times):
            event_indices.remove(int(event_label.lstrip('event')))

        #only whites or only greens
        elif unique_methods == {'MC'} or unique_methods == {'AT'}:
            event_indices.remove(int(event_label.lstrip('event')))

        #one black in lower diameters
        elif len(event) == 1 and unique_methods == {'MF'} and all(d < mgsc for d in all_diams):
            event_indices.remove(int(event_label.lstrip('event')))

        elif len(event) < 1:
            event_indices.remove(int(event_label.lstrip('event')))

    events = {f'event{event_num+1}': events[f'event{i}'] for event_num,i in enumerate(event_indices) if f'event{i}' in events}

    return events

############## GROWTH RATE ESTIMATION ###############
#growth rate estimation
def estimate_growth_rate(lines):
    '''
    Estimates growth rates of lines depending on
    their order and using a weighted average.
    '''
    #PAULI MUOKKAA LOPPUUN PETRIN KANSSA
    #growth rates
    growth_rates = [line['growth rate'] for line in lines]
    mf_grs = [gr for i,gr in enumerate(growth_rates) if lines[i]['method'] == 'MF']
    mc_grs = [gr for i,gr in enumerate(growth_rates) if lines[i]['method'] == 'MC']
    at_grs = [gr for i,gr in enumerate(growth_rates) if lines[i]['method'] == 'AT']

    #classify event depending on line order
    all_times_MC = [t for line in lines if line['method'] == 'MC' for t in list(zip(*line['fitted points']))[0]]
    all_times_MF = [t for line in lines if line['method'] == 'MF' for t in list(zip(*line['fitted points']))[0]]
    all_times_AT = [t for line in lines if line['method'] == 'AT' for t in list(zip(*line['fitted points']))[0]]
    event_type = None
    start_margin = 45/60/24  #45mins in days

    event_type = 1
    #missing black
    if not all_times_MF:
        event_type = 1

    #missing white, green starts at same time or after black starts (with margin)
    elif not all_times_MC:
        # event_type = 2 if all_times_MF[0] - start_margin <= all_times_AT[0] else 1
        if all_times_AT:
            event_type = 2 if all_times_MF[0] - start_margin <= all_times_AT[0] else 1
        else:
            event_type = 1

    #missing green, white starts at same time or after black starts (with margin)
    elif not all_times_AT:
        # event_type = 2 if all_times_MF[0] - start_margin <= all_times_MC[0] else 1
        if all_times_MC:
            event_type = 2 if all_times_MF[0] - start_margin <= all_times_MC[0] else 1
        else:
            event_type = 1

    #all colors
    else:
        #white and green start at same time or after black starts (within 1,5h margin)
        if all_times_MF[0]-start_margin <= all_times_MC[0] and all_times_MF[0]-start_margin <= all_times_AT[0]:
            event_type = 2

        #all other orders
        elif np.mean(all_times_AT) <= np.mean(all_times_MC) and np.mean(all_times_MC) <= np.mean(all_times_MF) or \
            np.mean(all_times_MC) <= np.mean(all_times_AT) and np.mean(all_times_AT) <= np.mean(all_times_MF) or \
            np.mean(all_times_MC) <= np.mean(all_times_MF) and np.mean(all_times_MF) <= np.mean(all_times_AT) or \
            np.mean(all_times_AT) <= np.mean(all_times_MF) and np.mean(all_times_MF) <= np.mean(all_times_MC):
            event_type = 1

    #different weights for averages depending on situation
    if event_type == 1:
        mf_weight, mc_weight, at_weight = 1, 1, 1 #all just as important
    elif event_type == 2:
        mf_weight, mc_weight, at_weight = 2, 1, 1 #black more important

    #WHEIGHTED AVERAGE
    weighted_avg_gr = (sum(mf_weight*mf_grs) + sum(mc_weight*mc_grs) + sum(at_weight*at_grs)) \
                            /(mf_weight*len(mf_grs)+mc_weight*len(mc_grs)+at_weight*len(at_grs))
    min_gr = min(growth_rates)
    max_gr = max(growth_rates)

    return weighted_avg_gr, min_gr, max_gr
def add_event_info(events):
    '''
    Calculates growth rates for growth events and includes an estimate of
    the reliability of the result. Classifies events to different situations
    that affect the weighted average.
    '''
    for i, lines in enumerate(events.values()):
        # bp()
        #estimate growth rates
        weighted_avg_gr, min_gr, max_gr = estimate_growth_rate(lines)

        #calculate AFE (absolute fractional error) for the event
        growth_rates = [line['growth rate'] for line in lines]
        N = len(growth_rates)

        if N == 2:
            abs_error = [np.abs(growth_rates[0]-growth_rates[1])]
            MAFE = np.abs(2 * abs_error[0] / (growth_rates[0]+growth_rates[1]))
        else:
            abs_error = np.abs(growth_rates - weighted_avg_gr)
            MAFE = np.abs(2 / N * np.sum(abs_error) / weighted_avg_gr)

        #and all lines separately
        AFEs = []
        method_pairs = {('MC', 'MF'), ('AT', 'MF'), ('AT', 'MC')}

        for ii, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if ii >= j:
                    continue  #avoid duplicate or self-comparison

                method1 = line1['method']
                method2 = line2['method']
                pair = (method1, method2)
                reversed_pair = (method2, method1)

                if pair in method_pairs or reversed_pair in method_pairs:
                    gr1 = line1['growth rate']
                    gr2 = line2['growth rate']
                    AFE = np.abs(2 * np.abs(gr1 - gr2) / (gr1 + gr2))
                    AFEs.append((f'{method1} & {method2}', AFE))


        #average location in PSD
        datapoints = flatten([line['points'] for line in lines])
        all_t = [point[0] for point in datapoints]
        all_d = [point[1] for point in datapoints]
        mid_x = np.average(all_t)
        mid_y = np.average(all_d)

        #format dictionary differently and add info
        events[f'event{str(i+1)}'] = {'lines': lines}
        events[f'event{str(i+1)}'].update({"avg growth rate": weighted_avg_gr, "min growth rate": min_gr, "max growth rate": max_gr,
                                          "MAFE": MAFE, "respective AFEs": AFEs, "num of lines": len(growth_rates), "mid location": (mid_x,mid_y)})

    return events
def init_events(df_data,df_plot,MF_gr_points,MC_gr_points,AT_gr_points,mc_area_edges,mgsc):
    '''
    Initialize all functions to calculate final results.
    Format of resulting dictinary:
    events = {'event1': {'lines': [...], 'avg growth rate': ..., etc. }, 'event2': {etc.}}
    '''

    #find events
    all_events = detect_events(df_data,MF_gr_points,MC_gr_points,AT_gr_points,mc_area_edges,mgsc)
    final_events = split_events(all_events)

    #filter events
    all_events = filter_events(all_events,df_plot,mgsc)
    final_events = filter_events(final_events,df_plot,mgsc)

    #add more info
    all_events = add_event_info(all_events)
    final_events = add_event_info(final_events)

    return all_events, final_events

def timestamp_info(events):
    '''
    Similar to event_info, but for every timestamp in an event.
    Format of results:
    ts_info = {'event1': {ts1: {'lines': [...], 'avg growth rate': ..., etc. }, ts2: etc.}, 'event2': {etc.}}
    '''

    ts_info = {}

    for event_label,event in events.items():
        #create a timestamp list with rounded times to :15 or :45
        all_ts = [t for line in event['lines'] for t in list(zip(*line['fitted points']))[0]]

        #convert to datetime and round
        min_ts = num2date(min(all_ts)).replace(tzinfo=None)
        max_ts = num2date(max(all_ts)).replace(tzinfo=None)
        min_ts = round_up_to_quarter(min_ts)
        max_ts = round_down_to_quarter(max_ts)

        even_ts = []
        while min_ts <= max_ts:
            even_ts.append(min_ts)
            min_ts += timedelta(minutes=30)

        stamps = {}
        for ts in even_ts:
            #lines in this timestamp
            lines_in_ts = [deepcopy(line) for line in event['lines'] if min(list(zip(*line['fitted points']))[0]) <= date2num(ts) <= max(list(zip(*line['fitted points']))[0])]
            [line.pop('points',None) for line in lines_in_ts] #delete unnecessary info (points)

            #estimate average growth rate
            if len(lines_in_ts) >= 2:
                # bp()
                weighted_avg_gr, min_gr, max_gr = estimate_growth_rate(lines_in_ts)
            else:
                weighted_avg_gr, min_gr, max_gr = None, None, None

            #store info
            stamps.update({ts.strftime('%Y-%m-%d %H:%M:%S'):
                                {'lines': [deepcopy(line) for line in lines_in_ts],
                                "avg growth rate": weighted_avg_gr, "min growth rate": min_gr, "max growth rate": max_gr},
                           })

        #fill dictonary
        ts_info[event_label] = stamps

    return ts_info

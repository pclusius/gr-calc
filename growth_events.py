from matplotlib.dates import set_epoch, num2date, date2num
from datetime import timedelta
import numpy as np
from collections import defaultdict
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]

def detect_events(MF_gr_points,MC_gr_points,AT_gr_points,mc_area_edges,at_area_edges):
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
        grouped_dict = {f"event{i}": group for i, group in enumerate(groups)}
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

    #any black lines with time length of more than 4h
    for MF_line in MF_gr_points.values():
        MF_t, MF_d = zip(*MF_line['points'])
        time_len = max(MF_t)-min(MF_t) 
        
        if time_len > timedelta(hours=2):
            MF_line['method'] = 'MF'
            pairs.append([MF_line])
    
    # #add connecting black lines to events
    # for MF_line1 in MF_gr_points.values():
    #     for MF_line2 in MF_gr_points.values():
    #         MF_t1, MF_d1 = zip(*MF_line1['points'])
    #         MF_t2, MF_d2 = zip(*MF_line2['points'])
    #         gr1 = MF_line1['growth rate']
    #         gr2 = MF_line2['growth rate']
    #         abs_error = abs(gr1-gr2)
            
    #         #if any(t2 in MF_t1 for t2 in MF_t2):
    #         if (MF_t1[0] == MF_t2[-1] or MF_t1[-1] == MF_t2[0]) and \
    #             (MF_d1[0] == MF_d2[-1] or MF_d1[-1] == MF_d2[0]) and abs_error <= 2:
    #                 MF_line1['method'] = 'MF'
    #                 MF_line2['method'] = 'MF'
    #                 pairs.append([MF_line1, MF_line2])

    #maximum concentration and appearance time
    for MC_line in MC_gr_points.values(): 
        for AT_line in AT_gr_points.values():
            MC_t, MC_d = zip(*MC_line['points'])
            AT_t, AT_d = zip(*AT_line['points'])

            mc_areas_with_at_point = [area for area in mc_area_edges if area[0] in AT_d and any(list(AT_t) >= date2num(area[1])) and any(list(AT_t) <= date2num(area[2]))]
            
            #find matching areas
            matching_areas = [
                area for area in mc_areas_with_at_point 
                if area[0] in MC_d and any(list(MC_t) >= date2num(area[1])) and any(list(MC_t) <= date2num(area[2]))]
            
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
            #check if white lines start from under 6nm
            all_diams_MC = [d for line in event if line['method'] == 'MC' for d in list(zip(*line['points']))[1]]

            if any(diam <= 6 for diam in all_diams_MC): 
                #check if thresholds are exceeded
                # print("diams under 6nm")
                # print('MAFE:',MAFE)
                if MAFE <= 2/3:# and all(error <= 10 for error in abs_error):
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
                        valid_lines_i.extend([MF_line[0], MC_line[0]])

            #remove lines that don't overlap
            filtered_events[event_label] = [event[i] for i in valid_lines_i]

    
    ######################################
    #3 appearance time lines associated with white lines and overlapping with black lines
    # # filtered_MC_lines = [line for event in filtered_events.values() for line in event if line['method'] == 'MC']
    # filtered_MF_lines = [line for event in filtered_events.values() for line in event if line['method'] == 'MF']

    # # pairs = []
     
    # #add rest of the event lines to pairs
    # #[pairs.append([line for line in event]) for event in filtered_events.values()]
    # [pairs.append([line]) for line in filtered_MF_lines]
    # filtered_events = group_lines(pairs)
    
    
    #filter events
    event_indices = list(range(len(filtered_events)))

    for i, event in enumerate(filtered_events.values()):
        methods = [line['method'] for line in event]
        unique_methods = set(methods)

        #only whites or only greens
        if unique_methods == {'MC'} or unique_methods == {'AT'}:
            event_indices.remove(i)

        if len(event) < 1:
            event_indices.remove(i)

    filtered_events = {f'event{i}': filtered_events[f'event{i}'] for i in event_indices if f'event{i}' in filtered_events}

    return filtered_events
    
def event_growth_rates(events):
    '''
    Calculates growth rates for growth events and includes an estimate of 
    the reliability of the result. 
    '''
    
    event_grs = pd.DataFrame()
    
    for i, (event_label,lines) in enumerate(events.items()):
        #growth rates
        growth_rates = [line['growth rate'] for line in lines]
        #WHEIGHTED AVERAGE??
        avg_gr = np.average(growth_rates)
        min_gr = min(growth_rates)
        max_gr = max(growth_rates)
        
        #average location in PSD
        datapoints = flatten([line['points'] for line in lines])
        all_t_days = [date2num(point[0]) if isinstance(point[0],pd.Timestamp) else point[0] for point in datapoints]
        all_d = [point[1] for point in datapoints]
        mid_x = np.average(all_t_days)
        mid_y = np.average(all_d)
        
        #add to dataframe
        new_row = pd.DataFrame({"avg growth rate": [avg_gr], "min growth rate": [min_gr], "max growth rate": [max_gr],
                                "mid location": [(mid_x,mid_y)]})                           
        event_grs = pd.concat([event_grs,new_row],ignore_index=True)

    return event_grs
        
        
        
         
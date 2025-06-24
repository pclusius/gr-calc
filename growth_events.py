from matplotlib.dates import set_epoch, num2date, date2num
import numpy as np
from collections import defaultdict
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]

def detect_events(MF_gr_points,MC_gr_points,AT_gr_points,mc_area_edges,at_area_edges):
    '''
    Groups lines to the same growth event with multiple conditions.
    '''

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
    #1 mode fitting and maximum concentration line pairs
    for MF_line in MF_gr_points.values():
        for MC_line in MC_gr_points.values():
            MF_t, MF_d = zip(*MF_line['points'])
            MC_t_fit, MC_d_fit = zip(*MC_line['fitted points'])
            
            if any(t >= min(date2num(MF_t)) for t in MC_t_fit) and any(t <= max(date2num(MF_t)) for t in MC_t_fit) and \
                any(d >= min(MF_d) for d in MC_d_fit) and any(d <= max(MF_d) for d in MC_d_fit):
            
                MF_line['method'] = 'MF'
                MC_line['method'] = 'MC'
                pairs.append([MF_line, MC_line])
    
    #add connecting black lines to events
    for MF_line1 in MF_gr_points.values():
        for MF_line2 in MF_gr_points.values():
            MF_t1, MF_d1 = zip(*MF_line1['points'])
            MF_t2, MF_d2 = zip(*MF_line2['points'])
            gr1 = MF_line1['growth rate']
            gr2 = MF_line2['growth rate']
            abs_error = abs(gr1-gr2)
            
            #if any(t2 in MF_t1 for t2 in MF_t2):
            if (MF_t1[0] == MF_t2[-1] or MF_t1[-1] == MF_t2[0]) and abs_error <= 2:
                    MF_line1['method'] = 'MF'
                    MF_line2['method'] = 'MF'
                    pairs.append([MF_line1, MF_line2])

    #group white and black lines to events
    events = group_lines(pairs)

    ######################################
    #2 mean absolute fractional error (MAFE) of white and black lines
    filtered_events = {}

    for event_label, event in events.items():
        growth_rates = np.array([line['growth rate'] for line in event])
        grs_copy = growth_rates.copy()
        remaining_lines_i = list(range(len(growth_rates)))  #track indices of remaining lines

        print([line['method'] for line in event])
        
        while True:
            avg_gr = np.average(grs_copy)
            N = len(grs_copy)
            abs_error = np.abs(grs_copy - avg_gr)
            avg_abs_error = np.average(abs_error)
            MAFE = np.abs(2 / N * np.sum(abs_error) / avg_gr)
            
            #check if white lines start from under 6nm
            all_diams_MC = [d for line in event if line['method'] == 'MC' for d in list(zip(*line['points']))[1]]

            if any(diam <= 6 for diam in all_diams_MC): 
                #check if thresholds are exceeded
                print("diams under 6nm")
                print('MAFE:',MAFE,'average abs error',avg_abs_error)
                if MAFE <= 2/3:# and avg_abs_error <= 1.5:
                    break
                
                biggest_abs_error_i = np.argmax(abs_error)
                
                print("abs_error",abs_error,"biggest_abs_error_i",biggest_abs_error_i)
                print("before:",grs_copy)
                
                removed = remaining_lines_i.pop(biggest_abs_error_i)
                grs_copy = np.delete(grs_copy, biggest_abs_error_i)
                
                print("after:",grs_copy)
                print()
                
                #stop if only one growth rate remains
                if len(grs_copy) == 1:
                    remaining_lines_i.pop(0)  #remove the last remaining line
                    break
                
            else:
                #check if thresholds are exceeded
                print('MAFE:',MAFE,'average abs error',avg_abs_error)
                if MAFE <= 3/2:# and avg_abs_error <= 1.5:
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
                print("abs_error",abs_error,"biggest_abs_error_i",biggest_abs_error_i)
                print("before:",grs_copy)
                
                removed = remaining_lines_i.pop(biggest_abs_error_i)
                grs_copy = np.delete(grs_copy, biggest_abs_error_i)
                
                print("after:",grs_copy)
                print()
                
                
                #stop if only one growth rate remains
                if len(grs_copy) == 1:
                    remaining_lines_i.pop(0)  #remove the last remaining line
                    break

        #only include events with more than one line    
        if remaining_lines_i:
            filtered_events[event_label] = [event[i] for i in remaining_lines_i]

    
    ######################################
    #3 appearance time lines associated with white lines and overlapping with black lines
    filtered_MC_lines = [line for event in filtered_events.values() for line in event if line['method'] == 'MC']
    filtered_MF_lines = [line for event in filtered_events.values() for line in event if line['method'] == 'MF']

    pairs = []
    #maximum concentration and appearance time
    for MC_line in filtered_MC_lines: 
        for AT_line in AT_gr_points.values():
            MC_t, MC_d = zip(*MC_line['points'])
            AT_t, AT_d = zip(*AT_line['points'])

            mc_areas_with_at_point = [area for area in mc_area_edges if area[0] in AT_d and any(list(AT_t) >= date2num(area[1])) and any(list(AT_t) <= date2num(area[2]))]
            
            for area in mc_areas_with_at_point:
                if area[0] in MC_d and any(list(MC_t) >= date2num(area[1])) and any(list(MC_t) <= date2num(area[2])):

                    AT_line['method'] = 'AT'
                    pairs.append([MC_line, AT_line])
    
                    break 
    
    #mode fitting and appearance time
    for MF_line in filtered_MF_lines:
        for AT_line in AT_gr_points.values():
            MF_t, MF_d = zip(*MF_line['points'])
            AT_t_fit, AT_d_fit = zip(*AT_line['fitted points'])

            if any(t >= min(date2num(MF_t)) for t in AT_t_fit) and any(t <= max(date2num(MF_t)) for t in AT_t_fit) and \
                any(d >= min(MF_d) for d in AT_d_fit) and any(d <= max(MF_d) for d in AT_d_fit):
                
                AT_line['method'] = 'AT'
                pairs.append([MF_line, AT_line])
    
    #add rest of the event lines to pairs
    [pairs.append([line for line in event]) for event in filtered_events.values()]
    events = group_lines(pairs)
    
    
    #filter events consisting of only one method
    event_indices = list(range(len(events)))

    for i, event in enumerate(events.values()):
        methods = [line['method'] for line in event]
        unique_methods = set(methods)
        
        if len(unique_methods) == 1:
            event_indices.remove(i)
    
    events = {f'event{i}': events[f'event{i}'] for i in event_indices if f'event{i}' in events}

    return events
    
def event_growth_rates(events):
    '''
    Calculates growth rates for growth events and includes an estimate of 
    the reliability of the result. 
    '''
    
    event_grs = pd.DataFrame()
    
    for i, (event_label,lines) in enumerate(events.items()):
        #growth rates
        growth_rates = [line['growth rate'] for line in lines]
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
        
        
        
         
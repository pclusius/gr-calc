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
        
    pairs = []

    for i, MF_line in enumerate(MF_gr_points.values()):
        
        #mode fitting and maximum concentration
        for ii, MC_line in enumerate(MC_gr_points.values()):
            MF_t, MF_d = zip(*MF_line['points'])
            MC_t_fit, MC_d_fit = zip(*MC_line['fitted points'])
            
            if any(t >= min(date2num(MF_t)) for t in MC_t_fit) and any(t <= max(date2num(MF_t)) for t in MC_t_fit) and \
                any(d >= min(MF_d) for d in MC_d_fit) and any(d <= max(MF_d) for d in MC_d_fit):
            
                MF_line['method'] = 'MF'
                MC_line['method'] = 'MC'
                pairs.append([MF_line, MC_line])

        #mode fitting and appearance time
        for ii, AT_line in enumerate(AT_gr_points.values()):
            MF_t, MF_d = zip(*MF_line['points'])
            AT_t_fit, AT_d_fit = zip(*AT_line['fitted points'])

            if any(t >= min(date2num(MF_t)) for t in AT_t_fit) and any(t <= max(date2num(MF_t)) for t in AT_t_fit) and \
                any(d >= min(MF_d) for d in AT_d_fit) and any(d <= max(MF_d) for d in AT_d_fit):
                MF_line['method'] = 'MF'
                AT_line['method'] = 'AT'
                pairs.append([MF_line, AT_line])
    
    #maximum concentration and appearance time
    for i, MC_line in enumerate(MC_gr_points.values()):
        for ii, AT_line in enumerate(AT_gr_points.values()):
            
            MC_t, MC_d = zip(*MC_line['points'])
            AT_t, AT_d = zip(*AT_line['points'])

            mc_areas_with_at_point = [area for area in mc_area_edges if area[0] in AT_d and any(list(AT_t) >= date2num(area[1])) and any(list(AT_t) <= date2num(area[2]))]
            
            for area in mc_areas_with_at_point:
                if area[0] in MC_d and any(list(MC_t) >= date2num(area[1])) and any(list(MC_t) <= date2num(area[2])):
                    MC_line['method'] = 'MC'
                    AT_line['method'] = 'AT'
                    pairs.append([MC_line, AT_line])
    
                    break
    

    #group lines to events
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
    events = group_lines(pairs)
    
    events_without_AT = {event_key: [line for line in event if line['method'] != 'AT'] for event_key, event in events.items()}
    
    ######################################
    #mean absolute fractional error (MAFE)
    filtered_events = {}

    for event_label, event in events.items():
        growth_rates = np.array([line['growth rate'] for line in event])# if line['method'] != 'AT'])
        grs_copy = growth_rates.copy()
        remaining_lines = list(range(len(event)))  #track indices of remaining lines

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
                if MAFE <= 2/3 and avg_abs_error <= 1:
                    break
                
                biggest_abs_error_i = np.argmax(abs_error)
                remaining_lines.pop(biggest_abs_error_i)
                grs_copy = np.delete(grs_copy, biggest_abs_error_i)
                
                #stop if only one growth rate remains
                if len(grs_copy) == 1:
                    remaining_lines.pop(0)  #remove the last remaining line
                    break
                
            else:
                #check if thresholds are exceeded
                if MAFE <= 3/2 and avg_abs_error <= 1:
                    break
                #break
                #index of the growth rate with the largest error
                biggest_abs_error_i = np.argmax(abs_error)
                #corres_gr = grs_copy[biggest_abs_error_i]
                #corres_gr_i = list(growth_rates).index(corres_gr) #index in gr list

                #print(growth_rates, corres_gr, corres_gr_i)
                #remove the value with the largest error
                removed_index = remaining_lines.pop(biggest_abs_error_i)
                grs_copy = np.delete(grs_copy, biggest_abs_error_i)

                #stop if only one growth rate remains
                if len(grs_copy) == 1:
                    remaining_lines.pop(0)  #remove the last remaining line
                    break

        #only include events with more than one line
        if remaining_lines:
            filtered_events[event_label] = [event[i] for i in remaining_lines]
        
    
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
        
        
        
         
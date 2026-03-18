import math
import collections
import numpy as np
from pymatching import Matching


def get_complementary_gap(matching, detection_events, obs_flips, nodes_to_LB_X, nodes_to_RB_X):
    '''
    Get the complementary gap for surface code (X memory). Note depending on the memory experiment,
    nodes from left/right boundary will need to turn into nodes from top/right boundary.

    Inputs: 
    matching: the pymatching graph
    detection_events: the detection events
    obs_flips: the observable flips
    nodes_to_LB_X: the detector nodes to the left boundary (list of ints)
    nodes_to_RB_X: the detector nodes close to the right boundary (list of ints)

    Outputs:
    Gap:                complementary gap
    Signed_Gap:         signed complementary gap
    gap_conditioned_PL: gap conditioned logical error rate
    '''    
    
    num_shots = np.shape(detection_events)[0]
    all_edges = matching.edges()
    Comp_matching = Matching()


    LB = max(nodes_to_RB_X)+1
    RB = LB+1
    
    for edge in all_edges:
        node1 = edge[0]
        node2 = edge[1]

        if node2 is not None: #Regular edge we just add it
            
            Comp_matching.add_edge(node1=node1,node2=node2,
                               fault_ids = edge[2]['fault_ids'],
                               weight=edge[2]['weight'],
                               error_probability=edge[2]['error_probability'])
        else: 
            
            #Match to LB
            if node1 in nodes_to_LB_X:
                node2 = LB 
            if node1 in nodes_to_RB_X:
                node2 = RB 

            if node2 is None: #If it remained a None, then node1 \in Z_nodes - None

                Comp_matching.add_boundary_edge(node=node1,
                                                fault_ids = edge[2]['fault_ids'],
                                                weight=edge[2]['weight'],
                                                error_probability=edge[2]['error_probability'])

            else:

                Comp_matching.add_edge(node1=node1,node2=node2,
                                fault_ids = edge[2]['fault_ids'],
                                weight=edge[2]['weight'],
                                error_probability=edge[2]['error_probability'])            
            
    
    Comp_matching.set_boundary_nodes({RB})      
            
    
    pred_reg, W_reg = matching.decode_batch(detection_events,return_weights=True) #This is the regular matching

    new_array = np.zeros((num_shots,1),dtype=int)
    det0      = np.hstack((detection_events,new_array)) #set boundary to 0 (will not match)
    
    new_array = np.ones((num_shots,1),dtype=int)
    det1      = np.hstack((detection_events,new_array)) #set boundary to 1 (will match)

    pred0, W0 = Comp_matching.decode_batch(det0,return_weights=True) #One logical class
    pred1, W1 = Comp_matching.decode_batch(det1,return_weights=True) #The other logical class

    
    edge = next(iter(matching.to_networkx().edges.values()))
    edge_w = edge['weight']
    edge_p = edge['error_probability']
    decibels_per_w = -math.log10(edge_p / (1 - edge_p)) * 10 / edge_w   #Conversion to db, taken from: https://github.com/Strilanc/yoked-surface-codes/blob/main/src/yoked/gap/_gap_worker_handler.py#L54

    #Unsigned gap
    Gap = []
    for k in range(num_shots):
        if W1[k]<W0[k]:
            
            Gap.append( (W0[k]-W1[k]) * decibels_per_w)
        else:
            Gap.append( (W1[k]-W0[k]) * decibels_per_w )     

    Signed_Gap   = []

    for k in range(num_shots):
        if pred0[k]==obs_flips[k]:
            Signed_Gap.append( (W1[k]-W0[k]) * decibels_per_w)
        else:
            Signed_Gap.append( (W0[k]-W1[k]) * decibels_per_w)


    errors = np.any(pred_reg != obs_flips, axis=1)

    # Classify all shots by their error + gap.
    custom_counts = collections.Counter()
    Gap  = np.round(Gap).astype(dtype=np.int64)
    for k in range(len(Gap)):
        g = Gap[k]
        key = f'E{g}' if errors[k] else f'C{g}'
        custom_counts[key] += 1/num_shots

    # P_L(e | g) = E_g / (E_g + C_g) -> gap conditioned logical error rate

    gap_conditioned_PL = {}

    # collect all gap values that appear
    gaps = set()
    for key in custom_counts:
        gaps.add(int(key[1:]))

    for g in gaps:
        E = custom_counts.get(f'E{g}', 0.0)
        C = custom_counts.get(f'C{g}', 0.0)

        if E + C > 0:
            gap_conditioned_PL[g] = E / (E + C)
        else:
            gap_conditioned_PL[g] = np.nan    


    return Gap,Signed_Gap,gap_conditioned_PL

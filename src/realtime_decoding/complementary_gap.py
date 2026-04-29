import math
import collections
import numpy as np
from pymatching import Matching

def get_detector_inds_for_sc(d,rds):
    #Get the detector indices which are L/R boundary (X)
    
    rds_eff = rds+1
    n_anc   = d**2-1


    X_det_inds = []
    Z_det_inds = []

    X_det_inds_LB = []
    X_det_inds_RB = []

    X_dets_inds_LB_per_rd = []
    X_dets_inds_RB_per_rd = []

    for rd in range(rds_eff):

        
        for k in range(n_anc//2): 
            X_det_inds.append(k+n_anc*rd)
            
        temp_X_LB = []
        temp_X_RB = []
        for k in range(n_anc//4): #LB
            X_det_inds_LB.append(k+n_anc*rd) 
            temp_X_LB.append(k+n_anc*rd)

        for k in range(n_anc//4,n_anc//2):
            X_det_inds_RB.append(k+n_anc*rd) 
            temp_X_RB.append(k+n_anc*rd) 


        X_dets_inds_LB_per_rd.append(temp_X_LB)
        X_dets_inds_RB_per_rd.append(temp_X_RB)
        

        if rd == rds_eff-1:
            break

        for k in range(n_anc//2,n_anc):
            Z_det_inds.append(k + n_anc*rd)

    return X_det_inds,Z_det_inds,X_det_inds_LB,X_det_inds_RB,X_dets_inds_LB_per_rd,X_dets_inds_RB_per_rd


from collections import defaultdict

def get_boundary_detectors(circuit, side="left", z_parity=0):
    """
    Returns IDs of boundary stabilizers with fixed top/bottom orientation.
    side: "left", "right", "top", or "bottom"
    z_parity: 0 or 2 (The value of (x+y)%4 that identifies a Z-stabilizer)
    """
    detector_coords = circuit.get_detector_coordinates()
    if not detector_coords:
        return []

    # 1. Determine target type for the requested side
    # Z-type for Left/Right; X-type for Top/Bottom
    if side in ["left", "right"]:
        target_parity = z_parity
    else:
        target_parity = (z_parity + 2) % 4  # X is the opposite parity

    # 2. Filter detectors by the target parity
    filtered_detectors = {}
    for det_id, coords in detector_coords.items():
        x, y = round(coords[0]), round(coords[1])
        if (x + y) % 4 == target_parity:
            filtered_detectors[det_id] = (x, y)
    
    if not filtered_detectors:
        return []

    # 3. Group by row (y) or column (x) to find the absolute edge
    groups = defaultdict(list)
    if side in ["left", "right"]:
        for det_id, (x, y) in filtered_detectors.items():
            groups[y].append((x, det_id))
    else: # top or bottom
        for det_id, (x, y) in filtered_detectors.items():
            groups[x].append((y, det_id))

    # 4. Pick the extreme detector for each row/column
    # Left: min x | Right: max x | Top: max y | Bottom: min y
    if side == "left":
        find_extreme = min
    elif side == "right":
        find_extreme = max
    elif side == "top":
        find_extreme = max  # Fixed: Use max for Top
    elif side == "bottom":
        find_extreme = min  # Fixed: Use min for Bottom
    else:
        raise ValueError("Side must be 'left', 'right', 'top', or 'bottom'.")

    boundary_ids = []
    for axis_key in groups:
        extreme_val = find_extreme(val for val, det_id in groups[axis_key])
        for val, det_id in groups[axis_key]:
            if val == extreme_val:
                boundary_ids.append(det_id)
                
    return boundary_ids


def get_complementary_gap(circuit, detection_events, obs_flips, basis):
    '''
    Get the complementary gap for surface code (X memory). Note depending on the memory experiment,
    nodes from left/right boundary will need to turn into nodes from top/right boundary.

    Inputs: 
    matching: the pymatching graph
    detection_events: the detection events
    obs_flips: the observable flips

    Outputs:
    Gap:                complementary gap
    Signed_Gap:         signed complementary gap
    gap_conditioned_PL: gap conditioned logical error rate
    '''    
    
    num_shots = np.shape(detection_events)[0]
    dem = circuit.detector_error_model()
    matching = Matching.from_detector_error_model(dem)
    all_edges = matching.edges()
    Comp_matching = Matching()

    if basis == 'x': # in stim, XL runs top to bottom
        b1_nodes = get_boundary_detectors(circuit, "top")
        b2_nodes = get_boundary_detectors(circuit, "bottom")
    elif basis == 'z':
        b1_nodes = get_boundary_detectors(circuit, "left")
        b2_nodes = get_boundary_detectors(circuit, "right")
    else:
        ValueError("Improper choice of basis")
    
    b1 = max(b2_nodes)+1
    b2 = b1+1
    
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
            if node1 in b1_nodes:
                node2 = b1 
            if node1 in b2_nodes:
                node2 = b2 

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
            
    
    Comp_matching.set_boundary_nodes({b2})      
            
    
    pred_reg, W_reg = matching.decode_batch(detection_events,return_weights=True) #This is the regular matching
    print(matching.num_detectors)
    print(Comp_matching.num_detectors)
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
    W_min = np.zeros(W_reg.shape)
    W_comp = np.zeros(W_reg.shape)
    pred_min = np.zeros(pred0.shape)
    for k in range(num_shots):
        if W1[k]<W0[k]:
            
            Gap.append( (W0[k]-W1[k]) * decibels_per_w)
        else:
            Gap.append( (W1[k]-W0[k]) * decibels_per_w )     

    Signed_Gap   = []

    for k in range(num_shots):

        # check whether min path belongs to nodes off / on
        if pred_reg[k] == pred0[k]:
            pred_min[k] = pred0[k]
            W_min[k] = W0[k]
            W_comp[k] = W1[k]
        else:
            pred_min[k] = pred1[k]
            W_min[k] = W1[k]
            W_comp[k] = W0[k]


        if pred_min[k]==obs_flips[k]:
            Signed_Gap.append( (W_comp[k]-W_min[k]) * decibels_per_w)
        else:
            Signed_Gap.append( (W_min[k]-W_comp[k]) * decibels_per_w)


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

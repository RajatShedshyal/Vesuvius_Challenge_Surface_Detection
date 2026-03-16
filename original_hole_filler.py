# !pip install connected-components-3d 
import cc3d

import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import heapq
print('IMPORT OK')

def find_segment_start_endpoint(skel):

    skel = (skel > 0).astype(np.uint8)
    #kernel = np.ones((2, 2), dtype=np.uint8)
    kernel= np.array([
        0,0,0,
        1,1,0,
        1,1,0,
    ]).reshape(3,3)
    neighbor_count = convolve(skel, kernel, mode='constant', cval=0) - skel
    endpoint_mask = (skel == 1) & (neighbor_count ==0)
    ey, ex = np.where(endpoint_mask)

    kernel= np.array([
        0,1,1,
        0,1,1,
        0,0,0,
    ]).reshape(3,3)
    neighbor_count = convolve(skel, kernel, mode='constant', cval=0) - skel
    startpoint_mask = (skel == 1) & (neighbor_count ==0)
    sy, sx = np.where(startpoint_mask)

    return (sy, sx), (ey, ex)


# todo: image boundary to endpoint and startpoint to boundary
def pair_segment_start_endpoint(startpoint, endpoint):
    """
    sy, sx : 1D arrays of start-point coordinates
    ey, ex : 1D arrays of end-point coordinates

    Returns:
        pairs : list of (sy, sx, ey, ex) for matched start/end pairs

    Rules:
      - each start used at most once
      - each end used at most once
      - allowed if (end_y >= start_y and end_x >= start_x) OR same pixel
      - isolated pixel (start==end) will pair with itself and dropped
    """

    sy, sx = startpoint
    ey, ex = endpoint

    start = np.column_stack([sy, sx])  # [Ns, 2]
    end = np.column_stack([ey, ex])  # [Ne, 2]

    Ns, Ne = len(start), len(end)
    if Ns == 0 or Ne == 0:
        return []

    # base geometric distance
    dist = cdist(start, end, metric="euclidean")  # [Ns, Ne]

    # initial cost
    cost = dist

    # direction constraint: end must be right-bottom of start
    INF = 1e6
    for i in range(Ns):
        for j in range(Ne):
            sy_i, sx_i = start[i]
            ey_j, ex_j = end[j]

            same_pixel = (sy_i == ey_j) and (sx_i == ex_j)
            forward = (ey_j >= sy_i) and (ex_j >= sx_i)

            if same_pixel:
                cost[i, j] = INF  # disallow this pairing

            if not (same_pixel or forward):
                cost[i, j] = INF  # disallow this pairing

    # Hungarian assignment (works for rectangular cost matrices)
    row_ind, col_ind = linear_sum_assignment(cost)

    pair = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= INF:  # no valid forward end for this start
            continue
        sy_i, sx_i = start[r]
        ey_j, ex_j = end[c]
        pair.append((sy_i, sx_i, ey_j, ex_j))

    return pair





# ---------- 1. Orientation field from the red probability image ----------

def compute_orientation_field(prob, sigma=1.0):
    """
    prob : 2D array, higher = more probable (your red background)
    Returns:
        dir_y, dir_x : orientation unit vectors pointing *along* ridges
    """
    prob = prob.astype(np.float64)
    # smooth a bit, then gradient
    if sigma>0:
        ps = gaussian_filter(prob, sigma=sigma)
    else:
        ps = prob
    gy, gx = np.gradient(ps)

    # gradient points across ridge; rotate by 90° to get along-ridge direction
    vy = -gx
    vx = gy

    norm = np.sqrt(vx ** 2 + vy ** 2) + 1e-12
    vx /= norm
    vy /= norm

    return vy, vx  # (dir_y, dir_x)


# ---------- 2. Single-source / multi-target shortest path with energy ----------

# 8-neighbor moves: (dy, dx, step_length)
NEIGH = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, np.sqrt(2.0)),
    (-1, 1, np.sqrt(2.0)),
    (1, -1, np.sqrt(2.0)),
    (1, 1, np.sqrt(2.0)),
]


def shortest_energy_path(
    startpoint, endpoint,
    prob,
    dir_y, dir_x,
    w_len=1.0,
    w_prob=1.0,
    w_dir=1.0
):
    """
    start    : (sy, sx)
    end_list : list of (ey, ex) pixels (includes isolated points if you want)
    prob     : 2D probability image (higher = better)
    dir_y,x  : orientation field from compute_orientation_field

    weights:
        w_len  : cost per geometric length
        w_prob : weight for (1 - probability)
        w_dir  : weight for deviation from local direction
    """
    H, W = prob.shape
    sy, sx = startpoint
    ey, ex = endpoint

    # Normalize probability to [0, 1]
    p = prob.astype(np.float64)
    p = (p - p.min()) / (p.max() - p.min() + 1e-12)



    # Dijkstra
    dist   = np.full((H, W), np.inf, dtype=np.float64)
    prev_y = np.full((H, W), -1, dtype=np.int32)
    prev_x = np.full((H, W), -1, dtype=np.int32)

    pq = []
    dist[sy, sx] = 0.0
    heapq.heappush(pq, (0.0, sy, sx))

    reached_end = None
    while pq:
        cur_cost, y, x = heapq.heappop(pq)
        if cur_cost > dist[y, x]:
            continue

        if (y, x) == (ey,ex):
            reached_end = (y, x)
            break

        for dy, dx, step_len in NEIGH:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue

            # geometric length cost
            c_len = w_len * step_len

            # probability cost (prefer high probability => low cost)
            c_prob = w_prob * (1.0 - p[ny, nx])

            # direction coherence cost
            step_vec_y = dy / step_len
            step_vec_x = dx / step_len
            # local preferred direction
            vy = dir_y[ny, nx]
            vx = dir_x[ny, nx]
            cosang = abs(step_vec_y * vy + step_vec_x * vx)  # abs = both directions ok
            c_dir = w_dir * (1.0 - cosang)

            step_cost = c_len + c_prob + c_dir
            new_cost = cur_cost + step_cost

            if new_cost < dist[ny, nx]:
                dist[ny, nx] = new_cost
                prev_y[ny, nx] = y
                prev_x[ny, nx] = x
                heapq.heappush(pq, (new_cost, ny, nx))

    if reached_end is None:
        return []  # no path ???

    # backtrack path
    path = []
    y, x = reached_end
    while not (y == -1 and x == -1):
        path.append((y, x))
        py, px = prev_y[y, x], prev_x[y, x]
        y, x = py, px
    path.reverse()
    return path


# ---------- 3. Example usage with your endpoints / startpoints ----------

def connect_curve_segment(
    prob,
    startpoint,
    endpoint,
    w_len=1.0,
    w_prob=1.0,
    w_dir=1.0,
    sigma_orient=0, #1.5
):
    """
    prob       : red background image
    startpoint : sy, sx
    endpoint   : ey, ex
    Returns path
    """
    sy, sx = startpoint
    ey, ex = endpoint
    dir_y, dir_x = compute_orientation_field(prob, sigma=sigma_orient)
    path = shortest_energy_path(
        (sy, sx), (ey,ex), prob, dir_y, dir_x,
        w_len=w_len, w_prob=w_prob, w_dir=w_dir
    )
    return path

print('HELPER OK')  

from tqdm import tqdm

print('Loading probability volume...')
prob = np.load(r'E:\MyProjects\Vesu\exp_5\102536988_prob (1).npy')
binary = np.load(r"E:\MyProjects\Vesu\exp_5\102536988_pred (1).npy")
D, H, W = prob.shape

# 1. Generate Seeds and CCs
print('Running Initial CC3D...')
predict = binary
cc = cc3d.connected_components(predict)
cc = cc3d.dust(cc, threshold=100)

# 2. Prepare Output Volume
# 0 = Background, 1 = Original Seed, 2 = New Connections
final_volume = (cc > 0).astype(np.uint8)

n_comp = cc.max()

unique_labels = np.arange(1, n_comp + 1) # Remove background
print(f'Found {len(unique_labels)} components to process.')

# 3. Process every component
for label in tqdm(unique_labels, desc="Processing Components"):
    # Create a mask for just this specific component
    one_component = (cc == label).astype(np.uint8)
    
    for z in range(D):
        slice_p = one_component[z]
        
        # Skip empty slices for this component to save time
        if not np.any(slice_p):
            continue
            
        # Step A: Find endpoints
        (sy, sx), (ey, ex) = find_segment_start_endpoint(slice_p)
        
        # Step B: Pair them
        pairs = pair_segment_start_endpoint(
            startpoint=(sy, sx), endpoint=(ey, ex),
        )
        
        # Step C: Connect and update final_volume
        for psy, psx, pey, pex in pairs:
            # Use the original probability map slice for guidance
            con = connect_curve_segment(
                prob=prob[z],

                startpoint=(psy, psx), endpoint=(pey, pex),
                w_len=1.0, w_prob=1.0, w_dir=1.0, sigma_orient=1.5
            )
            
            if len(con) == 0: 
                continue
                
            for y, x in con:
                # Mark as '2' to distinguish from original predicted seeds
                if final_volume[z, y, x] == 0:
                    final_volume[z, y, x] = 2

print('All components processed. Saving...')
np.save('finale_7.npy', final_volume)
print('COMPLETED!!!')

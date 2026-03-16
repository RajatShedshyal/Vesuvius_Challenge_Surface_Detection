import numpy as np
import cc3d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve, gaussian_filter
import heapq
from tqdm import tqdm

# =========================================================
# 1️⃣ Directional Kernels (8 directions)
# =========================================================

KERNELS = [
np.array([[0,0,0],[0,1,1],[0,0,0]]),  # right
np.array([[0,0,0],[1,1,0],[0,0,0]]),  # left
np.array([[0,1,0],[0,1,0],[0,1,0]]),  # down
np.array([[0,1,0],[0,1,0],[0,1,0]]),  # up
np.array([[0,0,0],[0,1,0],[0,0,1]]),  # down right
np.array([[0,0,0],[0,1,0],[1,0,0]]),  # down left
np.array([[0,0,1],[0,1,0],[0,0,0]]),  # up right
np.array([[1,0,0],[0,1,0],[0,0,0]])   # up left
]

# =========================================================
# 2️⃣ Endpoint Detection
# =========================================================

def find_endpoints_directional(skel):

    skel = (skel > 0).astype(np.uint8)

    responses = []
    for K in KERNELS:
        resp = convolve(skel, K, mode='constant', cval=0)
        responses.append(resp == 2)

    total_matches = np.sum(responses, axis=0)

    endpoint_mask = (skel == 1) & (total_matches == 1)

    y, x = np.where(endpoint_mask)
    return y, x


# =========================================================
# 3️⃣ Orientation Field
# =========================================================

def compute_orientation_field(prob, sigma=1.5):

    ps = gaussian_filter(prob.astype(np.float64), sigma=sigma)

    gy, gx = np.gradient(ps)

    vy = -gx
    vx = gy

    norm = np.sqrt(vx**2 + vy**2) + 1e-12
    vx /= norm
    vy /= norm

    return vy, vx


# =========================================================
# 4️⃣ Start / End Classification
# =========================================================

def classify_start_end(ys, xs, dir_y, dir_x):

    starts = []
    ends = []

    for y, x in zip(ys, xs):

        flow = dir_y[y,x] + dir_x[y,x]

        if flow >= 0:
            starts.append((y,x))
        else:
            ends.append((y,x))

    return starts, ends


# =========================================================
# 5️⃣ Hungarian Pairing
# =========================================================

def pair_start_end(starts, ends):

    if len(starts)==0 or len(ends)==0:
        return []

    start = np.array(starts)
    end = np.array(ends)

    cost = cdist(start, end)

    INF = 1e6

    for i in range(len(start)):
        for j in range(len(end)):

            sy, sx = start[i]
            ey, ex = end[j]

            if not (ey >= sy and ex >= sx):
                cost[i,j] = INF

    r,c = linear_sum_assignment(cost)

    pairs = []
    for i,j in zip(r,c):
        if cost[i,j] < INF:
            pairs.append((*start[i], *end[j]))

    return pairs


# =========================================================
# 6️⃣ Energy Shortest Path
# =========================================================

NEIGH = [
(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),
(-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),
(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))
]

def shortest_energy_path(start, end, prob, dir_y, dir_x):

    H,W = prob.shape
    sy,sx = start
    ey,ex = end

    p = (prob - prob.min())/(prob.max()-prob.min()+1e-12)

    dist = np.full((H,W), np.inf)
    py = np.full((H,W), -1, int)
    px = np.full((H,W), -1, int)

    pq = []
    dist[sy,sx] = 0
    heapq.heappush(pq,(0,sy,sx))

    while pq:

        cost,y,x = heapq.heappop(pq)

        if (y,x)==(ey,ex):
            break

        if cost > dist[y,x]:
            continue

        for dy,dx,l in NEIGH:

            ny,nx = y+dy, x+dx

            if not (0<=ny<H and 0<=nx<W):
                continue

            c_len = l
            c_prob = 1 - p[ny,nx]

            step_y = dy/l
            step_x = dx/l

            cosang = abs(step_y*dir_y[ny,nx] + step_x*dir_x[ny,nx])
            c_dir = 1 - cosang

            new = cost + c_len + c_prob + c_dir

            if new < dist[ny,nx]:
                dist[ny,nx] = new
                py[ny,nx] = y
                px[ny,nx] = x
                heapq.heappush(pq,(new,ny,nx))

    path = []
    y,x = ey,ex
    while y != -1:
        path.append((y,x))
        y,x = py[y,x], px[y,x]

    return path[::-1]


# =========================================================
# 7️⃣ Slice Connector
# =========================================================

def connect_slice(prob_slice, comp_slice):

    fy, fx = find_endpoints_directional(comp_slice)

    if len(fy) == 0:
        return []

    dir_y, dir_x = compute_orientation_field(prob_slice)

    starts, ends = classify_start_end(fy, fx, dir_y, dir_x)

    pairs = pair_start_end(starts, ends)

    paths = []

    for sy,sx,ey,ex in pairs:

        p = shortest_energy_path(
            (sy,sx),(ey,ex),
            prob_slice,
            dir_y,dir_x
        )

        if len(p) > 0:
            paths.append(p)

    return paths


# =========================================================
# 8️⃣ FULL VOLUME PIPELINE
# =========================================================

print("Loading probability volume...")
prob = np.load(r"kaggle/working/102536988_prob.npy")

input_shape = prob.shape
print("Input shape:", input_shape)

D,H,W = prob.shape

print("Running CC3D...")
predict = (prob > 0.3).astype(np.uint8)

cc = cc3d.connected_components(predict)
cc = cc3d.dust(cc, threshold=100)

final = (cc > 0).astype(np.uint8)

labels = np.unique(cc)
labels = labels[labels != 0]

print("Components:", len(labels))

total_lines_connected = 0
total_voxels_filled = 0

for label in tqdm(labels):

    comp = (cc == label).astype(np.uint8)

    for z in range(D):

        if not np.any(comp[z]):
            continue

        paths = connect_slice(prob[z], comp[z])

        total_lines_connected += len(paths)

        for path in paths:
            for y,x in path:
                if final[z,y,x] == 0:
                    final[z,y,x] = 2
                    total_voxels_filled += 1


# =========================================================
# 9️⃣ SAVE + SHAPE CHECK
# =========================================================

print("\n==== RESULTS ====")
print("Total lines connected:", total_lines_connected)
print("Total voxels filled:", total_voxels_filled)
print("Output shape:", final.shape)

assert final.shape == input_shape, "❌ Shape mismatch!"
print("✅ Shape preserved")

np.save("finale3.npy", final)
print("Saved finale3.npy")
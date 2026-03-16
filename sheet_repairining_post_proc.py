import numpy as np
from scipy import ndimage
import cc3d
from sklearn.decomposition import PCA

def get_oriented_disk_struct(normal, radius=3):
    size = radius * 2 + 1
    zz, yy, xx = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing='ij'
    )
    dist_from_plane = np.abs(zz * normal[0] + yy * normal[1] + xx * normal[2])
    dist_from_center = np.sqrt(zz**2 + yy**2 + xx**2)
    struct = (dist_from_plane <= 0.8) & (dist_from_center <= radius)
    return struct.astype(bool)

def seal_and_repair_split_labels(split_labels,comp_normal, radius=15, min_size=100):
    """
    Takes split_labels (the output from dissolve_sheet_merges) and repairs 
    each sheet individually to prevent re-merging.
    """
    # Initialize a global mask for the final output
    final_output = np.zeros_like(split_labels, dtype=np.uint8)
    
    # Identify unique sheets (ignoring background 0)
    sheet_ids = np.unique(split_labels)
    sheet_ids = sheet_ids[sheet_ids != 0]
    
    print(f"🛠️  Repairing {len(sheet_ids)} individual sheets...")

    for sheet_id in sheet_ids:
        # 1. Isolate the specific sheet mask
        sheet_mask = (split_labels == sheet_id)
        
        # Skip tiny fragments that don't need repairing
        if sheet_mask.sum() < min_size:
            final_output[sheet_mask] = 1
            continue
        
        # 3. Create a thin (1-pixel) oriented disk structuring element
        # Using a radius of 2 is safer for tight gaps
        struct = get_oriented_disk_struct(comp_normal, radius=radius)
        
        # 4. Apply Planar Closing
        # This fills small cracks/lace-like holes strictly within the sheet plane
        repaired = ndimage.binary_closing(sheet_mask, structure=struct)
        
        # 5. Fill Voids
        # Because we only have ONE sheet in context, this cannot bridge to neighbors
        repaired = ndimage.binary_fill_holes(repaired)
        
        # 6. Fuse into the final output
        final_output[repaired > 0] = 1
        
    return final_output

# --------------------------------------------------
# 1️⃣ NORMAL ESTIMATION USING PCA
# --------------------------------------------------
def estimate_sheet_normal(mask):
    """
    Estimate sheet normal using PCA.
    Smallest variance axis = normal direction.
    """
    coords = np.column_stack(np.where(mask))
    
    if len(coords) < 50:
        return np.array([0,0,1], dtype=float)
    
    pca = PCA(n_components=3)
    pca.fit(coords)
    
    normal = pca.components_[-1]
    normal = normal / np.linalg.norm(normal)
    
    return normal


# --------------------------------------------------
# 2️⃣ FULL REPAIR PIPELINE
# --------------------------------------------------
def repair_binary_sheet_volume(
        binary_volume,
        radius=15,
        min_size=100,
        min_sheet_size=500
):
    """
    FULL PIPELINE:
    binary → split → per-sheet normal → repair → merge
    """
    
    print("🔹 Step 1: Ensure binary")
    binary = binary_volume > 0
    
    print("🔹 Step 2: Connected Components (Split Sheets)")
    split_labels = cc3d.connected_components(binary)
    
    sheet_ids = np.unique(split_labels)
    sheet_ids = sheet_ids[sheet_ids != 0]
    
    print(f"🔹 Found {len(sheet_ids)} sheets")
    
    final_output = np.zeros_like(binary, dtype=np.uint8)
    
    print("🔹 Step 3: Per-Sheet Repair")
    
    for sheet_id in sheet_ids:
        
        sheet_mask = (split_labels == sheet_id)
        
        if sheet_mask.sum() < min_sheet_size:
            final_output[sheet_mask] = 1
            continue
        
        print(f"   Repairing sheet {sheet_id}")
        
        # --- Normal estimation ---
        normal = estimate_sheet_normal(sheet_mask)
        
        # --- Structuring element ---
        struct = get_oriented_disk_struct(normal, radius=radius)
        
        # --- Planar closing ---
        repaired = ndimage.binary_closing(sheet_mask, structure=struct)
        
        # --- Hole fill ---
        repaired = ndimage.binary_fill_holes(repaired)
        
        final_output[repaired] = 1
    
    print("✅ Repair Done")
    
    return final_output.astype(np.uint8)

volume = np.load("E:\\MyProjects\\Vesu\\exp_5\\102536988_pred (1).npy")

repaired_volume = repair_binary_sheet_volume(
    volume,
    radius=15,
    min_size=100,
    min_sheet_size=500
)

np.save("repaired_volume.npy", repaired_volume)


import os

for fils in os.listdir("E:\\MyProjects\\Vesu\\exp_5\\pred_maps"):
    if fils.endswith("_pred.npy"):
        print(f"Processing {fils}...")
        vol = np.load(os.path.join("E:\\MyProjects\\Vesu\\exp_5\\pred_maps", fils))
        repaired = repair_binary_sheet_volume(
            vol,
            radius=15,
            min_size=100,
            min_sheet_size=500
        )
        print(f"Finished repairing {fils}. Saving output...")
        save_path = os.path.join("E:\\MyProjects\\Vesu\\exp_5\\repaired_maps", fils.replace("_pred.npy", "_repaired.npy"))
        np.save(save_path, repaired)
        print(f"Saved repaired volume to {save_path}")
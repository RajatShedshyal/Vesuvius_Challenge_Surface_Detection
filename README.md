# Vesuvius Challenge: Ancient Scroll Surface Detection

A complete solution for the Kaggle Vesuvius Challenge - detecting and reconstructing the surface of an ancient Roman scroll from 3D X-ray CT scans. This repository implements two primary segmentation approaches (**SwinUNETR Baseline** and **nnUNet**) combined with advanced post-processing techniques for hole filling, sheet repair, and topological refinement.

**Link:** [Kaggle_Vesuvius_Surface_Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/leaderboard)

## Core Solutions

### Segmentation Models

- **[SwinUNETR Baseline]** - Vision Transformer-based architecture for surface segmentation with strong baseline performance
- **nnUNet** - Self-configuring segmentation framework that automatically designs optimal network architecture for this dataset

### Post-Processing Pipeline

- **Directional Hole Filling**: Sophisticated hole-filling algorithm using directional kernels, skeleton analysis, and shortest-path search
- **Ant March Post-Processing**: Dijkstra-based bridge splitting to separate incorrectly merged surfaces and preserve topology
- **Sheet Repair & Merging**: Tools for repairing sheet-like structures using oriented morphological operations
- **Topological Analysis**: Skeleton-based endpoint detection and orientation field computation
- **3D Visualization**: Real-time 3D visualization and inspection of volumetric predictions
- **Vesuvius Metric Integration**: Support for official competition scoring and evaluation

## Project Structure

```
├── base-line/                           # SOLUTION 1: SwinUNETR Baseline Model
│   ├── Baseline-SwinUNETR.ipynb         # Training pipeline for SwinUNETR architecture
│   ├── Inference-SwinUNETR.ipynb        # Inference and prediction generation
│   ├── PostProcessing/                  # Post-processing on baseline predictions
│   │   └── output-simulation.ipynb      # Post-processing evaluation
│   ├── marching-ants-postproc/          # Marching cubes post-processing
│   ├── Outputs/                         # Model checkpoints and training metrics
│   └── graphs.jpeg                      # Training/validation visualization
│
├── nnunet.ipynb                         # SOLUTION 2: nnUNet integration & metrics
│
├── Post-Processing Tools (Applied to both models)
│   ├── filler_all_dir.py                # Main hole-filling algorithm (directional kernels)
│   ├── original_hole_filler.py          # Alternative hole-filling with skeleton tracking
│   ├── sheet_repairining_post_proc.py   # Sheet repair and morphological operations
│   ├── base-line/marching-ants-postproc/         # Ant March post-processing with Dijkstra bridge splitting
│   └── full_viewer.py                   # 3D visualization tool using PyVista
│
├── Data & Experiments
│   ├── Kaggle_Notebooks/                # Competition submissions & experimental approaches
│   │   ├── ant-march-post-processs.ipynb     # **KEY**: Dijkstra-based bridge splitting for surface separation
│   │   ├── hole-filler-by-rajat.ipynb       # Testing hole-filling implementations
│   │   ├── prob-map-generator.ipynb         # Probability map generation
│   │   └── metrics.ipynb                    # Metric evaluation
│   ├── pred_maps/                       # Generated probability predictions (.npy)
│   ├── repaired_maps/                   # Post-processed output volumes (.npy)
│   ├── seeds/                           # Seed data and reference labels
│   └── Images/                          # Some Images
```

## Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy scipy cc3d pyvista scikit-learn matplotlib jupyter torch monai tifffile pillow dijkstra3d
   ```

   Key packages:
   - **Model Training**: `torch`, `monai` (for SwinUNETR and nnUNet)
   - **Image Processing**: `numpy`, `scipy`, `cc3d`, `scikit-learn`, `dijkstra3d`
   - **Visualization**: `pyvista`, `matplotlib`
   - **Data I/O**: `tifffile`, `pillow`

### Quick Start: Full Pipeline

#### Step 1: Train/Inference with SwinUNETR Baseline

```bash
jupyter notebook base-line/Baseline-SwinUNETR.ipynb  # Train the model
jupyter notebook base-line/Inference-SwinUNETR.ipynb  # Generate predictions
```

Or use nnUNet for automatic architecture optimization:

```bash
jupyter notebook nnunet.ipynb
```

#### Step 2: Post-Process Predictions

Apply hole-filling and sheet repair to clean up segmentation outputs:

```python
import numpy as np
from filler_all_dir import find_endpoints_directional, shortest_energy_path, compute_orientation_field

# Load segmentation predictions
prob_map = np.load('pred_maps/post_102536988_pred.npy')

# Compute orientation field
dir_y, dir_x = compute_orientation_field(prob_map, sigma=1.5)

# Find endpoints and apply hole filling
y_coords, x_coords = find_endpoints_directional(prob_map)
```

#### Step 3: Evaluate with Vesuvius Metric

```bash
jupyter notebook nnunet.ipynb  # Includes Vesuvius metric evaluation
```

### Model Comparison

| Model | Approach | Configuration |
|-------|----------|---|
| **SwinUNETR** | Vision Transformer | See [Baseline-SwinUNETR.ipynb](base-line/Baseline-SwinUNETR.ipynb) |
| **nnUNet** | Auto-configured | See [nnunet.ipynb](nnunet.ipynb) |

### Post-Processing Techniques

#### Directional Hole Filling

The core post-processing algorithm in `filler_all_dir.py` handles disconnected regions:

```python
from filler_all_dir import (
    find_endpoints_directional, 
    compute_orientation_field, 
    classify_start_end, 
    pair_start_end,
    shortest_energy_path
)

# Load probability map
prob_map = np.load('prediction.npy')

# 1. Detect skeleton endpoints
endpoints_y, endpoints_x = find_endpoints_directional(prob_map)

# 2. Compute flow direction
dir_y, dir_x = compute_orientation_field(prob_map, sigma=1.5)

# 3. Classify endpoints
starts, ends = classify_start_end(endpoints_y, endpoints_x, dir_y, dir_x)

# 4. Pair starts with ends optimally
pairs = pair_start_end(starts, ends)

# 5. Connect pairs with energy-weighted shortest paths
for start, end in pairs:
    path = shortest_energy_path(start, end, prob_map, dir_y, dir_x)
```

#### Sheet Repair

For sheet-like structures (common in scroll surfaces), use morphological operations aligned to surface normals:

```python
from sheet_repairining_post_proc import seal_and_repair_split_labels

# Repair individual sheet structures detected by the segmentation model
repaired = seal_and_repair_split_labels(
    split_labels=label_volume,
    comp_normal=[0, 0, 1],  # Sheet normal (z-direction for scroll)
    radius=15,              # Structuring element radius
    min_size=100            # Minimum component size to process
)
```

#### Ant March Post-Processing: Dijkstra-Based Bridge Splitting

A critical technique for separating incorrectly merged surfaces using efficient shortest-path computation:

```python
import dijkstra3d
import cc3d
import numpy as np

# Identify merged segments that should be separate
merged_volume = segmentation_output

# Find all connected components
labels = cc3d.connected_components(merged_volume, connectivity=26)

# For components with multiple surfaces, use Dijkstra to find optimal split
def split_merged_surfaces(volume, start_point, end_point):
    """
    Uses Dijkstra's algorithm to find the shortest path (minimum "bridge")
    connecting two points, then splits the volume along this path.
    
    Args:
        volume: 3D binary/probability volume
        start_point: First surface endpoint (y, x, z)
        end_point: Second surface endpoint (y, x, z)
    
    Returns:
        Dijkstra cost field and optimal path
    """
    # Create cost field (inverse of probability for pathfinding)
    cost_field = 1.0 - volume.astype(np.float32)
    
    # Compute shortest path using Dijkstra
    dist, path = dijkstra3d.dijkstra(
        cost_field,
        start=start_point,
        end=end_point,
        connectivity=26
    )
    
    return dist, path

# Example: split merged surfaces
dist_field, optimal_path = split_merged_surfaces(
    volume=segmentation_output,
    start_point=(10, 10, 5),
    end_point=(50, 50, 25)
)

# The path defines where to cut merged surfaces
```

**Key Advantages**:
- Preserves surface topology by finding optimal separation lines
- Computationally efficient using Dijkstra's algorithm
- Works on probability maps to make intelligent splitting decisions
- Handles complex merged regions with multiple layers

#### 3D Visualization & Inspection

Use PyVista for interactive 3D visualization of predictions and repairs:

```python
import numpy as np
import pyvista as pv
import cc3d

# Load volume
vol = np.load('repaired_maps/post_102536988_repaired.npy')
binary = vol > 0.3

# Connected components analysis
labels = cc3d.connected_components(binary, connectivity=26)

# Visualize
plotter = pv.Plotter()
mesh = pv.voxelize(binary)
plotter.add_mesh(mesh, color='white')
plotter.show()
```

## Solution Overview

### Primary Approaches

#### 1. SwinUNETR Baseline ([base-line/](base-line/))

Vision Transformer-based architecture for volumetric surface detection:

- **Architecture**: Combines Swin Transformer with 3D UNet decoder
- **Training**: End-to-end learning on CT scan volumes
- **Output**: Probability maps of scroll surface

#### 2. nnUNet

Self-configuring segmentation framework:

- **Automatic Configuration**: Learns optimal architecture from dataset
- **Robustness**: Self-tuning hyperparameters and preprocessing
- **Metric Integration**: Built-in support for Vesuvius Challenge metrics
- **Flexibility**: Can be applied to different sub-volumes and regions

### Post-Processing Pipeline

Applied **after** segmentation to refine predictions:

1. **Ant March (Bridge Splitting)** - Separates incorrectly merged surfaces using Dijkstra shortest-path algorithm
2. **Hole Filling** - Connects disconnected components using skeleton analysis
3. **Sheet Repair** - Fixes breaks in surface topology  
4. **Connected Components Analysis** - Validates and measures final output
5. **Metric Evaluation** - Scores against ground truth using official Vesuvius metrics

## Algorithm Details

### Directional Hole Filling

The core post-processing algorithm in `filler_all_dir.py` handles disconnected regions:

1. **Endpoint Detection**: Uses 8 directional kernels to identify skeleton endpoints
2. **Orientation Field**: Computes gradient-based flow direction using Gaussian filtering
3. **Start/End Classification**: Classifies endpoints as start or end points based on flow direction
4. **Hungarian Pairing**: Optimally pairs starts and ends using the Hungarian algorithm
5. **Energy-Weighted Shortest Path**: Finds paths connecting pairs using Dijkstra's algorithm with probability-weighted costs

### Ant March Post-Processing: Dijkstra-Based Bridge Splitting

1. **Connected Components Analysis**: Identifies potentially merged regions
2. **Endpoint Detection via Raycast**: Finds surface boundaries using ray marching techniques
3. **Dijkstra Shortest Path**: Computes minimum-cost path between merged surfaces
4. **Bridge Splitting**: Removes thin bridge regions to separate surfaces cleanly
5. **Topology Preservation**: Ensures split doesn't create artificial artifacts

**Why it works**: 
- The Dijkstra algorithm finds the weakest connection (bridge) between merged surfaces
- By removing low-confidence bridge pixels, surfaces naturally separate
- Maintains topological consistency better than naive threshold-based splitting

### Sheet Repair

The `sheet_repairining_post_proc.py` module provides:

1. **Oriented Disk Structuring**: Creates morphological elements aligned to sheet normals
2. **Planar Closing**: Fills small cracks within the sheet plane
3. **Void Sealing**: Prevents re-merging of adjacent sheets

## Data Formats

- **Input**: 3D CT scan volumes (TIFF stacks or NumPy arrays)
- **Model Output**: Probability maps indicating scroll surface location (0-1 float values)
- **Repaired Output**: Binary or multi-label 3D volumes after post-processing
- **Storage**: NumPy .npy format for efficient I/O

### Data Directories

- `Images/` - Original CT scan input data
- `pred_maps/` - Raw predictions from segmentation models
- `repaired_maps/` - Post-processed and refined predictions
- `seeds/` - Ground truth labels and reference data

## Competition Context

This is a complete submission to the **Kaggle Vesuvius Challenge** - a competition to develop AI methods for recovering the text from damaged ancient Roman scrolls using 3D X-ray CT scans.

**Challenge Goal**: Automatically detect and reconstruct the ink-covered surface of a scroll from volumetric CT data, making the written text readable.

**Link**: [Kaggle Vesuvius Challenge](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/leaderboard)

## Contributing

Contributions welcome! Potential improvements:

- GPU-accelerated hole-filling (CuPy)
- Additional architectures beyond SwinUNETR and nnUNet
- Ensemble methods combining models
- Metric-specific optimizations
- Comprehensive unit tests and benchmarks

## Acknowledgments

Built as a competitive solution to the Kaggle Vesuvius Challenge. Thanks to:
- Kaggle community for dataset and competition platform
- MONAI and PyTorch for deep learning frameworks
- Connected components 3D library for topology analysis

## Project Status

**Active Competition Submission**

- **Primary Models**: SwinUNETR baseline and nnUNet approaches fully implemented
- **Post-Processing**: Advanced hole-filling and sheet repair validated on test data
- **Evaluation**: Integration with official Vesuvius metrics for scoring

import numpy as np
import pyvista as pv
import cc3d
import math

print("Loading volume...")
# vol = np.load(r"E:\MyProjects\Vesu\val_preds_post_process_plz (4)\kaggle\working\1029212680_pred.npy")

# vol = np.load(r"E:\\MyProjects\\Vesu\\val_preds_post_process_plz (4)\\kaggle\\working\\102536988_pred.npy")
vol = np.load(r"E:\MyProjects\Vesu\exp_5\repaired_maps\post_102536988_repaired.npy")


binary = vol > 0.3

print("Shape: " + str(binary.shape))

print("Running CC3D...")
labels = cc3d.connected_components(binary, connectivity=26)
n_comp = labels.max()
print("Total components:", n_comp)


# -----------------------------
# Subplot grid size
# -----------------------------
cols = int(math.ceil(math.sqrt(n_comp)))
rows = int(math.ceil(n_comp / cols))

plotter = pv.Plotter(shape=(rows, cols))


# Random colors
rng = np.random.default_rng(42)
colors = rng.random((n_comp + 1, 3))


# -----------------------------
# Loop components
# -----------------------------
plot_index = 0

for comp_id in range(1, n_comp + 1):

    mask = labels == comp_id
    if mask.sum() < 100:
        continue

    r = plot_index // cols
    c = plot_index % cols
    plotter.subplot(r, c)

    grid = pv.ImageData()
    grid.dimensions = mask.shape
    grid.spacing = (1,1,1)
    grid.point_data["values"] = mask.astype(np.uint8).flatten(order="F")

    surface = grid.contour([0.5])

    plotter.add_mesh(surface, color=colors[comp_id])
    plotter.add_text(f"Sheet {comp_id}", font_size=10)

    plot_index += 1


print("\n--- GPU Info ---")
plotter.show(auto_close=False)

ren_win = plotter.ren_win
print(ren_win.ReportCapabilities())
import slam.io as sio
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

directory = '/home/INT/dienye.h/python_files/Babofet/sub-Borgne/sub-Borgne-seg/sub-Borgne-hemi/new_hemi_meshes_10_smoothing_iterations/left'
analysis_dir = directory

vertices_counts = []
surface_area_values = []
volume_values = []
filenames = []

mL_in_MM3 = 1000
CM2_in_MM2 = 100

# Recursive search, tolerant of naming; tighten the pattern once you confirm it
mesh_files = sorted(Path(directory).rglob('*.gii'))
if not mesh_files:
    raise FileNotFoundError(f"No .gii meshes found under {directory}")

for mesh_file in mesh_files:
    mesh = sio.load_mesh(str(mesh_file))
    num_vertices = len(mesh.vertices)
    volume = np.floor(mesh.volume / mL_in_MM3)
    surface_area = np.floor(mesh.area / CM2_in_MM2)

    vertices_counts.append(num_vertices)
    surface_area_values.append(surface_area)
    volume_values.append(volume)
    filenames.append(mesh_file.name)

# Create DataFrame
info_df = pd.DataFrame({
    'File name': filenames,
    'Number of Vertices': vertices_counts,
    'Surface Area Values': surface_area_values,
    'Volume Values': volume_values
})

if info_df.empty:
    raise ValueError("No meshes were loaded — check the directory and filename filter.")

info_df.to_csv(os.path.join(analysis_dir, 'vertices_surface_area_and_volume.csv'), index=False)

# --- Identify meshes with the highest vertices, volume, and surface area ---
max_vertices_row = info_df.loc[info_df['Number of Vertices'].idxmax()]
max_volume_row   = info_df.loc[info_df['Volume Values'].idxmax()]
max_area_row     = info_df.loc[info_df['Surface Area Values'].idxmax()]

top_meshes_df = pd.DataFrame([
    {
        'Metric':    'Most Vertices',
        'File name': max_vertices_row['File name'],
        'Value':     max_vertices_row['Number of Vertices'],
        'Unit':      'vertices'
    },
    {
        'Metric':    'Largest Volume',
        'File name': max_volume_row['File name'],
        'Value':     max_volume_row['Volume Values'],
        'Unit':      'mL'
    },
    {
        'Metric':    'Largest Surface Area',
        'File name': max_area_row['File name'],
        'Value':     max_area_row['Surface Area Values'],
        'Unit':      'cm²'
    },
])

top_meshes_df.to_csv(os.path.join(analysis_dir, 'top_meshes.csv'), index=False)
print("Top meshes saved to top_meshes.csv")
print(top_meshes_df.to_string(index=False))

# ── Plots ────────────────────────────────────────────────────────────────────

def plot_histogram(df, column, title, xlabel, save_path):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=df, x=column, bins=30,
        color='#2E86C1', alpha=0.8
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count of Meshes', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_histogram(info_df, 'Number of Vertices',  'Distribution of Vertex Counts Across Meshes',       'Number of Vertices',  os.path.join(analysis_dir, 'distribution_of_vertices.png'))
plot_histogram(info_df, 'Surface Area Values', 'Distribution of Surface Area Values Across Meshes', 'Surface Area Values', os.path.join(analysis_dir, 'distribution_of_surface_area.png'))
plot_histogram(info_df, 'Volume Values',       'Distribution of Volume Values Across Meshes',       'Volume Values',       os.path.join(analysis_dir, 'distribution_of_volume.png'))
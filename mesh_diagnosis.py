import nibabel as nib
import numpy as np
import trimesh
import os

OUTPUT_TXT = '/home/INT/dienye.h/python_files/Babofet/sub-Borgne/analysis/mesh_quality_report.txt'

def check_mesh(path, output_txt):
    gii = nib.load(path)
    verts = gii.darrays[0].data
    faces = gii.darrays[1].data

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    lines = [
        f"File: {os.path.basename(path)}",
        f"Vertices: {len(verts)}, Faces: {len(faces)}",
        f"Watertight:            {mesh.is_watertight}",
        f"Winding consistent:    {mesh.is_winding_consistent}",
        f"Degenerate faces:      {(mesh.area_faces == 0).sum()}",
        f"Connected components:  {len(list(trimesh.graph.connected_components(mesh.edges_unique)))}",
        f"Vertex coordinate range: {verts.min():.2f} to {verts.max():.2f} mm",
        "-" * 50,
    ]

    with open(output_txt, 'a') as f:
        f.write('\n'.join(lines) + '\n')

# Clear file before loop so you get a fresh report each run
open(OUTPUT_TXT, 'w').close()

mesh_dir = "/home/INT/dienye.h/python_files/Babofet/sub-Borgne/all_mesh_combined"

for i in os.listdir(mesh_dir):
    path = os.path.join(mesh_dir, i)
    check_mesh(path, OUTPUT_TXT)
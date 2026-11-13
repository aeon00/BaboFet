
import numpy as np
import slam.io as sio
import slam.texture as stex
import slam.plot as splt
from scipy.stats import pearsonr
from collections import Counter, defaultdict
import slam.distortion as sdst


def sphere_normalization(sphere_source, sphere_deform):
    """
    Normalize the deformed sphere so that it has the same maximum radius
    as the source sphere.

    Parameters
    ----------
    sphere_source : trimesh.base.Trimesh
        Reference sphere used for normalization.

    sphere_deform : trimesh.base.Trimesh
        Deformed sphere to be normalized.

    Returns
    -------
    None
        The deformed sphere is modified in place.
    """

    # Compute the maximum distance from the origin to the vertices
    # of the source sphere.
    r_source = np.linalg.norm(sphere_source.vertices, axis=1).max()

    # Compute the maximum distance from the origin to the vertices
    # of the deformed sphere.
    r_out = np.linalg.norm(sphere_deform.vertices, axis=1).max()

    # Compute the scaling factor and apply it to the deformed sphere
    # so that both spheres have the same maximum radius.
    sphere_deform.apply_scale(r_source / r_out)


def vertex_average_from_faces(vertices, faces, face_values, face_areas):
    """
    Distribute face-based values and face areas among their three vertices.

    Each face contributes one third of its value and area to each
    of its three vertices.

    Parameters
    ----------
    vertices : numpy.ndarray
        Array containing the mesh vertex coordinates.

    faces : numpy.ndarray
        Array containing the vertex indices defining each triangular face.

    face_values : numpy.ndarray
        Values associated with each face.

    face_areas : numpy.ndarray
        Areas associated with each face.

    Returns
    -------
    vertex_values : numpy.ndarray
        Accumulated face values assigned to each vertex.

    vertex_areas : numpy.ndarray
        Accumulated face areas assigned to each vertex.
    """

    # Get the total number of vertices in the mesh.
    num_vertices = vertices.shape[0]

    # Initialize arrays to store the accumulated values and areas
    # associated with each vertex.
    vertex_values = np.zeros(num_vertices)
    vertex_areas = np.zeros(num_vertices)

    # Iterate over each triangular face, its associated value and area.
    for (v1, v2, v3), val, A in zip(faces, face_values, face_areas):

        # Assign one third of the face area to each of its vertices.
        share_A = A / 3.0

        # Assign one third of the face value to each of its vertices.
        share_val = val / 3.0

        # Accumulate the contributions at the three vertices.
        for v in (v1, v2, v3):
            vertex_values[v] += share_val
            vertex_areas[v] += share_A

    # Return the accumulated values and areas for each vertex.
    return vertex_values, vertex_areas


def compute_percent_intensities(intensities_in, intensities_out):
    """
    Compute the percentage of distortion relative to the input values.

    Parameters
    ----------
    intensities_in : numpy.ndarray
        Original or reference values.

    intensities_out : numpy.ndarray
        Computed differences or distortion values.

    Returns
    -------
    numpy.ndarray
        Distortion values expressed as percentages of the input values.
    """

    # Express the output values as a percentage of the input values.
    return (intensities_out / intensities_in) * 100


def overlap(sphere_in, sphere_out, parcels):
    """
    Resample a parcellation map from the deformed sphere onto a target sphere
    using a nearest-neighbor approach.

    Parameters
    ----------
    sphere_in : trimesh.base.Trimesh
        Target mesh onto which the parcellation will be projected.

    sphere_out : trimesh.base.Trimesh
        Deformed mesh containing the source vertex positions.

    parcels : numpy.ndarray
        Parcellation labels associated with the vertices of the deformed mesh.

    Returns
    -------
    numpy.ndarray
        Resampled parcellation map associated with the target mesh vertices.
    """

    # Build or retrieve the KD-tree associated with the deformed sphere.
    # This spatial structure allows efficient nearest-neighbor searches.
    kdtree_deformed = sphere_out.kdtree

    # For each vertex of the target sphere, find the closest vertex
    # on the deformed sphere.
    _, closest_vertex_ids = kdtree_deformed.query(sphere_in.vertices)

    # Transfer the parcellation labels using the nearest-neighbor indices.
    return parcels[closest_vertex_ids]


def face_parcels(parcels, faces):
    """
    Assign a parcellation label to each face based on the majority
    label among its three vertices.

    Parameters
    ----------
    parcels : numpy.ndarray
        Parcellation labels associated with mesh vertices.

    faces : numpy.ndarray
        Array containing the vertex indices defining each triangular face.

    Returns
    -------
    numpy.ndarray
        Parcellation labels associated with mesh faces.
    """

    # For each face, retrieve the labels of its three vertices
    # and assign the most frequent label to the face.
    return np.array([
        Counter(parcels[face]).most_common(1)[0][0]
        for face in faces
    ])


def dice(source_parcels, target_parcels):
    """
    Compute Dice similarity coefficients between two parcellation maps.

    The Dice coefficient is calculated separately for each parcel.
    A global vertex-wise agreement percentage is also computed.

    Parameters
    ----------
    source_parcels : numpy.ndarray
        Reference parcellation map.

    target_parcels : numpy.ndarray
        Target parcellation map to compare with the reference.

    Returns
    -------
    dict
        Dictionary containing the Dice scores for each parcel
        and the global agreement percentage under the key "Mean".
    """

    # Initialize dictionaries to store the vertex indices
    # belonging to each parcel in the source and target maps.
    indices_source = defaultdict(list)
    indices_target = defaultdict(list)

    # Group source vertex indices according to their parcel labels.
    for idx, p in enumerate(source_parcels):
        indices_source[p].append(idx)

    # Group target vertex indices according to their parcel labels.
    for idx, p in enumerate(target_parcels):
        indices_target[p].append(idx)

    # Initialize the dictionary containing the Dice scores.
    dice_scores = {}

    # Compute the Dice coefficient independently for each source parcel.
    for p in indices_source:

        # Retrieve the vertex indices associated with parcel p
        # in the source and target maps.
        a = set(indices_source[p])
        b = set(indices_target.get(p, []))

        # Compute the Dice similarity coefficient:
        # Dice = 2 * intersection / (source size + target size).
        # The result is expressed as a percentage.
        dice_scores[p] = (
            2 * len(a & b) / (len(a) + len(b)) * 100
            if (len(a) + len(b)) > 0 else 0
        )

    # Compute the global percentage of vertices having identical
    # parcellation labels in both maps.
    mean_overlap = np.mean(source_parcels == target_parcels) * 100

    # Store the global agreement percentage, rounded to one decimal place.
    dice_scores["Mean"] = round(mean_overlap, 1)

    # Return the Dice scores for all parcels and the global agreement.
    return dice_scores


# ============================================================
base_work = "/envau/work/meca/users/2025_laffite.z"
base_data = "/envau/work/meca"
save_results = "data/results"
hcp_data = "/envau/work/meca/data/HCP_latest/data"
glasser_atlas = "/envau/work/meca/users/2025_laffite.z/data/HCP_datatset/atlas_glasser2016"
dhcp_data = "data/dHCP/rel3_dhcp_anat_pipeline"
dhcp_metadata = "data/dHCP/dhcp_rel3_metadata_official.tsv"
dpf_star_data = "/hpc/meca/data/dHCP_dpf_star"
template_data = "data/dhcpSym_template"
abide_data = "/envau/work/meca/data/ABIDE_database/2026_subsample_for_zoe"
freesurfer_atlas = "/envau/work/meca/data/ABIDE_database/fsaverage"

# Directory containing the registration results.
FOLDER = "/envau/work/meca/users/2025_laffite.z/data/results/dhcp-sub_dhcpSym-dens-32k_corrected/2AFF"

# Hemisphere to analyze.
LOC = "hemi-left"
TEX = "sulc"
WEEK = 34
SUB = "CC00218BN12"
SES = "74800"
# ============================================================

# Path to the original subject spherical mesh.
source_sphere_path = (
    f"{base_data}/{dhcp_data}/sub-{SUB}/ses-{SES}/anat/"
    f"sub-{SUB}_ses-{SES}_{LOC}_sphere.surf.gii"
)

# Path to the target symmetric template spherical mesh.
target_sphere_path = (
    f"{base_work}/{template_data}/week-{WEEK}_{LOC}_"
    "space-dhcpSym_dens-32k_sphere.surf.gii"
)

# Path to the registered subject spherical mesh.
sphere_reg_path = (
    f"{FOLDER}/{WEEK}/sub-{SUB}_ses-{SES}_dhcpSym_dens-32k/"
    f"{LOC}/{TEX}/{LOC}.sphere.reg.surf.gii"
)

# Path to the original subject white matter surface.
source_white_path = (
    f"{base_data}/{dhcp_data}/sub-{SUB}/ses-{SES}/anat/"
    f"sub-{SUB}_ses-{SES}_{LOC}_wm.surf.gii"
)

# Path to the target template white matter surface.
target_white_path = (
    f"{base_work}/{template_data}/week-{WEEK}_{LOC}_"
    "space-dhcpSym_dens-32k_wm.surf.gii"
)

# Path to the original subject sulcal depth texture.
source_sulc_path = (
    f"{base_data}/{dhcp_data}/sub-{SUB}/ses-{SES}/anat/"
    f"sub-{SUB}_ses-{SES}_{LOC}_{TEX}.shape.gii"
)

# Path to the target template sulcal depth texture.
target_sulc_path = (
    f"{base_work}/{template_data}/week-{WEEK}_{LOC}_"
    f"space-dhcpSym_dens-32k_{TEX}.shape.gii"
)

# Path to the registered and resampled sulcal depth texture.
reg_sulc_path = (
    f"{FOLDER}/{WEEK}/sub-{SUB}_ses-{SES}_dhcpSym_dens-32k/"
    f"{LOC}/{TEX}/{LOC}.transformed_and_reprojected.func.gii"
)

# Path to the original subject anatomical parcellation map.
source_parcels_path = (
    f"{base_data}/{dhcp_data}/sub-{SUB}/ses-{SES}/anat/"
    f"sub-{SUB}_ses-{SES}_{LOC}_desc-drawem_dseg.label.gii"
)

# Path to the target template parcellation map (currently unused).
# target_parcels_path = (
#     f"{base_work}/{template_data}/week-{WEEK}_{LOC}_"
#     "space-dhcpSym_dens-32k_desc-drawem_dseg.label.gii"
# )


# ============================================================
source_sphere = sio.load_mesh(source_sphere_path)
target_sphere = sio.load_mesh(target_sphere_path)
sphere_reg = sio.load_mesh(sphere_reg_path)

source_white = sio.load_mesh(source_white_path)
target_white = sio.load_mesh(target_white_path)

source_sulc = sio.load_texture(source_sulc_path).darray[0]
target_sulc = sio.load_texture(target_sulc_path).darray[0]
reg_sulc = sio.load_texture(reg_sulc_path).darray[0]

source_parcels = sio.load_texture(source_parcels_path).darray[0]
# target_parcels = sio.load_texture(target_parcels_path).darray[0]

# ============================================================

# NORMALIZATION
sphere_normalization(source_sphere, sphere_reg)

# JACOBIAN DISTORTION
## Calcul des jacobiens par face
source_face_areas = source_sphere.area_faces
reg_face_areas = sphere_reg.area_faces
jacob_faces = abs(np.log2(reg_face_areas / source_face_areas)) * source_face_areas

## calcul des jacobien par vertex
jacob_vertex, vertex_area = vertex_average_from_faces(
    source_sphere.vertices,
    source_sphere.faces,
    jacob_faces,
    source_face_areas
)

## intégrale des jacobiens
int_vertex = np.sum(jacob_vertex)


# DISTORSIONS
area_diff = sdst.area_difference(sphere_reg, source_sphere)
area_source = source_sphere.area_faces
area_diff_percent = compute_percent_intensities(area_source, np.array(area_diff))

# PEARSON CORR
res_sulc = np.asarray(reg_sulc)
target_sulc = np.asarray(target_sulc)
corr, _ = pearsonr(reg_sulc, target_sulc)

# DICE SCOTRES
# interpolation parcels par plus proches voisins
reg_parcels = overlap(target_sphere, sphere_reg, np.array(source_parcels))

# Obternir les parcels par face et non par vertex
# source_face = face_parcels(source_parcels, source_white.faces)
# target_face = face_parcels(target_parcels, target_white.faces)
# res_face = face_parcels(reg_parcels, target_white.faces)

# dice_scores = dice(target_parcels, reg_parcels)" pour "# Calcul des jacobiens par face
# BaboFet — MSM registration and SPANGY on registered meshes

Longitudinal surface registration of the BaboFet baboon fetal white-matter meshes with
MSM, followed by SPANGY on meshes that share one vertex numbering per animal and
hemisphere, and a timelapse viewer for the SPANGY dominant bands.

- **Code:** `/envau/work/meca/users/dienye.h/BaboFet/msm_registration/` (this folder)
- **Input meshes (read only):** `/envau/work/meca/users/dienye.h/Pymesh/mesh/sub-<NAME>/ses-<NN>/anat/sub-<NAME>_ses-<NN>_manualseg_extract_surf_hemi_<left|right>_wm.gii`
- **All outputs:** `/envau/work/meca/users/dienye.h/python_files/Babofet/`

The original meshes are never modified. Every step writes new files.

---

## Contents

1. [Overview](#overview)
2. [Environment](#environment)
3. [Step 1: prepare MSM inputs](#step-1--prepare-msm-inputs)
4. [Step 2: QC and repairs](#step-2--qc-and-repairs)
5. [Step 3: MSM registration](#step-3--msm-registration-session-to-next-session)
6. [Step 4: chain to one reference session](#step-4--chain-to-one-reference-session-per-animal)
7. [Step 5: SPANGY](#step-5--spangy)
8. [Step 6: timelapse viewer](#step-6--timelapse-viewer)
9. [Output folders](#output-folders)
10. [File reference](#file-reference)
11. [Key concepts](#key-concepts)
12. [Known issues and open points](#known-issues-and-open-points)
13. [Troubleshooting](#troubleshooting)

---

## Overview

```
Pymesh meshes ──prep──▶ sphere + sulc + DPF-star ──QC──▶ MSM (session → next session)
                                                              │  "links": *.sphere.reg.surf.gii
                                                          chaining
                                                              ▼
                                   every session positioned on its animal's reference session
                                                              │
                         ┌────────────────────────────────────┴───────────────────────────┐
                 resample textures                                              resample surfaces
          (SPANGY on original meshes,                                   (surfaces on reference mesh,
           maps moved onto reference)                                      then SPANGY on them)
                         └────────────────────────────────────┬───────────────────────────┘
                                                              ▼
                                       vertex-wise maps per animal / hemisphere + timelapse
```

The reference session is each animal's **latest** session, the oldest gestational age and
the most folded brain (e.g. Aziza L → ses-07, Bibi → ses-10). Use `REF=first` in the
chaining step for the earliest session instead.

### Quick start (full run)

```bash
cd /envau/work/meca/users/dienye.h/BaboFet/msm_registration

# 1. prepare spheres, sulc, DPF-star (one SLURM task per surface)
bash run_prep_msm_slurm.sh check
bash run_prep_msm_slurm.sh list
bash run_prep_msm_slurm.sh submit

# 2. QC, repair folded spheres
bash rerun_qc.sh
bash fix_failed_spheres.sh <prefix> [<prefix> ...]

# 3. register each session to the next (sulc and DPF-star)
bash run_msm_slurm.sh check
TEXTURE=both bash run_msm_slurm.sh list
TEXTURE=both bash run_msm_slurm.sh submit
TEXTURE=both bash run_msm_slurm.sh summary

# 4. chain to the reference session
bash chain_to_reference.sh                       # sulc
TEXTURE=dpfstar bash chain_to_reference.sh       # DPF-star

# 5. surfaces on the reference mesh, then SPANGY on them
export NEED_FREESURFER=0; source env_setup.sh
$PYTHON resample_to_ref.py                       # sulc
$PYTHON resample_to_ref.py --texture dpfstar     # DPF-star
sbatch --export=ALL,SPANGY_ARGS=--onref run_spangy_pymesh.sh
sbatch --export=ALL,SPANGY_ARGS="--onref --reg-texture dpfstar" run_spangy_pymesh.sh

# 6. timelapse (build on the cluster, view locally)
$PYTHON build_spangy_timelapse.py --onref
python view_spangy_timelapse.py <file>.npz       # on your own machine
```

---

## Environment

All cluster jobs source **`env_setup.sh`**. It does, in this order (the order that works
on the INT cluster):

1. `module purge`, `module load all`, then `FSL/6.0.6.2` (for `msm`) and `workbench/1.5.0`.
2. FreeSurfer (`freesurfer/7.4.1` by default, override with `FS_MODULE`), only when
   `NEED_FREESURFER=1`, which is the default.
3. FSL setup (`fsl.sh`), **before** conda. FSL ships its own `python3`, which would
   otherwise hide the conda one.
4. `source ~/.bashrc`, `conda activate babofet` (override with `CONDA_ENV`).
5. Exports `PYTHON=$CONDA_PREFIX/bin/python`. Every Python step runs with `$PYTHON`.
6. Finds the FreeSurfer licence (`FS_LICENSE`, auto-detected in `$FREESURFER_HOME`).
7. Checks every required tool and that `nibabel` and `slam` import; fails with a clear
   message otherwise.

| Variable | Default | Meaning |
|---|---|---|
| `NEED_FREESURFER` | `1` | load FreeSurfer and check the `mris_*` tools (only the prep step needs it) |
| `NEED_MSM` | `0` | check that `msm` is on PATH |
| `FS_MODULE` | `freesurfer/7.4.1` (set in `run_prep_msm_slurm.sh`) | FreeSurfer module name |
| `FS_LICENSE` | auto | FreeSurfer licence file |
| `CONDA_ENV` | `babofet` | conda env with slam + nibabel |

Check the environment on the login node before submitting anything:

```bash
bash run_prep_msm_slurm.sh check      # prep: FreeSurfer, Workbench, python
bash run_msm_slurm.sh check           # registration: msm, Workbench, python
```

For manual work in a shell: `export NEED_FREESURFER=0; source env_setup.sh`, then use `$PYTHON`.

---

## Step 1 — Prepare MSM inputs

**Scripts:** `run_prep_msm_slurm.sh` (batch), `prep_msm_surface.sh` (one surface),
`compute_dpf_star.py`, `dpf_star_lib.py`, `qc_msm_inputs.py`

```bash
bash run_prep_msm_slurm.sh list     # builds msm_inputs/joblist.txt from the Pymesh folders
bash run_prep_msm_slurm.sh submit   # SLURM array, 1 task per surface (1 CPU, 8 GB, 4 h)
```

For each hemisphere surface, `prep_msm_surface.sh`:

1. Converts the GIFTI to FreeSurfer format, smooths it lightly (3 iterations), **inflates** it
   (this also produces **sulc**) and maps it to a **sphere** (`mris_sphere`, the slow step).
2. Converts back to GIFTI. Sulc is converted with nibabel, because `mris_convert -c`
   prepends `lh.`/`rh.` to the output name.
3. Recentres the sphere at the origin with **radius 100**, as MSM expects
   (`wb_command -surface-modify-sphere`).
4. Sets hemisphere and surface-type metadata, computes mean curvature and the
   anatomy→sphere areal distortion.
5. Computes **DPF-star** on the original mesh (`compute_dpf_star.py`, alpha 500,
   `volume_hull`). It uses `dpf_star` from the installed slam if available, otherwise the
   bundled copy `dpf_star_lib.py`.
6. Runs QC (`qc_msm_inputs.py`).

**Why spheres:** MSM registers spheres, not folded surfaces. Every sphere has the same
shape, so it serves as a common space. The sphere is **the same mesh** as the original:
same vertices, same order, same triangles, only the coordinates differ. So every texture
computed on the original mesh is valid on the sphere.

**Outputs:** `msm_inputs/<sub>/<ses>/<sub>_<ses>_hemi-<L|R>.*`

| File | Use |
|---|---|
| `.sphere.surf.gii` | `--inmesh` / `--refmesh` for MSM |
| `.sulc.func.gii` | registration texture (FreeSurfer convention: sulci positive) |
| `.dpfstar.func.gii` (+ `.dpfstar.func.json`, which holds alpha, adaptation and scale `lc`) | alternative registration texture (opposite sign to sulc, r ≈ −0.9) |
| `.anat.surf.gii` | copy of the original mesh, `--inanat` / `--refanat` |
| `.inflated.surf.gii` | viewing only |
| `.curv.func.gii` | mean curvature (QC / evaluation) |
| `.sphere_distortion.func.gii` | areal distortion anatomy → sphere |
| `.qc.txt` | QC report |
| `fswork_<prefix>/` | FreeSurfer working files. A rerun reuses them; set `REDO_FS=1` to recompute |

Options (environment variables) for `prep_msm_surface.sh`: `SMOOTH_ITERS` (3),
`SPHERE_QUICK` (0), `SULC_HCP_SIGN` (0), `SEED` (1234), `KEEP_WORK` (1), `REDO_FS` (0),
`COMPUTE_DPFSTAR` (1), `DPF_ALPHA` (500), `DPF_ADAPTATION` (`volume_hull`).

---

## Step 2 — QC and repairs

**Scripts:** `qc_msm_inputs.py`, `rerun_qc.sh`, `fix_sphere_flips.py`, `fix_failed_spheres.sh`

```bash
bash rerun_qc.sh                 # re-check every prepared surface, PASS/FAIL summary
bash rerun_qc.sh sub-Aziza       # one animal
```

QC checks:
- the anatomy and sphere have the same vertices and triangles;
- the surface is closed and **genus 0** (Euler characteristic 2);
- the sphere is **centred with radius 100**, using a least-squares sphere fit (the
  average of vertex positions is not a reliable centre, because `mris_sphere` spreads
  vertices unevenly);
- **no folded-over triangles** on the sphere;
- no NaNs, and the correlation between sulc and DPF-star.

A folded sphere does not make MSM fail; it silently gives wrong correspondences.
**Surfaces that fail QC are excluded automatically from registration.**

**Repairing folded spheres** (surface genus 0, but a few flipped triangles):

```bash
bash fix_failed_spheres.sh sub-Aziza_ses-08_hemi-L sub-Fabienne_ses-08_hemi-L sub-Fabienne_ses-10_hemi-L
```

`fix_sphere_flips.py` moves only the vertices in and around the folded patches to the
average of their neighbours, projects them back onto the sphere, and repeats until nothing
is folded. Vertex order and textures are unchanged. The original sphere is kept as
`<prefix>.sphere_orig.surf.gii`; distortion and QC are recomputed. If a large region has
to move, rerunning `mris_sphere` with another seed is cleaner: delete `fswork_<prefix>/`,
then run with `SEED=4321`.

**Not repairable here:** surfaces that fail the genus-0 check. They have handles from
segmentation errors; fix the segmentation, re-extract the surface, rerun the prep with `REDO_FS=1`.

---

## Step 3 — MSM registration (session to next session)

**Scripts:** `run_msm_slurm.sh`, `msm_baboon_pairwise.conf` (`run_msm_example.sh` and
`msm_strain_sulc.conf` are kept as a single-pair example)

```bash
bash run_msm_slurm.sh check
TEXTURE=both bash run_msm_slurm.sh list      # pair list, skips surfaces whose QC did not pass
TEXTURE=both bash run_msm_slurm.sh submit    # 1 task per pair (1 CPU, 8 GB, 8 h)
TEXTURE=both bash run_msm_slurm.sh summary   # registration QC table
```

`TEXTURE=sulc` (default), `dpfstar` or `both` (every pair once per texture).

Each usable session is registered to the **next usable session** of the same animal and
hemisphere (ses-01 → ses-02 → …). Excluded sessions are jumped over (e.g. Aziza L
ses-05 → ses-07). Adjacent sessions look the most alike, so each step is the most
accurate possible.

**Configuration (`msm_baboon_pairwise.conf`)**, adapted from the group config:
- 2 affine levels, then 3 non-rigid levels with control-point grids at icosphere levels 2/3/4;
- **equal smoothing** on input and reference, since both are individual brains (the
  original config smoothed the reference less, as for an averaged template);
- **λ = 0, 0, 0.2, 0.3, 0.4** (raised from 0.1/0.2/0.3), because folds appear between
  sessions and weak regularisation would stretch the surface to chase them.

**Outputs:** `msm_registered/<texture>/<sub>/hemi-<H>/<sub>_hemi-<H>_ses-AA_to_ses-BB.*`

| File | What it is |
|---|---|
| `.sphere.reg.surf.gii` | **the result, the "link"**: ses-AA's sphere (ses-AA's vertices) with each vertex moved to where it corresponds on ses-BB's sphere |
| `.transformed_and_reprojected.func.gii` | MSM's own check: ses-AA texture on ses-BB's mesh |
| `.sphere.LR.reg.surf.gii`, `.logdir/` | MSM internal low-resolution output and logs |
| `.reg_distortion.func.gii` | areal distortion caused by the registration |
| `.<texture>.on_ref.before/after.func.gii` | input texture on the reference mesh before/after MSM |
| `.regqc.txt` | `corr_before`, `corr_after`, `distortion_p95`, `distortion_max` |

Pair lists: `msm_registered/pairlist_<TEXTURE>.txt`. Logs:
`msm_registered/logs/msm_reg_<TEXTURE>_<jobid>_<task>.out`. The task number is the line
number in the pair list (`cat -n pairlist_both.txt`).

**Reading `summary`:** `after` should be clearly above `before`. Low `before` values
(~0.5–0.6) are mostly orientation differences that the affine step corrects. A pair
whose distortion is far above the others deserves a look in `wb_view`.

**First results (74 pairs per texture):**
- mean `corr_after` 0.970 (sulc) / 0.956 (DPF-star);
- mean p95 distortion 0.53 / 0.60;
- pairs that skip a session have higher distortion (p95 0.79 vs 0.46).

Each registration is scored on its own texture, so this is **not** a fair comparison of
sulc vs DPF-star (see open points).

---

## Step 4 — Chain to one reference session per animal

**Script:** `chain_to_reference.sh`

```bash
bash chain_to_reference.sh                   # all animals, sulc registrations
bash chain_to_reference.sh sub-Aziza         # one animal
TEXTURE=dpfstar bash chain_to_reference.sh   # DPF-star registrations
REF=first bash chain_to_reference.sh         # reference = earliest session
```

Combines the links so every session maps directly onto the reference session, using
`wb_command -surface-sphere-project-unproject` (no new MSM run). One step, from session a
to session b:

- **forward:** project the current sphere onto `sphere_a`, then unproject from `reg(a→b)`;
  the sphere is now in b's space;
- **backward** (for `REF=first`): project onto `reg(a→b)`, then unproject from `sphere_a`.

Each landing point lies inside a triangle of one sphere; the next link says where that
triangle went, and the point keeps its position within the triangle.

An animal whose links do not form one unbroken chain is skipped with a warning.

**Outputs:** `msm_chained/<texture>/<sub>/hemi-<H>/`

| File | What it is |
|---|---|
| `<sub>_hemi-<H>_ses-XX_in_ses-RR.sphere.surf.gii` | ses-XX's vertices positioned on the reference's sphere. **Used by everything after this step.** |
| `…_ses-XX_in_ses-RR.distortion.func.gii` | areal distortion of the whole chain |
| `…_ses-XX_in_ses-RR.<texture>.on_ref.func.gii` | QC: ses-XX texture on the reference mesh |
| `<sub>_hemi-<H>_chain_qc.txt` | per session: links, correlation with reference, distortion |

Early sessions correlate less with the reference, because the brain really is smoother
then. Look for sudden drops or distortion jumps at one session instead: they indicate a weak link.

**Vertex counts:** every sphere keeps its own session's vertex count. Registration and
chaining move vertices; they never add or remove them. Only resampling (Step 5) puts data
on a common vertex set.

---

## Step 5 — SPANGY

**Scripts:** `spangy_babofet_pymesh.py`, `run_spangy_pymesh.sh`, `resample_to_ref.py`

There are two routes to SPANGY results that share one vertex numbering per
animal/hemisphere:

| | Route A: resample the textures | Route B: resample the surfaces |
|---|---|---|
| SPANGY runs on | the original Pymesh meshes | each session's surface resampled onto the reference mesh |
| Interpolated | the finished band maps | the geometry, before SPANGY |
| Pros | SPANGY results exact | all outputs share one vertex numbering, nothing to resample afterwards |
| Cons | band labels can shift at band borders | fine folds slightly flattened (B5–B6), triangle quality can drop |

Both use the same registration, so the **vertex correspondence is identical**. Keep the
global band powers from route A for reporting, since they involve no interpolation.

### Route A: SPANGY on the original meshes

```bash
python spangy_babofet_pymesh.py --list                 # 90 meshes expected
sbatch run_spangy_pymesh.sh                            # 30 tasks (3 meshes each), ≤10 at once
python spangy_babofet_pymesh.py --merge                # per-subject + pooled CSVs
```

Output: `<sub>/analysis/spangy_pymesh/` (textures `spangy_dom_band_<sub>_<ses>_hemi-<H>.gii`,
mean/principal curvature, spectra plots, per-file JSON, CSVs); pooled table
`analysis_pooled/all_subjects_spangy_pymesh_results.csv`.

`spangy_babofet_pymesh.py` is the BaboFet SPANGY script, adapted to the Pymesh layout.
The SPANGY computation is unchanged: 4000 eigenpairs, z-filtered mean curvature,
spectrum, local dominance map. It applies the principal-inertia rotation, which keeps
vertex order, and checks that the dominance map has one value per vertex.

The older SPANGY outputs in `<sub>/analysis/spangy/` were computed on **different meshes**
and do not match the registered meshes. Don't use them with this pipeline.

### Route B: SPANGY on surfaces resampled onto the reference mesh

```bash
export NEED_FREESURFER=0; source env_setup.sh
$PYTHON resample_to_ref.py                         # sulc chain → surfaces_on_ref/sulc/
$PYTHON resample_to_ref.py --texture dpfstar       # DPF-star chain → surfaces_on_ref/dpfstar/

python spangy_babofet_pymesh.py --onref --list
sbatch --export=ALL,SPANGY_ARGS=--onref run_spangy_pymesh.sh
sbatch --export=ALL,SPANGY_ARGS="--onref --reg-texture dpfstar" run_spangy_pymesh.sh
python spangy_babofet_pymesh.py --onref --merge                       # (add --reg-texture dpfstar)
```

`resample_to_ref.py` writes `surfaces_on_ref/<texture>/<sub>/hemi-<H>/<sub>_<ses>_hemi-<H>_on_<ref>.anat.surf.gii`:
each session's **own shape** on the reference mesh's vertices and triangles. It also
writes `<sub>_hemi-<H>_resample_qc.txt`, which you should check before SPANGY:
- `area_ratio` (after/before) should be ≈ 1. A drop means folds were flattened.
- `bad_tri_%` is the share of triangles with a minimum angle < 5°, next to the original
  mesh's value. A large rise means the triangulation fits that session poorly.

SPANGY output: `<sub>/analysis/spangy_onref_<texture>/`; pooled
`analysis_pooled/all_subjects_spangy_onref_<texture>_results.csv`.

### Moving any texture onto the reference mesh by hand

```bash
wb_command -metric-resample  tex_sesXX.func.gii \
    msm_chained/sulc/<sub>/hemi-L/<sub>_hemi-L_ses-XX_in_ses-RR.sphere.surf.gii \
    msm_inputs/<sub>/ses-RR/<sub>_ses-RR_hemi-L.sphere.surf.gii \
    BARYCENTRIC  out.func.gii  -largest            # categorical (band labels)
# continuous values: ADAP_BARY_AREA out.func.gii -area-surfs <sesXX anat> <ses-RR anat>
```

---

## Step 6 — Timelapse viewer

**Scripts:** `build_spangy_timelapse.py` (cluster), `view_spangy_timelapse.py` (local, PyVista)

### Build (once, on the cluster)

```bash
export NEED_FREESURFER=0; source env_setup.sh
$PYTHON build_spangy_timelapse.py --check              # route A: does every texture match its mesh?
$PYTHON build_spangy_timelapse.py                      # route A: resample maps + surfaces
$PYTHON build_spangy_timelapse.py --save-gii           # also keep the resampled GIFTIs (spangy_on_ref/)
$PYTHON build_spangy_timelapse.py --onref              # route B: nothing resampled
$PYTHON build_spangy_timelapse.py --onref --texture dpfstar
```

The script writes one `.npz` per animal/hemisphere in `spangy_timelapse/`:
`<sub>_hemi-<H>_<kind>_<texture>[_onref]_on_<ref>.npz`. It holds the reference mesh's
triangles, every session's anatomy on the reference vertices (so the shape can morph),
and every session's texture.

Options:
- `--kind label` (default, band labels, resampled with `-largest` so they stay whole numbers)
  or `--kind continuous` (area-weighted, for band power or curvature);
- `--pattern` to read other textures;
- `--sub` for one animal.

Sessions whose texture vertex count doesn't match the registered mesh are skipped. The
message says whether it looks like a left/right swap or a different mesh.

### View (locally)

```bash
python view_spangy_timelapse.py sub-Bibi_hemi-L_label_sulc_onref_on_ses-10.npz
python view_spangy_timelapse.py FILE.npz --show sulcal          # start with sulcal bands only
python view_spangy_timelapse.py FILE.npz --geometry ref         # bands on the fixed reference brain
python view_spangy_timelapse.py FILE.npz --same-size            # shape change without growth
python view_spangy_timelapse.py FILE.npz --export movie.mp4     # write a movie (needs imageio-ffmpeg)
```

| Control | Action |
|---|---|
| mouse | rotate / zoom / pan at any time, also while playing |
| **Space** / button (bottom left) | play / pause (it starts playing; `--paused` to start stopped) |
| slider | jump to any time, including between sessions (pauses) |
| **← / →** | previous / next session (pauses) |
| **↑ / ↓** | faster / slower |
| **B** | bands shown: both → sulcal only → gyral only |
| **L** | loop on/off |
| **R** | reset the view |

Colours: B3 gold, B4 blue, B5 green, B6 red, below B3 grey. In "both" mode, sulcal
vertices are drawn darker. Sulcal/gyral follows the sign of slam's local dominance map
(negative where the analysed curvature ≤ 0).

Frames between sessions are display blends: the shape is interpolated and the bands
switch at the midpoint. **Read data at whole sessions only** (←/→).

---

## Output folders

All under `/envau/work/meca/users/dienye.h/python_files/Babofet/`:

| Folder | Step | Contents |
|---|---|---|
| `msm_inputs/<sub>/<ses>/` | 1 | spheres, sulc, DPF-star, anat copy, QC, `fswork_*` |
| `msm_inputs/joblist.txt`, `msm_inputs/logs/` | 1 | prep job list and logs |
| `msm_registered/<texture>/<sub>/hemi-<H>/` | 3 | MSM links and registration QC |
| `msm_registered/pairlist_*.txt`, `msm_registered/logs/` | 3 | pair lists and logs |
| `msm_chained/<texture>/<sub>/hemi-<H>/` | 4 | each session on the reference sphere, chain QC |
| `surfaces_on_ref/<texture>/<sub>/hemi-<H>/` | 5B | surfaces resampled onto the reference mesh, resample QC |
| `<sub>/analysis/spangy_pymesh/` | 5A | SPANGY on original Pymesh meshes |
| `<sub>/analysis/spangy_onref_<texture>/` | 5B | SPANGY on reference-mesh surfaces |
| `analysis_pooled/` | 5 | pooled SPANGY CSVs |
| `spangy_timelapse/` | 6 | `.npz` files for the viewer |
| `spangy_on_ref/<texture>/` | 6 (`--save-gii`) | resampled textures and surfaces as GIFTI |
| `logs/` | 5 | SPANGY job logs (`spangy_pymesh_<job>_<task>.out/.err`) |

---

## File reference

| File | Runs on | Purpose |
|---|---|---|
| `env_setup.sh` | sourced | modules, conda env, FreeSurfer licence, tool checks; exports `$PYTHON` |
| `run_prep_msm_slurm.sh` | login → SLURM | Step 1 batch: `check` / `list` / `submit` |
| `prep_msm_surface.sh` | SLURM task | Step 1 for one surface |
| `compute_dpf_star.py` | called by prep | DPF-star texture on one mesh |
| `dpf_star_lib.py` | imported | fallback copy of the DPF / DPF-star implementation |
| `qc_msm_inputs.py` | called by prep / QC scripts | QC of one prepared surface |
| `rerun_qc.sh` | login | QC of all prepared surfaces, PASS/FAIL summary |
| `fix_sphere_flips.py` | called by fix script | removes folded triangles from one sphere |
| `fix_failed_spheres.sh` | login | repair + recompute distortion + QC for listed surfaces |
| `msm_baboon_pairwise.conf` | MSM config | session-to-session registration (used by default) |
| `msm_strain_sulc.conf` | MSM config | HCP-style strain config (reference/example) |
| `run_msm_example.sh` | login | single-pair MSM example (`TEXTURE=`, `CONF=`) |
| `run_msm_slurm.sh` | login → SLURM | Step 3: `check` / `list` / `submit` / `summary` |
| `chain_to_reference.sh` | login | Step 4 |
| `resample_to_ref.py` | login | Step 5B: surfaces onto the reference mesh + QC |
| `spangy_babofet_pymesh.py` | SLURM | Step 5: SPANGY (`--onref`, `--reg-texture`, `--list`, `--merge`, `--force`) |
| `run_spangy_pymesh.sh` | SLURM | Step 5 job array (pass options via `SPANGY_ARGS`) |
| `build_spangy_timelapse.py` | login | Step 6: build `.npz` (`--onref`, `--check`, `--save-gii`, `--kind`, `--pattern`) |
| `view_spangy_timelapse.py` | local | Step 6: interactive viewer / movie export |

---

## Key concepts

- **Sphere = your mesh.** The prep moves vertices onto a sphere, without adding, removing
  or reordering any. Vertex N on the sphere is vertex N of the original mesh.
- **Link** (`sphere.reg`). One MSM output: the input session's sphere with each vertex moved
  to its corresponding position on the next session's sphere. It is a lookup table and
  changes no mesh.
- **Chaining.** Composing links (01→02, 02→04, …) into one mapping per session onto the
  reference session. Errors add up along the chain.
- **Resampling.** Moving data, or a surface's geometry, from one mesh onto another,
  using two aligned spheres. Labels use the nearest/largest weight (`-largest`);
  continuous values use area-weighted barycentric interpolation (`ADAP_BARY_AREA`).
- **Registration vs analysis texture.** Registration is driven by sulc or DPF-star.
  SPANGY analyses mean curvature. Evaluate registrations on a feature that was not used to
  drive them.
- **Within vs across animals.** Everything here is *within* each animal. Comparing or
  pooling animals needs a further registration of each animal's reference session to a
  common template (not implemented yet).

---

## Known issues and open points

- **Excluded surfaces:**
  - `sub-Aziza_ses-06_hemi-L` is not genus 0 (Euler −10, 6 handles). Its segmentation
    needs fixing; until then its chain jumps ses-05 → ses-07.
  - `sub-Aziza_ses-08_hemi-L`, `sub-Fabienne_ses-08_hemi-L` and `sub-Fabienne_ses-10_hemi-L`
    have folded spheres. Repair them with `fix_failed_spheres.sh`, then register their pairs.
- **Weakest registrations:** pairs that skip a session (Fabienne L and Aziza L ses-05→07,
  Borgne ses-01→03 and ses-07→09). Inspect `reg_distortion` maps before relying on them.
- **sulc vs DPF-star:** not settled. Compare both on an independent feature, such as
  curvature, or the consistency of SPANGY maps across consecutive sessions.
- **λ tuning:** run one early and one late pair at ×0.5 / ×1 / ×2 λ; choose where
  correlation stops improving before distortion rises.
- **Route A vs route B:** compare band powers and dominance maps between `spangy_pymesh` and
  `spangy_onref_<texture>` on a few sessions.
- **Across-animal template:** not built yet (needed for pooled vertex-wise analyses).
- **Timelapse viewer:** the colour and interpolation logic is tested; the PyVista window
  and movie export have not yet been run on a real file.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unable to locate a modulefile` | wrong module name; check `module avail 2>&1 \| grep -i <name>`, set `FS_MODULE` |
| `MISSING: nibabel/slam` while they are installed | wrong Python; `env_setup.sh` uses `$CONDA_PREFIX/bin/python`, so always call `$PYTHON` |
| sulc file named `lh.…`/`rh.…` | old prep version; the current one converts sulc with nibabel |
| QC "sphere not centred" on many surfaces | old QC version (vertex-mean centre); use the current `qc_msm_inputs.py` |
| `input metric has different number of nodes than input sphere` | texture computed on a different mesh, or left/right swapped; run `build_spangy_timelapse.py --check` |
| bash shows a `>` prompt after a command | unclosed quote; press Ctrl+C and fix the quoting |
| MSM job timeout | dense mesh (e.g. Borgne ses-01); raise `--time` in the `submit` block of `run_msm_slurm.sh` |
| failed/missing registrations | `grep -L "\[msm\] done" msm_registered/logs/msm_reg_*.out` |

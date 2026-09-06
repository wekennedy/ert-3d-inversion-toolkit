# -*- coding: utf-8 -*-
"""
True3DInversionMain.py
-----------------------
Entry point for true 3D ERT inversion with pyGIMLi, driven directly from
native 3D Terrameter LS/LS2 exports (not the collinear 2D pseudo-sections
those exports are normally reduced to). Companion module:
True3DInversionUtilities.py (imported below as `u`).

Workflow (see the numbered steps printed by main()):
    1. read and merge one or many native 3D exports by electrode coordinate
    2. compute geometric factors for the real 3D electrode layout
    3. quality control: reciprocal errors, optional automatic filters,
       optional interactive QC, error model, optional reciprocal stacking
    4. build one 3D PLC + tetrahedral mesh, reused across every inversion
    5. run the inversion sweep over every (cType, zWeight, lambda) combination,
       then write misfit/metadata CSVs for the successful runs
    6. plot the L-curve, report the chi2-closest-to-1 model, and export
       VTKs (and coverage-slice plots) for ParaView

Run:  python True3DInversionMain.py
"""

import os
import csv
import glob
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

# --- TetGen -----------------------------------------------------------------
# Locate a TetGen install so mesh generation works even where TetGen is not on
# the system PATH: try, in order, (1) an extracted TetGen CLI binary next to
# this script, (2) a locally built binary under a cloned tetgen/build, and
# (3) a cloned tetgen source tree added to sys.path. TETGEN_DIR records which
# one (if any) was found, and is passed on to ensure_tetgen()/build_3d_mesh()
# further down.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TETGEN_ROOT = os.path.join(SCRIPT_DIR, 'tetgen')
LOCAL_TETGEN_BUILD = os.path.join(LOCAL_TETGEN_ROOT, 'build')
LOCAL_TETGEN_BIN = os.path.join(SCRIPT_DIR, 'tetgen_bin', 'usr', 'bin')
TETGEN_DIR = None

# Option 1: an already-extracted TetGen CLI binary.
if os.path.isdir(LOCAL_TETGEN_BIN) and os.path.isfile(os.path.join(LOCAL_TETGEN_BIN, 'tetgen')):
    os.environ['PATH'] = LOCAL_TETGEN_BIN + os.pathsep + os.environ.get('PATH', '')
    TETGEN_DIR = LOCAL_TETGEN_BIN
    print(f"  [tetgen] using extracted TetGen binary at {LOCAL_TETGEN_BIN}")

# Option 2: a binary built locally from the cloned tetgen source.
if TETGEN_DIR is None and os.path.isdir(LOCAL_TETGEN_BUILD):
    if os.path.isfile(os.path.join(LOCAL_TETGEN_BUILD, 'tetgen')):
        os.environ['PATH'] = LOCAL_TETGEN_BUILD + os.pathsep + os.environ.get('PATH', '')
        TETGEN_DIR = LOCAL_TETGEN_BUILD
        print(f"  [tetgen] using built local tetgen at {LOCAL_TETGEN_BUILD}")

# Option 3: fall back to importing the tetgen Python package from source.
if TETGEN_DIR is None and os.path.isdir(LOCAL_TETGEN_ROOT):
    if os.path.isdir(os.path.join(LOCAL_TETGEN_ROOT, 'src')):
        sys.path.insert(0, os.path.join(LOCAL_TETGEN_ROOT, 'src'))
    sys.path.insert(0, LOCAL_TETGEN_ROOT)
    os.environ['PATH'] = LOCAL_TETGEN_ROOT + os.pathsep + os.environ.get('PATH', '')
    TETGEN_DIR = LOCAL_TETGEN_ROOT
    print(f"  [tetgen] using local clone at {LOCAL_TETGEN_ROOT}")

import tetgen
import pygimli as pg
import pygimli.physics.ert as ert

# All survey-processing, meshing and inversion helpers live in the companion
# utilities module, imported once here as `u` and re-bound to local names
# below.
import True3DInversionUtilities as u

# Minimum True3DInversionUtilities version this script was written against,
# and the exact set of names it expects to find in that module. Both are
# checked immediately below so a stale or incompatible utilities file fails
# fast with a clear message instead of an obscure AttributeError deep inside
# main().
REQUIRED_UTILS_VERSION = (2, 0)
_NEEDED = ['load_3d_survey', 'compute_geometric_factors', 'apply_filters',
           'interactive_qc', 'set_error_model', 'reciprocal_errors',
           'fit_reciprocal_error_model',
           'stack_reciprocals', 'check_3d_coupling', 'build_3d_mesh',
           'run_inversion_3d', 'plot_survey_3d', 'plot_coverage_slices',
           'plot_lcurve', 'ensure_tetgen',
           'apply_electrode_layout', 'report_address_blocks', 'Survey',
           'make_line_layout']

# Compare the utilities module's own __version__ (major, minor) against the
# required version, and check every name in _NEEDED is actually defined
# there; either failing raises with a message naming the file, its version,
# and any missing functions.
_have = tuple(int(v) for v in getattr(u, '__version__', '0').split('.')[:2])
_missing = [n for n in _NEEDED if not hasattr(u, n)]
if _have < REQUIRED_UTILS_VERSION or _missing:
    raise ImportError(
        f"\n\nThe utilities file is out of date.\n"
        f"  file    : {getattr(u, '__file__', '?')}\n"
        f"  version : {getattr(u, '__version__', 'unknown')} "
        f"(this script needs {'.'.join(map(str, REQUIRED_UTILS_VERSION))})\n"
        + (f"  missing : {', '.join(_missing)}\n" if _missing else "")
        + "Replace it with the matching version and re-run.\n")

# Local names for the utilities functions used below, so the rest of this
# file can call them directly instead of through `u.`.
load_3d_survey = u.load_3d_survey
compute_geometric_factors = u.compute_geometric_factors
apply_filters = u.apply_filters
interactive_qc = u.interactive_qc
set_error_model = u.set_error_model
reciprocal_errors = u.reciprocal_errors
fit_reciprocal_error_model = u.fit_reciprocal_error_model
stack_reciprocals = u.stack_reciprocals
check_3d_coupling = u.check_3d_coupling
build_3d_mesh = u.build_3d_mesh
run_inversion_3d = u.run_inversion_3d
plot_survey_3d = u.plot_survey_3d
plot_coverage_slices = u.plot_coverage_slices
make_line_layout = u.make_line_layout
plot_lcurve = u.plot_lcurve

# plot_lcurve (used in step 6 below) lives in True3DInversionUtilities and
# writes figures with matplotlib's savefig, so it works unattended on a
# headless machine with no display.


# =====================================================================
# ============================ CONFIG =================================
# =====================================================================

# ---- input ----------------------------------------------------------
DATA_DIR = r'./data'   # EDIT THIS: folder holding your native 3D Terrameter exports

# One or more native 3D export files, merged into a single survey by
# electrode coordinate (so protocol files, roll-alongs and crossing lines can
# be listed in any order). A glob pattern also works here, e.g.
# DATA_FILES = os.path.join(DATA_DIR, '3D_*.txt')
DATA_FILES = [
    os.path.join(DATA_DIR, 'survey_3d.txt'),   # EDIT THIS: your export filename(s)
   # os.path.join(DATA_DIR, 'survey_3d_line2.txt'),
]

# Only needed when the export records electrode NUMBERS rather than
# coordinates: an 'id,x,y,z' CSV or an ABEM Spread XML that resolves those
# numbers to positions. Leave as None for coordinate-style exports.
ELEC_LOOKUP_FILE = None

# Explicit column-position map for the export's ASCII table, e.g.
# {'ax':4,'ay':5,'az':6,'bx':8,...,'r':28,'err':29} (0-based positions).
# Leave as None to auto-detect the delimiter, header row and columns.
COLMAP = None

# Reposition individual files before merging, keyed by file basename, e.g.
#   TRANSFORMS = {'survey_3d_line2.txt': dict(offset=(0.0, 1.5, 0.0))}
# Only needed when a file's own coordinates need shifting/rotating before the
# merge (for example if the instrument recorded every line in the same local
# coordinates). Leave empty when the exports already carry true survey
# coordinates.
TRANSFORMS = {}

# Two electrode entries within this distance (m) are treated as the same
# physical electrode when merging files. Keep it comfortably below the real
# electrode spacing but above the rounding noise in the export.
ELECTRODE_TOL = 0.05          # m

# Is the error column a percentage (e.g. the LS 'Var %' column) or already a
# fraction (0.02 == 2%)?
ERR_IS_PERCENT = True

# ---- output ---------------------------------------------------------
OUTPUT_DIR = r'./output/3D_inversion'   # EDIT THIS: where results should be written
BASE_NAME = 'survey3D'

# ---- geometric factors ----------------------------------------------
# False = analytical geometric factor for a 3D half-space (flat ground).
# True  = numerical k computed on a homogeneous mesh instead (needed once
#         real topography is involved).
NUMERICAL_K = False

# ---- electrode layout -------------------------------------------------
# Rebuilds electrode coordinates from the switch/take-out ADDRESSES recorded
# in the export rather than trusting the coordinates the instrument wrote,
# since the address is what identifies the physical take-out regardless of
# how the spread was defined.
#
# LINE_RUNS gives the address at each END of each line, in the order you walk
# along it, one tuple per line, lines listed in offset order. Write it exactly
# as you would on a field sheet - a cable that doubled back is just a reversed
# tuple, no special flag needed. Below: 4 lines of 16 electrodes each.
LINE_RUNS = [(1, 16),
             (32, 17),      # line 1, addresses counting up
             (33, 48),
             (64, 49)]     # line 4, doubled back
ELEC_SPACING = 0.5            # m between electrodes along each line
LINE_SPACING = 0.5            # m between adjacent lines
GRID_ORIGIN = (0, 0, 0.0)  # position of the first electrode of line 1

# Built once here from LINE_RUNS/ELEC_SPACING/LINE_SPACING/GRID_ORIGIN and
# passed into load_3d_survey() as the `layout` argument.
ELECTRODE_LAYOUT = make_line_layout(LINE_RUNS, ELEC_SPACING, LINE_SPACING, GRID_ORIGIN, verbose=False)
# Set ELECTRODE_LAYOUT = None to trust the coordinates in the export instead.

# ---- coordinates -----------------------------------------------------
# Multiplies every electrode coordinate after the layout above is applied.
# Use it when the instrument's spread definition did not match the true
# electrode spacing on the ground - e.g. set to 2.0 if electrodes were
# actually planted 0.5 m apart but the spread was set up as 0.25 m. This
# rescales the geometry and, with it, the geometric factor k and hence rhoa.
# Leave at 1.0 once the spread/layout is confirmed correct.
COORDINATE_SCALE = 1.0

# ---- quality control -------------------------------------------------
# Interactive QC: click points to reject, panels grouped by line and by
# configuration size. Set False to run headless (no plot windows).
INTERACTIVE_QC = True
QC_GROUP_BY = 'line+size'     # any of 'line', 'size', 'file', combined with '+'
QC_MASK_FILE = None           # path to reuse a previously saved mask, e.g. 'qc_mask.txt'

# Automatic filters, applied in apply_filters(). ALL DEFAULT TO OFF - nothing
# is rejected unless a threshold is set below. Non-finite and non-positive
# rhoa are always removed regardless of these settings, since they cannot be
# inverted at all.
MIN_RHOA = None                # reject rhoa below this, ohm.m
MAX_RHOA = None                # reject rhoa above this, ohm.m
K_MAX = None                  # reject |k| above this, m
MAX_INSTRUMENT_ERR = None     # reject on the LS 'Var %' column, as a fraction
MAX_RECIPROCAL_ERR = 0.15     # reject normal/reciprocal pairs disagreeing by more than this fraction
MIN_U = None                  # reject |U| below this, V, e.g. 1e-4
MIN_I = None                  # reject |I| below this, mA, e.g. 1.0
NEIGHBOUR_THR = None          # local outlier rejection threshold, in MAD, e.g. 3.0

# ---- error model ------------------------------------------------------
# Controls where each datum's inversion error (data['err'], what chi^2 is
# measured against) comes from:
#   'model' fits pyGIMLi's reciprocal error model (dR = a + b|R|, or the
#           relative form below) to the normal/reciprocal disagreements and
#           uses that smooth fit as the error, falling back to the noise
#           model below wherever the fit has no finite value (including when
#           no reciprocals are found at all).
#   'auto'  (or True) take max(noise model, each pair's own disagreement).
#   False   noise model only (RELATIVE_ERROR + the voltage term below).
USE_RECIPROCALS = 'model'
# True  fits the RELATIVE errors directly (dR/R = a + b/|R|), usually the
#       more realistic fit across a wide range of R. False fits absolute
#       residuals (dR = a + b|R|) instead. Only used when USE_RECIPROCALS
#       is 'model'.
RECIPROCAL_REL_FIT = True
# Collapses each normal/reciprocal pair into a single averaged datum. In
# 'model' mode the stacked datum keeps the fitted-model error (rather than
# being overwritten by the raw per-pair disagreement) because keep_err is
# passed as (USE_RECIPROCALS == 'model') further down. No effect on data
# with no reciprocal pairs.
STACK_RECIPROCALS = True
# The noise model: err = max(ERROR_FLOOR, sqrt(RELATIVE_ERROR^2 +
# (ABSOLUTE_U_ERROR/U)^2)). Used directly when USE_RECIPROCALS is False, and
# as the fallback wherever USE_RECIPROCALS='model' has no reciprocal-based
# estimate for a datum.
RELATIVE_ERROR = 0.02         # base relative error, as a fraction
ABSOLUTE_U_ERROR = 1e-4       # voltage-measurement error, V (100 uV)
ERROR_FLOOR = 0.02            # minimum error assigned to any datum, as a fraction
# ---- mesh -----------------------------------------------------------
PARA_DEPTH = 7              # depth of the parameter (inversion) domain below the electrodes, m
# Target cell VOLUME in m^3 (3D mesh, not an area) used by build_3d_mesh() in
# main(). Raise this if meshing runs out of memory.
PARA_MAX_CELL_SIZE = 0.25
# paraBoundary is a FRACTION of the survey extent used to pad the parameter
# domain outward (not a number of electrode spacings). None lets pyGIMLi use
# its own default padding.
PARA_BOUNDARY = None          # None -> pyGIMLi default padding
SURFACE_QUALITY = 34
MESH_QUALITY = 1.3

# ---- inversion ------------------------------------------------------
# The sweep runs every combination of CTYPES x ZWEIGHTS x LAMBDAS (see
# itertools.product in main()), one inversion per combination.
CTYPES = [1]                                  # constraint type: 0 damping, 1 first-order smoothness, 2 second-order smoothness
ZWEIGHTS = [0.3]                              # vertical (z) smoothness weight relative to horizontal; 1.0 is isotropic
# Candidate regularisation strengths (lambda) to sweep, log-spaced from 2 to
# 300 across 24 points. np.unique(np.round(..., 2)) removes duplicate values
# after rounding to 2 dp so the sweep never wastes a run (and a VTK filename)
# on two lambdas that rounded to the same number. To run a single lambda
# instead, set num=1 with matching bounds (e.g. np.logspace(np.log10(3),
# np.log10(3), num=1)).
LAMBDAS = np.unique(np.round(
    np.logspace(np.log10(2), np.log10(300), num=24), 2))
MAX_ITER = 12
ROBUST_DATA = True           # L1 data norm (robust to outlying data)
BLOCKY_MODEL = True          # L1 model norm (sharper, blockier boundaries) instead of the smooth L2 default
START_MODEL_FILE = None


# =====================================================================
# ============================= MAIN ==================================
# =====================================================================

def main():

    # --------------------------- directories -------------------------
    # Shared INPUTS (reused across sweeps) live at the top level, so the
    # expensive mesh build, the unified data file and the QC mask are not
    # regenerated every run.
    dirs = {k: os.path.join(OUTPUT_DIR, v) for k, v in {
        'data': 'Data', 'qc': 'QC', 'mesh': 'Mesh',
    }.items()}

    # Per-sweep OUTPUTS go in a timestamped run folder so successive sweeps
    # never overwrite or bleed into each other. This is what stops a stale
    # 'lam0.0.vtk' from an earlier experiment masquerading as part of this run.
    RUN_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_DIR, 'Runs', RUN_ID)
    dirs.update({k: os.path.join(run_dir, v) for k, v in {
        'vtk': 'VTKs', 'residual': 'Residuals', 'misfit': 'ModelMisfits',
        'meta': 'ModelMetadata', 'lcurve': 'LCurves', 'model': 'Models',
    }.items()})

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"  run outputs -> {run_dir}")

    DATA_UNIFIED = os.path.join(dirs['data'], f'{BASE_NAME}_3D.dat')
    MESH_FILE = os.path.join(dirs['mesh'], f'{BASE_NAME}_mesh.bms')
    MISFIT_FILE = os.path.join(dirs['misfit'], 'misfit.csv')
    META_FILE = os.path.join(dirs['meta'], 'metadata.csv')

    u.ensure_tetgen([TETGEN_DIR] if TETGEN_DIR else None)

    # ------------------ 1. read the native 3D data -------------------
    print("\n=== 1. Reading native 3D data ===")
    files = DATA_FILES
    if isinstance(files, str):
        files = sorted(glob.glob(files))
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    survey = load_3d_survey(files,
                            elec_lookup_file=ELEC_LOOKUP_FILE,
                            tol=ELECTRODE_TOL,
                            colmap=COLMAP,
                            transforms=TRANSFORMS,
                            layout=ELECTRODE_LAYOUT,
                            coordinate_scale=COORDINATE_SCALE,
                            err_is_percent=ERR_IS_PERCENT)

    # how much of this data actually constrains the model across the lines?
    check_3d_coupling(survey)

    # ------------------ 2. geometric factors in 3D -------------------
    print("=== 2. Geometric factors ===")
    survey = compute_geometric_factors(survey, numerical=NUMERICAL_K)

    # ------------------ 3. quality control ---------------------------
    print("=== 3. Quality control ===")
    reciprocal_errors(survey)

    survey = apply_filters(survey,
                           min_rhoa=MIN_RHOA, max_rhoa=MAX_RHOA,
                           k_max=K_MAX,
                           max_instrument_err=MAX_INSTRUMENT_ERR,
                           max_reciprocal_err=MAX_RECIPROCAL_ERR,
                           min_u=MIN_U, min_i=MIN_I,
                           neighbour_thr=NEIGHBOUR_THR)

    if QC_MASK_FILE and os.path.exists(QC_MASK_FILE):
        survey.load_mask(QC_MASK_FILE)
    elif INTERACTIVE_QC:
        interactive_qc(survey, group_by=QC_GROUP_BY,
                       summary_path=os.path.join(dirs['qc'],
                                                 'QC_final_selection.png'))
        survey.save_mask(os.path.join(dirs['qc'], 'qc_mask.txt'))

    survey = set_error_model(survey,
                             relative=RELATIVE_ERROR,
                             absolute_u=ABSOLUTE_U_ERROR,
                             use_reciprocals=USE_RECIPROCALS,
                             reciprocal_rel_fit=RECIPROCAL_REL_FIT,
                             use_instrument=False,
                             floor=ERROR_FLOOR)

    if STACK_RECIPROCALS:
        # in 'model' mode, keep the fitted error rather than overwriting it with
        # the raw per-pair disagreement while averaging the pair
        survey = stack_reciprocals(survey,
                                   keep_err=(USE_RECIPROCALS == 'model'))

    if survey.n_valid == 0:
        print("No valid measurements left after QC - stopping.")
        return

    plot_survey_3d(survey, dirs['qc'], BASE_NAME)
    survey.summary()
    data = survey.save(DATA_UNIFIED)

    # ------------------ 4. build the 3D mesh once --------------------
    print("=== 4. Building 3D mesh ===")
    mesh = build_3d_mesh(data,
                         para_depth=PARA_DEPTH,
                         para_max_cell_size=PARA_MAX_CELL_SIZE,
                         para_boundary=PARA_BOUNDARY,
                         surface_quality=SURFACE_QUALITY,
                         quality=MESH_QUALITY,
                         tetgen_dirs=[TETGEN_DIR] if TETGEN_DIR else None,
                         save_path=MESH_FILE)

    # ------------------ 5. inversion sweep ---------------------------
    print("=== 5. Inversions ===")
    params = list(product(CTYPES, ZWEIGHTS, LAMBDAS))
    print(f"  cTypes  : {CTYPES}")
    print(f"  zWeights: {ZWEIGHTS}")
    print(f"  lambdas : {list(LAMBDAS)}")
    print(f"  -> {len(params)} inversions on {mesh.cellCount()} cells")

    results = []
    with tqdm(total=len(params), desc='Running 3D inversions', leave=True) as bar:
        for ctype, zw, lam in params:
            results.append(run_inversion_3d((
                ctype, zw, lam, MAX_ITER, ROBUST_DATA, BLOCKY_MODEL,
                DATA_UNIFIED, MESH_FILE,
                dirs['vtk'], dirs['residual'], START_MODEL_FILE)))
            bar.update(1)

    valid = [r for r in results if r[-1] != "FAILED"]
    print(f"\n  {len(valid)} / {len(results)} inversions succeeded")
    for r in results:
        status = "FAILED" if r[-1] == "FAILED" else f"chi2={r[4]:.2f}"
        print(f"    cType={r[0]} zW={r[1]} lam={r[2]}: {status}")

    if not valid:
        print("All inversions failed - nothing further to do.")
        return

    # ------------------ save the misfit statistics --------------------
    with open(MISFIT_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cType', 'zWeight', 'lambda', 'Iterations', 'Chi2', 'RMS',
                    'RRMS', 'Phi', 'phiModel', 'phiData', 'modelNorm',
                    'weightedNorm'])
        for r in valid:
            w.writerow(r[:12])

    with open(META_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cType', 'zWeight', 'lambda', 'vtk_file'])
        for r in valid:
            w.writerow([r[0], r[1], r[2], r[12]])

    plt.close('all')

    # ------------------ 6. L-curve + quick slices ---------------------
    # An L-curve needs at least 3 points to have a corner to find; report why
    # it was skipped rather than skipping silently.
    if len(valid) > 2:
        print("=== 6. L-curve ===")
        try:
            suggested = plot_lcurve(MISFIT_FILE, META_FILE, dirs['lcurve'],
                                    BASE_NAME, lam_idx=len(valid) // 2,
                                    logplot=False, normplot=False,
                                    geom_avg=True, arith_avg=False,
                                    pct_start=False, mark_corner=True)
            for (ct, zw), lam in suggested.items():
                print(f"  L-curve corner suggests lambda={lam:.2f} "
                      f"(cType={ct}, zWeight={zw})")
        except Exception as e:
            print(f"  [lcurve] plotting failed: {e}")
    else:
        print(f"  [lcurve] skipped: only {len(valid)} successful inversion(s), "
              f"need >2 for an L-curve. Widen LAMBDAS.")

    best = min(valid, key=lambda r: abs(r[4] - 1.0))   # chi2 closest to 1
    print(f"\n  chi2-closest-to-1 model: cType={best[0]}, zW={best[1]}, "
          f"lam={best[2]}, chi2={best[4]:.2f}, rRMS={best[6]:.2f}%")
    try:
        plot_coverage_slices(best[12], dirs['model'], BASE_NAME)
    except Exception as e:
        print(f"  (slice plot skipped: {e})")

    print(f"\nDone. VTKs for ParaView are in {dirs['vtk']}")


if __name__ == "__main__":
    main()
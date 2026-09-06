# -*- coding: utf-8 -*-
"""
True3DTimelapseInversionMain.py
--------------------------------
Entry point for true 3D timelapse (repeat-survey) ERT inversion with
pyGIMLi, run over two or more native 3D Terrameter exports of the same
electrode layout collected at different times. Companion module:
True3DInversionUtilities.py (imported below as `u`).

Workflow (see the numbered steps printed by main()):
    1. load and QC each timestep independently
    2. resolve one acquisition time per timestep
    3. put every timestep on a shared electrode table, then reduce all
       timesteps to the configurations common to every one of them
    4. build one tetrahedral mesh, shared by every timestep and every lambda
    5. optionally sweep a range of lambdas and locate the L-curve corner
    6. run the timelapse inversion at the chosen lambda
       (STRATEGY = 'full' or 'sequential')
    7. compute coverage
    8. export per-timestep VTKs, a change-summary VTK, slice plots, and a
       per-timestep misfit CSV

Adapted from the 2D timelapse workflow (ERTtimelapse_original_v1, S. Bithell).
Carried over from that code:
  * cross-timestep QC - a measurement rejected in ANY timestep is dropped
    from EVERY timestep (see match_configurations() in the utilities
    module), so a point present at one time and absent at another cannot
    masquerade as change
  * driving pyGIMLi's TimelapseERT with explicit per-timestep datetime objects
  * preferring the joint ("full") inversion over independent per-timestep
    ("sequential") inversions, with scalef controlling how much change is
    allowed between timesteps
  * change expressed as (model / other) - 1, a signed fractional change
  * per-timestep VTKs carrying resistivity, log resistivity, a normalised
    gradient and pairwise change ratios, plus one averaged-change VTK

Adapted for 3D:
  * timesteps are matched on the electrode quadruple (the current and
    potential pair) rather than on row position, so files may differ in
    acquisition order or length
  * one shared 3D electrode table, rebuilt from switch addresses
  * a single tetrahedral parameter mesh, built once from the first timestep
    and reused, unchanged, by every timestep and every lambda
  * fullInversion computes no sensitivity of its own, so coverage is
    obtained separately, from a one-iteration single-timestep inversion

Run:  python True3DTimelapseInversionMain.py
"""

import os
import csv
import glob
import time
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- TetGen -------------------------------------------------------------
# TetGen (the tetrahedral mesher pyGIMLi shells out to) is located in one of
# three places, tried in order: an extracted CLI binary, a locally built
# binary, or a local source clone importable as the `tetgen` package. The
# winning location is prepended to PATH (and, for the source clone, to
# sys.path) so pyGIMLi's mesh builder can find it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TETGEN_ROOT = os.path.join(SCRIPT_DIR, 'tetgen')
LOCAL_TETGEN_BUILD = os.path.join(LOCAL_TETGEN_ROOT, 'build')
LOCAL_TETGEN_BIN = os.path.join(SCRIPT_DIR, 'tetgen_bin', 'usr', 'bin')
TETGEN_DIR = None

if os.path.isdir(LOCAL_TETGEN_BIN) and os.path.isfile(os.path.join(LOCAL_TETGEN_BIN, 'tetgen')):
    os.environ['PATH'] = LOCAL_TETGEN_BIN + os.pathsep + os.environ.get('PATH', '')
    TETGEN_DIR = LOCAL_TETGEN_BIN
    print(f"  [tetgen] using extracted TetGen binary at {LOCAL_TETGEN_BIN}")

if TETGEN_DIR is None and os.path.isdir(LOCAL_TETGEN_BUILD):
    if os.path.isfile(os.path.join(LOCAL_TETGEN_BUILD, 'tetgen')):
        os.environ['PATH'] = LOCAL_TETGEN_BUILD + os.pathsep + os.environ.get('PATH', '')
        TETGEN_DIR = LOCAL_TETGEN_BUILD
        print(f"  [tetgen] using built local tetgen at {LOCAL_TETGEN_BUILD}")

if TETGEN_DIR is None and os.path.isdir(LOCAL_TETGEN_ROOT):
    if os.path.isdir(os.path.join(LOCAL_TETGEN_ROOT, 'src')):
        sys.path.insert(0, os.path.join(LOCAL_TETGEN_ROOT, 'src'))
    sys.path.insert(0, LOCAL_TETGEN_ROOT)
    os.environ['PATH'] = LOCAL_TETGEN_ROOT + os.pathsep + os.environ.get('PATH', '')
    TETGEN_DIR = LOCAL_TETGEN_ROOT
    print(f"  [tetgen] using local clone at {LOCAL_TETGEN_ROOT}")

import pygimli as pg
import pygimli.physics.ert as ert

# True3DInversionUtilities is imported as `u` everywhere below; if that file
# is ever renamed, this is the only line that needs to change.
import True3DInversionUtilities as u

REQUIRED_UTILS_VERSION = (1, 9)
_NEEDED = ['Survey', 'load_3d_survey', 'make_line_layout', 'apply_filters',
           'interactive_qc', 'set_error_model', 'stack_reciprocals',
           'fit_reciprocal_error_model',
           'compute_geometric_factors', 'check_3d_coupling', 'build_3d_mesh',
           'unify_surveys', 'match_configurations', 'to_timelapse',
           'timelapse_coverage', 'export_timelapse_vtks',
           'timelapse_change_summary', 'plot_timelapse_slices',
           'times_from_filenames', 'plot_survey_3d', 'ensure_tetgen',
           'resolve_times', 'times_from_surveys']
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


# =====================================================================
# ============================ CONFIG =================================
# =====================================================================

BASE_NAME = 'SUDS-TL'

# ---- input --------------------------------------------------------------
# False: read the paths listed in DATA_FILES below, in the order given -
# that order becomes the timestep order, so list files in date order.
# True: main() instead pops up a file picker (see USE_FILE_DIALOG below).
USE_FILE_DIALOG = False

DATA_DIR = r'./data'   # EDIT THIS: folder containing your native 3D Terrameter exports
DATA_FILES = [
    os.path.join(DATA_DIR, 'timelapse_t1.txt'),   # EDIT THIS: your timestep filenames, in date order
    os.path.join(DATA_DIR, 'timelapse_t2.txt'),
    os.path.join(DATA_DIR, 'timelapse_t3.txt'),
]

OUTPUT_DIR = r'./output/3D_timelapse'   # EDIT THIS: where results should be written

# One acquisition time per file, in the same order as DATA_FILES. Leave as
# None and resolve_times() fills them in automatically, trying in order:
#   1. dates parsed out of the filenames (YYYYMMDD / YYMMDD)
#   2. the instrument's own Time column (the LS2 writes one per reading)
#   3. evenly-spaced placeholders one day apart, with a warning
# Set TIMES to an explicit array (like the commented-out example below) to
# skip all of that and use exact, known acquisition timestamps instead.
TIMES = None
# TIMES = np.array([
#     datetime(2026, 7, 16, 9, 30, 0),
#     datetime(2026, 7, 17, 9, 30, 0),
# ])

# ---- electrode layout (see True3DInversionMain.py for more background) --
# The field layout, rebuilt from switch addresses rather than trusted from
# the export - see make_line_layout(). LINE_RUNS gives the address at each
# end of each line, in line order; here two lines of 32 electrodes each,
# the first walked from address 32 down to 1, the second from 33 up to 64.
LINE_RUNS = [(32, 1), (33, 64)]
ELEC_SPACING = 0.5             # m, along each line
LINE_SPACING = 0.5             # m, between adjacent lines
GRID_ORIGIN = (0, 0, 0.0)      # position of the first electrode of line 1
ELECTRODE_LAYOUT = u.make_line_layout(LINE_RUNS, ELEC_SPACING, LINE_SPACING,
                                      GRID_ORIGIN, verbose=False)
# ELECTRODE_LAYOUT = None   # trust the coordinates in the export instead

# Electrode positions within this distance of each other are treated as the
# same physical electrode - used both when a file is first loaded and again
# when unify_surveys() builds the shared electrode table across timesteps.
ELECTRODE_TOL = 0.05
# Uniform multiplier applied to every coordinate on load; leave at 1.0 unless
# the export's spread definition used the wrong electrode spacing.
COORDINATE_SCALE = 1.0
# Is the error column a percentage (LS 'Var %') or already a fraction?
ERR_IS_PERCENT = True

# ---- QC (applied per timestep, then intersected across them) -----------
# Each timestep is filtered and QC'd independently below; match_configurations()
# afterwards keeps only the configurations that survived in every timestep.
INTERACTIVE_QC = False        # True opens the click-to-toggle window per file
QC_GROUP_BY = 'line+size'     # any of 'line', 'size', 'file', combined with '+'
REUSE_QC_MASKS = True         # True: reload a saved mask file instead of re-clicking

# Automatic filters, all off by default - see apply_filters() for exactly
# what each threshold rejects. Non-finite/non-positive rhoa is always
# removed regardless of these settings, since it cannot be inverted.
MIN_RHOA = None
MAX_RHOA = None
K_MAX = None
MAX_INSTRUMENT_ERR = None
MAX_RECIPROCAL_ERR = 0.15
MIN_U = None
MIN_I = None
NEIGHBOUR_THR = None

# Controls where each timestep's data error (data['err'], the denominator
# chi^2 is measured against) comes from - see set_error_model():
#   'model' fits pyGIMLi's reciprocal error model (dR = a + b|R|) and uses
#           that fit directly as the error. Recommended for reciprocal
#           protocols; it is the usual fix for a chi^2 stuck around 10, which
#           generally means the flat noise model below understated the real
#           noise. Applied independently per timestep, so each acquisition
#           gets its own fitted model. Falls back to the noise model
#           wherever the fit cannot produce a value, or no reciprocals exist.
#   'auto'  older behaviour: max(noise model, each pair's own normal/
#           reciprocal disagreement).
#   False   noise model only.
USE_RECIPROCALS = 'model'
RECIPROCAL_REL_FIT = True     # fit relative errors (dR/R = a + b/|R|) rather
                              # than absolute residuals (dR = a + b|R|)
STACK_RECIPROCALS = True      # collapse each normal/reciprocal pair into one datum
# Noise model: the base error, and the fallback wherever reciprocals are
# unavailable. err = max(ERROR_FLOOR, sqrt(RELATIVE_ERROR^2 + (ABSOLUTE_U_ERROR/U)^2))
RELATIVE_ERROR = 0.02
ABSOLUTE_U_ERROR = 1e-4
ERROR_FLOOR = 0.02

# ---- mesh ----------------------------------------------------------------
# This is the 2x32 gradient layout defined above: current dipoles reach up
# to ~15.5 m apart, so the data constrain several metres of depth - a
# too-shallow domain would squash the model into a flat slab.
PARA_DEPTH = 7.0
# A VOLUME in m^3, not an area (0.25 m^3 works out to roughly 0.63 m cells,
# a little over one electrode spacing of 0.5 m). Coarsened to control memory
# as much as speed: under STRATEGY='full' every timestep's model is solved
# jointly, so peak memory scales with n_timesteps x one 3D solve rather than
# just one. Change detection is smooth, so a mesh coarser than a
# single-survey inversion would use is usually still adequate. Raise this
# further (and/or switch to STRATEGY='sequential' below) if meshing or
# inversion runs out of memory.
PARA_MAX_CELL_SIZE = 0.25
# A FRACTION of the survey extent, not a number of electrode spacings. None
# lets pyGIMLi fall back to its own default padding around the electrodes.
PARA_BOUNDARY = None
SURFACE_QUALITY = 34
MESH_QUALITY = 1.3

# ---- inversion ------------------------------------------------------------
# 'full'       - every timestep solved jointly in one inversion, with
#                temporal regularisation (see SCALEF) coupling neighbouring
#                timesteps. More robust, but the model vector is
#                n_timesteps x n_cells, so it is far heavier in 3D than a
#                single-survey inversion.
# 'sequential' - timesteps inverted one at a time as independent
#                single-survey inversions (see IS_REFERENCE / CREEP below
#                for how each one is started). Cheaper, but the resulting
#                "change" is then just the difference of independently
#                noisy models rather than a jointly-regularised one.
STRATEGY = 'full'

LAM = 6                        # regularisation strength used when LAM_SWEEP is None
ZWEIGHT = 0.3                  # vertical vs. horizontal smoothness weighting;
                              # carried over from the 2D workflow - for a
                              # genuinely 3D layout, values nearer 1.0 are
                              # often preferred
MAX_ITER = 12
ROBUST_DATA = True            # L1 data norm
BLOCKY_MODEL = True            # L1 model norm, sharper boundaries
SCALEF = 0.5                  # temporal smoothing ('full' only): larger
                              # values allow less change between timesteps
# sequential only - how each timestep's starting model is chosen:
IS_REFERENCE = True           # start every timestep from timestep 0's model;
                              # only takes effect when CREEP is False
CREEP = True                  # start each timestep from the PRECEDING
                              # timestep's model instead; when True this
                              # takes priority over IS_REFERENCE

# ---- lambda sweep / L-curve ----------------------------------------------
# A list of lambdas to invert in turn, tracing an L-curve of data misfit
# (chi^2) against model roughness ||C m||. lcurve_corner() below finds the
# corner - the point of maximum curvature on the log-log curve - the classic
# heuristic optimum: past it, more regularisation barely lowers roughness
# while misfit climbs; before it, misfit barely falls while the model
# roughens. Set to None (or an empty list) to skip the sweep and invert once
# at LAM.
LAM_SWEEP = None

# Which lambda's models/VTKs get exported as the final result:
#   None  -> the corner lambda found by the sweep (or LAM, if no sweep ran)
#   float -> use this value regardless of what the sweep found
LAM_FINAL = None
# Keep every swept model in RAM so the one finally chosen is reused instead
# of re-inverted. Cheaper in time, but holds n_lambda x n_timesteps 3D
# models at once - leave False in 3D unless the run comfortably fits in memory.
CACHE_SWEEP_MODELS = False

# ---- outputs ---------------------------------------------------------------
ALL_DIFFS = True               # per-timestep VTKs: every timestep vs every
                              # other (False: only vs the baseline and vs
                              # the immediately preceding timestep)
FROM_BASELINE = True           # change summary relative to BASELINE_INDEX
                              # (False: averaged over every ordered pair of
                              # timesteps instead of one fixed baseline)
BASELINE_INDEX = 0              # index into DATA_FILES/models used as the reference timestep
COMPUTE_COVERAGE = True        # run one extra single-iteration inversion to
                              # estimate coverage (fullInversion computes
                              # none of its own)


# =====================================================================
# ================= single-lambda inversion + L-curve =================
# =====================================================================

def run_timelapse_once(alldata, mesh, times, lam, keep_models=True,
                       verbose=True):
    # Run one timelapse inversion at a single lambda and report its metrics.
    #
    # alldata     : list of per-timestep DataContainerERT (from to_timelapse())
    # mesh        : the shared 3D parameter mesh, assigned to every timestep
    # times       : one datetime per timestep, same length/order as alldata
    # lam         : regularisation strength (lambda) for this one run
    # keep_models : if False, the returned 'models' entry is discarded (set
    #               to None) to save memory - useful during a lambda sweep
    #               where only the misfit/roughness numbers are needed
    # verbose     : forwarded to pyGIMLi's own progress printing
    #
    # Reads the module-level inversion settings (STRATEGY, ZWEIGHT, MAX_ITER,
    # ROBUST_DATA, BLOCKY_MODEL, SCALEF, IS_REFERENCE, CREEP) rather than
    # taking them as arguments.
    #
    # Returns a dict:
    #   models      : list of per-timestep model arrays, or None (see keep_models)
    #   para        : the parameter-domain mesh the models are defined on
    #   chi2s       : per-timestep chi^2 values, or None if they could not be
    #                 recomputed
    #   rrmss       : per-timestep relative RMS misfit (%), matching chi2s
    #   global_chi2 : for STRATEGY='full', the joint chi^2 actually minimised;
    #                 for 'sequential', the mean of the per-timestep chi2s
    #   roughness   : sqrt(phi_model) - the model-roughness axis of the L-curve
    #   phi_model   : the raw model objective ||C m||^2 (summed over
    #                 timesteps for 'sequential')
    tl = ert.TimelapseERT(alldata, times=times)
    tl.mesh = mesh
    if verbose:
        print(tl)

    t0 = time.time()
    models = None
    para = None
    chi2s = rrmss = None
    global_chi2 = None
    phi_model = None

    if STRATEGY == 'full':
        # cType is deliberately left out here: passing it to fullInversion
        # causes problems, so it is omitted rather than set explicitly.
        tl.fullInversion(verbose=verbose, zWeight=ZWEIGHT, lam=lam,
                         blockyModel=BLOCKY_MODEL, robustData=ROBUST_DATA,
                         maxIter=MAX_ITER, scalef=SCALEF)
        models = list(tl.models)
        para = tl.pd if hasattr(tl, 'pd') and tl.pd is not None \
            else tl.mgr.paraDomain
        try:
            # fullInversion reports one joint response for all timesteps
            # stacked together; slice it back into per-timestep blocks (each
            # of size nper) to recover a chi2/rrms per timestep for the
            # misfit table, in addition to the joint chi2 below.
            resp = np.asarray(tl.inv.response)
            tl_frames = [tl.chooseTime(t) for t in range(len(alldata))]
            nper = tl_frames[0].size()
            chi2s, rrmss = [], []
            for t, d in enumerate(tl_frames):
                block = resp[t * nper:(t + 1) * nper]
                rhoa = np.asarray(d['rhoa'], float)
                err = np.asarray(d['err'], float)
                res = (np.log(block) - np.log(rhoa)) / err
                chi2s.append(float(np.mean(res ** 2)))
                rrmss.append(float(np.sqrt(np.mean(
                    ((block - rhoa) / rhoa) ** 2)) * 100.0))
            global_chi2 = float(tl.inv.chi2())
        except Exception as e:
            print(f"  !  [tl] chi2 recomputation failed ({e})")
            chi2s = rrmss = None
        try:
            phi_model = float(tl.inv.phiModel())   # ||C m||^2
        except Exception as e:
            print(f"  !  [tl] roughness (phiModel) unavailable ({e})")
            phi_model = None

    elif STRATEGY == 'sequential':
        models = []
        chi2s, rrmss = [], []
        phi_model = 0.0
        ref_model = None
        for i, d in enumerate(alldata):
            if verbose:
                print(f"  [tl] timestep {i + 1}/{len(alldata)} ...")
            mgr_i = ert.ERTManager(d)
            mgr_i.setMesh(mesh)
            try:
                mgr_i.inv.inv.setDeltaPhiAbortPercent(0)
            except Exception:
                pass
            # ref_model supplies this timestep's starting model; CREEP and
            # IS_REFERENCE (module config) decide how it is updated below.
            start = ref_model
            mgr_i.invert(mesh=mesh, verbose=verbose, cType=1, zWeight=ZWEIGHT,
                         lam=lam, maxIter=MAX_ITER, robustData=ROBUST_DATA,
                         blockyModel=BLOCKY_MODEL, startModel=start)
            pdi = mgr_i.paraDomain
            if pdi.dim() != 3:
                # invert() can silently fall back to a 2D problem if the mesh
                # assignment above did not take - fail loudly instead.
                raise RuntimeError(f"timestep {i} inverted in 2D despite a 3D "
                                   f"mesh - the mesh was not applied.")
            models.append(np.asarray(mgr_i.model))
            try:
                chi2s.append(float(mgr_i.inv.chi2()))
            except Exception:
                chi2s.append(None)
            try:
                rrmss.append(float(mgr_i.inv.relrms()))
            except Exception:
                rrmss.append(None)
            try:
                phi_model += float(mgr_i.inv.phiModel())   # summed over steps
            except Exception:
                phi_model = None if phi_model is None else phi_model
            if para is None:
                para = pg.Mesh(pdi)
            # CREEP always advances ref_model to this timestep's model, so it
            # takes priority whenever both flags are True. IS_REFERENCE only
            # has an effect when CREEP is False, and even then only sets
            # ref_model once, from timestep 0, leaving it fixed afterwards.
            if CREEP:
                ref_model = np.asarray(mgr_i.model)
            elif IS_REFERENCE and ref_model is None:
                ref_model = np.asarray(mgr_i.model)
            del mgr_i
            gc.collect()
        # No single objective is shared across independent inversions here,
        # so the mean per-timestep chi2 stands in for a "joint" metric.
        good = [c for c in chi2s if isinstance(c, float)]
        global_chi2 = float(np.mean(good)) if good else None
    else:
        raise ValueError(f"STRATEGY must be 'full' or 'sequential', "
                         f"got {STRATEGY!r}")

    if verbose:
        print(f"  [tl] lam={lam:g} took {(time.time() - t0) / 60:.2f} min"
              + (f", joint chi2={global_chi2:.3f}" if global_chi2 else ""))

    # roughness is the L-curve's model axis. Kept at 0.0 (not None) when
    # phi_model is exactly zero so a genuinely zero-roughness point can still
    # be plotted; None only when phi_model itself is unknown.
    roughness = float(np.sqrt(phi_model)) if phi_model not in (None, 0.0) \
        else (0.0 if phi_model == 0.0 else None)

    del tl
    gc.collect()
    return {
        'models': models if keep_models else None,
        'para': para,
        'chi2s': chi2s, 'rrmss': rrmss,
        'global_chi2': global_chi2,
        'roughness': roughness, 'phi_model': phi_model,
    }


def lcurve_corner(lams, misfits, roughness):
    """Index of the maximum-curvature corner of a log-log L-curve.

    Uses the Menger curvature of consecutive point triples on
    (log10 roughness, log10 misfit) - a standard, derivative-free L-curve
    corner heuristic (after Hansen). Needs at least three finite points;
    returns None if it cannot be evaluated, so the caller can fall back to
    a fixed lambda instead.
    """
    lams = np.asarray(lams, float)
    x = np.log10(np.asarray(roughness, float))
    y = np.log10(np.asarray(misfits, float))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    idx = np.where(ok)[0]
    best_i, best_c = None, -np.inf
    for k in range(1, len(idx) - 1):
        i, j, l = idx[k - 1], idx[k], idx[k + 1]
        p1 = np.array([x[i], y[i]])
        p2 = np.array([x[j], y[j]])
        p3 = np.array([x[l], y[l]])
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        area2 = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                    (p3[0] - p1[0]) * (p2[1] - p1[1]))
        curv = 2.0 * area2 / (a * b * c) if a * b * c > 0 else 0.0
        if curv > best_c:
            best_c, best_i = curv, j
    return best_i


def save_lcurve(rows, corner_i, outdir, base_name):
    """Write the L-curve points to a CSV and a log-log PNG with the corner marked.

    rows      : list of dicts with keys 'lam', 'chi2', 'roughness', 'rrms'
                (one dict per swept lambda, in sweep order)
    corner_i  : index into rows of the chosen corner point, or None
    outdir    : directory the CSV/PNG are written into (created if missing)
    base_name : filename prefix for both outputs

    Returns (csv_path, png_path). png_path is None if plotting failed - the
    CSV is still written either way.
    """
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f'{base_name}_Lcurve.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['lambda', 'chi2', 'roughness_norm', 'rrms_percent',
                    'is_corner'])
        for i, r in enumerate(rows):
            w.writerow([f"{r['lam']:g}",
                        f"{r['chi2']:.4f}" if r['chi2'] is not None else '',
                        f"{r['roughness']:.6g}"
                        if r['roughness'] is not None else '',
                        f"{r['rrms']:.4f}" if r['rrms'] is not None else '',
                        'yes' if i == corner_i else ''])

    png_path = os.path.join(outdir, f'{base_name}_Lcurve.png')
    try:
        lam = [r['lam'] for r in rows]
        chi2 = [r['chi2'] for r in rows]
        rough = [r['roughness'] for r in rows]
        # only points with both a finite chi2 and a non-zero roughness can be
        # placed on the log-log plot
        finite = [i for i in range(len(rows))
                  if chi2[i] is not None and rough[i] not in (None, 0.0)]
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        if finite:
            xf = [rough[i] for i in finite]
            yf = [chi2[i] for i in finite]
            ax.plot(xf, yf, '-o', color='0.4', zorder=1)
            for i in finite:
                ax.annotate(f"{lam[i]:g}", (rough[i], chi2[i]),
                            textcoords='offset points', xytext=(6, 4),
                            fontsize=8)
            if corner_i is not None and chi2[corner_i] is not None:
                # highlight the chosen corner with an open circle + legend
                ax.plot(rough[corner_i], chi2[corner_i], 'o', ms=12,
                        mfc='none', mec='crimson', mew=2, zorder=3,
                        label=f"corner  lam={lam[corner_i]:g}")
                ax.legend(loc='best', fontsize=9)
            ax.set_xscale('log')
            ax.set_yscale('log')
        # reference line at chi2=1 (a perfectly-fitted-to-its-error-estimate model)
        ax.axhline(1.0, ls='--', lw=0.8, color='steelblue')
        ax.text(ax.get_xlim()[0], 1.0, ' chi2 = 1', va='bottom',
                ha='left', fontsize=8, color='steelblue')
        ax.set_xlabel('model roughness  ||C m||')
        ax.set_ylabel('data misfit  chi$^2$')
        ax.set_title(f'{base_name} L-curve')
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  !  [Lcurve] plot failed ({e}) - CSV still written")
        png_path = None
    return csv_path, png_path


# =====================================================================
# ============================= MAIN ==================================
# =====================================================================

def main():
    # Drives the full timelapse workflow described in the module docstring
    # above, in the numbered steps printed as it runs. Takes no arguments
    # and returns nothing; all inputs come from the CONFIG block, and all
    # outputs are written under OUTPUT_DIR.

    # --------------------------- inputs ------------------------------
    files = DATA_FILES
    out_dir = OUTPUT_DIR
    if USE_FILE_DIALOG:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        files = sorted(filedialog.askopenfilenames(
            title="Select 3D data files, one per timestep",
            filetypes=[("Text/RES", "*.txt *.res *.csv"), ("All", "*.*")]))
        out_dir = filedialog.askdirectory(title="Select output directory")

    files = list(files)
    if len(files) < 2:
        raise ValueError(f"Timelapse needs at least two files, got {len(files)}.")
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    # One subfolder per kind of output, all under out_dir.
    dirs = {k: os.path.join(out_dir, v) for k, v in {
        'data': 'UnifiedFiles', 'qc': 'QC', 'mesh': 'Mesh', 'vtk': 'VTKs',
        'misfit': 'ModelMisfits', 'meta': 'ModelMetadata',
        'model': 'Models', 'change': 'Change',
    }.items()}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Short label per file (used in filenames, QC mask names and printouts).
    labels = [os.path.splitext(os.path.basename(f))[0] for f in files]

    print("\n--- Timelapse inputs ---")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print(f"  output: {out_dir}")
    print("------------------------\n")

    # ------------------ 1. load and QC each timestep -----------------
    u.ensure_tetgen([TETGEN_DIR] if TETGEN_DIR else None)

    print("=== 1. Loading timesteps ===")
    surveys = []
    for f, lab in zip(files, labels):
        print(f"\n--- {lab} ---")
        sv = u.load_3d_survey([f],
                              tol=ELECTRODE_TOL,
                              layout=ELECTRODE_LAYOUT,
                              coordinate_scale=COORDINATE_SCALE,
                              err_is_percent=ERR_IS_PERCENT,
                              verbose=True)
        u.compute_geometric_factors(sv, numerical=False)
        u.apply_filters(sv,
                        min_rhoa=MIN_RHOA, max_rhoa=MAX_RHOA, k_max=K_MAX,
                        max_instrument_err=MAX_INSTRUMENT_ERR,
                        max_reciprocal_err=MAX_RECIPROCAL_ERR,
                        min_u=MIN_U, min_i=MIN_I,
                        neighbour_thr=NEIGHBOUR_THR)

        # Reuse a saved QC mask if one exists for this file; otherwise open
        # the interactive tool (if enabled) and save the resulting mask so
        # the next run can reuse it.
        mask_file = os.path.join(dirs['qc'], f'{lab}_qc_mask.txt')
        if REUSE_QC_MASKS and os.path.exists(mask_file):
            sv.load_mask(mask_file)
        elif INTERACTIVE_QC:
            u.interactive_qc(sv, group_by=QC_GROUP_BY,
                             summary_path=os.path.join(
                                 dirs['qc'], f'{lab}_QC_selection.png'))
            sv.save_mask(mask_file)

        u.set_error_model(sv, relative=RELATIVE_ERROR,
                          absolute_u=ABSOLUTE_U_ERROR,
                          use_reciprocals=USE_RECIPROCALS,
                          reciprocal_rel_fit=RECIPROCAL_REL_FIT,
                          use_instrument=False, floor=ERROR_FLOOR)
        if STACK_RECIPROCALS:
            # keep the fitted-model error rather than the raw per-pair value
            u.stack_reciprocals(sv, keep_err=(USE_RECIPROCALS == 'model'))

        u.plot_survey_3d(sv, dirs['qc'], lab)
        surveys.append(sv)

    # ------------------ 2. acquisition times -------------------------
    print("=== 2. Acquisition times ===")
    times = u.resolve_times(len(files), explicit=TIMES, files=files,
                            surveys=surveys)

    # ------------------ 3. align the timesteps -----------------------
    print("=== 3. Cross-timestep alignment ===")
    # Put every timestep's electrode indices on one shared table, reduce all
    # timesteps to the configurations common to every one of them (the
    # cross-timestep QC rule described in the module docstring), then check
    # how much of the surviving data is non-collinear.
    u.unify_surveys(surveys, tol=ELECTRODE_TOL)
    u.match_configurations(surveys, labels=labels)
    u.check_3d_coupling(surveys[0])

    alldata = u.to_timelapse(surveys, times)
    for lab, d in zip(labels, alldata):
        path = os.path.join(dirs['data'], f'{lab}_3D.dat')
        d.save(path)
    print(f"  [tl] per-timestep unified files written to {dirs['data']}")

    # ------------------ 4. one mesh for every timestep ---------------
    print("=== 4. Building the 3D mesh ===")
    mesh_file = os.path.join(dirs['mesh'], f'{BASE_NAME}_mesh.bms')
    mesh = u.build_3d_mesh(alldata[0],
                           para_depth=PARA_DEPTH,
                           para_max_cell_size=PARA_MAX_CELL_SIZE,
                           para_boundary=PARA_BOUNDARY,
                           surface_quality=SURFACE_QUALITY,
                           quality=MESH_QUALITY,
                           tetgen_dirs=[TETGEN_DIR] if TETGEN_DIR else None,
                           save_path=mesh_file)
    if STRATEGY == 'full':
        print(f"  [tl] full inversion will solve "
              f"{len(alldata)} x {mesh.cellCount()} = "
              f"{len(alldata) * mesh.cellCount()} parameters")

    # ------------------ 5. lambda sweep / L-curve --------------------
    lam_final = LAM_FINAL if LAM_FINAL is not None else LAM
    sweep_cache = {}       # lam -> result dict, only populated if CACHE_SWEEP_MODELS
    if LAM_SWEEP:
        print(f"=== 5a. Lambda sweep / L-curve ({STRATEGY}) ===")
        print(f"  [Lcurve] lambdas: "
              f"{', '.join(f'{x:g}' for x in LAM_SWEEP)}")
        rows = []
        for k, lam in enumerate(LAM_SWEEP):
            print(f"\n--- L-curve point {k + 1}/{len(LAM_SWEEP)}: "
                  f"lam = {lam:g} ---")
            res = run_timelapse_once(alldata, mesh, times, lam,
                                     keep_models=CACHE_SWEEP_MODELS,
                                     verbose=True)
            if CACHE_SWEEP_MODELS:
                sweep_cache[lam] = res
            rows.append({'lam': lam, 'chi2': res['global_chi2'],
                         'roughness': res['roughness'],
                         'rrms': (np.mean([r for r in (res['rrmss'] or [])
                                           if isinstance(r, float)])
                                  if res['rrmss'] else None)})
            if not CACHE_SWEEP_MODELS:
                res['models'] = None
                gc.collect()

        corner_i = lcurve_corner([r['lam'] for r in rows],
                                 [r['chi2'] for r in rows],
                                 [r['roughness'] for r in rows])
        csv_path, png_path = save_lcurve(rows, corner_i, dirs['misfit'],
                                         BASE_NAME)
        print(f"\n  [Lcurve] table -> {csv_path}")
        if png_path:
            print(f"  [Lcurve] plot  -> {png_path}")
        print("  [Lcurve]  lambda      chi2     ||C m||")
        for i, r in enumerate(rows):
            mark = '  <- corner' if i == corner_i else ''
            c = f"{r['chi2']:.3f}" if r['chi2'] is not None else '   -'
            g = f"{r['roughness']:.4g}" if r['roughness'] is not None else '  -'
            print(f"  [Lcurve]  {r['lam']:>8g}  {c:>8}  {g:>10}{mark}")

        # LAM_FINAL, when set, always wins over whatever the sweep found.
        if LAM_FINAL is not None:
            lam_final = LAM_FINAL
            print(f"  [Lcurve] LAM_FINAL overrides the corner: "
                  f"exporting lam = {lam_final:g}")
        elif corner_i is not None:
            lam_final = LAM_SWEEP[corner_i]
            print(f"  [Lcurve] exporting the corner lambda = {lam_final:g}")
        else:
            print(f"  [Lcurve] no corner found - exporting LAM = {lam_final:g}")

    # ------------------ 5b. final inversion --------------------------
    print(f"=== 5. Timelapse inversion ({STRATEGY}, lam = {lam_final:g}) ===")
    if lam_final in sweep_cache:
        # already inverted during the sweep (only possible when
        # CACHE_SWEEP_MODELS was True) - reuse it instead of redoing the work
        print("  [tl] reusing the cached model from the sweep")
        result = sweep_cache.pop(lam_final)
    else:
        result = run_timelapse_once(alldata, mesh, times, lam_final,
                                    keep_models=True, verbose=True)
    sweep_cache.clear()
    gc.collect()

    models = result['models']
    para = result['para']
    chi2s = result['chi2s']
    rrmss = result['rrmss']
    global_chi2 = result['global_chi2']
    if global_chi2 is not None:
        print(f"  [tl] joint 4D chi2 = {global_chi2:.3f}"
              + (f"; per-timestep chi2 = "
                 f"[{', '.join(f'{c:.2f}' for c in chi2s)}]"
                 if chi2s else ""))

    if para is None or para.dim() != 3:
        raise RuntimeError("The parameter domain came back 2D - the inversion "
                           "silently fell back to a 2D problem.")
    print(f"  [tl] {len(models)} models on {para.cellCount()} cells")

    # ------------------ 6. coverage ----------------------------------
    coverage = None
    if COMPUTE_COVERAGE:
        print("=== 6. Coverage ===")
        try:
            coverage, cov_pd = u.timelapse_coverage(alldata[0], mesh,
                                                     lam=lam_final)
            if len(coverage) != para.cellCount():
                print(f"  !  [tl] coverage has {len(coverage)} cells but the "
                      f"models have {para.cellCount()} - not attached")
                coverage = None
            else:
                np.savetxt(os.path.join(dirs['model'],
                                        f'{BASE_NAME}_coverage.vec'), coverage)
        except Exception as e:
            print(f"  !  [tl] coverage failed ({e}) - continuing without it")
            coverage = None

    # ------------------ 7. outputs -----------------------------------
    print("=== 7. Exports ===")
    u.export_timelapse_vtks(models, para, dirs['vtk'], BASE_NAME,
                            all_diffs=ALL_DIFFS, coverage=coverage,
                            times=times)
    u.timelapse_change_summary(models, para, dirs['change'], BASE_NAME,
                               baseline_index=BASELINE_INDEX,
                               from_baseline=FROM_BASELINE,
                               coverage=coverage)
    try:
        u.plot_timelapse_slices(models, para, dirs['model'], BASE_NAME,
                                times=times, baseline_index=BASELINE_INDEX)
    except Exception as e:
        print(f"  (slice figure skipped: {e})")

    # ------------------ 8. misfit table ------------------------------
    misfit_file = os.path.join(dirs['misfit'], f'{BASE_NAME}_misfit.csv')
    with open(misfit_file, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['timestep', 'label', 'time', 'chi2', 'rrms_percent',
                    'median_resistivity', 'median_change_vs_baseline'])

        def _fmt(v, spec):
            return format(v, spec) if isinstance(v, (int, float)) else ''

        for i, m in enumerate(models):
            chi2 = chi2s[i] if chi2s is not None and i < len(chi2s) else ''
            rrms = rrmss[i] if rrmss is not None and i < len(rrmss) else ''
            change = np.median(m / models[BASELINE_INDEX] - 1)
            w.writerow([i, labels[i],
                        f"{times[i]:%Y-%m-%d %H:%M}" if times is not None else '',
                        _fmt(chi2, '.3f'), _fmt(rrms, '.3f'),
                        f"{np.median(m):.3f}", f"{change:+.5f}"])

        # Joint chi2: for STRATEGY='full' this is the objective actually
        # minimised (the honest overall fit metric, more so than any one
        # timestep's row above); for 'sequential' it is simply the mean of
        # the per-timestep chi2 values, since independent inversions share
        # no single objective.
        if global_chi2 is not None:
            w.writerow(['joint', 'all_frames', '',
                        f"{global_chi2:.3f}", '', '', ''])
    print(f"  [tl] misfit table -> {misfit_file}")
    if global_chi2 is not None:
        print(f"  [tl] joint 4D chi2 = {global_chi2:.3f}")

    print(f"\nDone. Timestep VTKs in {dirs['vtk']}, change summary in "
          f"{dirs['change']}")


if __name__ == "__main__":
    main()

"""
ResolutionTester.py -- cross-line recovery / resolution-vs-location test.

For a chosen electrode GRID ("2x32" or "4x16"), this script buries a single
synthetic resistivity anomaly, forward-models synthetic ERT data (with noise)
for one or more candidate electrode designs on that grid, inverts each
design's data independently with pyGIMLi, and compares how well each design
recovers the anomaly's position and shape in the cross-line (y) direction.
The idea is that a density/sensitivity plot only shows where a design has
coverage -- this actually tries to reconstruct a known feature.

Two modes, selected by DO_SWEEP:
  * single-position mode (DO_SWEEP=False): bury the anomaly once, at REF_Y,
    forward-model + invert it for every design, and plot the recovered
    cross-line and in-line slices side by side against the true anomaly.
  * sweep mode (DO_SWEEP=True): additionally repeat the forward+invert step
    for a row of anomaly y-positions (SWEEP_Y) per design, so recovered
    cross-line spread and mislocation can be plotted as a function of where
    the anomaly sits relative to the electrode lines. Each design's mesh and
    geometric factors are built once and reused across every sweep position,
    so the added cost of the sweep is just the extra forward+inversion runs.

Outputs (written to OUT_DIR):
  - recovery_crossline.png      true vs. each design's recovered cross-line (y-z) slice, at REF_Y
  - recovery_inline.png         true vs. each design's recovered in-line (x-z) slice, at REF_Y
  - recovery_metrics.csv        one row per design, metrics at REF_Y
  - resolution_vs_location.png  sweep mode only: per-design curves of y-spread / y-error vs. true y
  - resolution_sweep.csv        sweep mode only: one row per (design, sweep y-position)

Run (from this directory, with the pyGIMLi environment active):
    python ResolutionTester.py

This follows the standard pyGIMLi closed/insulated-box 3D ERT workflow used
elsewhere in this project (mesh via createParaMeshPLC3D + createMesh, as in
ert_3d_inversion/True3DInversionUtilities.py's build_3d_mesh) -- if that
module's mesh/boundary/geometric-factor/noise handling changes, consider
mirroring the change here; the relevant knobs are grouped under CONFIG below.
Total runtime scales with (#designs x #sweep-positions) inversions, so try a
short SWEEP_Y first to confirm the pipeline runs before committing to a full
sweep.
"""

import os
import csv
import traceback
import numpy as np

# ============================ CONFIG ============================

OUT_DIR = "./output/ResolutionOutput2x32"   # EDIT THIS: directory to write plots/CSVs into (created if missing)

# --- TetGen -------------------------------------------------------------
# pyGIMLi builds the 3D mesh by shelling out to the `tetgen` executable, so a
# usable tetgen binary must be locatable. Prefer a copy bundled next to this
# script (tetgen_bin/usr/bin), otherwise fall back to a `tetgen` folder next
# to it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TETGEN_BIN = os.path.join(SCRIPT_DIR, 'tetgen_bin', 'usr', 'bin')
LOCAL_TETGEN_ROOT = os.path.join(SCRIPT_DIR, 'tetgen')
TETGEN_DIR = LOCAL_TETGEN_BIN if os.path.isdir(LOCAL_TETGEN_BIN) else LOCAL_TETGEN_ROOT

# --- GRID switch ----------------------------------------------------------
# Selects which electrode grid's preset (below) drives this run: the grid
# dimensions, the anomaly's default position/size, and the sweep positions
# all come from GRID_PRESETS[GRID]. Compare designs from only ONE grid per
# run -- 4x16 and 2x32 have different row counts, so their cross-line
# sampling isn't directly comparable on one plot.
GRID = "2x32"                # "4x16" or "2x32"

GRID_PRESETS = {
    # 4 electrode lines at y = 0, 0.32, 0.64, 0.96 (rows=4, dy=0.32). Sweep
    # positions are symmetric about the midline y=0.48 (0.16<->0.80,
    # 0.32<->0.64 are mirror pairs); 0.16/0.48/0.80 fall between lines, while
    # 0.32/0.64 land exactly on the 2nd/3rd lines.
    "4x16": dict(rows=4, cols=16, dx=0.18, dy=0.32,
                 cx=1.44, anom_cy=0.48, anom_sy=0.45, ref_y=0.48,
                 sweep_y=[0.16, 0.32, 0.48, 0.64, 0.80]),
    # 2 electrode lines at y = 0, 0.32 (rows=2, dy=0.32). NOTE: ref_y/sweep_y
    # below are centred on y=0.25, not on y=0.16 -- the midpoint this pair's
    # actual 0.32 m gap implies. Worth checking before assuming the sweep is
    # centred between the two lines.
    "2x32": dict(rows=2, cols=32, dx=0.09, dy=0.32,
                 cx=1.44, anom_cy=1, anom_sy=0.30, ref_y=0.15,
                 sweep_y=[0.15, 0.20, 0.25, 0.30, 0.35]),
}
if GRID not in GRID_PRESETS:
    raise ValueError(f"GRID must be one of {list(GRID_PRESETS)}; got {GRID!r}")
_P = GRID_PRESETS[GRID]

# What to do if a loaded design's ACTUAL electrode geometry doesn't match
# GRID (e.g. XML_DESIGNS points at the wrong grid's files, or GRID wasn't
# updated to match): "abort" (default) stops immediately, before any
# meshing, since a mismatched run would otherwise silently produce a
# wrong-grid result and burn hours of compute. "warn" logs loudly and
# continues -- useful for a deliberate cross-grid comparison.
GRID_MISMATCH_ACTION = "abort"       # "abort" or "warn"

# --- Which designs to compare ---------------------------------------------
USE_XML = True
XML_DESIGNS = [
    # (label, spread_xml, protocol_xml) tuples for the SAME grid as GRID
    # above, e.g. for GRID="2x32": ("2x32_grad_rc", "spread_2x32_rc.xml", "protocol_2x32_rc.xml"),
    ("2x32r", "2x32rspread.xml", "2x32rprotocol.xml"),
]
GRID_ROWS, GRID_COLS = _P["rows"], _P["cols"]
GRID_DX, GRID_DY = _P["dx"], _P["dy"]
# the design labels to build in programmatic_designs() when USE_XML is False
# (subject to what's geometrically possible for the current grid -- see that
# function). Trim this tuple to a subset if you only want some of them.
PROGRAMMATIC_DESIGNS = ("rows", "rows+cols", "rows+cols+diag")

# --- Soil box (insulated cuboid). Computed from the electrode extent plus a
# margin at runtime (box_from_sensors), so it always fully contains every
# electrode -- a hand-set box smaller than the array causes "requested
# electrode does not match the given mesh" errors. The margin also keeps
# electrodes off the domain walls (better FEM behaviour near the edges).
# x=length, y=width, z=depth (z<=0).
BOX_MARGIN = 0.05            # lateral margin (m) added around the electrode array
BOX = None                  # per-design slice-region bounds; filled at runtime

# --- TRUE anomaly (block) --------------------------------------------------
ANOM = dict(
    cx=_P["cx"],             # x centre (within the grid's x-extent)
    cy=_P["anom_cy"],        # default y centre from the GRID preset; overridden via anom_at() at REF_Y / each SWEEP_Y
    cz=-0.12,                # depth centre (shallow, so it's recoverable)
    sx=0.90, sy=_P["anom_sy"], sz=0.2,   # half-extent-defining sizes (m); sy comes from the GRID preset to fit that grid's line spacing
    rho_bg=100.0,            # background resistivity (ohm-m)
    rho_anom=25.0,           # anomaly resistivity (ohm-m); < rho_bg => conductive anomaly
)

# --- Resolution-vs-location sweep -----------------------------------------
DO_SWEEP = False
# anomaly y-centres to test (m); taken from the GRID preset so they suit that
# grid's line spacing and stay inside the electrode footprint.
SWEEP_Y = list(_P["sweep_y"])
SWEEP_DEPTH = -0.12          # depth centre used for both the REF_Y panel and every sweep position (currently equal to ANOM["cz"])
REF_Y = _P["ref_y"]          # y-position used for the single-position slice panels / recovery_metrics.csv
# Average each (design, position) over this many independent noise
# realisations. >1 washes out noise-driven asymmetry and adds an error bar
# (y_spread_sd), at a proportional runtime cost.
NUM_SEEDS = 1

# --- Mesh (mirrors True3DInversionUtilities.build_3d_mesh's defaults) -----
PARA_DEPTH = 1.0             # para-domain depth (m); must be >= the anomaly's depth extent
PARA_BOUNDARY = None         # None -> let pyGIMLi choose its default background/boundary extent
PARA_MAX_CELL_FWD = 0.0002    # forward (data-generation) mesh: max cell volume (m^3) -- finer than the inversion mesh
PARA_MAX_CELL_INV = 0.0004    # inversion mesh: max cell volume (m^3) -- coarser than the forward mesh
PARA_QUALITY = 1.3           # tetgen quality passed to createMesh
SURFACE_QUALITY = 34         # surfaceMeshQuality passed to createParaMeshPLC3D

# --- Forward / inversion knobs ---------------------------------------------
NOISE_LEVEL = 0.03           # relative noise fraction passed to ert.simulate
NOISE_ABS = 1e-4             # absolute noise floor passed to ert.simulate
INV_LAMBDA = 6           # regularisation strength passed to ERTManager.invert (lam=); lower values smooth less, preserving more of the anomaly's contrast -- try roughly 2-10
INV_MAXITER = 12
SEED = 1337                  # base RNG seed for the noise added in ert.simulate; bumped by seed index across NUM_SEEDS

# --- Slicing / plotting -----------------------------------------------------
SLICE_NX, SLICE_NY, SLICE_NZ = 160, 90, 90   # grid resolution used when sampling slices for plots/metrics
CMAP = "Spectral_r"          # colormap used for the recovery slice panels

# --- Debugging ---------------------------------------------------------------
# DEBUG: on a forward/invert failure, re-raise (with full traceback) instead
#        of logging it and continuing -- useful while diagnosing a problem,
#        but leave False for a full run so one flaky position/design doesn't
#        abort the whole run (failures are logged and recorded as NaN).
# QUICK_TEST: True runs only the first design at the reference position, no
#        sweep (fast smoke test). False runs all designs plus the sweep (if
#        DO_SWEEP).
DEBUG = False
QUICK_TEST = False

# ========================== END CONFIG ==========================


def ensure_tetgen_on_path(log):
    """Prepend TETGEN_DIR to PATH (in this process) so pyGIMLi's mesher can
    find the tetgen executable, mirroring the PATH fix in
    Generator_automator.py. Afterwards checks shutil.which('tetgen') and logs
    a warning if it's still not resolvable, since that's the usual cause of
    meshing failing with a missing '...-1.node' file."""
    import shutil
    if TETGEN_DIR and os.path.isdir(TETGEN_DIR):
        os.environ["PATH"] = TETGEN_DIR + os.pathsep + os.environ.get("PATH", "")
        log(f"tetgen dir on PATH: {TETGEN_DIR}")
    elif TETGEN_DIR:
        log(f"WARNING: TETGEN_DIR does not exist: {TETGEN_DIR}")
    if shutil.which("tetgen") is None:
        log("WARNING: 'tetgen' still not found on PATH. Set TETGEN_DIR to the folder "
            "containing tetgen.exe (the same one your Generator_automator.py uses), "
            "or add it to PATH. Meshing will fail until it resolves.")
    else:
        log(f"tetgen resolves to: {shutil.which('tetgen')}")


def anom_at(cy=None, cz=None, cx=None):
    """Return a copy of ANOM with cx/cy/cz optionally overridden, leaving the
    module-level ANOM untouched. Used to move the anomaly to REF_Y or a given
    SWEEP_Y position while keeping its size and resistivities fixed."""
    a = dict(ANOM)
    if cy is not None: a["cy"] = cy
    if cz is not None: a["cz"] = cz
    if cx is not None: a["cx"] = cx
    return a


# --------------------------------------------------------------------------
# Design construction (programmatic + XML), independent of pyGIMLi
# --------------------------------------------------------------------------
# Build a regular rows x cols electrode grid at the surface (z=0): cols
# electrodes per line spaced dx apart along x, rows parallel lines spaced dy
# apart along y. Returns the (n,3) sensor array and ix(row, col) -> flat
# index into it.
def build_grid(rows, cols, dx, dy):
    xs = np.arange(cols) * dx
    ys = np.arange(rows) * dy
    sensors = np.array([[x, y, 0.0] for y in ys for x in xs], float)
    return sensors, (lambda r, c: r * cols + c)


def box_from_sensors(sensors):
    """Slice-region bounds for a design: its sensors' x/y extent expanded by
    BOX_MARGIN on each side, with depth from the surface (z=0) down to
    PARA_DEPTH -- matching the para-mesh's depth so slices stay inside the
    meshed domain."""
    x, y = sensors[:, 0], sensors[:, 1]
    return dict(x0=float(x.min()) - BOX_MARGIN, x1=float(x.max()) + BOX_MARGIN,
                y0=float(y.min()) - BOX_MARGIN, y1=float(y.max()) + BOX_MARGIN,
                ztop=0.0, zbot=-PARA_DEPTH)


# Convert a list of (a, b, m, n) electrode-index tuples into the
# {'a':.., 'b':.., 'm':.., 'n':..} column-array form pyGIMLi's
# DataContainerERT expects.
def _scheme(quads):
    q = np.array(quads, int)
    return {k: q[:, j].astype(int) for j, k in enumerate("abmn")}


def programmatic_designs():
    """Build the requested subset of PROGRAMMATIC_DESIGNS ('rows',
    'rows+cols', 'rows+cols+diag') for the current GRID_ROWS x GRID_COLS
    grid, labelled with the actual dimensions (e.g. '2x32_rows') so a design
    can never end up mislabelled with the wrong grid.

    A cross-line (column or diagonal) gradient array needs at least 4
    collinear electrodes, so A, B, M, N can all be distinct. With
    GRID_ROWS < 4 there's no room for interior M/N along a column or
    diagonal, so cols_q/diag_q come out empty; in that case only the 'rows'
    design is built (with a log line explaining why) rather than emitting
    'rows+cols'/'rows+cols+diag' as duplicates of 'rows'."""
    sensors, ix = build_grid(GRID_ROWS, GRID_COLS, GRID_DX, GRID_DY)
    R, C = GRID_ROWS, GRID_COLS
    prefix = f"{R}x{C}"
    rows_q = []
    for r in range(R):
        a, b = ix(r, 0), ix(r, C - 1)
        for m in range(1, C - 2):
            rows_q.append((a, b, ix(r, m), ix(r, m + 1)))
    cols_q = []
    diag_q = []
    if R >= 4:
        for c in range(C):
            a, b = ix(0, c), ix(R - 1, c)
            for m in range(0, R - 2):
                cols_q.append((a, b, ix(m, c), ix(m + 1, c)))
        for s in (1, 2, 3):
            for c in range(0, C - 3 * s):
                line = [ix(r, c + r * s) for r in range(R)]
                a, b = line[0], line[-1]
                for k in range(len(line) - 2):
                    diag_q.append((a, b, line[k], line[k + 1]))
    else:
        print(f"    NOTE: GRID_ROWS={R} < 4, so no column/diagonal gradient array is "
              f"geometrically possible (needs >=4 collinear electrodes across the "
              f"lines) -- only the 'rows' design is meaningful for this grid; "
              f"'rows+cols(+diag)' would be identical to 'rows'.")

    out = {}
    if "rows" in PROGRAMMATIC_DESIGNS:
        out[f"{prefix}_rows"] = (sensors, _scheme(rows_q))
    if "rows+cols" in PROGRAMMATIC_DESIGNS and cols_q:
        out[f"{prefix}_rows+cols"] = (sensors, _scheme(rows_q + cols_q))
    if "rows+cols+diag" in PROGRAMMATIC_DESIGNS and (cols_q or diag_q):
        out[f"{prefix}_rows+cols+diag"] = (sensors, _scheme(rows_q + cols_q + diag_q))
    return out


# Load electrode/reading definitions from each (label, spread_xml,
# protocol_xml) tuple in XML_DESIGNS via measurementdensity.py (imported here
# as md): parse the spread file for electrode positions, parse + validate the
# protocol file for readings, then build the sensor array and a-b-m-n scheme.
# `label` has any directory and .xml suffix stripped to form the design's
# display name. Returns {label: (sensors, scheme)}, matching
# programmatic_designs()'s output shape.
def xml_designs():
    import os as _os
    import measurementdensity as md
    out = {}
    for label, spread, protocol in XML_DESIGNS:
        clean = _os.path.splitext(_os.path.basename(str(label)))[0]  # drop dir + .xml
        electrodes = md.parse_spread_xml(spread)
        readings = md.validate_readings(md.parse_protocol_xml(protocol), lambda *_: None)
        sensors, sch = md.build_sensors_and_scheme(electrodes, readings)
        out[clean] = (sensors, {k: sch[k].astype(int) for k in "abmn"})
    return out


def describe_grid(sensors):
    """Summarize a design's ACTUAL electrode-line layout from its built/loaded
    sensors: the sorted distinct (rounded) y-values ('lines'), how many there
    are, and the total y-extent (max - min)."""
    ys = np.sort(np.unique(np.round(sensors[:, 1], 6)))
    extent = float(ys.max() - ys.min()) if len(ys) else 0.0
    return dict(n_lines=len(ys), ys=ys.tolist(), extent=extent)


def check_grid_match(label, sensors, log):
    """Compare a design's ACTUAL electrode-line count/extent (via
    describe_grid) against what the selected GRID preset expects (GRID_ROWS
    lines spanning (GRID_ROWS-1)*GRID_DY m). This matters because USE_XML
    loads real electrodes independently of GRID, so nothing else would catch
    GRID pointing at the wrong grid's XML files (or vice versa). On a
    mismatch, raises (GRID_MISMATCH_ACTION='abort', the default, since a
    mismatched run wastes hours of compute) or logs a warning and continues
    (GRID_MISMATCH_ACTION='warn')."""
    g = describe_grid(sensors)
    exp_lines = GRID_ROWS
    exp_extent = (GRID_ROWS - 1) * GRID_DY
    log(f"    detected grid: {g['n_lines']} line(s) at y={[round(y, 3) for y in g['ys']]} "
        f"(extent {g['extent']:.3f} m)")
    ok = (g["n_lines"] == exp_lines and abs(g["extent"] - exp_extent) < 0.03)
    if ok:
        return
    msg = (f"GRID MISMATCH for '{label}': GRID={GRID!r} expects {exp_lines} "
          f"electrode line(s) spanning ~{exp_extent:.3f} m (spacing {GRID_DY} m), "
          f"but the loaded electrodes have {g['n_lines']} line(s) spanning "
          f"{g['extent']:.3f} m -- this looks like a DIFFERENT grid. Check that "
          f"XML_DESIGNS points at files for GRID={GRID!r}, or set GRID to match "
          f"what these files actually are.")
    if GRID_MISMATCH_ACTION == "abort":
        raise RuntimeError(msg + " (set GRID_MISMATCH_ACTION='warn' to run anyway.)")
    log("    WARNING: " + msg)


# Collapse readings that use the exact same 4 electrodes, regardless of which
# are assigned as the current (a,b) vs. potential (m,n) pair, keeping only
# the first occurrence -- this removes reciprocal duplicates (a/b and m/n
# swapped) as well as any other repeated 4-electrode combination.
def dedupe_reciprocals(sch):
    abmn = np.column_stack([sch[k] for k in "abmn"]).astype(int)
    seen, keep = set(), []
    for i, row in enumerate(abmn):
        key = frozenset(row.tolist())
        if key not in seen:
            seen.add(key); keep.append(i)
    keep = np.array(keep, int)
    return {k: sch[k][keep] for k in "abmn"}


# --------------------------------------------------------------------------
# Anomaly geometry (pure numpy)
# --------------------------------------------------------------------------
# Boolean mask over `pts`: True where a point lies inside the anomaly's
# axis-aligned box (within half-widths sx/2, sy/2, sz/2 of its centre
# cx, cy, cz).
def in_anomaly(pts, anom):
    dx = np.abs(pts[:, 0] - anom["cx"]) <= anom["sx"] / 2
    dy = np.abs(pts[:, 1] - anom["cy"]) <= anom["sy"] / 2
    dz = np.abs(pts[:, 2] - anom["cz"]) <= anom["sz"] / 2
    return dx & dy & dz


# --------------------------------------------------------------------------
# pyGIMLi-dependent pieces (isolated)
# --------------------------------------------------------------------------
def _tetgen_binary(log):
    """Resolve an explicit path to the tetgen executable, rather than relying
    on PATH. createMesh(tetgen=...) needs an explicit path because pyGIMLi's
    mesher shells out through its C++ layer using an environment snapshot
    that does not see a PATH edited from Python afterwards, so name-based
    ('tetgen') resolution can fail there even when shutil.which finds it
    fine. Falls back to the bare string 'tetgen' (relying on PATH) if no
    binary file can be located."""
    import shutil
    tg = None
    if TETGEN_DIR:
        for name in ("tetgen.exe", "tetgen"):
            cand = os.path.join(TETGEN_DIR, name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                tg = cand
                break
    if tg is None:
        tg = shutil.which("tetgen") or shutil.which("tetgen.exe")
    if tg is None:
        log("    WARNING: could not locate tetgen; falling back to bare 'tetgen'.")
        return "tetgen"
    return f'"{tg}"' if " " in tg else tg          # quote for spaces in the path


def build_para_mesh(sensors, max_cell, log):
    """Build the 3D ERT parameter mesh for a set of sensors, the same way
    True3DInversionUtilities.build_3d_mesh does: createParaMeshPLC3D (a para
    domain plus a background/boundary region with well-posed boundary
    conditions) followed by createMesh. `max_cell` sets paraMaxCellSize
    (smaller = finer mesh) -- pass PARA_MAX_CELL_FWD for the forward/
    data-generation mesh and PARA_MAX_CELL_INV for the inversion mesh. Falls
    back to calling createMesh without an explicit tetgen= argument if the
    installed pyGIMLi build doesn't accept that keyword."""
    import pygimli as pg
    import pygimli.meshtools as mt
    d = pg.DataContainerERT()
    for p in sensors:
        d.createSensor([float(p[0]), float(p[1]), float(p[2])])
    kwargs = dict(sensors=d, paraDepth=PARA_DEPTH, paraMaxCellSize=max_cell,
                  surfaceMeshQuality=SURFACE_QUALITY)
    if PARA_BOUNDARY is not None:
        kwargs["paraBoundary"] = PARA_BOUNDARY
    plc = mt.createParaMeshPLC3D(**kwargs)
    tg = _tetgen_binary(log)
    log(f"    meshing (tetgen={tg}, paraMaxCellSize={max_cell}) ...")
    try:
        mesh = mt.createMesh(plc, quality=PARA_QUALITY, tetgen=tg)
    except TypeError:
        log("    (this pyGIMLi build takes no tetgen= arg; relying on PATH instead)")
        mesh = mt.createMesh(plc, quality=PARA_QUALITY)
    log(f"    mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes")
    return mesh


# Build a pyGIMLi DataContainerERT from a sensor array and an
# {'a','b','m','n'} index scheme: register every sensor, size the container
# to the number of readings, set the four electrode-index arrays, and mark
# every reading valid.
def make_data(sensors, sch):
    import pygimli as pg
    data = pg.DataContainerERT()
    for p in sensors:
        data.createSensor([float(p[0]), float(p[1]), float(p[2])])
    n = len(sch["a"])
    data.resize(n)
    for k in "abmn":
        data[k] = sch[k].astype(int)
    data["valid"] = np.ones(n, int)
    return data


def prepare_design(sensors, sch, log):
    """Build this design's DataContainerERT (via make_data) and attach its
    geometric factors k, using pyGIMLi's analytic dim=3 factors -- the
    default in True3DInversionUtilities.compute_geometric_factors, correct
    for a flat surface, and fast since it needs no mesh."""
    from pygimli.physics import ert
    data = make_data(sensors, sch)
    log("    geometric factors (analytic, dim=3)...")
    data["k"] = ert.createGeometricFactors(data, dim=3)
    return data


# Per-cell resistivity model on `mesh`: rho_bg everywhere except inside the
# anomaly box (per in_anomaly), where it's rho_anom.
def true_model_on(mesh, anom):
    centers = np.array(mesh.cellCenters())
    res = np.full(mesh.cellCount(), anom["rho_bg"], float)
    res[in_anomaly(centers, anom)] = anom["rho_anom"]
    return res


def invert_for_anomaly(data, fwd_mesh, inv_mesh, anom, log, seed=SEED):
    """Forward-model `anom` on fwd_mesh with noise (NOISE_LEVEL/NOISE_ABS,
    given seed), then invert the resulting synthetic data on inv_mesh with
    pyGIMLi's ERTManager, following the same steps as
    True3DInversionUtilities.run_inversion_3d (setDeltaPhiAbortPercent(0),
    lam=INV_LAMBDA, maxIter=INV_MAXITER, starting from the background
    resistivity). Drops any non-finite or non-positive simulated apparent
    resistivities before inverting, and raises if fewer than 4 readings
    remain usable. Returns (mgr.paraDomain, model) -- the inversion mesh and
    its recovered per-cell resistivity."""
    from pygimli.physics import ert
    res = true_model_on(fwd_mesh, anom)
    sim = ert.simulate(fwd_mesh, res=res, scheme=data,
                       noiseLevel=NOISE_LEVEL, noiseAbs=NOISE_ABS, seed=seed,
                       verbose=False)
    try:
        has_k = sim.haveData("k")
    except Exception:
        has_k = "k" in sim.dataMap().keys()
    if not has_k:
        sim["k"] = data["k"]
    rhoa = np.array(sim["rhoa"], float)
    good = np.isfinite(rhoa) & (rhoa > 0)
    log(f"      simulated rhoa: {len(rhoa)} total, {int(good.sum())} usable "
        f"(finite & >0); range [{np.nanmin(rhoa):.3g}, {np.nanmax(rhoa):.3g}]")
    if good.sum() < 4:
        raise RuntimeError(f"only {int(good.sum())} usable readings after simulate")
    if not good.all():
        sim.remove(~good)
    mgr = ert.ERTManager(sim)
    mgr.setMesh(inv_mesh)
    mgr.inv.inv.setDeltaPhiAbortPercent(0)          # matches True3DInversionUtilities.run_inversion_3d
    mgr.invert(mesh=inv_mesh, lam=INV_LAMBDA, maxIter=INV_MAXITER,
               startModel=anom["rho_bg"], verbose=False)
    return mgr.paraDomain, np.array(mgr.model)


def averaged_metrics(data, fwd_mesh, inv_mesh, anom, log):
    """Invert `anom` under NUM_SEEDS independent noise realisations (seeds
    SEED, SEED+1, ...), score each inversion's cross-line recovery with
    recovery_metrics, and average the scores across seeds so noise-driven
    asymmetry washes out (also records the std of y_spread across seeds when
    NUM_SEEDS > 1). Returns (avg_metrics, para_mesh, model), where para_mesh
    and model come from the FIRST seed only (used for the slice panels)."""
    yy = np.linspace(BOX["y0"], BOX["y1"], SLICE_NY)
    zz = np.linspace(BOX["zbot"], BOX["ztop"], SLICE_NZ)
    seeds = [SEED + i for i in range(max(1, NUM_SEEDS))]
    mlist, pd0, model0 = [], None, None
    for i, sd in enumerate(seeds):
        pd, model = invert_for_anomaly(data, fwd_mesh, inv_mesh, anom, log, seed=sd)
        _, _, g = sample_slice(pd, model, "yz", anom)
        mlist.append(recovery_metrics(yy, zz, g, anom))
        if i == 0:
            pd0, model0 = pd, model
    keys = ["contrast_pct", "y_centroid_err", "y_spread", "peak_at_true"]
    avg = {k: float(np.nanmean([m[k] for m in mlist])) for k in keys}
    if len(seeds) > 1:
        avg["y_spread_sd"] = float(np.nanstd([m["y_spread"] for m in mlist]))
    return avg, pd0, model0


# --------------------------------------------------------------------------
# Slicing + metrics (pure numpy once the model is in hand)
# --------------------------------------------------------------------------
# Interpolate `model` (on inv_mesh) onto a regular grid spanning the current
# BOX: a y-z slice through anom['cx'] when plane='yz', otherwise an x-z slice
# through anom['cy']. Returns (axis_values, z_values, grid), with grid shaped
# (len(axis_values), len(z_values)).
def sample_slice(inv_mesh, model, plane, anom):
    import pygimli as pg
    b = BOX
    z = np.linspace(b["zbot"], b["ztop"], SLICE_NZ)
    if plane == "yz":
        A = np.linspace(b["y0"], b["y1"], SLICE_NY)
        AA, ZZ = np.meshgrid(A, z, indexing="ij")
        pts = np.column_stack([np.full(AA.size, anom["cx"]), AA.ravel(), ZZ.ravel()])
    else:
        A = np.linspace(b["x0"], b["x1"], SLICE_NX)
        AA, ZZ = np.meshgrid(A, z, indexing="ij")
        pts = np.column_stack([AA.ravel(), np.full(AA.size, anom["cy"]), ZZ.ravel()])
    vals = np.array(pg.interpolate(inv_mesh, model, pts))
    return A, z, vals.reshape(AA.shape)


# Same grid/plane construction as sample_slice, but evaluates the TRUE
# resistivity model (rho_bg / rho_anom via in_anomaly) instead of an inverted
# one -- used for the "TRUE" reference panel.
def _true_slice(plane, anom):
    b = BOX
    z = np.linspace(b["zbot"], b["ztop"], SLICE_NZ)
    if plane == "yz":
        A = np.linspace(b["y0"], b["y1"], SLICE_NY)
        AA, ZZ = np.meshgrid(A, z, indexing="ij")
        pts = np.column_stack([np.full(AA.size, anom["cx"]), AA.ravel(), ZZ.ravel()])
    else:
        A = np.linspace(b["x0"], b["x1"], SLICE_NX)
        AA, ZZ = np.meshgrid(A, z, indexing="ij")
        pts = np.column_stack([AA.ravel(), np.full(AA.size, anom["cy"]), ZZ.ravel()])
    grid = np.full(AA.size, anom["rho_bg"])
    grid[in_anomaly(pts, anom)] = anom["rho_anom"]
    return A, z, grid.reshape(AA.shape)


def recovery_metrics(y, z, grid_yz, anom):
    """Score a recovered cross-line (y-z) slice against the true anomaly at
    (anom['cy'], anom['cz']). Sums the anomaly-signed deviation from
    background (positive whether the anomaly is more resistive or more
    conductive than background) over depths within one sz of anom['cz'], to
    get a per-y recovery profile; if BOX is set, the profile is also
    restricted to y inside the electrode footprint (BOX minus BOX_MARGIN) so
    far-field inversion artefacts near the domain edge don't skew the
    centroid/spread. Returns a dict:
      contrast_pct   -- signed % change from background at the (y,z) grid
                        point nearest the true anomaly location
      y_centroid_err -- |recovered profile centroid y - true anomaly y|
      y_spread       -- std of the recovered profile about its centroid
                        (lower = tighter cross-line localisation)
      peak_at_true   -- recovered profile value at the true y, as a fraction
                        of the profile's peak (1.0 = true position IS the peak)
    If the profile sums to <=0 (no usable recovered signal), y_centroid_err
    and y_spread are NaN and the other two are 0.0."""
    bg = anom["rho_bg"]
    dev = grid_yz - bg
    sign = -1.0 if anom["rho_anom"] < bg else 1.0
    strength = np.clip(sign * dev, 0, None)
    zband = np.abs(z - anom["cz"]) <= anom["sz"]
    prof = strength[:, zband].sum(axis=1)
    if BOX is not None:                                  # keep only the electrode span
        inside = (y >= BOX["y0"] + BOX_MARGIN) & (y <= BOX["y1"] - BOX_MARGIN)
        prof = prof * inside
    total = prof.sum()
    if total <= 0:
        return dict(contrast_pct=0.0, y_centroid_err=np.nan, y_spread=np.nan, peak_at_true=0.0)
    y_centroid = float((y * prof).sum() / total)
    y_var = float((prof * (y - y_centroid) ** 2).sum() / total)
    y_spread = float(np.sqrt(max(y_var, 0.0)))
    iy = int(np.argmin(np.abs(y - anom["cy"])))
    iz = int(np.argmin(np.abs(z - anom["cz"])))
    contrast = float((grid_yz[iy, iz] - bg) / bg * 100.0)
    peak_at_true = float(prof[iy] / prof.max()) if prof.max() > 0 else 0.0
    return dict(contrast_pct=contrast,
                y_centroid_err=abs(y_centroid - anom["cy"]),
                y_spread=y_spread, peak_at_true=peak_at_true)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
# Multi-panel figure: one pcolormesh panel per (label, A, z, grid) entry in
# `results` (the first entry is typically "TRUE"), sharing one log-scaled
# color axis, with the true anomaly's footprint outlined by a dashed box.
# plane='yz' labels the panel axis as y (across the lines); anything else is
# treated as x (along the lines). Saves the figure to out_path.
def plot_panels(results, plane, anom, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    vmin = min(g.min() for _, _, _, g in results)
    vmax = max(g.max() for _, _, _, g in results)
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-2), vmax=vmax)
    a_lab = "y across the lines (m)" if plane == "yz" else "x along the lines (m)"
    for ax, (label, A, z, grid) in zip(axes[0], results):
        pm = ax.pcolormesh(A, z, grid.T, cmap=CMAP, norm=norm, shading="auto")
        if plane == "yz":
            a0, a1 = anom["cy"] - anom["sy"] / 2, anom["cy"] + anom["sy"] / 2
        else:
            a0, a1 = anom["cx"] - anom["sx"] / 2, anom["cx"] + anom["sx"] / 2
        z0, z1 = anom["cz"] - anom["sz"] / 2, anom["cz"] + anom["sz"] / 2
        ax.plot([a0, a1, a1, a0, a0], [z0, z0, z1, z1, z0], "k--", lw=1.2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(a_lab); ax.set_ylabel("depth (m)")
        fig.colorbar(pm, ax=ax, shrink=0.85, label="resistivity (ohm-m)")
    ttl = ("Cross-line (y-z) recovery: dashed = true anomaly"
           if plane == "yz" else "In-line (x-z) recovery: dashed = true anomaly")
    fig.suptitle(ttl, y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"Wrote {out_path}")


def plot_resolution_curves(sweep, line_ys, out_path):
    """Plot the resolution-vs-location sweep: two side-by-side panels
    (recovered cross-line spread, and cross-line mislocation |dy|) vs. true
    anomaly y-position, one line per design. `sweep` is
    {design: [(true_y, metrics), ...]} as built in main(); `line_ys` (the
    union of every design's electrode-line y-positions) are drawn as
    vertical dotted reference lines. Saves the figure to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#e6194B", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#42d4f4"]
    fig, (axS, axE) = plt.subplots(1, 2, figsize=(13, 5))
    for i, (design, rows) in enumerate(sweep.items()):
        ys = [r[0] for r in rows]
        spread = [r[1]["y_spread"] for r in rows]
        err = [r[1]["y_centroid_err"] for r in rows]
        c = colors[i % len(colors)]
        axS.plot(ys, spread, "-o", color=c, label=design, lw=1.8, ms=5)
        axE.plot(ys, err, "-o", color=c, label=design, lw=1.8, ms=5)
    for ax in (axS, axE):
        for ly in line_ys:
            ax.axvline(ly, color="#cbd5e0", ls=":", lw=1, zorder=0)
        ax.set_xlabel("true anomaly y-position (m)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axS.set_ylabel("recovered cross-line spread (m)  —  lower = sharper")
    axS.set_title("Cross-line resolution vs location")
    axE.set_ylabel("cross-line mislocation |Δy| (m)  —  lower = truer")
    axE.set_title("Cross-line positioning vs location")
    fig.suptitle("Resolution vs location  (dotted grey = electrode lines)", y=1.01, fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"Wrote {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
# Orchestrates a full run: load/build every design, mesh each one once, run
# the single-position (REF_Y) comparison (always), and then the
# resolution-vs-location sweep (only if DO_SWEEP and not QUICK_TEST).
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log = print
    ensure_tetgen_on_path(log)                       # Windows: make tetgen findable
    try:
        import pygimli as pg
        log(f"pyGIMLi version: {pg.__version__}")
    except Exception as e:
        log(f"could not import pygimli: {e}")

    designs = xml_designs() if USE_XML else programmatic_designs()
    designs = {lbl: (s, dedupe_reciprocals(sch)) for lbl, (s, sch) in designs.items()}
    if QUICK_TEST:
        first = next(iter(designs))
        designs = {first: designs[first]}
        log(f"QUICK_TEST: only {first}, reference position only, no sweep")
    log(f"GRID = {GRID}  (anomaly cy={ANOM['cy']} sy={ANOM['sy']}, sweep {SWEEP_Y})")
    log(f"Comparing {len(designs)} design(s): {', '.join(designs)}")

    global BOX

    # Each design gets its OWN mesh + box (rather than sharing one mesh from
    # the first design), since designs can have different electrode
    # footprints (e.g. 2x32 vs 4x16) -- sharing a mesh would silently put
    # another design's electrodes, or the anomaly, outside its own domain.
    prepared = {}
    all_line_ys = set()
    for label, (sensors, sch) in designs.items():
        log(f"\nPreparing {label} ({len(sch['a'])} readings)...")
        check_grid_match(label, sensors, log)   # abort fast on a GRID/XML mismatch
        box = box_from_sensors(sensors)
        ely = sorted(np.unique(np.round(sensors[:, 1], 6)).tolist())
        all_line_ys.update(ely)
        ey0, ey1 = min(ely), max(ely)
        # Flag if any anomaly y-position this run will use pokes outside this
        # design's box -- beyond the box it simply can't be recovered. (A
        # small poke past the outermost electrode line but still inside the
        # box is fine, just edge-degraded.)
        y_positions = list(SWEEP_Y) + [REF_Y] if (DO_SWEEP and not QUICK_TEST) else [REF_Y]
        lo = min(y_positions) - ANOM["sy"] / 2
        hi = max(y_positions) + ANOM["sy"] / 2
        if lo < box["y0"] - 1e-9 or hi > box["y1"] + 1e-9:
            log(f"    WARNING: anomaly y-range [{lo:.2f},{hi:.2f}] extends outside "
                f"{label}'s domain (box y [{box['y0']:.2f},{box['y1']:.2f}], electrodes "
                f"[{ey0:.2f},{ey1:.2f}]) -- those recoveries will be pinned to the edge "
                f"and unreliable. Check GRID matches these designs' grid.")
        BOX = box
        data = prepare_design(sensors, sch, log)
        fmesh = build_para_mesh(sensors, PARA_MAX_CELL_FWD, log)
        imesh = build_para_mesh(sensors, PARA_MAX_CELL_INV, log)
        prepared[label] = dict(data=data, fwd=fmesh, inv=imesh, box=box, sch=sch)
    line_ys = sorted(all_line_ys)

    # Display box spanning every design's box, used only for the TRUE
    # reference panels (each design's own recovered slice still uses its own
    # box, set again below).
    disp = dict(x0=min(P["box"]["x0"] for P in prepared.values()),
                x1=max(P["box"]["x1"] for P in prepared.values()),
                y0=min(P["box"]["y0"] for P in prepared.values()),
                y1=max(P["box"]["y1"] for P in prepared.values()),
                ztop=0.0, zbot=-PARA_DEPTH)

    # ---- reference-position slice panels ----
    ref = anom_at(cy=REF_Y, cz=SWEEP_DEPTH)
    BOX = disp
    tyz = _true_slice("yz", ref); txz = _true_slice("xz", ref)
    yz_panels = [("TRUE", *tyz)]; xz_panels = [("TRUE", *txz)]
    ref_rows = []
    for label, P in prepared.items():
        BOX = P["box"]                                    # slice/metric in THIS design's frame
        log(f"\n[REF y={REF_Y}] {label} (averaging {NUM_SEEDS} seed(s)) ...")
        try:
            m, pd, model = averaged_metrics(P["data"], P["fwd"], P["inv"], ref, log)
        except Exception as e:
            log(f"    FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            if DEBUG:
                raise            # stop here so the real error is visible, not swallowed
            continue
        yz_panels.append((label, *sample_slice(pd, model, "yz", ref)))
        xz_panels.append((label, *sample_slice(pd, model, "xz", ref)))
        m["design"] = label; m["n_readings"] = len(P["sch"]["a"]); ref_rows.append(m)
        log(f"    contrast={m['contrast_pct']:.0f}%  y-err={m['y_centroid_err']:.3f}  "
            f"y-spread={m['y_spread']:.3f}")
    plot_panels(yz_panels, "yz", ref, os.path.join(OUT_DIR, "recovery_crossline.png"))
    plot_panels(xz_panels, "xz", ref, os.path.join(OUT_DIR, "recovery_inline.png"))
    _write_csv(os.path.join(OUT_DIR, "recovery_metrics.csv"), ref_rows)

    # ---- resolution-vs-location sweep ----
    if DO_SWEEP and not QUICK_TEST:
        log(f"\n=== SWEEP over y = {SWEEP_Y} (depth {SWEEP_DEPTH}) ===")
        sweep = {label: [] for label in prepared}
        sweep_rows = []
        for label, P in prepared.items():
            BOX = P["box"]
            for yc in SWEEP_Y:
                a = anom_at(cy=yc, cz=SWEEP_DEPTH)
                log(f"  {label} @ y={yc:.2f} (averaging {NUM_SEEDS} seed(s)) ...")
                try:
                    m, _, _ = averaged_metrics(P["data"], P["fwd"], P["inv"], a, log)
                except Exception as e:
                    log(f"    FAILED: {type(e).__name__}: {e}")
                    if DEBUG:
                        log(traceback.format_exc()); raise
                    m = dict(contrast_pct=np.nan, y_centroid_err=np.nan,
                             y_spread=np.nan, peak_at_true=np.nan)
                sweep[label].append((yc, m))
                row = dict(m); row["design"] = label; row["true_y"] = yc
                sweep_rows.append(row)
        plot_resolution_curves(sweep, line_ys,
                               os.path.join(OUT_DIR, "resolution_vs_location.png"))
        _write_csv(os.path.join(OUT_DIR, "resolution_sweep.csv"), sweep_rows,
                   cols=["design", "true_y", "y_spread", "y_centroid_err",
                         "contrast_pct", "peak_at_true"])
        log("\nResolution-vs-location: lower y-spread AND lower |dy| = better "
            "cross-line resolution. Compare designs, and watch for dips BETWEEN the "
            "electrode lines (dotted grey) where cross-line power is weakest.")


# Write `rows` (a list of dicts) to `path` as CSV using `cols` as the column
# order (defaults to the recovery_metrics.csv columns); a row missing a key
# in `cols` writes '' for that cell. No-op if `rows` is empty.
def _write_csv(path, rows, cols=None):
    if not rows:
        return
    cols = cols or ["design", "n_readings", "contrast_pct", "y_centroid_err",
                    "y_spread", "peak_at_true"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

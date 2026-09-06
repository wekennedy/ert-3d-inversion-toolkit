"""
Density-vs-Predicted-Field-Time Comparison
==========================================

Ranks candidate (spread, protocol) survey designs by two independent axes:

  1. DENSITY = a per-cell MEASUREMENT COUNT, i.e. how many readings' geometric
     reach includes that cell (see estimate_reach_coverage()). This is a
     geometric heuristic, not a Jacobian/sensitivity calculation.

  2. COMPARABILITY = each design's density is normalised against its OWN
     footprint (the region reached by >= 1 reading), so designs of different
     shape/size (e.g. a squat 4x16 vs. a long 2x32) are compared on "of the
     ground you can see at all, how much do you see DENSELY (>= K overlapping
     readings)?" rather than on raw cell counts.

  3. COST AXIS = predicted field acquisition time on an ABEM Terrameter LS2,
     modelled from current injections rather than raw reading count: the LS2
     reads up to LS2_CHANNELS potential dipoles per current injection, so
     readings that share a current pair (A,B) are channel-packed into the same
     energisation. Standard arrays (one reading per current pair) reduce to
     roughly one injection per reading; gradient-type arrays (many readings
     per current pair) are amortised across fewer injections. Reciprocal
     readings swap the current/potential roles, so they fall into their own
     injection groups rather than reusing the forward reading's group.

WHAT IT PRODUCES (written into OUT_DIR by run())
-------------------------------------------------
  - density_vs_time.png          Predicted field time (x) vs. % of own
                                  footprint densely probed (y), at
                                  PRIMARY_THRESHOLD. Better designs sit
                                  toward the top-left.
  - density_vs_time_topleft.png  Only written if MAKE_ZOOM_PLOT. A zoomed
                                  view of the ZOOM_SHOW_N designs nearest the
                                  ideal (fastest, densest) corner, labelled,
                                  with the non-dominated (Pareto) frontier
                                  drawn.
  - density_vs_depth.png         Density vs. depth, one line per design, each
                                  normalised to its own surface value.
  - threshold_sensitivity.png    % of own footprint dense vs. the K threshold
                                  (one line per design, over THRESHOLDS).
  - density_slices.png           Top-N only: a vertical section (length x
                                  depth) through each design's densest row and
                                  a plan view (length x width) at its densest
                                  depth, coloured by reach count.
  - sensitivity_slices.png       Only if MAKE_SENSITIVITY_SLICES. Top-N only:
                                  the same section/plan layout, coloured by
                                  analytic homogeneous-half-space sensitivity
                                  instead of reach count.
  - density_runtime_comparison.csv   All per-design numbers (see below).
  - density_runtime_report.txt   Plain-text log of everything printed while
                                  run() executed.

CSV COLUMNS (density_runtime_comparison.csv)
---------------------------------------------
Rows are sorted best-first (highest primary-threshold density, then shortest
time). Columns:

  label                 candidate name/key as given in CANDIDATES.
  n_readings            number of readings parsed from the protocol file
                        (one A/B/M/N quadrupole = one reading; if the
                        protocol includes reciprocals they are counted here
                        too, minus any dropped by validate_readings()).
  n_injections          number of current energisations after channel-packing
                        readings that share a current pair (A,B), up to
                        LS2_CHANNELS per injection. This is what drives field
                        time; for standard arrays it tracks n_readings
                        closely, for gradient-type arrays it is much smaller.
  time_min              predicted LS2 field time in minutes:
                        n_injections * (CURRENT_ON_S + DELAY_S) / 60.
  footprint_pct_of_box  % of the modelled box VOLUME reached by >= 1 reading.
                        This is absolute reach (how much ground the design
                        sees at all) and is not footprint-normalised, so read
                        it alongside the dense_pct_K<k> columns when the
                        question is "how much of the box" rather than "how
                        densely within what it does reach".
  dense_pct_K<k>        the density metric: % of THIS design's own footprint
                        (its >= 1-reading region) that is reached by >= k
                        readings, one column per value in THRESHOLDS
                        (dense_pct_K1, _K3, _K5, _K10, _K20 with the defaults
                        below). dense_pct_K1 is always 100 by definition,
                        since the footprint is defined as the >= 1 region.
                        The hero plot's y-axis is dense_pct_K<PRIMARY_THRESHOLD>.

HOW TO USE
----------
1. Fill in CANDIDATES with the (spread, protocol) XML file pairs to compare.
2. Run this file directly, or call run(candidates, out_dir) programmatically
   (e.g. from a sweep script) to reuse the same plotting/CSV pipeline on a
   different set of candidates.

NOTES ON THE UNDERLYING HEURISTICS
-----------------------------------
The XML parsing, sensor/scheme construction, and the estimate_reach_coverage()
geometric reach heuristic (DOI_FACTOR / lateral-pad logic) are adapted from an
existing forward-modelling pipeline used elsewhere in this project, so the
density field is computed the same way there. The LS2 timing model and the
footprint-normalised density metric are specific to this script; treat both
as design-comparison heuristics rather than as guaranteed absolute predictions
of real survey time or resolvable resistivity. The pyGIMLi mesh calls
(mt.createCube / mt.createMesh / mesh.cellSizes / mesh.cellCenters) are
standard pyGIMLi meshtools usage.
"""

import os
import gc
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pygimli as pg
    import pygimli.meshtools as mt
except ImportError as e:
    raise ImportError(
        "pyGIMLi could not be imported. Run this in your ERT environment.\n"
        f"Original error: {e}"
    )


# ============ USER INPUTS: EDIT THIS SECTION ============

# One entry per (spread, protocol) pair to compare. The dict key is the label
# used everywhere else (plots, CSV, console log). Each value needs "spread"
# and "protocol" file paths; an optional "array" key (e.g. "wenner") overrides
# the array-type guess that infer_array() would otherwise make from the label,
# and controls which ARRAY_COLORS entry a non-top-N design is drawn with.
CANDIDATES = {
    "4x16 W-S":  {"spread": "spread_wenner_s.xml", "protocol": "protocol_wenner_s_rcd.xml"},
    # "2x32 W-S":  {"spread": "spread_2x32.xml",      "protocol": "protocol_2x32_ws.xml"},
    # "4x16 DD":   {"spread": "spread_wenner_s.xml",  "protocol": "protocol_4x16_dd.xml"},
    # "4x16 GRAD": {"spread": "spread_gradient.xml",  "protocol": "protocol_4x16_grad.xml"},
}

OUT_DIR = "./Output/DensityRuntime"

# --- Density metric ---
# A cell counts as "densely probed" once at least this many readings reach it.
# PRIMARY_THRESHOLD is the K used for the hero plot's y-axis and must be one
# of the values listed in THRESHOLDS. THRESHOLDS is the full list of K values
# swept for the CSV's dense_pct_K<k> columns and the threshold-sensitivity plot.
PRIMARY_THRESHOLD = 5
THRESHOLDS = [1, 3, 5, 10, 20]

# --- Plot readability ---
# The TOP_N best-ranked designs (by RANK_METRIC) are drawn with a distinct
# vivid colour and get their own legend entry, plotted title/labels, and
# section figures; every other design is drawn faded, coloured only by its
# array type (ARRAY_COLORS).
TOP_N = 5
# How the TOP_N are chosen:
#   "density"    -> (default; also the fallback for any other value) highest
#                   % dense at PRIMARY_THRESHOLD.
#   "efficiency" -> highest % dense at PRIMARY_THRESHOLD per minute of
#                   predicted field time.
RANK_METRIC = "density"

# On the scatter plots, points that would land on (nearly) the same spot are
# nudged onto a small ring around their true location so none is hidden behind
# another -- purely cosmetic, the CSV always holds the true values. Set to
# False to plot exact positions. OVERLAP_SPREAD sets the ring radius as a
# fraction of each axis's data range.
SEPARATE_OVERLAPS = True
OVERLAP_SPREAD = 0.013

# --- Zoomed 'best designs' plot (top-left corner of the hero plot) ---
# If True, writes a second figure, density_vs_time_topleft.png, zoomed on the
# ZOOM_SHOW_N designs whose (time, density) is closest to the ideal (fastest
# time, highest density) corner, each labelled, with the non-dominated
# (Pareto) trade-off frontier drawn dashed for context. The axis limits are
# fit to just those designs plus ZOOM_MARGIN of padding, with extra headroom
# above so a design sitting near 100% density isn't clipped. Raise
# ZOOM_SHOW_N to include more designs in the zoom, lower it to zoom in tighter
# on just the best few.
MAKE_ZOOM_PLOT = True
ZOOM_SHOW_N = 8             # how many of the closest-to-ideal designs to show
ZOOM_MARGIN = 0.12          # padding around those designs, fraction of their span
ZOOM_LABEL_POINTS = True    # write each shown design's label beside its point

# --- Density slice sections (rendered for the top-N only) ---
# For each top design: a vertical section (length x depth) through its
# densest row, and a plan view (length x width) at its densest depth. Both
# are sampled on a regular grid with estimate_reach_coverage (the same
# density definition used for the metric above) and rendered in
# render_density_slices().
SLICE_NX, SLICE_NY, SLICE_NZ = 120, 40, 40   # sampling resolution (x, y, depth)
# Colour ramp for the slice fill, running light (sparse) to dark (dense). Any
# sequential colormap whose dark end is its high end works ("Blues",
# "Purples", "Greys", "mako", or a reversed map like "magma_r").
# SLICE_LOW_CLIP trims the palest end of that colormap so low-but-nonzero
# cells still show up against the grey "no coverage" background.
SLICE_CMAP = "Blues"
SLICE_LOW_CLIP = 0.12
# Slice colour scale:
#   "per_design" -> (default; also the fallback for any other value) each row
#                   gets its own colour scale and its own colourbar, spanning
#                   that design's sparsest -> densest covered cell in
#                   absolute reading counts. The full ramp is used inside
#                   every panel, so within-design patterns stay visible even
#                   for low-count designs, but shades are NOT comparable
#                   between rows -- read each row against its own colourbar.
#                   Each row's label shows its own count range (e.g.
#                   "12-37 readings").
#   "shared"     -> one absolute readings-per-cell scale shared by every row:
#                   the same shade means the same count everywhere, so
#                   designs become directly comparable, but a high-peak
#                   design (e.g. a dipole-dipole reaching the hundreds) can
#                   push low-count designs toward one flat pale colour. Keep
#                   SLICE_SHARED_LOG on to soften that.
SLICE_SCALE = "per_design"
# On the shared scale, use log spacing (counts commonly span roughly 1 to
# several hundred, so log keeps sparse designs legible while still showing
# the magnitude gap). Set False for a plain linear count ramp -- reasonable
# only when the compared designs have similar peak counts.
SLICE_SHARED_LOG = True
# On the per-design scale, each row is normalised over its own [sparsest,
# densest] count range. Linear (False, default) spreads that range evenly;
# log (True) expands the low end, which only helps when a design's own count
# range is very wide (e.g. a dipole-dipole with a handful of readings at the
# rim and hundreds at the core).
SLICE_PERDESIGN_LOG = False
# Optional iso-density contour lines drawn over the slice fill. These read
# well on smooth fields (e.g. a dipole-dipole row, where coverage decays
# gradually), but the reach field of a collinear line array is coarse and
# piecewise-constant, so on those designs the contours turn into jagged
# rectangular staircases -- left off by default for that reason. Levels are
# read from SLICE_CONTOUR_PCTS (% of that row's own peak) in per_design mode,
# or SLICE_CONTOUR_COUNTS (absolute reading counts) in shared mode.
SLICE_CONTOURS = False
SLICE_CONTOUR_PCTS = [10, 25, 50, 75, 90]        # per_design: % of that row's peak
SLICE_CONTOUR_COUNTS = [2, 5, 10, 20, 50, 100]   # shared: absolute readings/cell
# In the two slice figures (density_slices.png and sensitivity_slices.png)
# only, collapse readings that share the same four electrodes down to one: a
# reciprocal reading uses the identical four electrodes as its forward
# reading, so it marks the same reach box rather than adding new ground, and
# counting both just double-counts that footprint. Since reciprocals are
# often only a random subset of readings, that double-counting can make an
# otherwise-symmetric design look lopsided in the slice views. Leave this ON
# for a cleaner coverage picture; it does NOT affect the density metric/CSV
# upstream, which counts every reading (reciprocals included) as a measure of
# redundancy, a different question from "where does coverage reach".
SLICE_DEDUPE_RECIPROCALS = True

# --- Real sensitivity slices (analytic homogeneous half-space) ---
# If True, writes a second set of slice figures (sensitivity_slices.png)
# built from the true Fréchet sensitivity (estimate_sensitivity_coverage)
# rather than the geometric reach count -- i.e. where each design can
# actually resolve resistivity contrast, including its natural depth decay.
# Uses the same top-N designs and the same section + plan-view layout as the
# density slices. The 1/r^3 electrode singularity is handled by masking cells
# within SENS_MASK_FRAC of the minimum electrode spacing and clipping the
# (log) colour scale at the SENS_CAP_PCTL percentile, which is what keeps the
# plot legible where an unclipped sensitivity field would be all-black with
# hot spikes at the electrodes. This is a homogeneous-half-space, design-stage
# sensitivity, so it should be read as a RELATIVE comparison between designs,
# not as an absolute resolution in ohm-m.
MAKE_SENSITIVITY_SLICES = True
SENS_SCALE = "per_design"    # "per_design" (own scale + own bar per row) or "shared"
SENS_MASK_FRAC = 0.5         # mask cells closer than this * minimum electrode spacing
SENS_CAP_PCTL = 98           # clip colour scale at this percentile (drops the singular tail)
SENS_FLOOR_DECADES = 3.0     # show this many log-decades of sensitivity below the cap
SENS_CMAP = "magma"          # sequential map; reads well for a decaying field

# Colour per array type, used for the faded (non-top-N) background points and
# lines. Kept bright and well-separated so those dimmed points still stay
# distinguishable from each other even at reduced opacity.
ARRAY_COLORS = {
    "wenner":              "#00a5b5",   # teal
    "wenner_schlumberger": "#4f6bed",   # blue
    "dipole_dipole":       "#5cb85c",   # green
    "gradient":            "#f5872e",   # orange
    "other":              "#9e9e9e",   # grey
}
# Vivid, well-separated colours for the top-N designs, assigned in rank order.
HIGHLIGHT_PALETTE = ["#e6194B", "#4363d8", "#3cb44b", "#f58231", "#911eb4",
                     "#42d4f4", "#f032e6", "#bfef45"]

# Short array-type tags, as they might appear as tokens in a candidate label
# (e.g. "4x16_ws_rows"), mapped to the full array-type name used elsewhere.
# Used by infer_array() to colour hand-written CANDIDATES that don't carry an
# explicit "array" key.
TAG_TO_ARRAY = {"w": "wenner", "ws": "wenner_schlumberger",
                "dd": "dipole_dipole", "grad": "gradient"}

# --- Modelling domain ---
BOX_DEPTH = 1.0          # m, modelled soil-box depth; the mesh (and hence any DOI) is capped here.
FOOTPRINT_PAD = 0.2      # m, lateral padding added around each design's own electrodes so
                         # edge reach isn't clipped by the mesh boundary. The domain is
                         # auto-fit per design; set CLIP_TO_PHYSICAL_BOX to hard-limit it instead.
CLIP_TO_PHYSICAL_BOX = False
PHYSICAL_BOX = (3.0, 1.0)   # (size_x, size_y) used instead of the auto-fit domain when clipping is on

MESH_QUALITY = 1.2
MESH_VOLUME = 0.001      # m^3, target mesh cell size; a single resolution is used for every design.

# --- Reach heuristic ---
DOI_FACTOR = 0.17          # depth of investigation for a reading = this * the reading's own max electrode spacing
LATERAL_PAD_FACTOR = 1.0   # half-electrode-spacing lateral padding applied to a reading whose
                           # electrodes are collinear along one axis, so it still covers some width

# --- LS2 field-time model ---
CURRENT_ON_S = 5.0       # seconds of measure/current-on time per injection
DELAY_S = 0.5            # seconds of delay per injection
LS2_CHANNELS = 12        # potential dipoles the LS2 can read in parallel per current injection
MERGE_AB_POLARITY = True # treat (A,B) and (B,A) as the same current pair when grouping for channel-packing

# ============ END OF USER INPUTS ============

CYCLE_S = CURRENT_ON_S + DELAY_S


# ---------------------------------------------------------------------------
# XML parsing + sensor/scheme build
# ---------------------------------------------------------------------------

# Parse a spread XML file into {electrode switch address: (x, y)}. Reads
# every <Electrode> element that has SwitchAddress/X/Y children; elements
# missing any of those are skipped. Raises FileNotFoundError if the file
# doesn't exist and ValueError if no usable electrodes are found.
def parse_spread_xml(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Spread file not found: {path} (cwd: {os.getcwd()})")
    root = ET.parse(path).getroot()
    electrodes = {}
    for elec in root.iter("Electrode"):
        a = elec.find("SwitchAddress"); x = elec.find("X"); y = elec.find("Y")
        if a is None or x is None or y is None:
            continue
        electrodes[int(a.text)] = (float(x.text), float(y.text))
    if not electrodes:
        raise ValueError(f"No usable <Electrode> entries in {path}.")
    return electrodes


# Parse a protocol XML file into a list of (a, b, m, n) readings. Each
# <Measure> under <Sequence> supplies one current pair (its <Tx>) and one
# reading per <Rx> it contains, so a single Measure can expand into several
# readings. Raises FileNotFoundError if the file doesn't exist, and
# ValueError if there is no <Sequence> element or no usable readings.
def parse_protocol_xml(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Protocol file not found: {path} (cwd: {os.getcwd()})")
    root = ET.parse(path).getroot()
    seq = root.find("Sequence")
    if seq is None:
        raise ValueError(f"No <Sequence> element in {path}.")
    readings = []
    for meas in seq.findall("Measure"):
        tx = meas.find("Tx")
        if tx is None or not (tx.text or "").strip():
            continue
        a, b = (int(v) for v in tx.text.split())
        for rx in meas.findall("Rx"):
            if not (rx.text or "").strip():
                continue
            m, n = (int(v) for v in rx.text.split())
            readings.append((a, b, m, n))
    if not readings:
        raise ValueError(f"No usable readings in {path}.")
    return readings


# Drop degenerate readings (any electrode reused across a/b/m/n). A reading
# is kept only if its four electrode addresses are all distinct. Dropped
# readings are counted and reported through log(); returns the cleaned list.
def validate_readings(readings, log):
    clean, dropped = [], []
    for r in readings:
        (dropped if len(set(r)) < 4 else clean).append(r)
    if dropped:
        log(f"  WARNING: dropped {len(dropped)} degenerate reading(s) (reused electrode).")
    return clean


# Build the (sensors, scheme) pair used by the rest of the pipeline. sensors
# is an (N, 3) array of electrode (x, y, 0) positions, ordered by ascending
# switch address. scheme is a dict with keys "a", "b", "m", "n", each an
# array of sensor-array row indices (not raw switch addresses), one entry
# per reading. Raises ValueError if the protocol references an electrode
# address that isn't present in the spread.
def build_sensors_and_scheme(electrodes, readings):
    addrs = sorted(electrodes)
    idx = {a: i for i, a in enumerate(addrs)}
    sensors = np.array([[electrodes[a][0], electrodes[a][1], 0.0] for a in addrs])
    missing = set(x for r in readings for x in r) - set(electrodes)
    if missing:
        raise ValueError(f"Protocol references electrodes {sorted(missing)} absent from spread.")
    return sensors, {k: np.array([idx[r[j]] for r in readings], float)
                     for j, k in enumerate("abmn")}


# ---------------------------------------------------------------------------
# Domain + mesh
# ---------------------------------------------------------------------------

def build_domain_mesh(sensors, log):
    """Build the pyGIMLi mesh for one design's modelled box.

    The box footprint is the bounding box of this design's electrodes plus
    FOOTPRINT_PAD on every side, with depth BOX_DEPTH; auto-fitting the
    footprint per design keeps each design's full reach inside its own mesh
    so differently-shaped spreads are compared on equal footing. If
    CLIP_TO_PHYSICAL_BOX is True, the footprint is instead recentred on the
    same point but forced to PHYSICAL_BOX's fixed size (logging a warning if
    that clips any electrode). Logs the resulting domain extents and cell
    count, and returns the mesh.
    """
    xmn, xmx = sensors[:, 0].min() - FOOTPRINT_PAD, sensors[:, 0].max() + FOOTPRINT_PAD
    ymn, ymx = sensors[:, 1].min() - FOOTPRINT_PAD, sensors[:, 1].max() + FOOTPRINT_PAD

    if CLIP_TO_PHYSICAL_BOX:
        cx, cy = (xmn + xmx) / 2.0, (ymn + ymx) / 2.0
        sx, sy = PHYSICAL_BOX
        xmn, xmx = cx - sx / 2.0, cx + sx / 2.0
        ymn, ymx = cy - sy / 2.0, cy + sy / 2.0
        if (sensors[:, 0].min() < xmn or sensors[:, 0].max() > xmx or
                sensors[:, 1].min() < ymn or sensors[:, 1].max() > ymx):
            log(f"  WARNING: electrodes extend outside the physical box {PHYSICAL_BOX} "
                "-- reach near the edges will be clipped.")

    size = [xmx - xmn, ymx - ymn, BOX_DEPTH]
    pos = [(xmn + xmx) / 2.0, (ymn + ymx) / 2.0, -BOX_DEPTH / 2.0]
    world = mt.createCube(size=size, pos=pos, marker=1)
    mesh = mt.createMesh(world, quality=MESH_QUALITY, area=MESH_VOLUME)
    log(f"  domain x=[{xmn:.2f},{xmx:.2f}] y=[{ymn:.2f},{ymx:.2f}] z=[-{BOX_DEPTH:.2f},0], "
        f"{mesh.cellCount()} cells")
    return mesh


def estimate_reach_coverage(cell_centers, sensors, sch, doi_factor, lateral_pad_factor, tol=1e-6):
    """Geometric per-cell reading count (the density heuristic).

    For each reading, the "reach" region is a box in x/y spanning its four
    electrodes' bounding box, extended down to a depth doi = doi_factor *
    (the reading's own largest pairwise electrode distance). If the four
    electrodes are collinear along x or y (span < tol on that axis), the box
    is widened on that axis by lateral_pad_factor * (median electrode
    spacing) / 2 on each side, since a zero-width box would otherwise reach
    no cells at all for a straight-line reading. Every cell inside a
    reading's box has its count incremented; returns an integer array, one
    count per cell, of how many readings reach it. This is a geometric
    heuristic, not a computed sensitivity, intended for comparing designs
    against each other rather than as an absolute resolution measure.
    """
    xs = np.unique(np.round(sensors[:, 0], 6)); ys = np.unique(np.round(sensors[:, 1], 6))
    xr = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.0
    yr = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.0
    if yr <= 0: yr = xr
    if xr <= 0: xr = yr

    a, b, m, n = (sch[k].astype(int) for k in "abmn")
    cx, cy, cz = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]
    cov = np.zeros(len(cell_centers), dtype=int)

    for i in range(len(a)):
        pts = sensors[[a[i], b[i], m[i], n[i]]][:, :2]
        dmax = max(float(np.hypot(pts[p, 0] - pts[q, 0], pts[p, 1] - pts[q, 1]))
                   for p in range(4) for q in range(p + 1, 4))
        doi = doi_factor * dmax
        xrng = pts[:, 0].max() - pts[:, 0].min()
        yrng = pts[:, 1].max() - pts[:, 1].min()
        xp = (lateral_pad_factor * xr / 2.0) if xrng < tol else 0.0
        yp = (lateral_pad_factor * yr / 2.0) if yrng < tol else 0.0
        xmn, xmx = pts[:, 0].min() - xp, pts[:, 0].max() + xp
        ymn, ymx = pts[:, 1].min() - yp, pts[:, 1].max() + yp
        inside = (cx >= xmn) & (cx <= xmx) & (cy >= ymn) & (cy <= ymx) & (cz <= 0) & (cz >= -doi)
        cov[inside] += 1
    return cov


def estimate_sensitivity_coverage(cell_centers, sensors, sch, mask_radius=0.0):
    """Per-cell cumulative Fréchet sensitivity for a homogeneous half-space.

    Sums, over every reading, coverage(r) = |grad(u_current) . grad(u_potential)|
    at each cell centre r, where the potential from a unit point source at
    electrode p is taken proportional to 1/|r-p| (surface electrodes, so the
    half-space image doubles it -- a constant factor that drops out of a
    relative map), giving grad(1/|r-p|) = -(r-p)/|r-p|^3. The current dipole
    is A(+)/B(-) and the potential dipole is M(+)/N(-). This is the same
    quantity pyGIMLi's coverage() would give on a homogeneous model; it is
    computed directly here so the 1/r^3 electrode singularity can be masked
    cleanly -- any cell within mask_radius of ANY electrode is set to NaN.
    Because it assumes a homogeneous half-space, treat the result as a
    RELATIVE map between designs, not an absolute sensitivity.
    """
    a, b, m, n = (sch[k].astype(int) for k in "abmn")
    used = np.unique(np.concatenate([a, b, m, n]))
    grad = {}
    for e in used:
        d = cell_centers - sensors[e]
        r = np.sqrt((d * d).sum(axis=1))
        r3 = np.where(r < 1e-9, np.nan, r ** 3)
        grad[e] = d / r3[:, None]                      # ~ grad(1/|r-p|), up to sign
    total = np.zeros(len(cell_centers))
    for ai, bi, mi, ni in zip(a, b, m, n):
        gc = grad[ai] - grad[bi]                       # current-dipole field gradient
        gp = grad[mi] - grad[ni]                       # potential-dipole field gradient
        total += np.abs((gc * gp).sum(axis=1))         # |S_k(r)|, summed
    if mask_radius > 0:                                # blank the singular near-electrode cells
        dmin = np.full(len(cell_centers), np.inf)
        for e in range(len(sensors)):
            d = cell_centers - sensors[e]
            dmin = np.minimum(dmin, np.sqrt((d * d).sum(axis=1)))
        total[dmin < mask_radius] = np.nan
    return total


# ---------------------------------------------------------------------------
# The two new pieces: footprint-normalised density + LS2 field time
# ---------------------------------------------------------------------------

def dense_fraction(coverage, volumes, k):
    """Volume-weighted % of this design's own footprint reached by >= k readings.

    The footprint is the volume where coverage >= 1. Dividing the >= k volume
    by that footprint volume (rather than by the whole modelled box) makes
    the result shape-independent, so designs with different footprints are
    directly comparable. Returns 0.0 if the design has no footprint at all.
    """
    foot = volumes[coverage >= 1].sum()
    if foot <= 0:
        return 0.0
    return float(volumes[coverage >= k].sum() / foot)


def predict_field_time(readings):
    """Predict LS2 acquisition time (seconds) for a set of readings.

    Groups readings by current pair (a, b) -- sorted, i.e. (A,B) and (B,A)
    treated as the same pair, if MERGE_AB_POLARITY -- and channel-packs each
    pair's readings into ceil(count / LS2_CHANNELS) injections, since the LS2
    can read that many potential dipoles per current injection. Standard
    arrays (one reading per current pair) end up near one injection per
    reading; gradient-type arrays (many readings per current pair) are
    amortised into far fewer injections. Reciprocal readings, having swapped
    current/potential roles, land in their own current-pair groups.

    Returns (time_seconds, n_injections, per_pair) where per_pair maps each
    current pair to its reading count.
    """
    per_pair = defaultdict(int)
    for (a, b, m, n) in readings:
        pair = tuple(sorted((a, b))) if MERGE_AB_POLARITY else (a, b)
        per_pair[pair] += 1
    injections = sum(math.ceil(c / LS2_CHANNELS) for c in per_pair.values())
    return injections * CYCLE_S, injections, per_pair


def depth_profile(cell_centers, coverage, volumes, n_bins=20):
    """Volume-weighted mean reading count vs. depth, within the footprint.

    Bins cell depth (0 to BOX_DEPTH, positive downward) into n_bins equal
    bands and, within each band, averages `coverage` over cells that are part
    of the footprint (coverage >= 1), weighted by cell volume. Returns
    (bin_midpoints, mean_coverage_per_bin); a bin with no footprint volume
    gets 0.0.
    """
    z = -cell_centers[:, 2]                      # depth, positive downward
    edges = np.linspace(0, BOX_DEPTH, n_bins + 1)
    mids, means = [], []
    foot = coverage >= 1
    for i in range(n_bins):
        sel = foot & (z >= edges[i]) & (z < edges[i + 1])
        vw = volumes[sel]
        mids.append((edges[i] + edges[i + 1]) / 2.0)
        means.append(float((coverage[sel] * vw).sum() / vw.sum()) if vw.sum() > 0 else 0.0)
    return np.array(mids), np.array(means)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_density_slices(top, geom, highlight, out_dir, log):
    """Write density_slices.png: reach-count sections for the top-N designs.

    For each design in `top`, samples estimate_reach_coverage on a regular
    SLICE_NX x SLICE_NY x SLICE_NZ grid over that design's own domain, then
    draws two panels: a vertical section (length x depth) through the row
    with the highest total coverage, and a plan view (length x width) at the
    depth with the highest total coverage. Colour scale and contouring follow
    SLICE_SCALE / SLICE_SHARED_LOG / SLICE_PERDESIGN_LOG / SLICE_CONTOURS.
    Saves the figure to out_dir, logs the write, and returns the file path.
    """
    import numpy as np
    import matplotlib.colors as mcolors

    def mids(a, b, nn):
        e = np.linspace(a, b, nn + 1)
        return (e[:-1] + e[1:]) / 2.0

    def sample_field(sensors, sch, pts):
        return estimate_reach_coverage(pts, sensors, sch, DOI_FACTOR, LATERAL_PAD_FACTOR).astype(float)

    def unique_geometry(sch):
        """Keep only the first reading for each unique set of four electrodes.

        A reciprocal reading (m,n,a,b) uses the same four electrodes as its
        forward reading (a,b,m,n), so it marks the identical reach box rather
        than adding new ground -- counting both would double-count that
        footprint in the slice view. Because reciprocals are often only a
        random subset of readings, that double-counting would land on a
        random subset of the design and break what should be left-right /
        front-back symmetry. This only affects how the slice figures are
        built; the density metric/CSV upstream still counts every reading,
        reciprocals included, since redundancy is a different question there.
        """
        abmn = np.column_stack([sch[k].astype(int) for k in "abmn"])
        seen, keep = set(), []
        for i, row in enumerate(abmn):
            key = frozenset(row.tolist())
            if key not in seen:
                seen.add(key); keep.append(i)
        keep = np.array(keep, dtype=int)
        return {k: sch[k][keep] for k in "abmn"}

    # ---- pass 1: compute every design's density field + its densest planes ----
    fields = []
    for r in top:
        label = r["label"]
        ge = geom[label]
        sensors, sch = ge["sensors"], ge["sch"]
        if SLICE_DEDUPE_RECIPROCALS:
            sch = unique_geometry(sch)
        cx = mids(sensors[:, 0].min() - FOOTPRINT_PAD, sensors[:, 0].max() + FOOTPRINT_PAD, SLICE_NX)
        cy = mids(sensors[:, 1].min() - FOOTPRINT_PAD, sensors[:, 1].max() + FOOTPRINT_PAD, SLICE_NY)
        cd = mids(0.0, BOX_DEPTH, SLICE_NZ)          # depth, positive downward
        X, Y, D = np.meshgrid(cx, cy, cd, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), (-D).ravel()])
        cov = sample_field(sensors, sch, pts).reshape(X.shape)
        peak = int(cov.max())
        fields.append(dict(label=label, sensors=sensors, cx=cx, cy=cy, cd=cd, cov=cov, peak=peak,
                           yi=int(np.argmax(cov.sum(axis=(0, 2)))) if peak > 0 else 0,
                           zi=int(np.argmax(cov.sum(axis=(0, 1)))) if peak > 0 else 0))

    gmax = max((f["peak"] for f in fields), default=0)

    base = plt.get_cmap(SLICE_CMAP)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "density", base(np.linspace(SLICE_LOW_CLIP, 1.0, 256)))
    cmap.set_bad("#e9e9e9")                          # cells no reading reaches -> light grey

    shared = SLICE_SCALE == "shared" and gmax > 0
    if shared and SLICE_SHARED_LOG:
        norm = mcolors.LogNorm(vmin=1, vmax=max(gmax, 2))
        cbar_label = "readings reaching cell (shared, log)"
    elif shared:
        norm = mcolors.Normalize(vmin=0, vmax=gmax)
        cbar_label = "readings reaching cell (shared)"
    else:
        norm = None                      # per_design: a fresh norm is built per row
        cbar_label = "readings reaching cell"

    def row_norm(cov):
        """Per-design absolute-count norm spanning that row's own sparsest ->
        densest covered cell, so the full colour ramp is used within the
        panel and small changes stay visible. Log-scaled if SLICE_PERDESIGN_LOG."""
        pos = cov[cov > 0]
        lo = int(pos.min()) if pos.size else 1
        hi = int(pos.max()) if pos.size else 2
        if hi <= lo:
            hi = lo + 1                  # avoid a zero-width scale on a flat field
        if SLICE_PERDESIGN_LOG:
            return mcolors.LogNorm(vmin=max(lo, 1), vmax=hi)
        return mcolors.Normalize(vmin=lo, vmax=hi)

    def field2d(cov2d):
        """Absolute reading counts, masked where no reading reaches the cell."""
        return np.ma.masked_where(cov2d <= 0, cov2d.astype(float))

    def contour_levels(peak, lo):
        """Iso-count contour levels for this row, or None if contours are off
        or the field is too flat (peak <= 1) to contour meaningfully."""
        if not SLICE_CONTOURS or peak <= 1:
            return None
        if shared:
            lv = [v for v in SLICE_CONTOUR_COUNTS if 0 < v < peak]
        else:
            lv = sorted({max(1, round(p / 100.0 * peak)) for p in SLICE_CONTOUR_PCTS})
            lv = [v for v in lv if lo < v < peak]
        return lv or None

    # ---- pass 2: plot ----
    n = len(top)
    fig, axes = plt.subplots(n, 2, figsize=(11.5, 2.15 * n + 0.9), squeeze=False)
    im = None
    for row, f in enumerate(fields):
        c = highlight[f["label"]]
        axV, axP = axes[row][0], axes[row][1]
        cx, cy, cd, cov, peak = f["cx"], f["cy"], f["cd"], f["cov"], f["peak"]

        if peak <= 0:
            axV.text(-0.32, 0.5, f"{f['label']}\n(no coverage)", transform=axV.transAxes,
                     ha="right", va="center", fontsize=8.5, fontweight="bold", color=c)
            for a in (axV, axP):
                a.text(0.5, 0.5, "no coverage", ha="center", va="center", fontsize=8)
                a.set_xticks([]); a.set_yticks([])
            continue

        lo = int(cov[cov > 0].min())
        rn = norm if shared else row_norm(cov)
        rng_txt = f"peak {peak}" if shared else f"{lo}–{peak} readings"
        axV.text(-0.32, 0.5, f"{f['label']}\n({rng_txt})", transform=axV.transAxes,
                 ha="right", va="center", fontsize=8.5, fontweight="bold", color=c)

        yi, zi = f["yi"], f["zi"]
        secV = field2d(cov[:, yi, :]).T           # (depth, x)
        planP = field2d(cov[:, :, zi]).T          # (y, x)
        lv = contour_levels(peak, lo)

        imV = axV.pcolormesh(cx, cd, secV, cmap=cmap, norm=rn, shading="auto")
        if lv:
            axV.contour(cx, cd, secV, levels=lv, colors=c, linewidths=0.5, alpha=0.55)
        axV.axhline(cd[zi], color=c, ls="--", lw=1.0)   # where the plan view is taken
        axV.invert_yaxis()
        axV.set_ylabel("depth (m)", fontsize=8)
        axV.tick_params(labelsize=7)

        axP.pcolormesh(cx, cy, planP, cmap=cmap, norm=rn, shading="auto")
        if lv:
            axP.contour(cx, cy, planP, levels=lv, colors=c, linewidths=0.5, alpha=0.55)
        axP.axhline(cy[yi], color=c, ls="--", lw=1.0)   # where the section is taken
        axP.scatter(f["sensors"][:, 0], f["sensors"][:, 1], s=7, c="white",
                    edgecolor="#1a202c", linewidth=0.3, zorder=3)
        axP.set_ylabel("y (m)", fontsize=8)
        axP.tick_params(labelsize=7)
        im = imV

        if not shared:                            # per_design: one colourbar PER ROW,
            cb = fig.colorbar(imV, ax=axP, fraction=0.046, pad=0.03)  # in this row's counts
            cb.ax.tick_params(labelsize=6)
            cb.set_label("readings", fontsize=6)

        if row == 0:
            axV.set_title("Vertical section (length × depth)\nthrough densest row", fontsize=9)
            axP.set_title("Plan view (length × width)\nat densest depth", fontsize=9)
        if row == n - 1:
            axV.set_xlabel("x (m)", fontsize=8)
            axP.set_xlabel("x (m)", fontsize=8)

    if shared and im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, location="right")
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    scale_note = ("shared absolute scale (comparable between rows)" if shared
                  else "readings per cell, each row on its OWN scale (see per-row bars)")
    fig.suptitle("Where each top design concentrates coverage\n" + scale_note,
                 fontsize=11, y=0.995)
    path = os.path.join(out_dir, "density_slices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    log(f"Wrote {path}")
    return path


def render_sensitivity_slices(top, geom, highlight, out_dir, log):
    """Write sensitivity_slices.png: analytic sensitivity sections for the top-N.

    Same section/plan-view layout as render_density_slices(), but the field
    sampled is estimate_sensitivity_coverage (real homogeneous-half-space
    Fréchet sensitivity) instead of the reach count. Always log-scaled, with
    the near-electrode singularity masked (SENS_MASK_FRAC) and the colour
    scale clipped at the SENS_CAP_PCTL percentile. SENS_SCALE="per_design"
    gives each row its own scale and colourbar; "shared" puts every row on
    one comparable scale.
    """
    import numpy as np
    import matplotlib.colors as mcolors

    def mids(a, b, nn):
        e = np.linspace(a, b, nn + 1)
        return (e[:-1] + e[1:]) / 2.0

    def min_spacing(sensors):
        pts = sensors[:, :2]
        best = np.inf
        for i in range(len(pts)):
            d = np.hypot(pts[i, 0] - pts[:, 0], pts[i, 1] - pts[:, 1])
            d[i] = np.inf
            best = min(best, d.min())
        return best if np.isfinite(best) else 1.0

    def unique_geometry(sch):
        abmn = np.column_stack([sch[k].astype(int) for k in "abmn"])
        seen, keep = set(), []
        for i, row in enumerate(abmn):
            key = frozenset(row.tolist())
            if key not in seen:
                seen.add(key); keep.append(i)
        return {k: sch[k][np.array(keep, dtype=int)] for k in "abmn"}

    # ---- pass 1: sensitivity field + densest planes per design ----
    fields = []
    for r in top:
        label = r["label"]
        ge = geom[label]
        sensors, sch = ge["sensors"], ge["sch"]
        if SLICE_DEDUPE_RECIPROCALS:
            sch = unique_geometry(sch)
        mask_r = SENS_MASK_FRAC * min_spacing(sensors)
        cx = mids(sensors[:, 0].min() - FOOTPRINT_PAD, sensors[:, 0].max() + FOOTPRINT_PAD, SLICE_NX)
        cy = mids(sensors[:, 1].min() - FOOTPRINT_PAD, sensors[:, 1].max() + FOOTPRINT_PAD, SLICE_NY)
        cd = mids(0.0, BOX_DEPTH, SLICE_NZ)
        X, Y, D = np.meshgrid(cx, cy, cd, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), (-D).ravel()])
        S = estimate_sensitivity_coverage(pts, sensors, sch, mask_radius=mask_r).reshape(X.shape)
        finite = S[np.isfinite(S) & (S > 0)]
        cap = float(np.percentile(finite, SENS_CAP_PCTL)) if finite.size else 1.0
        floor = cap / (10.0 ** SENS_FLOOR_DECADES)
        # densest planes: use nansum so masked cells don't dominate the argmax
        col_sum = np.nan_to_num(S).sum(axis=(0, 2))
        dep_sum = np.nan_to_num(S).sum(axis=(0, 1))
        # Take the plan view at the most-sensitive depth that lies BELOW the masked
        # near-electrode shell -- otherwise the plan lands in the surface layer where
        # the singularity mask has carved a hole around every electrode. Any depth
        # >= mask_r is hole-free (the nearest electrode is at the surface), and
        # sensitivity is still near its shallow peak there, so nothing is lost.
        below = cd >= mask_r
        dep_for_plan = np.where(below, dep_sum, -np.inf) if below.any() else dep_sum
        fields.append(dict(label=label, sensors=sensors, cx=cx, cy=cy, cd=cd, cov=S,
                           cap=cap, floor=floor,
                           yi=int(np.argmax(col_sum)) if finite.size else 0,
                           zi=int(np.argmax(dep_for_plan)) if finite.size else 0))
        gc.collect()

    caps = [f["cap"] for f in fields if f["cap"] > 0]
    gcap = max(caps) if caps else 1.0
    gfloor = gcap / (10.0 ** SENS_FLOOR_DECADES)

    base = plt.get_cmap(SENS_CMAP)
    cmap = mcolors.LinearSegmentedColormap.from_list("sens", base(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad("#e9e9e9")

    shared = SENS_SCALE == "shared"
    if shared:
        norm = mcolors.LogNorm(vmin=gfloor, vmax=gcap)

    def masked(sec):
        return np.ma.masked_invalid(np.ma.masked_less_equal(np.ma.masked_invalid(sec), 0.0))

    n = len(top)
    fig, axes = plt.subplots(n, 2, figsize=(11.5, 2.15 * n + 0.9), squeeze=False)
    for row, f in enumerate(fields):
        c = highlight[f["label"]]
        axV, axP = axes[row][0], axes[row][1]
        cx, cy, cd, S = f["cx"], f["cy"], f["cd"], f["cov"]
        rn = norm if shared else mcolors.LogNorm(vmin=f["floor"], vmax=max(f["cap"], f["floor"] * 1.01))

        axV.text(-0.32, 0.5, f["label"], transform=axV.transAxes, ha="right", va="center",
                 fontsize=8.5, fontweight="bold", color=c)

        yi, zi = f["yi"], f["zi"]
        imV = axV.pcolormesh(cx, cd, masked(S[:, yi, :]).T, cmap=cmap, norm=rn, shading="auto")
        axV.axhline(cd[zi], color=c, ls="--", lw=1.0)
        axV.invert_yaxis(); axV.set_ylabel("depth (m)", fontsize=8); axV.tick_params(labelsize=7)

        axP.pcolormesh(cx, cy, masked(S[:, :, zi]).T, cmap=cmap, norm=rn, shading="auto")
        axP.axhline(cy[yi], color=c, ls="--", lw=1.0)
        axP.scatter(f["sensors"][:, 0], f["sensors"][:, 1], s=7, c="white",
                    edgecolor="#1a202c", linewidth=0.3, zorder=3)
        axP.set_ylabel("y (m)", fontsize=8); axP.tick_params(labelsize=7)

        if not shared:
            cb = fig.colorbar(imV, ax=axP, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6); cb.set_label("sensitivity", fontsize=6)

        if row == 0:
            axV.set_title("Vertical section (length × depth)\nthrough most-sensitive row", fontsize=9)
            axP.set_title("Plan view (length × width)\nat most-sensitive depth", fontsize=9)
        if row == n - 1:
            axV.set_xlabel("x (m)", fontsize=8); axP.set_xlabel("x (m)", fontsize=8)

    if shared:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, location="right")
        cbar.set_label("sensitivity (shared, log)", fontsize=8); cbar.ax.tick_params(labelsize=7)
    scale_note = ("shared log scale (comparable between rows)" if shared
                  else "each row on its OWN log scale (see per-row bars)")
    fig.suptitle("Real sensitivity coverage (homogeneous half-space)\n" + scale_note,
                 fontsize=11, y=0.995)
    path = os.path.join(out_dir, "sensitivity_slices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    log(f"Wrote {path}")
    return path


def infer_array(label):
    """Guess a candidate's array type from tokens in its label.

    Splits the label on "_" (after normalising "-" to "_") and returns the
    array type for the first token that matches TAG_TO_ARRAY, else "other".
    Only used when a candidate's CANDIDATES entry doesn't carry an explicit
    "array" key.
    """
    for tok in str(label).replace("-", "_").split("_"):
        if tok in TAG_TO_ARRAY:
            return TAG_TO_ARRAY[tok]
    return "other"


def run(candidates, out_dir=OUT_DIR):
    """Compute the density/time comparison and write all plots + the CSV.

    `candidates` is a {label: {"spread": path, "protocol": path, ["array"]}}
    dict (the same shape as the module-level CANDIDATES, so this can also be
    called with a dict built programmatically, e.g. by a sweep script).
    Creates out_dir if needed, then for each candidate: parses the spread and
    protocol XML, validates the readings, builds the mesh and reach coverage,
    computes the footprint and dense-fraction metrics and the predicted LS2
    field time, and logs a per-design summary. Once every candidate has been
    processed it ranks them, writes density_vs_time.png, optionally
    density_vs_time_topleft.png, density_vs_depth.png,
    threshold_sensitivity.png, density_slices.png, optionally
    sensitivity_slices.png, density_runtime_comparison.csv, and
    density_runtime_report.txt into out_dir. main() below just calls this
    with the module-level CANDIDATES and OUT_DIR.
    """
    os.makedirs(out_dir, exist_ok=True)
    report = []
    def log(m):
        print(m); report.append(m)

    results = []
    profiles = {}
    geom = {}

    for label, paths in candidates.items():
        log(f"\n=== {label} ===")
        electrodes = parse_spread_xml(paths["spread"])
        readings = validate_readings(parse_protocol_xml(paths["protocol"]), log)
        sensors, sch = build_sensors_and_scheme(electrodes, readings)
        geom[label] = {"sensors": sensors, "sch": sch}
        log(f"  {len(sensors)} electrodes, {len(readings)} readings")

        mesh = build_domain_mesh(sensors, log)
        centers = np.array([[c.x(), c.y(), c.z()] for c in mesh.cellCenters()])
        volumes = np.array(mesh.cellSizes())

        coverage = estimate_reach_coverage(centers, sensors, sch, DOI_FACTOR, LATERAL_PAD_FACTOR)
        foot_frac_of_box = float(volumes[coverage >= 1].sum() / volumes.sum())

        fracs = {k: dense_fraction(coverage, volumes, k) for k in THRESHOLDS}
        t_s, n_inj, per_pair = predict_field_time(readings)
        reads_per_inj = np.array(list(per_pair.values()))

        log(f"  footprint = {foot_frac_of_box*100:.1f}% of modelled box; "
            f"max overlap on any cell = {int(coverage.max())}")
        log(f"  density (% of own footprint): " +
            ", ".join(f"K>={k}:{fracs[k]*100:.1f}%" for k in THRESHOLDS))
        log(f"  LS2 time: {len(per_pair)} current pairs, {n_inj} energisations "
            f"(reads/injection max={reads_per_inj.max()}, mean={reads_per_inj.mean():.2f}) "
            f"-> {t_s/60:.2f} min")

        profiles[label] = depth_profile(centers, coverage, volumes)
        results.append({
            "label": label, "n_readings": len(readings),
            "n_injections": n_inj, "time_min": t_s / 60.0,
            "footprint_pct_of_box": foot_frac_of_box * 100.0,
            "primary_dense_pct": fracs[PRIMARY_THRESHOLD] * 100.0,
            "fracs": fracs,
            "array": paths.get("array") or infer_array(label),
        })

    if not results:
        log("No candidates configured -- edit CANDIDATES at the top."); return

    # labels / x-positions / CSV column names
    level_x = {k: k for k in THRESHOLDS}
    primary_x = PRIMARY_THRESHOLD
    hero_ylabel = f"% of own footprint probed by ≥ {PRIMARY_THRESHOLD} readings"
    thr_xlabel = "Density threshold K (readings per cell)"
    thr_ylabel = "% of own footprint with ≥ K readings"
    def col_name(k): return f"dense_pct_K{k}"

    # ---- rank performers; top-N get vivid colours, the rest fade by array type ----
    from matplotlib.lines import Line2D

    def _score(r):
        if RANK_METRIC == "efficiency" and r["time_min"] > 0:
            return r["primary_dense_pct"] / r["time_min"]
        return r["primary_dense_pct"]

    ranked = sorted(results, key=lambda r: (-_score(r), r["time_min"]))
    top = ranked[:min(TOP_N, len(ranked))]
    highlight = {r["label"]: HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)]
                 for i, r in enumerate(top)}
    by_label = {r["label"]: r for r in results}
    arrays_bg = sorted({r["array"] for r in results if r["label"] not in highlight})

    def bg_color(r):
        return ARRAY_COLORS.get(r["array"], ARRAY_COLORS["other"])

    def array_legend(kind):   # kind: 'marker' or 'line'
        return [Line2D([0], [0],
                       marker="o" if kind == "marker" else None,
                       ls="" if kind == "marker" else "-",
                       color=ARRAY_COLORS.get(a, ARRAY_COLORS["other"]),
                       lw=2, alpha=0.9, label=a.replace("_", " ").title())
                for a in arrays_bg]

    def top_legend(kind):     # ranked top-N, named
        return [Line2D([0], [0],
                       marker="o" if kind == "marker" else None,
                       ls="" if kind == "marker" else "-",
                       color=highlight[r["label"]],
                       markeredgecolor="#1a202c" if kind == "marker" else None,
                       markersize=9, lw=2.4, label=f"{i}. {r['label']}")
                for i, r in enumerate(top, start=1)]

    def attach_legends(ax, kind):
        """Two stacked legends outside the axes: top-N (named) above, array key below."""
        l1 = ax.legend(handles=top_legend(kind),
                       title=f"Top {len(top)} by {RANK_METRIC}",
                       loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       fontsize=8, title_fontsize=9, framealpha=0.95, borderaxespad=0.0)
        ax.add_artist(l1)
        extra = [l1]
        if arrays_bg:
            l2 = ax.legend(handles=array_legend(kind), title="Others (by array type)",
                           loc="lower left", bbox_to_anchor=(1.02, 0.0),
                           fontsize=8, title_fontsize=9, framealpha=0.95, borderaxespad=0.0)
            extra.append(l2)
        return extra

    def fan_positions(items):
        """Return {label: (x, y)} with (near-)coincident scatter points fanned
        out onto a small ring so none is hidden behind another; points that
        aren't near any other point are left at their true position. Purely
        cosmetic -- callers should use the true values for anything else."""
        pts = {k: (x, y) for k, x, y in items}
        if not SEPARATE_OVERLAPS or len(items) < 2:
            return pts
        xs = [x for _, x, _ in items]; ys = [y for _, _, y in items]
        rx = (max(xs) - min(xs) or 1.0) * OVERLAP_SPREAD
        ry = (max(ys) - min(ys) or 1.0) * OVERLAP_SPREAD
        gx, gy = rx * 1.3, ry * 1.3                      # coincidence resolution
        buckets = {}
        for k, x, y in items:
            buckets.setdefault((round(x / gx), round(y / gy)), []).append(k)
        for group in buckets.values():
            if len(group) < 2:
                continue
            cx = sum(pts[k][0] for k in group) / len(group)
            cy = sum(pts[k][1] for k in group) / len(group)
            for j, k in enumerate(sorted(group)):
                ang = 2 * math.pi * j / len(group)
                pts[k] = (cx + rx * math.cos(ang), cy + ry * math.sin(ang))
        return pts

    log(f"\nTop {len(top)} by {RANK_METRIC}: " + ", ".join(r["label"] for r in top))

    # ---- HERO plot: predicted field time vs % of own footprint densely probed ----
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    pos = fan_positions([(r["label"], r["time_min"], r["primary_dense_pct"]) for r in results])
    for r in results:                                  # faded background, by array type
        if r["label"] in highlight:
            continue
        x, y = pos[r["label"]]
        ax.scatter(x, y, s=62, color=bg_color(r), alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=2)
    for r in top:                                      # vivid top-N (named in legend)
        x, y = pos[r["label"]]
        ax.scatter(x, y, s=150, color=highlight[r["label"]], alpha=0.95,
                   edgecolor="#1a202c", linewidth=1.1, zorder=4)
    ax.set_xlabel("Predicted LS2 field time (min)")
    ax.set_ylabel(hero_ylabel)
    ax.set_title("Density vs acquisition cost  (better → top-left)")
    ax.grid(True, alpha=0.3)
    ax.annotate("better", xy=(0.03, 0.97), xycoords="axes fraction",
                fontsize=9, color="#718096", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#718096"), xytext=(0.12, 0.88))
    extra = attach_legends(ax, "marker")
    fig.tight_layout()
    hero = os.path.join(out_dir, "density_vs_time.png")
    fig.savefig(hero, dpi=150, bbox_inches="tight", bbox_extra_artists=extra); plt.close(fig)
    log(f"Wrote {hero}")

    # ---- zoomed 'best designs' plot: the top-left corner of the hero plot ----
    if MAKE_ZOOM_PLOT and results:
        import matplotlib.patheffects as pe

        def pareto_front(rs):
            """Non-dominated designs: no other design is both faster AND denser."""
            front = []
            for r in rs:
                if not any(s is not r and
                           s["time_min"] <= r["time_min"] and
                           s["primary_dense_pct"] >= r["primary_dense_pct"] and
                           (s["time_min"] < r["time_min"] or
                            s["primary_dense_pct"] > r["primary_dense_pct"])
                           for s in rs):
                    front.append(r)
            return front

        times = np.array([r["time_min"] for r in results])
        dens = np.array([r["primary_dense_pct"] for r in results])
        tspan = (times.max() - times.min()) or 1.0
        dspan = (dens.max() - dens.min()) or 1.0
        ideal_t, ideal_d = times.min(), dens.max()   # the unreachable perfect corner

        # Show the N designs closest to that corner (fastest + densest). Using a
        # count rather than a hard crop keeps the plot populated however tightly
        # the winners cluster -- the earlier percentile crop could leave only one
        # or two points when the good designs all sit near 100%.
        dist = lambda r: (((r["time_min"] - ideal_t) / tspan) ** 2 +
                          ((r["primary_dense_pct"] - ideal_d) / dspan) ** 2)
        kept = sorted(results, key=dist)[:min(ZOOM_SHOW_N, len(results))]

        kx = [r["time_min"] for r in kept]; ky = [r["primary_dense_pct"] for r in kept]
        xs = (max(kx) - min(kx)) or max(max(kx), 1.0) * 0.10
        ys = (max(ky) - min(ky)) or max(max(ky), 1.0) * 0.10
        xlo = max(0.0, min(kx) - ZOOM_MARGIN * xs); xhi = max(kx) + ZOOM_MARGIN * xs
        # extra headroom on top so designs at ~100% keep their markers/labels; do
        # NOT clamp to 100 -- a little whitespace above the ceiling is fine.
        ylo = max(0.0, min(ky) - ZOOM_MARGIN * ys); yhi = max(ky) + (ZOOM_MARGIN + 0.12) * ys
        ywin = yhi - ylo

        figz, axz = plt.subplots(figsize=(9.6, 6.4))
        front = pareto_front(results)
        fsorted = sorted(front, key=lambda r: r["time_min"])
        if len(fsorted) > 1:
            axz.plot([r["time_min"] for r in fsorted], [r["primary_dense_pct"] for r in fsorted],
                     color="#a0aec0", ls="--", lw=1.3, zorder=1, label="best trade-off frontier")
        for r in results:                                # every point drawn; limits show the corner
            x, y = pos[r["label"]]
            vivid = r["label"] in highlight
            axz.scatter(x, y, s=170 if vivid else 72,
                        color=highlight[r["label"]] if vivid else bg_color(r),
                        alpha=0.95 if vivid else 0.85,
                        edgecolor="#1a202c" if vivid else "white",
                        linewidth=1.1 if vivid else 0.5, zorder=4 if vivid else 2)

        if ZOOM_LABEL_POINTS:
            # label the kept designs; put top-band labels BELOW their point (so
            # they don't collide with the title) and nudge apart any that clash.
            placed = []
            for r in sorted(kept, key=lambda r: (-pos[r["label"]][1], pos[r["label"]][0])):
                x, y = pos[r["label"]]
                top_band = y > yhi - 0.28 * ywin
                dx, dy, va = 6, (-12 if top_band else 8), ("top" if top_band else "bottom")
                for (px, py, pva) in placed:
                    if pva == va and abs(px - x) < 0.16 * (xhi - xlo) and abs(py - y) < 0.06 * ywin:
                        dy += (-11 if top_band else 11)
                axz.annotate(r["label"], (x, y), xytext=(dx, dy), textcoords="offset points",
                             fontsize=7.6, zorder=6, ha="left", va=va, color="#1a202c",
                             path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
                placed.append((x, y, va))

        axz.set_xlim(xlo, xhi); axz.set_ylim(ylo, yhi)
        axz.set_xlabel("Predicted LS2 field time (min)")
        axz.set_ylabel(hero_ylabel)
        axz.set_title("Best designs — top-left of the density/cost plot  (better → top-left)",
                      fontsize=11, pad=12)
        axz.grid(True, alpha=0.3)
        if len(fsorted) > 1:
            axz.legend(loc="lower right", fontsize=8, framealpha=0.95)
        figz.tight_layout()
        zoom = os.path.join(out_dir, "density_vs_time_topleft.png")
        figz.savefig(zoom, dpi=150, bbox_inches="tight"); plt.close(figz)
        log(f"Wrote {zoom} ({len(kept)} design(s) shown)")

    # ---- density-vs-depth ----
    # Each design's curve is normalised to its OWN shallowest (surface) value,
    # so the comparison is about the SHAPE of the depth falloff (and how it
    # relates to the shared box depth), not absolute magnitude -- an
    # unnormalised plot would let one design's raw scale dwarf the rest.
    def norm_profile(means):
        ref = means.max()
        return (means / ref * 100.0) if ref > 0 else means

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for label, (mids, means) in profiles.items():      # faded background
        if label in highlight:
            continue
        ax.plot(norm_profile(means), mids, color=bg_color(by_label[label]), alpha=0.6, lw=1.6, zorder=2)
    for r in top:                                      # vivid top-N
        mids, means = profiles[r["label"]]
        ax.plot(norm_profile(means), mids, color=highlight[r["label"]], lw=2.4, marker="o", ms=3, zorder=4)
    ax.invert_yaxis()
    ax.set_xlabel("Density as % of that design's surface density", fontsize=10)
    ax.set_ylabel("Depth (m)")
    ax.set_title("Depth decay of coverage  (shared DOI ceiling ~0.45 m)")
    ax.grid(True, alpha=0.3)
    extra = attach_legends(ax, "line")
    fig.tight_layout()
    depth_png = os.path.join(out_dir, "density_vs_depth.png")
    fig.savefig(depth_png, dpi=150, bbox_inches="tight", bbox_extra_artists=extra); plt.close(fig)
    log(f"Wrote {depth_png}")

    # ---- threshold sensitivity ----
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    xs_lvl = [level_x[k] for k in THRESHOLDS]
    for r in results:                                  # faded background
        if r["label"] in highlight:
            continue
        ax.plot(xs_lvl, [r["fracs"][k] * 100 for k in THRESHOLDS],
                color=bg_color(r), alpha=0.6, lw=1.6, zorder=2)
    for r in top:                                      # vivid top-N
        ax.plot(xs_lvl, [r["fracs"][k] * 100 for k in THRESHOLDS],
                color=highlight[r["label"]], lw=2.4, marker="o", ms=4, zorder=4)
    ax.axvline(primary_x, color="#a0aec0", ls="--", lw=1)
    ax.set_xlabel(thr_xlabel)
    ax.set_ylabel(thr_ylabel)
    ax.set_title("How the density metric responds to the threshold")
    ax.grid(True, alpha=0.3)
    extra = attach_legends(ax, "line")
    fig.tight_layout()
    thr_png = os.path.join(out_dir, "threshold_sensitivity.png")
    fig.savefig(thr_png, dpi=150, bbox_inches="tight", bbox_extra_artists=extra); plt.close(fig)
    log(f"Wrote {thr_png}")

    # ---- density slice sections for the top-N ----
    render_density_slices(top, geom, highlight, out_dir, log)

    # ---- real sensitivity slice sections for the top-N ----
    if MAKE_SENSITIVITY_SLICES:
        render_sensitivity_slices(top, geom, highlight, out_dir, log)

    # ---- CSV ----
    csv = os.path.join(out_dir, "density_runtime_comparison.csv")
    with open(csv, "w") as f:
        f.write("label,n_readings,n_injections,time_min,footprint_pct_of_box," +
                ",".join(col_name(k) for k in THRESHOLDS) + "\n")
        for r in sorted(results, key=lambda x: (-x["primary_dense_pct"], x["time_min"])):
            f.write(f"{r['label']},{r['n_readings']},{r['n_injections']},"
                    f"{r['time_min']:.2f},{r['footprint_pct_of_box']:.2f}," +
                    ",".join(f"{r['fracs'][k]*100:.2f}" for k in THRESHOLDS) + "\n")
    log(f"Wrote {csv}")

    with open(os.path.join(out_dir, "density_runtime_report.txt"), "w") as f:
        f.write("\n".join(report))


def main():
    run(CANDIDATES, OUT_DIR)


if __name__ == "__main__":
    main()

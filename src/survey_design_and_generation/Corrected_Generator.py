"""
ABEM Terrameter LS/LS2 Spread + Protocol XML generator, with a Tkinter GUI.

Given a rectangular electrode grid (rows x cols, up to 64 electrodes total)
and a choice of array type, this script builds the full list of 4-electrode
(A, B, M, N) readings for that array, optionally adds reciprocal readings
(current/potential pairs swapped), optionally reorders the readings so that
consecutive measurements never use grid-neighbouring electrodes, and writes
the result out as a pair of XML files matching the schema the Terrameter LS
Toolbox expects (see the comment above CABLE_SIZE further down for the
reference layout):

    - a Spread file describing the electrode grid's physical layout and its
      mapping onto the instrument's two 32-channel cable connectors (see
      write_spread_xml()), and
    - a Protocol file describing the sequence of measurements to take (see
      write_protocol_xml()).

Supported array types are Wenner, Wenner-Schlumberger, Dipole-Dipole and
Gradient (see ARRAY_GENERATORS), plus an optional 'L'-shaped
perpendicular-dipole reading set at the 4 grid corners, based on
Tejero-Andrade et al. (2015, Near Surface Geophysics,
doi:10.3997/1873-0604.2015015) - see l_array_cross_arm_readings().

Only a 2-cable, <=64-electrode setup is supported: build_electrode_grid()
itself has no upper limit on electrode count, but asking for more than 64
will make write_spread_xml() emit more than the 2 <Cable> blocks a real
Terrameter LS/LS2 (2 x 32-channel connectors) can use, so treat any output
for >64 electrodes as invalid for that hardware.

--------------------------------------------------------------------------------
########## GUI WALKTHROUGH (run_gui(), used when this file is run directly) ##########

1. Array type popup (show_array_type_popup())

    A small popup asking which array type to generate - Wenner,
    Wenner-Schlumberger, Dipole-Dipole, or Gradient - each with a one-line
    description. Pick one and click "Next". Closing this popup without ever
    having picked an array type ends the program, since there is nothing
    sensible to build yet. The main configuration window stays hidden until
    an array type has been chosen.

2. Main configuration window

    - Array type: shows the currently selected array type, with a "Change"
      button that reopens the array type popup above at any time.

    - Grid Structure (Spread): total number of electrodes, number of rows,
      number of columns, and X/Y electrode spacing in metres. Rows x
      columns must equal the total electrode count, or you get an error
      message when you try to build or preview the grid.

    - Calling Procedure (Protocol):
        * checkboxes for which reading directions to include: rows,
          columns, diagonals (only diagonal lines of >= 4 electrodes are
          used, since every supported array type needs 4 distinct
          electrode positions), and the 'L'-shaped perpendicular-dipole
          corner readings (a different geometry from the line arrays -
          see l_array_cross_arm_readings()). At least one must be checked.
        * "Randomise order": reshuffles the reading sequence so consecutive
          readings avoid using grid-adjacent electrodes, including diagonal
          neighbours (see randomize_non_adjacent()) - meant to avoid
          residual charge from one reading affecting the next. When the
          grid is too small/dense to fully avoid this, the program reports
          how many violations were unavoidable. Randomising a Gradient
          survey breaks apart its shared-current-pair runs: the data is
          still valid, but the field-time benefit Gradient is normally
          chosen for is lost.
        * "Percentage of measurements with reciprocals": 0-100. Sets what
          fraction of readings also get a reciprocal (current/potential
          pairs swapped) added; which readings get one is chosen at random
          rather than just the first N%, so a reduced set still covers the
          whole survey representatively. Reciprocals let you estimate
          measurement error - more of them means a better error estimate
          at the cost of more field time.

    - Extras:
        * Spread/Protocol file name: the .xml filenames the files are saved
          under (".xml" is appended automatically if you leave it off).
          These are written relative to the directory the script is run
          from, not necessarily the script's own directory.
        * Spread/Protocol name (shown on Terrameter): the <Name> written
          inside each XML file - what the instrument itself displays in its
          spread/protocol lists. Independent of the file name above; left
          blank, an auto-generated name is used instead.

    - "Show Grid Layout" button: opens a static preview of just the
      electrode positions (show_grid_popup()) built from whatever grid
      settings are currently entered, with no need to build a full protocol
      first. Has Save (PNG/PDF/SVG) and Close.

    - "Show Animation" button: opens the animated calling-order popup
      (show_animation_popup()) for the most recently completed Build. If
      nothing has been built yet, it tells you to click Build first instead.

    - Animation speed slider: 50-1000 ms per step, used by the animation
      popup.

    - Timing field(s), used only for the runtime estimate below (never
      written to the XML): a single "seconds per measurement" field for
      Wenner/Wenner-Schlumberger/Dipole-Dipole, or, when Gradient is
      selected, two fields - "seconds per current injection" and "seconds
      per reading" - since a Gradient survey can take many readings per
      injection (see estimate_acquisition_time()).

    - "Build" button: validates the fields, generates the full reading
      list, writes the Spread and Protocol XML files, updates the status
      line with a short summary (electrode/reading counts, reciprocal
      coverage, randomisation result), and shows an "Estimated field
      acquisition time" dialog. That estimate comes from a simple,
      adjustable model, since ABEM doesn't publish one fixed per-reading
      time - actual speed depends on stacking settings, site conditions,
      and how many receiver channels your instrument reads in parallel.
      Treat it as a ballpark figure and adjust the timing field(s) above to
      match your own field experience, then rebuild for a refined estimate.

    The main window stays open after a Build, so settings can be changed and
    built again, or Show Grid Layout / Show Animation opened, as many times
    as needed. The program only ends when the main window is closed.

3. Animation popup (opened from "Show Animation")

    Shows the electrode grid with the current reading's electrodes
    highlighted - red for the current-injection pair (A, B), blue for the
    potential pair (M, N) - stepping through every reading, normal and
    reciprocal, in order. Has Play/Pause, Save (renders the animation to a
    GIF file), and Close.
--------------------------------------------------------------------------------
"""

import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ===========================================================================
# ============================ USER CONFIGURATION ==========================
# ===========================================================================
# Default values. These are only used as the initial contents of the GUI's
# input fields (see run_gui()) - editing them changes what the form shows
# when it opens, not what a given run actually builds; the user's own entries
# in the form are what get passed to build().

N_ELECTRODES = 64             # total electrode count; ROWS * COLS must equal this

ROWS = 8                       # number of rows in the electrode grid
COLS = 8                       # number of columns in the electrode grid

X_SPACING = 1.0                # electrode spacing along columns, in metres
Y_SPACING = 1.0                # electrode spacing along rows, in metres

DIRECTIONS = ("rows", "cols")  # which reading directions are pre-checked in the GUI:
                                #   "rows"    -> array-type sequence along each row, left to right
                                #   "cols"    -> array-type sequence along each column, top to bottom
                                #   "diag"    -> array-type sequence along every diagonal of >= 4 electrodes
                                #   "l_shape" -> 'L'-array perpendicular-dipole readings at the 4
                                #                grid corners (independent of array type - current/
                                #                potential pairs sit on two different perpendicular arms)
                                # combine any of them, e.g. ("rows", "cols", "l_shape")
                                # "array-type sequence" means whichever of Wenner/Wenner-Schlumberger/
                                # Dipole-Dipole/Gradient was chosen in the array-type popup

RANDOMIZE = False              # initial state of the "Randomise order" checkbox - if enabled,
                                # shuffles the reading order so consecutive measurements never
                                # use grid-adjacent (including diagonal) electrodes, to avoid
                                # residual charge from one measurement affecting the next
                                # (see randomize_non_adjacent())

SPREAD_OUTPUT_FILE = "spread.xml"      # default output filename for the Spread XML
PROTOCOL_OUTPUT_FILE = "protocol.xml"  # default output filename for the Protocol XML
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. ELECTRODE GRID
# ---------------------------------------------------------------------------
def build_electrode_grid(rows, cols, x_spacing=1.0, y_spacing=1.0, n_electrodes=N_ELECTRODES):
    """
    Build the list of electrode dicts for a rows x cols grid.

    Numbering follows a boustrophedon ("snake") pattern rather than plain
    row-major order: row 0 is numbered left-to-right, row 1 right-to-left,
    row 2 left-to-right again, and so on, alternating by row parity. This
    mirrors how a real multi-electrode cable is laid out in the field - it
    can't jump back to column 0 at the end of a row, it has to fold back and
    run the return row in the opposite direction.

    A useful side effect of the snake order: whenever cols evenly divides
    CABLE_SIZE (32) - e.g. 8 or 16 columns - each successive block of 32
    numbers stays spatially contiguous (a whole-row group of the grid),
    which is what keeps the per-cable electrode groupings in
    write_spread_xml() lined up with real physical cable segments instead
    of an arbitrary numeric slice. E.g. for a 4x16 grid, column 0's
    electrodes come out numbered 1, 32, 33, 64 from top to bottom - the
    numbering folds back on itself at the end of each row exactly as a
    real cable would.

    Each electrode dict has: number (the snake-order position, 1-based),
    switch_address (set equal to number - see write_spread_xml() for why
    this 1:1 mapping matters), switch_id (always 0, the internal relay
    switch), x/y (metres, from row/col * spacing), z (always 0.0), and the
    raw row/col grid indices.

    Raises ValueError if rows * cols != n_electrodes.
    """
    if rows * cols != n_electrodes:
        raise ValueError(
            f"rows x cols must equal {n_electrodes} (got {rows} x {cols} = {rows * cols})"
        )

    electrodes = []
    number = 1
    for r in range(rows):
        col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in col_range:
            electrodes.append({
                "number": number,
                "switch_address": number,  # matches the Terrameter's internal relay channel 1:1
                "switch_id": 0,            # 0 = internal relay switch (no external selector)
                "x": round(c * x_spacing, 4),
                "y": round(r * y_spacing, 4),
                "z": 0.0,
                "row": r,
                "col": c,
            })
            number += 1
    return electrodes


def electrode_lookup(electrodes):
    """Build a dict mapping electrode number -> its electrode dict."""
    return {e["number"]: e for e in electrodes}


def rows_of(electrodes, rows, cols):
    """Return each grid row as a list of electrode numbers, left to right."""
    lookup = electrode_lookup(electrodes)
    grid_by_rc = {(e["row"], e["col"]): e["number"] for e in electrodes}
    return [[grid_by_rc[(r, c)] for c in range(cols)] for r in range(rows)]


def cols_of(electrodes, rows, cols):
    """Return each grid column as a list of electrode numbers, top to bottom."""
    grid_by_rc = {(e["row"], e["col"]): e["number"] for e in electrodes}
    return [[grid_by_rc[(r, c)] for r in range(rows)] for c in range(cols)]


def diagonals_of(electrodes, rows, cols, min_length=4):
    """
    Return every diagonal line of the grid, in both diagonal directions
    (top-left -> bottom-right, and top-right -> bottom-left), each as a
    list of electrode numbers in physical order along that diagonal.

    Diagonals shorter than min_length are skipped, since every array type
    supported here needs at least 4 distinct electrodes on a line (A, B, M,
    N).

    Each step along a diagonal covers the same physical distance
    (sqrt(x_spacing^2 + y_spacing^2)) since the grid itself is evenly
    spaced, so these lines are evenly spaced too - which is what the
    line-array generators assume - even when x_spacing != y_spacing.
    """
    grid_by_rc = {(e["row"], e["col"]): e["number"] for e in electrodes}
    diagonals = []

    # Diagonals running top-left to bottom-right: each has a constant
    # (col - row) value k, ranging over every diagonal that touches the grid.
    for k in range(-(rows - 1), cols):
        line = []
        r, c = max(0, -k), max(0, k)
        while r < rows and c < cols:
            line.append(grid_by_rc[(r, c)])
            r += 1
            c += 1
        if len(line) >= min_length:
            diagonals.append(line)

    # Diagonals running top-right to bottom-left: each has a constant
    # (row + col) value k.
    for k in range(0, rows + cols - 1):
        line = []
        r, c = min(k, rows - 1), k - min(k, rows - 1)
        while r >= 0 and c < cols:
            line.append(grid_by_rc[(r, c)])
            r -= 1
            c += 1
        if len(line) >= min_length:
            diagonals.append(line)

    return diagonals


def corner_arms_of(electrodes, rows, cols):
    """
    For each of the 4 grid corners, return the pair of perpendicular
    electrode arms (row_arm, col_arm) that meet there - the corner's full
    row and full column - each ordered starting at the corner and running
    outward to the opposite edge. The corner electrode itself is only
    included in row_arm; col_arm starts at the next electrode along so it
    isn't duplicated.
    """
    grid_by_rc = {(e["row"], e["col"]): e["number"] for e in electrodes}
    result = []

    corners = [
        (0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)
    ]
    for corner_r, corner_c in corners:
        col_range = range(cols) if corner_c == 0 else range(cols - 1, -1, -1)
        row_arm = [grid_by_rc[(corner_r, c)] for c in col_range]

        row_range = range(rows) if corner_r == 0 else range(rows - 1, -1, -1)
        col_arm = [grid_by_rc[(r, corner_c)] for r in row_range][1:]

        result.append((row_arm, col_arm))

    return result


def l_array_cross_arm_readings(electrodes, rows, cols, max_dipole_spacing=2):
    """
    Generate 'L'-array (perpendicular dipole) readings for all 4 grid
    corners, following Tejero-Andrade et al. (2015, Near Surface
    Geophysics, doi:10.3997/1873-0604.2015015). This is a fundamentally
    different geometry from the line arrays below: the current pair and
    potential pair sit on two different, perpendicular arms rather than
    all four electrodes on one row/column/diagonal.

    For each corner (using corner_arms_of()): the current pair (A, B) is a
    dipole swept along row_arm, and the potential pair (M, N) is a dipole
    swept along col_arm, for every combination of current-dipole spacing
    1..max_dipole_spacing and potential-dipole spacing 1..max_dipole_spacing
    - i.e. a full sweep of current position x potential position for each
    spacing pair.

    max_dipole_spacing caps the electrode separation *within* each dipole
    (A-B or M-N), not the distance between the two dipoles, which varies
    naturally as they slide along their arms. It's capped because the
    reading count grows quickly: an 8x8 grid at max_dipole_spacing=2
    already produces roughly 140 readings per corner, 600+ total across all
    4 corners, before reciprocals.

    Returns a list of (A, B, M, N) electrode-number tuples.
    """
    readings = []
    for row_arm, col_arm in corner_arms_of(electrodes, rows, cols):
        max_a1 = min(max_dipole_spacing, len(row_arm) - 1)
        max_a2 = min(max_dipole_spacing, len(col_arm) - 1)

        for a1 in range(1, max_a1 + 1):
            ab_positions = [(row_arm[i], row_arm[i + a1])
                             for i in range(len(row_arm) - a1)]
            for a2 in range(1, max_a2 + 1):
                mn_positions = [(col_arm[j], col_arm[j + a2])
                                 for j in range(len(col_arm) - a2)]
                for A, B in ab_positions:
                    for M, N in mn_positions:
                        readings.append((A, B, M, N))

    return readings


# ---------------------------------------------------------------------------
# 2. LINE-ARRAY GENERATORS (each turns an ordered electrode list into
#    (A, B, M, N) readings, following a specific array type's geometry)
# ---------------------------------------------------------------------------
def generate_wenner_1d(line):
    """
    Standard Wenner-alpha sequence along one line of electrode numbers
    (given in physical order along that line).

    For every spacing factor a = 1, 2, 3, ... and valid start position s:
        A = line[s], M = line[s + a], N = line[s + 2a], B = line[s + 3a]
    i.e. A, M, N, B are equally spaced a apart.

    Returns a list of (A, B, M, N) electrode-number tuples - current
    electrodes first, then potential electrodes, the convention used
    throughout this file.
    """
    n = len(line)
    readings = []
    max_a = (n - 1) // 3
    for a in range(1, max_a + 1):
        for s in range(0, n - 3 * a):
            A = line[s]
            M = line[s + a]
            N = line[s + 2 * a]
            B = line[s + 3 * a]
            readings.append((A, B, M, N))
    return readings


def generate_wenner_schlumberger_1d(line):
    """
    Standard Wenner-Schlumberger sequence along one line of electrode
    numbers: the potential pair (M, N) is kept at a fixed 1-electrode
    spacing while the current pair (A, B) expands outward around it via a
    separation factor n = 1, 2, 3, ...

    For every n and every valid start position s:
        A = line[s], M = line[s + n], N = line[s + n + 1], B = line[s + 2n + 1]

    Returns a list of (A, B, M, N) electrode-number tuples.
    """
    n_electrodes = len(line)
    readings = []
    max_n = (n_electrodes - 2) // 2
    for n in range(1, max_n + 1):
        span = 2 * n + 1
        for s in range(0, n_electrodes - span):
            A = line[s]
            M = line[s + n]
            N = line[s + n + 1]
            B = line[s + 2 * n + 1]
            readings.append((A, B, M, N))
    return readings


def generate_dipole_dipole_1d(line):
    """
    Standard dipole-dipole sequence along one line of electrode numbers.
    Both the current dipole (A, B) and the potential dipole (M, N) use unit
    spacing, separated from each other by a factor n = 1, 2, 3, ...

    For every start position s and separation n:
        A = line[s], B = line[s + 1], M = line[s + 1 + n], N = line[s + 2 + n]

    Returns a list of (A, B, M, N) electrode-number tuples.
    """
    n_electrodes = len(line)
    readings = []
    for s in range(0, n_electrodes - 1):
        A = line[s]
        B = line[s + 1]
        max_n = n_electrodes - 3 - s
        for n in range(1, max_n + 1):
            M = line[s + 1 + n]
            N = line[s + 2 + n]
            readings.append((A, B, M, N))
    return readings


def generate_gradient_1d(line, mn_spacing=1, ab_span_fractions=(1.0,), ab_step=None):
    """
    Gradient array along one line of electrode numbers: for each current
    pair (A, B), potential pairs of fixed spacing mn_spacing sweep every
    valid position strictly between A and B. This is why Gradient needs far
    fewer current injections than Wenner/Wenner-Schlumberger/dipole-dipole
    for a similar reading count - one injection stays on while many
    potential pairs are read in sequence, matching the <Tx>-with-many-<Rx>
    structure written by write_protocol_xml().

    ab_span_fractions sets how wide each current pair is, as a fraction of
    the full line length (1.0 = the two end electrodes of the line); more
    than one fraction produces multiple current-pair widths. ab_step sets
    how far the current pair's start position slides between successive
    windows of the same width (defaults to the span itself, i.e.
    non-overlapping windows).
    """
    n_electrodes = len(line)
    readings = []
    for frac in ab_span_fractions:
        span = max(2, round((n_electrodes - 1) * frac))
        step = ab_step if ab_step else span
        a_idx = 0
        while a_idx + span <= n_electrodes - 1:
            b_idx = a_idx + span
            A, B = line[a_idx], line[b_idx]
            for m_idx in range(a_idx + 1, b_idx - mn_spacing):
                n_idx = m_idx + mn_spacing
                if n_idx < b_idx:
                    readings.append((A, B, line[m_idx], line[n_idx]))
            a_idx += step
    return readings


# Maps each array-type key (as chosen in the GUI's array-type popup) to the
# line-array generator function that turns an ordered electrode line into
# (A, B, M, N) readings for that array type.
ARRAY_GENERATORS = {
    "wenner": generate_wenner_1d,
    "wenner_schlumberger": generate_wenner_schlumberger_1d,
    "dipole_dipole": generate_dipole_dipole_1d,
    "gradient": generate_gradient_1d,
}

ARRAY_TYPE_LABELS = {
    "wenner": "Wenner",
    "wenner_schlumberger": "Wenner-Schlumberger",
    "dipole_dipole": "Dipole-Dipole",
    "gradient": "Gradient",
}


def build_array_readings(electrodes, rows, cols, directions=("rows", "cols"),
                           array_type="wenner"):
    """
    Build the full set of (A, B, M, N) readings for the grid by combining
    whichever of the following are present in `directions`:
        "rows"    - array_type sequence along each row, left to right
        "cols"    - array_type sequence along each column, top to bottom
        "diag"    - array_type sequence along every diagonal line of
                    >= 4 electrodes, in both diagonal directions
        "l_shape" - 'L'-array perpendicular-dipole readings at the 4 grid
                    corners (see l_array_cross_arm_readings()) - always
                    uses its own generator, independent of array_type

    array_type (one of the keys in ARRAY_GENERATORS) selects which
    line-array generator is used for "rows", "cols", and "diag".
    """
    generator = ARRAY_GENERATORS[array_type]
    readings = []
    if "rows" in directions:
        for line in rows_of(electrodes, rows, cols):
            readings.extend(generator(line))
    if "cols" in directions:
        for line in cols_of(electrodes, rows, cols):
            readings.extend(generator(line))
    if "diag" in directions:
        for line in diagonals_of(electrodes, rows, cols):
            readings.extend(generator(line))
    if "l_shape" in directions:
        readings.extend(l_array_cross_arm_readings(electrodes, rows, cols))
    return readings


# ---------------------------------------------------------------------------
# 3. RECIPROCALS
# ---------------------------------------------------------------------------
def add_reciprocals(readings, interleaved=True, reciprocal_percentage=100, seed=None):
    """
    Take a flat list of (A, B, M, N) normal readings and return a list of
    reading dicts, each with an added reciprocal (current/potential pairs
    swapped) for a chosen subset of them:
        normal:     inject A-B, measure M-N
        reciprocal: inject M-N, measure A-B

    Each output dict has keys: index (1-based position in the returned
    list), A/B/M/N, reciprocal (bool), and pair (the 1-based index of the
    original reading it came from, shared between a normal reading and its
    reciprocal).

    reciprocal_percentage (0-100) sets what fraction of the input readings
    get a reciprocal. Which ones is chosen at random (via rng.sample) rather
    than just the first N%, so a reduced set still covers the whole survey
    representatively instead of being concentrated at the start of the
    acquisition order. 100 (the default) gives every reading a reciprocal.

    interleaved=True (default) places each reciprocal right after its
    normal reading: [normal_1, reciprocal_1, normal_2, ...] - fine for
    Wenner/Wenner-Schlumberger/dipole-dipole, which essentially never
    repeat the same Tx (current) pair back-to-back anyway.

    interleaved=False instead appends all normal readings first, then all
    reciprocals afterward. Use this for Gradient: it relies on runs of
    consecutive readings sharing one Tx pair while potential pairs sweep
    between them (see write_protocol_xml()), and a reciprocal always has a
    *different* Tx, so interleaving one after every reading would break
    those runs apart.
    """
    rng = random.Random(seed)
    n = len(readings)
    num_recip = round(n * reciprocal_percentage / 100)
    num_recip = max(0, min(n, num_recip))
    recip_indices = set(rng.sample(range(n), num_recip)) if num_recip > 0 else set()

    full = []
    idx = 1
    if interleaved:
        for i, (A, B, M, N) in enumerate(readings):
            full.append({"index": idx, "A": A, "B": B, "M": M, "N": N,
                         "reciprocal": False, "pair": i + 1})
            idx += 1
            if i in recip_indices:
                full.append({"index": idx, "A": M, "B": N, "M": A, "N": B,
                             "reciprocal": True, "pair": i + 1})
                idx += 1
    else:
        for i, (A, B, M, N) in enumerate(readings):
            full.append({"index": idx, "A": A, "B": B, "M": M, "N": N,
                         "reciprocal": False, "pair": i + 1})
            idx += 1
        for i, (A, B, M, N) in enumerate(readings):
            if i in recip_indices:
                full.append({"index": idx, "A": M, "B": N, "M": A, "N": B,
                             "reciprocal": True, "pair": i + 1})
                idx += 1
    return full


def estimate_acquisition_time(full_readings, seconds_per_injection=3.0, seconds_per_reading=2.0):
    """
    Rough estimate of total field acquisition time for full_readings, using
    an adjustable model rather than one fixed per-reading time (ABEM's
    actual speed depends on stacking settings, site/noise conditions, and
    how many receiver channels the instrument reads in parallel):

        total_seconds = num_injections * seconds_per_injection
                       + len(full_readings) * seconds_per_reading

    num_injections counts how many times the current pair (Tx) changes
    across full_readings, i.e. how many separate current injections the
    sequence needs - seconds_per_injection is meant to cover relay
    switching and current stabilization before each one, and
    seconds_per_reading the voltage sampling for one reading. This is why
    array type affects the total even with the same seconds_per_reading:
    Wenner/Wenner-Schlumberger/dipole-dipole change Tx on nearly every
    reading, while Gradient reuses one Tx across many readings (matching
    the grouping in write_protocol_xml()), so it pays the injection cost
    far less often.

    This assumes every reading under one injection happens sequentially,
    which understates the speed of instruments that read several receiver
    channels in parallel per injection.

    Returns (total_seconds, num_injections).
    """
    num_injections = 0
    current_tx = None
    for r in full_readings:
        tx = (r["A"], r["B"])
        if tx != current_tx:
            num_injections += 1
            current_tx = tx
    total_seconds = num_injections * seconds_per_injection + len(full_readings) * seconds_per_reading
    return total_seconds, num_injections


def format_duration(total_seconds):
    """Format a number of seconds as a compact string, e.g. '1h 24m 10s' or '3m 5s' (hours omitted when zero)."""
    total_seconds = int(round(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def dedupe_readings(full_readings):
    """
    Drop reading dicts whose (A, B, M, N) combination already appeared
    earlier in full_readings, keeping the first occurrence and the
    original ordering. Guards against rare duplicate readings, e.g. where
    row and column sweeps happen to overlap.
    """
    seen = set()
    out = []
    for r in full_readings:
        key = (r["A"], r["B"], r["M"], r["N"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# 3.5 OPTIONAL: RANDOMIZE ORDER, KEEPING CONSECUTIVE SETS NON-ADJACENT
# ---------------------------------------------------------------------------
def randomize_non_adjacent(full_readings, electrodes, min_gap=2, attempts=None, seed=None):
    """
    Reorder full_readings so that consecutive measurements avoid sharing
    grid-adjacent electrodes, to reduce the chance that residual charge
    from one reading affects the next.

    "Adjacent" is judged on grid position (row/col index), not physical
    distance: min_gap=2 requires every electrode of one measurement to be
    at least 2 grid steps (Chebyshev distance - diagonal neighbours count
    as distance 1) from every electrode of the next measurement, i.e. at
    least one electrode's worth of gap between the two.

    Implementation is a randomized greedy construction: shuffle the start,
    then repeatedly pick a uniformly random still-unused reading that
    satisfies the gap requirement against the last one placed. When none
    qualify (possible near the end on small/dense grids, or with many
    readings such as diagonals included), the least-bad (most distant)
    remaining reading is used instead and counted as a violation. The whole
    process is retried up to `attempts` times, keeping whichever attempt
    has the fewest violations (stopping early on a violation-free result).

    Returns (reordered_list, violation_count); violation_count == 0 means
    every consecutive pair in the result satisfies the gap requirement.
    Reindexes the "index" field of each returned reading to match its new
    position. Returns (full_readings, 0) unchanged if it is empty.
    """
    import numpy as np

    rng = random.Random(seed)
    n = len(full_readings)
    if n == 0:
        return full_readings, 0

    if attempts is None:
        # Scale attempts down for very large protocols so this stays fast
        # (e.g. a 1x64 grid can produce 1000+ readings).
        attempts = max(2, min(15, 6000 // n))

    positions = {e["number"]: (e["row"], e["col"]) for e in electrodes}
    rc = np.array([
        [positions[r["A"]], positions[r["B"]], positions[r["M"]], positions[r["N"]]]
        for r in full_readings
    ], dtype=np.float32)  # shape (n, 4, 2)

    def dist_from(last_idx, mask):
        """Chebyshev distance from reading[last_idx] to every reading (n,)."""
        last_rc = rc[last_idx]  # (4, 2)
        diff = np.abs(rc[:, :, None, :] - last_rc[None, None, :, :])  # (n, 4, 4, 2)
        cheb = diff.max(axis=-1)  # (n, 4, 4)
        dist = cheb.min(axis=(1, 2))  # (n,)
        return np.where(mask, dist, -1.0)

    best_order = None
    best_violations = None

    for _ in range(attempts):
        used = np.zeros(n, dtype=bool)
        start = rng.randrange(n)
        sequence_idx = [start]
        used[start] = True
        violations = 0

        while not used.all():
            mask = ~used
            dist = dist_from(sequence_idx[-1], mask)
            candidates = np.where((dist >= min_gap) & mask)[0]
            if len(candidates) > 0:
                choice = int(candidates[rng.randrange(len(candidates))])
            else:
                valid_idx = np.where(mask)[0]
                choice = int(valid_idx[np.argmax(dist[valid_idx])])  # least-bad option
                violations += 1
            sequence_idx.append(choice)
            used[choice] = True

        if best_violations is None or violations < best_violations:
            best_order, best_violations = sequence_idx, violations
        if best_violations == 0:
            break

    reordered = [full_readings[i] for i in best_order]
    for i, r in enumerate(reordered, start=1):
        r["index"] = i
    return reordered, best_violations


# ---------------------------------------------------------------------------
# 4. XML WRITERS
# ---------------------------------------------------------------------------
# Reference layout for the two XML files this module writes (see
# write_spread_xml() / write_protocol_xml()), based on a Spread/Protocol
# pair that opens correctly in Terrameter LS Toolbox:
#
#   <Spread>
#     <Name>...</Name>
#     <Description>...</Description>
#     <CreateStation><Name>...</Name><X>0</X></CreateStation>
#     <Rollalong><X>...</X></Rollalong>
#     <Cable>
#       <Name>1</Name>
#       <Electrode><Id>1</Id><X>..</X><Y>..</Y><Name>1-1</Name><SwitchAddress>1</SwitchAddress></Electrode>
#       ... (up to 32 electrodes per cable)
#     </Cable>
#     <Cable>...</Cable>   (electrodes 33-64)
#   </Spread>
#
#   <Protocol>
#     <Name>...</Name>
#     <Description>...</Description>
#     <SpreadFile>spread.xml</SpreadFile>
#     <Arraycode>1</Arraycode>    <!-- 1 = Wenner (see ARRAY_CODES below for the others) -->
#     <Sequence>
#       <Measure><Tx>A B</Tx><Rx>M N</Rx></Measure>
#       ...
#     </Sequence>
#   </Protocol>
#
# Note the real format has no <SwitchId>, <Z>, per-electrode <Number>,
# per-reading <Index>, or <Reciprocal> flag - fields build_electrode_grid()
# and add_reciprocals() track internally but that never get written to the
# XML. Reciprocal readings are just additional plain <Measure> entries with
# Tx/Rx swapped, same as everything else here.
CABLE_SIZE = 32  # Terrameter LS/LS2 has two 32-channel internal connectors

# ABEM/Res2Dinv array codes used in the <Arraycode> element, per the ABEM
# Terrameter LS protocol reference. The codes relevant to this file:
#   1  = Wenner
#   3  = Dipole-dipole
#   7  = Schlumberger - used here for the Wenner-Schlumberger sequence, since
#        the Terrameter LS only ever measures on the fixed electrode grid
#        positions generated here, not a continuous VES-style Schlumberger
#        sweep
#   11 = General surface array - documented as the code to use for "an array
#        not defined in the list", which fits the 'L'-array (perpendicular
#        dipole) here: it's not Wenner, Wenner-Schlumberger, Dipole-Dipole,
#        or Gradient, so it gets its own code rather than borrowing whatever
#        line-array type happens to be selected alongside it
#   15 = Multiple gradient array
ARRAY_CODES = {
    "wenner": 1,
    "wenner_schlumberger": 7,
    "dipole_dipole": 3,
    "gradient": 15,
}
GENERAL_SURFACE_ARRAYCODE = 11  # used whenever 'l_shape' readings are included
DEFAULT_ARRAYCODE = ARRAY_CODES["wenner"]  # fallback if write_protocol_xml() is called directly


# Serialize an ElementTree element to an indented XML string via a
# round-trip through minidom (ElementTree itself has no pretty-printer).
def prettify(elem):
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="    ")


# Write electrodes as a Spread XML file at `path`. name/description/
# station_name become the corresponding <Name>/<Description>/<CreateStation>
# text; rollalong_x defaults to CABLE_SIZE, matching the reference file this
# schema is based on.
def write_spread_xml(electrodes, path, name="Generated_ERT_Spread",
                      description="Generated electrode grid",
                      station_name="ERT_GRID", rollalong_x=None):
    if rollalong_x is None:
        rollalong_x = float(CABLE_SIZE)

    root = ET.Element("Spread")
    ET.SubElement(root, "Name").text = name
    ET.SubElement(root, "Description").text = description

    create_station = ET.SubElement(root, "CreateStation")
    ET.SubElement(create_station, "Name").text = station_name
    ET.SubElement(create_station, "X").text = "0"

    rollalong = ET.SubElement(root, "Rollalong")
    ET.SubElement(rollalong, "X").text = str(rollalong_x)

    # Electrodes are already numbered 1..64 (== SwitchAddress) by
    # build_electrode_grid(); chunk them sequentially into groups of
    # CABLE_SIZE to match the instrument's two 32-channel connectors.
    for cable_index in range(0, len(electrodes), CABLE_SIZE):
        cable_electrodes = electrodes[cable_index:cable_index + CABLE_SIZE]
        cable_number = cable_index // CABLE_SIZE + 1

        cable_el = ET.SubElement(root, "Cable")
        ET.SubElement(cable_el, "Name").text = str(cable_number)

        for local_id, e in enumerate(cable_electrodes, start=1):
            electrode_el = ET.SubElement(cable_el, "Electrode")
            # <Id> is set equal to <SwitchAddress> and must stay globally
            # unique across the whole spread (i.e. must NOT restart at 1 for
            # each cable). The instrument resolves an electrode's physical
            # position from <Id>, not from the <X>/<Y> written here, so
            # duplicate Ids across cables would make cable 2's electrodes
            # resolve to cable 1's positions - collapsing the grid and
            # making readings that reference the ambiguous addresses
            # unresolvable.
            ET.SubElement(electrode_el, "Id").text = str(e["switch_address"])
            ET.SubElement(electrode_el, "X").text = str(e["x"])
            ET.SubElement(electrode_el, "Y").text = str(e["y"])
            # Name stays a human-readable "cable-position" label; only Id
            # carries resolution meaning for the instrument.
            ET.SubElement(electrode_el, "Name").text = f"{cable_number}-{local_id}"
            ET.SubElement(electrode_el, "SwitchAddress").text = str(e["switch_address"])

    with open(path, "w", encoding="utf-8") as f:
        f.write(prettify(root))


# Write full_readings as a Protocol XML file at `path`, referencing
# spread_filename as its <SpreadFile>. name/description become <Name>/
# <Description>; arraycode becomes <Arraycode> (see ARRAY_CODES).
def write_protocol_xml(full_readings, path, spread_filename,
                        name="Generated_Protocol",
                        description="Generated 4-electrode array protocol",
                        arraycode=DEFAULT_ARRAYCODE):
    root = ET.Element("Protocol")
    ET.SubElement(root, "Name").text = name
    ET.SubElement(root, "Description").text = description
    ET.SubElement(root, "SpreadFile").text = spread_filename
    ET.SubElement(root, "Arraycode").text = str(arraycode)

    sequence = ET.SubElement(root, "Sequence")
    # Consecutive readings sharing the same Tx (current pair) are grouped
    # under one <Measure> with multiple <Rx> children instead of a
    # <Measure> per reading - this is what lets a Gradient survey's shared
    # current injections show up as one injection with several potential
    # readings, matching real ABEM Gradient protocol files. Array types
    # that rarely repeat a Tx back-to-back (Wenner, Wenner-Schlumberger,
    # dipole-dipole) end up with effectively one <Rx> per <Measure> anyway.
    current_tx = None
    current_measure = None
    for r in full_readings:
        tx = (r["A"], r["B"])
        if tx != current_tx:
            current_measure = ET.SubElement(sequence, "Measure")
            ET.SubElement(current_measure, "Tx").text = f"{r['A']} {r['B']}"
            current_tx = tx
        ET.SubElement(current_measure, "Rx").text = f"{r['M']} {r['N']}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(prettify(root))


# ---------------------------------------------------------------------------
# 5. PLOTTING HELPERS (shared by the animated view, the static grid-only
#    view, and headless file-saving of either)
# ---------------------------------------------------------------------------
def frame_colors(electrodes, reading):
    """
    Build a {electrode_number: color} dict for one reading: current
    electrodes (A, B) red, potential electrodes (M, N) blue, every other
    electrode light grey.

    Has no matplotlib/tkinter dependency, so it's shared unchanged between
    the live animated popup (show_animation_popup()) and headless/file-only
    animation rendering (save_animation_to_file()).
    """
    colors = {e["number"]: "#c9c9c9" for e in electrodes}
    colors[reading["A"]] = "#d62728"  # current electrode A - red
    colors[reading["B"]] = "#d62728"  # current electrode B - red
    colors[reading["M"]] = "#1f77b4"  # potential electrode M - blue
    colors[reading["N"]] = "#1f77b4"  # potential electrode N - blue
    return colors


def _min_adjacent_gap(electrodes, rows, cols):
    """
    Smallest distance, in metres, between any two grid-adjacent electrodes
    (horizontal or vertical neighbours only, not diagonal). Computed
    directly from electrode x/y coordinates rather than assumed from the
    spacing inputs, so it stays correct even if X and Y spacing differ.
    """
    by_rc = {(e["row"], e["col"]): (e["x"], e["y"]) for e in electrodes}
    gaps = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = by_rc[(r, c)]
            if c + 1 < cols:
                x1, y1 = by_rc[(r, c + 1)]
                gaps.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
            if r + 1 < rows:
                x1, y1 = by_rc[(r + 1, c)]
                gaps.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    return min(gaps) if gaps else 1.0


def compute_figure_layout(electrodes, rows, cols, screen_height_in=None, reserve_controls=True):
    """
    Work out the matplotlib Figure size and marker/label scaling shared by
    the animated popup, the static grid popup, and their headless/
    file-saving equivalents.

    screen_height_in: the real screen height in inches, for a live popup.
    Treated as a preference, not a hard cap - if fitting the grid within
    that height would shrink electrode markers below a legible minimum
    size, the minimum size wins instead and the figure is allowed to end
    up taller than the screen (the caller then wraps it in a scrollable
    frame rather than rendering illegibly small numbers). Pass None for
    headless/file-only rendering, where there is no screen to fit.

    reserve_controls: whether to reserve extra vertical space below the
    plot for a row of buttons (Pause/Close/Save etc). Pass False for
    file-only saving, where there are no buttons.

    Returns (fig_width, fig_height, scale, pad, title_reserve_in,
    marker_radius_in) - scale is inches per metre of data, and pad is the
    axis padding (in data units) needed to keep markers from touching the
    plot edges.
    """
    xs = [e["x"] for e in electrodes]
    ys = [e["y"] for e in electrodes]
    x_extent = max(max(xs) - min(xs), 0.5) if xs else 1.0
    y_extent = max(max(ys) - min(ys), 0.5) if ys else 1.0

    INCHES_PER_METRE = 0.45
    PLOT_MARGIN_IN = 2.0
    TITLE_RESERVE_IN = 1.3
    CONTROLS_RESERVE_IN = 1.0 if reserve_controls else 0.0
    MIN_PLOT_DIM_IN = 4.0
    MAX_FIG_WIDTH_IN = 13.0
    DEFAULT_MAX_PLOT_HEIGHT_IN = 9.0
    DEFAULT_MARKER_RADIUS_IN = (160 / 3.14159) ** 0.5 / 72  # the original comfortable size
    MIN_READABLE_MARKER_RADIUS_IN = 0.093  # smallest a marker can get before its number stops being legible

    if screen_height_in is not None:
        max_plot_height_in = max(MIN_PLOT_DIM_IN, screen_height_in - TITLE_RESERVE_IN - CONTROLS_RESERVE_IN - 1.0)
    else:
        max_plot_height_in = DEFAULT_MAX_PLOT_HEIGHT_IN

    # X and Y must share one scale (inches per metre) since the plot is
    # drawn with aspect="equal" - a different scale per axis would distort
    # the grid or clip electrodes near the figure edge.
    scale = INCHES_PER_METRE
    if x_extent * scale + PLOT_MARGIN_IN > MAX_FIG_WIDTH_IN:
        scale = min(scale, (MAX_FIG_WIDTH_IN - PLOT_MARGIN_IN) / x_extent)
    screen_preferred_scale = scale
    if y_extent * screen_preferred_scale + PLOT_MARGIN_IN > max_plot_height_in:
        screen_preferred_scale = min(screen_preferred_scale, (max_plot_height_in - PLOT_MARGIN_IN) / y_extent)

    # Shrinking markers to fit the screen is fine down to a point - past
    # that, unreadable numbers help nobody. If the screen-preferred scale
    # would push markers below the legible floor, use whatever LARGER
    # scale is needed to keep them legible instead, even if that makes the
    # figure taller than the screen - the popup wraps this in a scrollable
    # frame so the extra height is just scrolled through, not squeezed.
    min_gap = _min_adjacent_gap(electrodes, rows, cols)
    scale_needed_for_legibility = MIN_READABLE_MARKER_RADIUS_IN / (0.35 * min_gap)
    scale = max(screen_preferred_scale, scale_needed_for_legibility)
    scale = max(scale, 0.01)  # floor so pathologically large grids don't divide by ~0

    fig_width = max(x_extent * scale + PLOT_MARGIN_IN, 5.0)
    plot_height = max(y_extent * scale + PLOT_MARGIN_IN, MIN_PLOT_DIM_IN)
    fig_height = plot_height + TITLE_RESERVE_IN

    # Marker size still capped at a fraction of the actual gap between
    # adjacent electrodes (converted through `scale`) so circles never
    # overlap - but now `scale` itself was already raised if needed to
    # keep that cap at or above the legible floor, so this rarely needs
    # to shrink markers below DEFAULT_MARKER_RADIUS_IN except for grids
    # that were already comfortable to begin with.
    max_marker_radius_in = 0.35 * min_gap * scale
    marker_radius_in = max(min(DEFAULT_MARKER_RADIUS_IN, max_marker_radius_in), MIN_READABLE_MARKER_RADIUS_IN)

    pad = (marker_radius_in * 1.5) / scale

    return fig_width, fig_height, scale, pad, TITLE_RESERVE_IN, marker_radius_in


def build_electrode_axes(fig, electrodes, rows, cols, fig_height, pad, title_reserve_in, marker_radius_in):
    """
    Draw the base electrode scatter plot (grey markers, each labeled with
    its electrode number) onto `fig`, using the sizing/padding values
    computed by compute_figure_layout(). Shared by the animated and static
    grid views so both look consistent.

    Returns (ax, scatter, title_text) so the caller can further customize
    the plot - e.g. update marker colors per-frame for the animation, or
    set a static title for the grid-only view.
    """
    ax = fig.add_subplot(111)
    title_band_frac = title_reserve_in / fig_height
    fig.subplots_adjust(top=1 - title_band_frac, bottom=0.10, left=0.10, right=0.95)

    xs = [e["x"] for e in electrodes]
    ys = [e["y"] for e in electrodes]

    # Marker area (matplotlib's "s") and label font size both scale down
    # together with marker_radius_in, so numbers never overflow the circle
    # they're supposed to sit inside, however small the circle gets.
    marker_s = 3.14159 * (marker_radius_in * 72) ** 2
    default_radius_in = (160 / 3.14159) ** 0.5 / 72
    label_fontsize = max(4.0, 7.0 * (marker_radius_in / default_radius_in))

    scatter = ax.scatter(xs, ys, s=marker_s, c="#c9c9c9", edgecolors="black", zorder=2)
    for e in electrodes:
        ax.annotate(str(e["number"]), (e["x"], e["y"]), ha="center", va="center",
                    fontsize=label_fontsize, zorder=3)

    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    title_text = fig.suptitle("", fontsize=10.5, y=0.97, va="top")
    return ax, scatter, title_text


# ---------------------------------------------------------------------------
# 6. GUI: configuration form, grid preview popup, and animated
#    "electrode calling order" popup
# ---------------------------------------------------------------------------
def _make_scrollable_root_frame(root):
    """
    Wrap root's content area in a scrollable frame capped to a fraction of
    the screen height, so that if the config form ends up taller than the
    screen (e.g. Gradient's extra timing fields push it over the top on a
    shorter screen) it scrolls instead of pushing the Build button - and
    anything below where it overflows - off-screen with no way to reach it,
    since the window itself does not resize.

    Returns the inner Frame that the rest of run_gui() should place its
    form content into, instead of packing/gridding directly into root.
    """
    import tkinter as tk
    from tkinter import ttk

    max_visible_px = int(root.winfo_screenheight() * 0.85)  # leave room for window chrome/taskbar

    container = ttk.Frame(root)
    container.pack(side="top", fill="both", expand=True)

    canvas = tk.Canvas(container, highlightthickness=0)
    vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)

    inner = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_inner_configure(_event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.configure(height=min(inner.winfo_reqheight(), max_visible_px),
                          width=inner.winfo_reqwidth())

    inner.bind("<Configure>", on_inner_configure)

    canvas.pack(side="left", fill="both", expand=True)
    vbar.pack(side="right", fill="y")

    # Let the mouse wheel scroll this canvas, but only while the pointer is
    # actually over it - bind_all is scoped to Enter/Leave so it doesn't
    # hijack scrolling in other windows (e.g. the animation popup, which
    # has its own separate scrollable canvas).
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_enter(_event):
        canvas.bind_all("<MouseWheel>", on_mousewheel)

    def on_leave(_event):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", on_enter)
    canvas.bind("<Leave>", on_leave)

    return inner


def run_gui():
    """
    Run the interactive configuration GUI end to end: array-type popup,
    then the main configuration window (grid/protocol fields plus Show
    Grid Layout / Show Animation / Build buttons), calling build() when
    Build is pressed to write the Spread/Protocol XML files. See the
    module docstring's GUI walkthrough for the full flow. Blocks on
    root.mainloop() until the main window is closed.

    Requires tkinter (ships with most Python installs; on some Linux
    distros install with `sudo apt install python3-tk`) and matplotlib
    (`pip install matplotlib`).
    """
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib
    matplotlib.use("TkAgg")

    root = tk.Tk()
    root.title("ERT Electrode Grid Builder")
    root.resizable(False, False)
    root.withdraw()  # stay hidden until an array type is chosen below

    # Make sure closing this window actually ends the program - see the
    # note in show_animation_popup() about why this matters.
    def on_root_close():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_root_close)

    array_type_holder = {"value": "wenner"}
    array_type_label_var = tk.StringVar(value=ARRAY_TYPE_LABELS["wenner"])

    ARRAY_TYPE_DESCRIPTIONS = {
        "wenner": "Wenner - robust, widely used, moderate depth range",
        "wenner_schlumberger": "Wenner-Schlumberger - hybrid array, greater depth range from the same electrodes",
        "dipole_dipole": "Dipole-Dipole - strong lateral resolution, more readings",
        "gradient": "Gradient - fewest current injections, fastest to acquire in the field",
    }

    def show_array_type_popup():
        popup = tk.Toplevel(root)
        popup.title("Select Array Type")
        popup.resizable(False, False)

        def on_popup_close():
            # No array type chosen yet - nothing sensible to do but end the program.
            root.quit()
            root.destroy()

        popup.protocol("WM_DELETE_WINDOW", on_popup_close)

        frame = ttk.Frame(popup, padding=16)
        frame.grid(row=0, column=0)
        ttk.Label(frame, text="Which electrode array type would you like to generate?",
                  font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10))

        array_var = tk.StringVar(value=array_type_holder["value"])
        for i, key in enumerate(ARRAY_TYPE_LABELS):
            ttk.Radiobutton(frame, text=ARRAY_TYPE_DESCRIPTIONS[key],
                             variable=array_var, value=key).grid(
                row=i + 1, column=0, sticky="w", pady=2)

        def on_next():
            array_type_holder["value"] = array_var.get()
            array_type_label_var.set(ARRAY_TYPE_LABELS[array_var.get()])
            popup.destroy()
            build_timing_fields()
            root.deiconify()

        ttk.Button(frame, text="Next", command=on_next).grid(
            row=len(ARRAY_TYPE_LABELS) + 1, column=0, pady=(12, 0))

    show_array_type_popup()

    scroll_root = _make_scrollable_root_frame(root)
    form = ttk.Frame(scroll_root, padding=16)
    form.grid(row=0, column=0)

    array_type_row = ttk.Frame(form)
    array_type_row.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
    ttk.Label(array_type_row, text="Array type: ").pack(side="left")
    ttk.Label(array_type_row, textvariable=array_type_label_var, font=("TkDefaultFont", 9, "bold")).pack(side="left")

    def on_change_array_type():
        root.withdraw()
        show_array_type_popup()

    ttk.Button(array_type_row, text="Change", command=on_change_array_type).pack(side="left", padx=(10, 0))

    fields = {}

    def add_field(label, row, default, width=12):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=4)
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(form, textvariable=var, width=width)
        entry.grid(row=row, column=1, pady=4, padx=(8, 0))
        fields[label] = var

    ttk.Label(form, text="Grid Structure (Spread)", font=("TkDefaultFont", 9, "bold")).grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(4, 2))

    add_field("Total number of electrodes", 2, N_ELECTRODES)
    add_field("Number of rows", 3, ROWS)
    add_field("Number of columns", 4, COLS)
    add_field("X electrode spacing (m)", 5, X_SPACING)
    add_field("Y electrode spacing (m)", 6, Y_SPACING)

    ttk.Label(form, text="Calling Procedure (Protocol)", font=("TkDefaultFont", 9, "bold")).grid(
        row=7, column=0, columnspan=2, sticky="w", pady=(10, 2))

    dir_rows = tk.BooleanVar(value="rows" in DIRECTIONS)
    dir_cols = tk.BooleanVar(value="cols" in DIRECTIONS)
    dir_diag = tk.BooleanVar(value="diag" in DIRECTIONS)
    dir_lshape = tk.BooleanVar(value="l_shape" in DIRECTIONS)
    ttk.Checkbutton(form, text="Include rows", variable=dir_rows).grid(
        row=8, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(form, text="Include columns", variable=dir_cols).grid(
        row=9, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(form, text="Include diagonals (4+ electrode lines)", variable=dir_diag).grid(
        row=10, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(form, text="Include 'L'-array perpendicular dipole (4 grid corners)", variable=dir_lshape).grid(
        row=11, column=0, columnspan=2, sticky="w")

    randomize_var = tk.BooleanVar(value=RANDOMIZE)
    ttk.Checkbutton(
        form, text="Randomise order (keeps neighbouring electrode sets apart)",
        variable=randomize_var
    ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(8, 0))

    # Which readings get a reciprocal is chosen RANDOMLY (not just the
    # first N%) so a reduced percentage still covers all lines/depths
    # representatively rather than concentrating wherever the acquisition
    # order happens to start - see add_reciprocals() docstring.
    add_field("Percentage of measurements with reciprocals (%)", 13, 100)

    ttk.Label(form, text="Extras", font=("TkDefaultFont", 9, "bold")).grid(
        row=14, column=0, columnspan=2, sticky="w", pady=(10, 2))

    add_field("Spread file name", 15, SPREAD_OUTPUT_FILE)
    add_field("Spread name (shown on Terrameter)", 16, "", width=28)
    add_field("Protocol file name", 17, PROTOCOL_OUTPUT_FILE)
    add_field("Protocol name (shown on Terrameter)", 18, "", width=28)

    def on_show_grid():
        try:
            n_electrodes = int(fields["Total number of electrodes"].get())
            rows = int(fields["Number of rows"].get())
            cols = int(fields["Number of columns"].get())
            x_sp = float(fields["X electrode spacing (m)"].get())
            y_sp = float(fields["Y electrode spacing (m)"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Rows/Columns/Spacing/Electrodes must be numbers.")
            return
        try:
            electrodes = build_electrode_grid(rows, cols, x_sp, y_sp, n_electrodes=n_electrodes)
        except ValueError as e:
            messagebox.showerror("Invalid grid", str(e))
            return
        show_grid_popup(root, electrodes, rows, cols)

    ttk.Button(form, text="Show Grid Layout", command=on_show_grid).grid(
        row=19, column=0, columnspan=2, sticky="w", pady=(8, 0))

    # Holds the result of the most recent successful build, so "Show
    # Animation" (a separate button, pressed whenever you want) can open
    # the popup for whatever was last built - mirrors "Show Grid Layout"
    # above, rather than the animation forcing itself open immediately
    # after every build. Saving happens from inside the popup itself
    # (its own Save... button), same as the grid popup.
    last_build = {"electrodes": None, "rows": None, "cols": None, "full_readings": None}

    def on_show_animation():
        if last_build["full_readings"] is None:
            messagebox.showinfo("Nothing built yet", "Click \"Build\" first, then Show Animation.")
            return
        show_animation_popup(root, last_build["electrodes"], last_build["rows"], last_build["cols"],
                              last_build["full_readings"], speed_var)

    ttk.Button(form, text="Show Animation", command=on_show_animation).grid(
        row=20, column=0, columnspan=2, sticky="w", pady=(4, 0))

    speed_row = ttk.Frame(form)
    speed_row.grid(row=21, column=0, columnspan=2, sticky="we", pady=(8, 0))
    ttk.Label(speed_row, text="Animation speed (ms/step)").pack(side="left")
    speed_var = tk.IntVar(value=250)
    speed_value_var = tk.StringVar(value=f"{speed_var.get()} ms")

    def on_speed_change(_event=None):
        speed_value_var.set(f"{int(float(speed_scale.get()))} ms")

    speed_scale = ttk.Scale(speed_row, from_=50, to=1000, variable=speed_var,
                             orient="horizontal", command=on_speed_change)
    speed_scale.pack(side="left", padx=(8, 8), fill="x", expand=True)
    ttk.Label(speed_row, textvariable=speed_value_var, width=6).pack(side="left")

    # Field acquisition timing estimate - see estimate_acquisition_time()
    # docstring for why this is adjustable rather than fixed: ABEM doesn't
    # publish one fixed reading time, since it depends on stacking, site
    # conditions, and how many receiver channels your instrument reads in
    # parallel. Adjust to match your own field experience.
    #
    # Only Gradient actually needs the injection/reading split - every
    # other array type pays both costs on every single reading anyway, so
    # showing them separately would just be two numbers that always get
    # added back together. This shows a single combined "seconds per
    # reading" field normally, and only splits it into two fields when
    # Gradient is selected (where the split reflects a real difference:
    # many readings share one current injection).
    timing_frame = ttk.Frame(form)
    timing_frame.grid(row=22, column=0, columnspan=2, sticky="w", pady=(8, 0))
    timing_vars = {}

    def build_timing_fields():
        for child in timing_frame.winfo_children():
            child.destroy()
        timing_vars.clear()

        if array_type_holder["value"] == "gradient":
            ttk.Label(timing_frame, text="Seconds per current injection (For runtime estimate)").grid(
                row=0, column=0, sticky="w", pady=4)
            inj_var = tk.StringVar(value="3.0")
            ttk.Entry(timing_frame, textvariable=inj_var, width=12).grid(
                row=0, column=1, pady=4, padx=(8, 0))
            timing_vars["injection"] = inj_var

            ttk.Label(timing_frame, text="Seconds per reading (For runtime estimate)").grid(
                row=1, column=0, sticky="w", pady=4)
            read_var = tk.StringVar(value="2.0")
            ttk.Entry(timing_frame, textvariable=read_var, width=12).grid(
                row=1, column=1, pady=4, padx=(8, 0))
            timing_vars["reading"] = read_var
        else:
            ttk.Label(timing_frame, text="Seconds per measurement (For runtime estimate)").grid(
                row=0, column=0, sticky="w", pady=4)
            combined_var = tk.StringVar(value="5.0")
            ttk.Entry(timing_frame, textvariable=combined_var, width=12).grid(
                row=0, column=1, pady=4, padx=(8, 0))
            timing_vars["combined"] = combined_var

    build_timing_fields()

    status_var = tk.StringVar(value="")
    ttk.Label(form, textvariable=status_var, foreground="#555", wraplength=280).grid(
        row=24, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def on_build():
        try:
            n_electrodes = int(fields["Total number of electrodes"].get())
            rows = int(fields["Number of rows"].get())
            cols = int(fields["Number of columns"].get())
            x_sp = float(fields["X electrode spacing (m)"].get())
            y_sp = float(fields["Y electrode spacing (m)"].get())
            reciprocal_percentage = float(fields["Percentage of measurements with reciprocals (%)"].get())
            if array_type_holder["value"] == "gradient":
                seconds_per_injection = float(timing_vars["injection"].get())
                seconds_per_reading = float(timing_vars["reading"].get())
            else:
                seconds_per_injection = 0.0
                seconds_per_reading = float(timing_vars["combined"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Numeric fields must contain numbers.")
            return

        if not (0 <= reciprocal_percentage <= 100):
            messagebox.showerror("Invalid input", "Percentage of measurements with reciprocals (%) must be between 0 and 100.")
            return

        spread_filename = fields["Spread file name"].get().strip() or SPREAD_OUTPUT_FILE
        protocol_filename = fields["Protocol file name"].get().strip() or PROTOCOL_OUTPUT_FILE
        if not spread_filename.lower().endswith(".xml"):
            spread_filename += ".xml"
        if not protocol_filename.lower().endswith(".xml"):
            protocol_filename += ".xml"

        # Instrument-facing names (the <Name> element that shows up in the
        # Terrameter LS/LS2's own list) - separate from the computer
        # filenames above. Left blank -> build() falls back to its
        # auto-generated name.
        spread_name = fields["Spread name (shown on Terrameter)"].get().strip() or None
        protocol_name = fields["Protocol name (shown on Terrameter)"].get().strip() or None

        directions = tuple(d for d, on in (
            ("rows", dir_rows.get()), ("cols", dir_cols.get()),
            ("diag", dir_diag.get()), ("l_shape", dir_lshape.get()),
        ) if on)
        if not directions:
            messagebox.showerror("Invalid input",
                                  "Select at least one direction (rows, columns, diagonals, and/or L-shapes).")
            return

        array_type = array_type_holder["value"]

        try:
            electrodes, full_readings, violations = build(
                rows, cols, x_sp, y_sp, directions,
                spread_filename, protocol_filename,
                randomize=randomize_var.get(),
                array_type=array_type,
                n_electrodes=n_electrodes,
                reciprocal_percentage=reciprocal_percentage,
                spread_name=spread_name,
                protocol_name=protocol_name,
            )
        except ValueError as e:
            messagebox.showerror("Invalid grid", str(e))
            return

        status = (f"{len(electrodes)} electrodes, {len(full_readings)} readings "
                   f"written to {spread_filename} / {protocol_filename}")
        if reciprocal_percentage < 100:
            n_normal = sum(1 for r in full_readings if not r["reciprocal"])
            n_recip = sum(1 for r in full_readings if r["reciprocal"])
            status += (f"\nReciprocals: {reciprocal_percentage:.0f}% requested "
                       f"({n_recip} of {n_normal} readings, chosen at random).")
        if randomize_var.get():
            if violations == 0:
                status += "\nOrder randomized - no adjacent-set violations."
            else:
                status += (f"\nOrder randomized - {violations} unavoidable adjacent-set "
                            f"violation(s) (grid too small/dense to fully separate every pair).")
        if array_type == "gradient" and randomize_var.get():
            status += ("\nNote: randomizing breaks apart Gradient's shared-current-pair "
                        "groupings - data is still valid, but the field efficiency benefit "
                        "(fewer current injections) is lost.")
        status_var.set(status)

        total_seconds, num_injections = estimate_acquisition_time(
            full_readings, seconds_per_injection, seconds_per_reading)

        if array_type == "gradient":
            basis_text = (f"Based on {seconds_per_injection:.1f}s per current injection + "
                           f"{seconds_per_reading:.1f}s per reading (only {num_injections} "
                           f"injections needed for {len(full_readings)} readings, since "
                           f"Gradient reuses the same current pair across many readings).")
        else:
            basis_text = f"Based on {seconds_per_reading:.1f}s per reading."

        messagebox.showinfo(
            "Estimated field acquisition time",
            f"{len(full_readings)} readings ({array_type_holder['value'].replace('_', ' ')} array)\n\n"
            f"Estimated time: {format_duration(total_seconds)}\n\n"
            f"{basis_text} Adjust the field(s) above and rebuild for a more "
            f"accurate estimate - actual ABEM acquisition speed depends on "
            f"stacking settings, site conditions, and how many receiver "
            f"channels your instrument reads in parallel, so treat this as a "
            f"ballpark figure, not a guarantee."
        )

        last_build["electrodes"] = electrodes
        last_build["rows"] = rows
        last_build["cols"] = cols
        last_build["full_readings"] = full_readings

    ttk.Button(form, text="Build", command=on_build).grid(
        row=23, column=0, columnspan=2, pady=(12, 0))

    root.mainloop()


def _embed_scrollable_figure(popup, fig, fig_width, fig_height):
    """
    Embed a matplotlib Figure into `popup` inside a scrollable area, so
    figures taller or wider than the screen scroll instead of being
    squeezed to fit (which would make electrode numbers illegible on
    dense/lopsided grids). The visible viewport is capped to the screen
    size; anything beyond that scrolls via the scrollbars.

    Call this AFTER packing any bottom-anchored controls (Pause/Close/Save
    etc.), so those controls keep their guaranteed space regardless of how
    tall the figure ends up being.

    Returns the drawn FigureCanvasTkAgg instance.
    """
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    dpi = fig.get_dpi()
    screen_w_in = popup.winfo_screenwidth() / max(popup.winfo_fpixels("1i"), 1)
    screen_h_in = popup.winfo_screenheight() / max(popup.winfo_fpixels("1i"), 1)
    visible_w_in = min(fig_width, max(screen_w_in - 1.5, 3.0))
    visible_h_in = min(fig_height, max(screen_h_in - 2.5, 3.0))

    container = ttk.Frame(popup)
    tk_canvas = tk.Canvas(container, width=int(visible_w_in * dpi), height=int(visible_h_in * dpi),
                          highlightthickness=0)
    vbar = ttk.Scrollbar(container, orient="vertical", command=tk_canvas.yview)
    hbar = ttk.Scrollbar(container, orient="horizontal", command=tk_canvas.xview)
    tk_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

    inner = ttk.Frame(tk_canvas)
    tk_canvas.create_window((0, 0), window=inner, anchor="nw")

    figure_canvas = FigureCanvasTkAgg(fig, master=inner)
    figure_canvas.get_tk_widget().pack()
    figure_canvas.draw()

    def on_inner_configure(_event):
        tk_canvas.configure(scrollregion=tk_canvas.bbox("all"))

    inner.bind("<Configure>", on_inner_configure)

    vbar.pack(side="right", fill="y")
    hbar.pack(side="bottom", fill="x")
    tk_canvas.pack(side="left", fill="both", expand=True)
    container.pack(side="top", fill="both", expand=True)

    return figure_canvas


def show_animation_popup(parent, electrodes, rows, cols, full_readings, speed_var):
    """
    Open a Toplevel window with an animated scatter plot of the electrode
    grid, stepping through full_readings in order and highlighting the
    current (red) and potential (blue) electrodes for each reading, with
    Play/Pause, Save (to GIF), and Close buttons.

    Deliberately builds the Figure directly rather than via
    matplotlib.pyplot (plt.subplots()): pyplot keeps every figure it
    creates registered in its own global manager until explicitly closed,
    which can leave things running in the background even after a window
    is closed. Building the Figure directly, and explicitly stopping the
    FuncAnimation's timer before the window is destroyed (see
    on_popup_close() below), means closing this popup cleans up fully.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.animation import FuncAnimation

    popup = tk.Toplevel(parent)
    popup.title("Electrode Calling Order")

    numbers = [e["number"] for e in electrodes]

    screen_height_in = popup.winfo_screenheight() / max(popup.winfo_fpixels("1i"), 1)
    fig_width, fig_height, scale, pad, title_reserve_in, marker_radius_in = compute_figure_layout(
        electrodes, rows, cols, screen_height_in=screen_height_in, reserve_controls=True)

    fig = Figure(figsize=(fig_width, fig_height))
    ax, scatter, title_text = build_electrode_axes(fig, electrodes, rows, cols, fig_height, pad, title_reserve_in, marker_radius_in)

    # Controls are packed at the bottom FIRST, before the canvas - packing
    # order matters here: whatever is packed first with side="bottom"
    # claims its space permanently, and the canvas (packed after, with
    # fill="both") only gets whatever space is left above it. This is what
    # actually guarantees the buttons stay visible regardless of how tall
    # the plot ends up being, on top of the screen-aware height cap above.
    controls_frame = ttk.Frame(popup)
    controls_frame.pack(side="bottom", fill="x")

    progress_var = tk.StringVar(value="")
    ttk.Label(controls_frame, textvariable=progress_var).pack(pady=(4, 4))

    button_row = ttk.Frame(controls_frame)
    button_row.pack(pady=(0, 8))

    canvas = _embed_scrollable_figure(popup, fig, fig_width, fig_height)

    state = {"paused": False}

    def update(frame_i):
        reading = full_readings[frame_i % len(full_readings)]
        colors = frame_colors(electrodes, reading)
        scatter.set_color([colors[n] for n in numbers])
        tag = "RECIPROCAL" if reading["reciprocal"] else "NORMAL"
        title_text.set_text(
            f"Reading {reading['index']} / {len(full_readings)}  ({tag})\n"
            f"Current: A={reading['A']}  B={reading['B']}   |   "
            f"Potential: M={reading['M']}  N={reading['N']}"
        )
        progress_var.set(f"Step {frame_i + 1} of {len(full_readings)}")
        return scatter, title_text

    ani = FuncAnimation(fig, update, frames=len(full_readings),
                         interval=speed_var.get(), repeat=True, blit=False)
    popup._animation_ref = ani  # keep a reference so it isn't garbage collected

    def toggle_pause():
        if state["paused"]:
            ani.event_source.start()
            pause_btn.config(text="Pause")
        else:
            ani.event_source.stop()
            pause_btn.config(text="Play")
        state["paused"] = not state["paused"]

    def on_save():
        path = filedialog.asksaveasfilename(
            title="Save animation as",
            defaultextension=".gif",
            filetypes=[("GIF animation", "*.gif")],
            initialfile="electrode_animation.gif",
        )
        if not path:
            return
        was_paused = state["paused"]
        if not was_paused:
            toggle_pause()  # pause playback in the live view while rendering the saved copy
        try:
            messagebox.showinfo("Saving", f"Rendering {len(full_readings)} frames to:\n{path}\n\n"
                                           f"This can take a while for large protocols - the window "
                                           f"will be unresponsive until it finishes.")
            ani.save(path, writer="pillow", fps=max(1, round(1000 / speed_var.get())))
            messagebox.showinfo("Saved", f"Animation saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
        if not was_paused:
            toggle_pause()

    def on_popup_close():
        # Stop the animation's timer BEFORE destroying the window. Without
        # this, the timer can keep trying to fire against a widget that no
        # longer exists, which is what caused the terminal to hang after
        # closing this window.
        ani.event_source.stop()
        popup.destroy()

    pause_btn = ttk.Button(button_row, text="Pause", command=toggle_pause)
    pause_btn.pack(side="left", padx=8)
    ttk.Button(button_row, text="Save...", command=on_save).pack(side="left", padx=8)
    ttk.Button(button_row, text="Close", command=on_popup_close).pack(side="left", padx=8)
    popup.protocol("WM_DELETE_WINDOW", on_popup_close)

    canvas.draw()

    # Center the window on screen. Freshly created Toplevel windows can
    # otherwise be placed by the window manager wherever it likes - on some
    # systems that's low enough that a tall window's bottom (with the
    # buttons) ends up below the visible screen area even though the
    # height cap above keeps the window itself a reasonable size.
    popup.update_idletasks()
    win_w = popup.winfo_width()
    win_h = popup.winfo_height()
    screen_w = popup.winfo_screenwidth()
    screen_h = popup.winfo_screenheight()
    x = max(0, (screen_w - win_w) // 2)
    y = max(0, (screen_h - win_h) // 2)
    popup.geometry(f"+{x}+{y}")


def save_animation_to_file(electrodes, rows, cols, full_readings, interval_ms, save_path):
    """
    Render the electrode-calling-order animation straight to a GIF file at
    save_path, without ever showing a live window - useful for batch/
    headless use. Uses Pillow as the animation writer, via matplotlib's
    FuncAnimation.

    Every reading in full_readings becomes one frame at interval_ms, same
    as the live popup - there's no frame reduction, so a large protocol
    (1000+ readings) can take a while to render and produce a large file.
    """
    from matplotlib.figure import Figure
    from matplotlib.animation import FuncAnimation

    numbers = [e["number"] for e in electrodes]
    fig_width, fig_height, scale, pad, title_reserve_in, marker_radius_in = compute_figure_layout(
        electrodes, rows, cols, screen_height_in=None, reserve_controls=False)

    fig = Figure(figsize=(fig_width, fig_height))
    ax, scatter, title_text = build_electrode_axes(fig, electrodes, rows, cols, fig_height, pad, title_reserve_in, marker_radius_in)

    def update(frame_i):
        reading = full_readings[frame_i % len(full_readings)]
        colors = frame_colors(electrodes, reading)
        scatter.set_color([colors[n] for n in numbers])
        tag = "RECIPROCAL" if reading["reciprocal"] else "NORMAL"
        title_text.set_text(
            f"Reading {reading['index']} / {len(full_readings)}  ({tag})\n"
            f"Current: A={reading['A']}  B={reading['B']}   |   "
            f"Potential: M={reading['M']}  N={reading['N']}"
        )
        return scatter, title_text

    ani = FuncAnimation(fig, update, frames=len(full_readings), interval=interval_ms, repeat=False, blit=False)
    ani.save(save_path, writer="pillow", fps=max(1, round(1000 / interval_ms)))


def show_grid_popup(parent, electrodes, rows, cols):
    """
    Open a Toplevel window with a static plot of just the electrode grid -
    no animation, no reading highlighted - so the physical layout can be
    checked without building a full protocol first. Has Save (PNG/PDF/SVG)
    and Close buttons.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    popup = tk.Toplevel(parent)
    popup.title("Electrode Grid Layout")

    screen_height_in = popup.winfo_screenheight() / max(popup.winfo_fpixels("1i"), 1)
    fig_width, fig_height, scale, pad, title_reserve_in, marker_radius_in = compute_figure_layout(
        electrodes, rows, cols, screen_height_in=screen_height_in, reserve_controls=True)

    fig = Figure(figsize=(fig_width, fig_height))
    ax, scatter, title_text = build_electrode_axes(fig, electrodes, rows, cols, fig_height, pad, title_reserve_in, marker_radius_in)
    title_text.set_text(f"{rows} rows x {cols} cols  ({len(electrodes)} electrodes)")

    controls_frame = ttk.Frame(popup)
    controls_frame.pack(side="bottom", fill="x")
    button_row = ttk.Frame(controls_frame)
    button_row.pack(pady=8)

    canvas = _embed_scrollable_figure(popup, fig, fig_width, fig_height)

    def on_save():
        path = filedialog.asksaveasfilename(
            title="Save grid layout as",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="electrode_grid_layout.png",
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=150)
            messagebox.showinfo("Saved", f"Grid layout saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    ttk.Button(button_row, text="Save...", command=on_save).pack(side="left", padx=8)
    ttk.Button(button_row, text="Close", command=popup.destroy).pack(side="left", padx=8)
    popup.protocol("WM_DELETE_WINDOW", popup.destroy)

    canvas.draw()

    popup.update_idletasks()
    win_w = popup.winfo_width()
    win_h = popup.winfo_height()
    screen_w = popup.winfo_screenwidth()
    screen_h = popup.winfo_screenheight()
    x = max(0, (screen_w - win_w) // 2)
    y = max(0, (screen_h - win_h) // 2)
    popup.geometry(f"+{x}+{y}")


def save_grid_to_file(electrodes, rows, cols, save_path):
    """Render just the static electrode grid straight to an image file (PNG/PDF/SVG chosen by save_path's extension), with no window shown."""
    from matplotlib.figure import Figure

    fig_width, fig_height, scale, pad, title_reserve_in, marker_radius_in = compute_figure_layout(
        electrodes, rows, cols, screen_height_in=None, reserve_controls=False)
    fig = Figure(figsize=(fig_width, fig_height))
    ax, scatter, title_text = build_electrode_axes(fig, electrodes, rows, cols, fig_height, pad, title_reserve_in, marker_radius_in)
    title_text.set_text(f"{rows} rows x {cols} cols  ({len(electrodes)} electrodes)")
    fig.savefig(save_path, dpi=150)



# End-to-end pipeline: build the electrode grid, generate readings for the
# requested array_type/directions, add reciprocals, dedupe, optionally
# randomize the order, then write both XML files and print a summary to
# stdout. This is what run_gui()'s Build button calls, and is also usable
# directly for scripted/non-GUI generation.
#
# spread_path/protocol_path are the output XML filenames (written relative
# to the current working directory); spread_name/protocol_name are the
# separate instrument-facing <Name> values (auto-generated from rows/cols/
# array_type if left as None).
#
# Returns (electrodes, full_readings, violations) - violations is None
# unless randomize=True (see randomize_non_adjacent()).
def build(rows, cols, x_spacing=1.0, y_spacing=1.0, directions=("rows", "cols"),
          spread_path="spread.xml", protocol_path="protocol.xml",
          randomize=False, random_seed=None, array_type="wenner", n_electrodes=N_ELECTRODES,
          reciprocal_percentage=100, spread_name=None, protocol_name=None):
    if array_type not in ARRAY_GENERATORS:
        raise ValueError(f"array_type must be one of {list(ARRAY_GENERATORS)}, got {array_type!r}")

    electrodes = build_electrode_grid(rows, cols, x_spacing, y_spacing, n_electrodes=n_electrodes)
    normal_readings = build_array_readings(electrodes, rows, cols, directions, array_type=array_type)

    # Gradient's whole efficiency premise is a run of consecutive readings
    # sharing the same current pair while potential pairs sweep between
    # them - interleaving reciprocals (which always use a different
    # current pair) would break that run apart, so keep all normals
    # together and all reciprocals afterward instead. Other array types
    # don't have runs to protect, so the usual interleaved order is fine.
    interleaved = (array_type != "gradient")
    full_readings = add_reciprocals(normal_readings, interleaved=interleaved,
                                     reciprocal_percentage=reciprocal_percentage, seed=random_seed)
    full_readings = dedupe_readings(full_readings)
    # re-index after dedupe so Index stays contiguous
    for i, r in enumerate(full_readings, start=1):
        r["index"] = i

    violations = None
    if randomize:
        full_readings, violations = randomize_non_adjacent(
            full_readings, electrodes, min_gap=2, seed=random_seed)

    array_label = ARRAY_TYPE_LABELS[array_type]
    name_suffix = f"{rows}x{cols}"

    # These are the instrument-facing <Name> values that the Terrameter
    # LS/LS2 displays in its own spread/protocol lists - independent of
    # the computer filenames (spread_path / protocol_path). If the caller
    # didn't supply one, fall back to the auto-generated name.
    if not spread_name:
        spread_name = f"Generated_ERT_Spread_{name_suffix}"
    if not protocol_name:
        protocol_name = f"Generated_{array_label.replace('-', '_')}_Protocol_{name_suffix}"

    # 'L'-array (perpendicular dipole) readings aren't Wenner, Wenner-
    # Schlumberger, Dipole-Dipole, or Gradient - if they're included, the
    # protocol as a whole no longer purely represents the selected
    # array_type, so it gets the documented "General surface array" code
    # instead of borrowing array_type's code for readings that aren't
    # actually that type.
    if "l_shape" in directions:
        arraycode = GENERAL_SURFACE_ARRAYCODE
    else:
        arraycode = ARRAY_CODES[array_type]

    write_spread_xml(
        electrodes, spread_path,
        name=spread_name,
        description=f"Generated {name_suffix} electrode grid",
        station_name=f"ERT_GRID_{name_suffix}",
    )
    write_protocol_xml(
        full_readings, protocol_path,
        spread_filename=os.path.basename(spread_path),
        name=protocol_name,
        description=f"Generated {array_label} array (with reciprocals)",
        arraycode=arraycode,
    )

    n_normal = sum(1 for r in full_readings if not r["reciprocal"])
    n_recip = sum(1 for r in full_readings if r["reciprocal"])
    print(f"Array type: {array_label}")
    print(f"Electrodes: {len(electrodes)} ({rows} rows x {cols} cols)")
    print(f"Readings written: {len(full_readings)}  "
          f"({n_normal} normal + {n_recip} reciprocal)")
    if reciprocal_percentage < 100:
        actual_pct = (n_recip / n_normal * 100) if n_normal else 0
        print(f"Reciprocal coverage: {reciprocal_percentage}% requested "
              f"({actual_pct:.0f}% actual, randomly selected across all readings).")
    if array_type == "gradient" and randomize:
        print("NOTE: randomizing order breaks apart Gradient's shared-current-pair "
              "groupings - readings are still valid, but the field efficiency benefit "
              "of Gradient (fewer current injections) is lost for this protocol.")
    if randomize:
        if violations == 0:
            print("Order randomized: no adjacent-electrode-set violations.")
        else:
            print(f"Order randomized: {violations} unavoidable adjacent-set "
                  f"violation(s) (grid too small/dense to fully separate every pair).")
    print(f"Spread file:   {spread_path}   (name on instrument: {spread_name})")
    print(f"Protocol file: {protocol_path}   (name on instrument: {protocol_name})")
    return electrodes, full_readings, violations


if __name__ == "__main__":
    # Launch the configuration GUI (see run_gui() and the module docstring's
    # GUI walkthrough). Requires tkinter and matplotlib.
    run_gui()
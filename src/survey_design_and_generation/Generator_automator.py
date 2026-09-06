"""
Batch-generates ERT spread/protocol XML files over a matrix of electrode
layouts, direction-sets, and array types (using Corrected_Generator.py's
build()), then feeds all of them to measurementdensity.py's run() so the
resulting designs can be compared on density/runtime.

To use: edit the USER INPUTS section below (layouts, direction sets, array
types, and the reciprocal/seed/output settings), then run this file. It
will create GEN_DIR if needed, write a spread/protocol XML pair per
surviving (layout, array type, direction-set) combination, print a summary
of what was generated and what was skipped, and finally call
measurementdensity.run() on the results to produce plots/CSV in OUT_DIR.
"""

import os
import importlib
import measurementdensity as cmp

# If the `tetgen` executable used by pygimli's mesh generation (invoked
# later via measurementdensity.run()) isn't already on PATH, set this to
# its directory so it can be found. Leave as "" to rely on PATH as-is.
TETGEN_DIR = r''
if TETGEN_DIR and os.path.isdir(TETGEN_DIR):
    os.environ['PATH'] = TETGEN_DIR + os.pathsep + os.environ.get('PATH', '')

# ============ USER INPUTS: EDIT THIS SECTION ============

GENERATOR_MODULE = "Corrected_Generator"

# Electrode grids to sweep over: rows/cols of the grid plus its electrode
# spacing (dx along a row, dy between rows).
LAYOUTS = {
    "4x16": dict(rows=4, cols=16, dx=0.18, dy=0.32),
    "2x32": dict(rows=2, cols=32, dx=0.09, dy=0.50),
}

# Reading-direction combinations to try for each layout. Valid tokens are
# "rows", "cols", "diag".
DIRECTION_SETS = [
    ("rows",),
    ("cols",),
    ("diag",),
    ("rows", "cols"),
    ("rows", "cols", "diag"),
]

# Array type(s) to generate each layout/direction-set combination with.
ARRAY_TYPES = ["wenner_schlumberger", "wenner", "gradient", "dipole_dipole"]

ARRAY_TAGS = {"wenner": "w", "wenner_schlumberger": "ws",
              "dipole_dipole": "dd", "gradient": "grad"}

RECIPROCAL_PERCENTAGE = 25
SEED = 0

GEN_DIR = "./Output/GeneratedSweep"
OUT_DIR = "./Output/DensityRuntime"

# ============ END OF USER INPUTS ============

gen = importlib.import_module(GENERATOR_MODULE)


def live_directions(electrodes, rows, cols, dset, array_type):
    """Filter dset down to the directions that actually produce at least one
    reading for this grid/array_type (e.g. 'diag' yields nothing on a grid
    with no line of >=4 electrodes).

    electrodes, rows, cols: as returned/used by gen.build_electrode_grid.
    dset: tuple of direction tokens to test.
    array_type: array type passed through to gen.build_array_readings.

    Returns a tuple of the direction tokens from dset that produced
    readings, in their original order.
    """
    return tuple(
        d for d in dset
        if len(gen.build_array_readings(electrodes, rows, cols, (d,), array_type=array_type)) > 0
    )


# Generate one spread/protocol XML pair per (layout, array type,
# direction-set) combination in the USER INPUTS section, skipping any
# combination that yields no readings and de-duplicating combinations whose
# live directions coincide with one already generated (e.g. on a grid where
# 'diag' is empty, "rows+cols+diag" reduces to the same readings as
# "rows+cols").
#
# Writes the XML files under GEN_DIR and prints a summary of what was
# generated and what was skipped (and why).
#
# Returns a dict keyed by a label like "<layout>_<array_tag>_<dirs>", each
# value {"spread": <path>, "protocol": <path>, "array": <array_type>} - the
# shape measurementdensity.run() expects as its candidates argument.
def build_sweep():
    os.makedirs(GEN_DIR, exist_ok=True)
    candidates = {}
    seen = set()
    skipped = []

    for lname, L in LAYOUTS.items():
        electrodes = gen.build_electrode_grid(
            L["rows"], L["cols"], L["dx"], L["dy"], n_electrodes=gen.N_ELECTRODES)

        for array_type in ARRAY_TYPES:
            for dset in DIRECTION_SETS:
                live = live_directions(electrodes, L["rows"], L["cols"], dset, array_type)
                if not live:
                    skipped.append((lname, array_type, "+".join(dset), "no readings on this grid"))
                    continue

                key = (lname, array_type, live)
                if key in seen:
                    skipped.append((lname, array_type, "+".join(dset),
                                    f"reduces to {'+'.join(live)} (already covered)"))
                    continue
                seen.add(key)

                arr_tag = ARRAY_TAGS.get(array_type, array_type)
                label = f"{lname}_{arr_tag}_{'+'.join(live)}"
                spread = os.path.join(GEN_DIR, f"spread_{label}.xml")
                protocol = os.path.join(GEN_DIR, f"protocol_{label}.xml")

                gen.build(
                    rows=L["rows"], cols=L["cols"],
                    x_spacing=L["dx"], y_spacing=L["dy"],
                    directions=live,
                    spread_path=spread, protocol_path=protocol,
                    array_type=array_type,
                    reciprocal_percentage=RECIPROCAL_PERCENTAGE,
                    random_seed=SEED,
                    n_electrodes=gen.N_ELECTRODES,
                )
                candidates[label] = {"spread": spread, "protocol": protocol,
                                     "array": array_type}

    print("\n================ SWEEP SUMMARY ================")
    print(f"Generated {len(candidates)} candidate(s):")
    for lbl in candidates:
        print(f"   {lbl}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for lname, at, dset, why in skipped:
            print(f"   {lname} [{at}] {dset}: {why}")
    print("==============================================\n")
    return candidates


if __name__ == "__main__":
    candidates = build_sweep()
    if candidates:
        cmp.run(candidates, out_dir=OUT_DIR)
    else:
        print("No valid candidates were generated -- check LAYOUTS / DIRECTION_SETS.")

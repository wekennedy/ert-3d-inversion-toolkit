# ERT 3D Inversion Toolkit

Python toolkit for designing, generating, and inverting 3D electrical
resistivity tomography (ERT) surveys, built around an ABEM Terrameter
LS/LS2 and [pyGIMLi](https://www.pygimli.org/). Developed as part of a
Research Assistant internship in the Department of Earth Sciences,
Durham University.

The toolkit covers two connected stages of an ERT survey:

1. **Survey design and generation** — build electrode spread/protocol
   XML files for the Terrameter, verify them before going to the field,
   and compare candidate electrode layouts/array types against each
   other (measurement density vs. predicted field time, and synthetic
   target-recovery/resolution tests) to choose the best design.
2. **3D ERT inversion** — take real field data collected with a design
   from stage 1 and invert it in true 3D with pyGIMLi, including a
   timelapse (repeat-survey) inversion mode.

## Repository structure

```
ert-3d-inversion-toolkit/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── 2x32_example_data/          # example native 3D exports, 2x32 electrode layout
│   └── 4x16_example_data/          # example native 3D exports, 4x16 electrode layout
└── src/
    ├── survey_design_and_generation/
    │   ├── Corrected_Generator.py      # builds Spread/Protocol XML for the Terrameter LS/LS2
    │   ├── Spread_Verifier.py          # checks a Spread+Protocol pair before going to the field
    │   ├── Generator_automator.py      # sweeps layouts/array types through the generator + comparison
    │   ├── measurementdensity.py       # scores designs: measurement density vs. predicted field time
    │   └── ResolutionTester.py         # synthetic anomaly recovery / cross-line resolution test
    └── ert_3d_inversion/
        ├── True3DInversionMain.py            # true 3D ERT inversion, single survey
        ├── True3DInversionUtilities.py       # shared utilities (data loading, mesh, QC, error model)
        └── True3DTimelapseInversionMain.py   # true 3D ERT inversion, timelapse (repeat surveys)
```

Each folder's scripts import each other directly (e.g.
`Generator_automator.py` imports `Corrected_Generator` and
`measurementdensity`; both inversion mains import
`True3DInversionUtilities`), so keep the files within each folder
together — that mirrors how they were developed and run.

## Installation

pyGIMLi is most reliably installed via conda/mamba rather than pip:

```bash
conda create -n ERTenv -c gimli -c conda-forge pygimli=1.* python=3.10
conda activate ERTenv
pip install -r requirements.txt
```

You will also need **TetGen** (the tetrahedral mesh generator pyGIMLi
calls out to for 3D meshing) available as a command-line executable,
either on your system `PATH` or pointed to explicitly via the
`TETGEN_DIR` variable near the top of the scripts that need it
(`Generator_automator.py`, `ResolutionTester.py`,
`True3DInversionMain.py`, `True3DTimelapseInversionMain.py`).

`Corrected_Generator.py`'s GUI uses `tkinter`, which ships with most
Python installs; on some Linux distributions you may need
`sudo apt install python3-tk`.

## Usage

Every script keeps its adjustable settings in a `USER INPUTS` /
`CONFIG` block near the top of the file — open the script and edit
that block rather than passing command-line arguments. Anywhere you
see an `# EDIT THIS` comment is a placeholder path or filename that
must be pointed at your own data before running.

- **Generate a Spread/Protocol XML pair:**
  `python Corrected_Generator.py` — opens a GUI to choose array type
  and grid/protocol options, then writes the two XML files.
- **Check a Spread/Protocol pair before fieldwork:**
  `python Spread_Verifier.py spread.xml protocol.xml [--fix]`
- **Compare candidate designs (density vs. field time):**
  fill in `CANDIDATES` in `measurementdensity.py` and run it directly,
  or run `python Generator_automator.py` to sweep a matrix of
  layouts/array types automatically and feed them straight into the
  comparison.
- **Test how well a design resolves a target:**
  `python ResolutionTester.py` (edit the `CONFIG` block first — grid,
  anomaly, and whether to run the single-position or sweep mode).
- **Invert a single 3D survey:**
  `python True3DInversionMain.py` (edit `DATA_DIR`/`DATA_FILES`/
  `OUTPUT_DIR` in the `CONFIG` block first).
- **Invert a timelapse (repeat) survey:**
  `python True3DTimelapseInversionMain.py` (same idea, with one file
  per timestep in `DATA_FILES`).

## Acknowledgements

Developed during a Research Assistant internship in the Department of
Earth Sciences, Durham University. The timelapse inversion workflow
adapts a 2D timelapse pipeline (`ERTtimelapse_original_v1`, S.
Bithell). The 'L'-shaped array reading scheme in
`Corrected_Generator.py` follows the perpendicular-dipole method of
Tejero-Andrade et al. (2015), *Near Surface Geophysics*,
doi:10.3997/1873-0604.2015015.

## License

See [LICENSE](LICENSE).

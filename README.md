# MAGNAS SSD Evo

MAGNAS SSD Evo (MAGnetic field Non-linear Amplification for the study of the Small Scale Dynamo Evolution) is a scientific analysis pipeline for cosmological AMR simulations (MASCLET-based), focused on magnetic-field induction, energy budgets, and production/dissipation diagnostics.

## What This Project Does

- Reads AMR snapshots and halo metadata.
- Computes induction terms and induction-energy terms.
- Applies configurable boundary handling (buffer, parent fill, blend).
- Computes radial profiles for induction and production/dissipation (P/D).
- Computes temporal evolution when enabled (integrated and/or differential).
- Produces publication-style plots and saves processed arrays.

## Repository Layout

```text
MAGNAS_SSD_Evo/
├── README.md
├── config.py
├── main.py
├── requirements.txt
├── data/
└── scripts/
    ├── amr2uniform.py
    ├── buffer.py
    ├── debug.py
    ├── diff.py
    ├── induction_evo.py
    ├── memory_utils.py
    ├── plot_fields.py
    ├── readers.py
    ├── spectral.py
    ├── test.py
    ├── units.py
    └── utils.py
```

## Installation

1. Clone repository:

```bash
git clone https://github.com/marjomol/MAGNAS_SSD_Evo.git
cd MAGNAS_SSD_Evo
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Edit `config.py`:
- Select simulation(s), snapshots, input/output folders.
- Set numerical options (stencil, interpolation, filter, levels).
- Enable/disable evolution and profile outputs.

2. Run pipeline:

```bash
python main.py
```

3. Check outputs:
- Plots in `OUTPUT_PARAMS["image_folder"]`.
- Raw processed arrays in `OUTPUT_PARAMS["data_folder"]`.
- Terminal logs in `OUTPUT_PARAMS["terminal_folder"]` when `save_terminal=True`.

## Configuration Guide

Main configuration blocks are in `config.py`:

- `IND_PARAMS`: numerical and physical analysis options.
- `OUTPUT_PARAMS`: IO paths, execution mode, output formatting.
- `EVO_PLOT_PARAMS`: temporal energy-evolution plot settings.
- `PROD_DISS_PLOT_PARAMS`: temporal production/dissipation plot settings.
- `INDUCTION_PROFILE_PLOT_PARAMS`: induction radial profile styling and selection.
- `PROD_DISS_PROFILE_PLOT_PARAMS`: P/D radial profile styling and selection.
- `DEBUG_PARAMS`: optional diagnostics modules.

### Recent Profile-Plot Controls

For animation-friendly profile postprocessing, these controls are available:

- `fixed_legend`: lock legend position inside axes.
- `component_alpha`: opacity of individual component curves.
- `area_alpha` (P/D only): opacity of shaded area between production and dissipation.
- `plot_density`, `plot_magnetic_energy`: optional reference curves in profile plots.

### Volumetric Export Controls (ParaView / Postprocessing)

In `IND_PARAMS["return"]` you can now configure per-snapshot volumetric exports:

- `enabled`: master switch for volumetric export.
- `format`: `npy`, `npz`, `vtk_ascii`, or `vtk_binary`.
- `grid`: `amr` (patch lists) or `uniform` (homogeneous grid).
- `fields`: selects which quantities are exported (`density`, `velocity`, magnetic fields, etc.).

Notes:

- `vtk` is directly compatible with ParaView and is exported as a homogeneous grid.
- For `vtk`, set `grid="uniform"`.
- Exported files are written under `OUTPUT_PARAMS["data_folder"] / volumetric_exports / <sim> / it_<snap> / level_<L>`.

## Processing Notes

- Profiles and temporal evolution are independently gated by configuration.
- `it_indx` selectors for profiles support both positional indices and explicit iteration IDs.
- P/D radial plotting supports absolute and fractional views.
- Fractional P/D profiles include net efficiency representation.

## Core Modules

- `scripts/induction_evo.py`: core physics pipeline and per-snapshot processing.
- `scripts/diff.py`: derivative operators and AMR differential calculus.
- `scripts/buffer.py` (if present in your branch): ghost-cell and boundary strategies.
- `scripts/plot_fields.py`: all plotting routines.
- `scripts/readers.py`: MASCLET/ASOHF data ingestion.
- `scripts/utils.py`: utility helpers, logging, and support functions.

## License

MIT License. See `LICENSE.md`.

## Acknowledgments

This project uses tools developed in the [MASCLET Framework](https://github.com/dvallesp/masclet_framework.git) to analyze and operate with the simulation data.

## Contact

For any questions or issues, please contact Marco José Molina Pradillo at [marco.molina@uv.es].

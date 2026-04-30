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
   - Set `OUTPUT_PARAMS["paths"]` to your preferred output root and organization labels.
   - Define `OUTPUT_PARAMS["simulations"]` with your simulation(s), input paths, and snapshot iterations.
   - Set `enabled=True` for simulations to include, `False` to skip.
   - Adjust `IND_PARAMS` for numerical options (stencil, interpolation, filter, levels).
   - Enable/disable `EVO_PLOT_PARAMS`, `PROD_DISS_PLOT_PARAMS`, and other output modules as needed.

2. Run pipeline:

```bash
python main.py
```

3. Check outputs:
   - Plots in `OUTPUT_PARAMS["paths"]["outdir"] / OUTPUT_PARAMS["paths"]["plotdir"]`.
   - Raw processed arrays in `OUTPUT_PARAMS["paths"]["outdir"] / OUTPUT_PARAMS["paths"]["rawdir"]`.
   - Terminal logs in `OUTPUT_PARAMS["paths"]["outdir"] / OUTPUT_PARAMS["paths"]["terminaldir"]` when `save_terminal=True`.

## Configuration Guide

Main configuration blocks are in `config.py`:

- `IND_PARAMS`: numerical and physical analysis options.
- `OUTPUT_PARAMS`: IO paths, execution mode, output formatting, and per-simulation settings.
- `EVO_PLOT_PARAMS`: temporal energy-evolution plot settings.
- `PROD_DISS_PLOT_PARAMS`: temporal production/dissipation plot settings.
- `INDUCTION_PROFILE_PLOT_PARAMS`: induction radial profile styling and selection.
- `PROD_DISS_PROFILE_PLOT_PARAMS`: P/D radial profile styling and selection.
- `DEBUG_PARAMS`: optional diagnostics modules.

### OUTPUT_PARAMS Structure

The `OUTPUT_PARAMS` dictionary now uses a nested structure to support multi-simulation analysis with independent configuration per simulation.

#### Overview

```python
OUTPUT_PARAMS = {
    "paths": {
        "outdir": "/absolute/path/to/outputs/",
        "plotdir": "plots/",
        "rawdir": "raw_data_out/",
        "terminaldir": "terminal_output/",
        "ID1": "dynamo/",
        "ID2": "ParaView/",
        "run": "MAGNAS_SSD_Evo_PV_1"
    },
    "simulations": {
        "cluster_B_low_res_paper_2020": {
            "enabled": True,
            "it": [1200, 1300, 1400],
            "paths": {
                "dir_DM": "/path/to/DM/snapshots/",
                "dir_gas": "/path/to/gas/snapshots/",
                "dir_grids": "/path/to/AMR/grids/",
                "dir_halos": "/path/to/halo/data/",
                "dir_vortex": "/path/to/vortex/data/"
            }
        },
        "another_simulation": {
            "enabled": False,
            "it": [900, 950],
            "paths": { ... }
        }
    }
}
```

#### Key Components

**`paths` (dict)**:
- Shared output roots used across all enabled simulations.
- `outdir`: base output directory (absolute path recommended).
- `plotdir`, `rawdir`, `terminaldir`: relative subdirectories for plots, raw arrays, and logs.
- `ID1`, `ID2`: additional organization labels for specific output types.
- `run`: run identifier used in plot titles and filenames.

**`simulations` (dict)**:
- Per-simulation configuration keyed by simulation name.
- Each simulation has:
  - `enabled` (bool): set to `True` to include in analysis, `False` to skip.
  - `it` (list of ints): snapshot iterations to process for this simulation.
  - `paths` (dict): simulation-specific data locations:
    - `dir_DM`, `dir_gas`, `dir_grids`: input AMR snapshot directories.
    - `dir_halos`, `dir_vortex`: halo and vortex analysis directories (if available).

#### Example: Multi-Simulation Run

To analyze 2 simulations but disable one:

```python
OUTPUT_PARAMS = {
    "paths": {
        "outdir": "/mnt/data/outputs/",
        "plotdir": "plots/",
        "rawdir": "processed_arrays/",
        "run": "Comparison_Run_Apr2026"
    },
    "simulations": {
        "sim_low_res": {
            "enabled": True,
            "it": [1000, 1100, 1200],
            "paths": {
                "dir_DM": "/data/sim_low_res/dm_snapshots/",
                "dir_gas": "/data/sim_low_res/gas_snapshots/",
                "dir_grids": "/data/sim_low_res/grids/",
                "dir_halos": "/data/sim_low_res/halos/",
                "dir_vortex": "/data/sim_low_res/vortex/"
            }
        },
        "sim_high_res": {
            "enabled": False,      # Skip this simulation for now
            "it": [950, 1050],
            "paths": {
                "dir_DM": "/data/sim_high_res/dm_snapshots/",
                "dir_gas": "/data/sim_high_res/gas_snapshots/",
                "dir_grids": "/data/sim_high_res/grids/",
                "dir_halos": "/data/sim_high_res/halos/",
                "dir_vortex": "/data/sim_high_res/vortex/"
            }
        }
    }
}
```

#### Index Mapping in Parallel/Serial Modes

The pipeline automatically:
1. **Filters active simulations**: only processes simulations with `enabled=True`.
2. **Preserves index alignment**: maintains mapping between `IND_PARAMS` (per-simulation parameter arrays) and the active simulation subset.
3. **Generates active lists**:
   - `active_sim_indices`: original config indices of enabled simulations.
   - `active_sim_names`: names of enabled simulations.
   - `active_sim_it`: iteration lists for enabled simulations.
   - `active_sim_paths`: path dicts for enabled simulations.

This allows per-simulation numerical parameters (e.g., different induction stencil settings) to be correctly aligned with the active simulations during execution.

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
- Exported files are written under `OUTPUT_PARAMS["paths"]["outdir"] / OUTPUT_PARAMS["paths"]["rawdir"] / volumetric_exports / <sim> / it_<snap> / level_<L>`.

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

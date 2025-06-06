# PRIMAL Seed Generator

## Overview

The PRIMAL Seed Gen (PRImordial MAgnetic FieLd Seed Generator) repository contains a set of Python scripts and functions to generate cosmological magnetic field seeds from scratch. The generated seeds can display different spectral indexes and randomness to be used in cosmological simulations. The repository also includes tools to plot and analyze the generated magnetic field seeds.

## Repository Structure

```
PRIMAL_Seed_Gen/
│
├── README.md
├── config.py
├── main.py
├── requirements.txt
│
├── data/
│   └── ... (generated seed files and live animations)
│
├── scripts/
│   ├── __init__.py
│   ├── diff.py
│   ├── plot_fields.py
│   ├── seed_generator.py
│   ├── spectral.py
│   ├── units.py
│   └── utils.py
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/marjomol/PRIMAL_Seed_Gen.git
    cd PRIMAL_Seed_Gen
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The configuration parameters for the simulation are stored in `config.py`. By deafoult this are chosen for MASCLET simulation. You can modify these parameters to suit your needs.

## Usage

1. Run the main script to generate the magnetic field seeds and plot the results:
    ```bash
    python main.py
    ```

2. The seed will be saved with the desired format in the data directory. 

3. The generated plots and animations will be saved in the specified `image_folder` directory.

4. For debugging purposes, run the script in debugging mode with a small resolution (recommended 32 cell size arrays) to save intermediary arrays.

## Scripts

- **diff.py**: Contains functions to compute the general differential calculus.
- **plot_fields.py**: Contains functions to plot the magnetic field seeds and create animations.
- **seed_generation.py**: Contains functions to generate the magnetic field seeds.
- **spectral.py**: Contains functions to perform spectral analysis on the generated magnetic field seeds.
- **units.py**: Provides unit conversion utilities for various physical quantities used in the simulations.
- **utils.py**: Includes general utility functions that support the main operations of the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/marjomol/PRIMAL_Seed_Gen/blob/master/LICENSE.md) file for details.

## Acknowledgments

This project uses tools developed in the [MASCLET Framework](https://github.com/dvallesp/masclet_framework.git) to analyze and operate with the simulation data.

## Contact

For any questions or issues, please contact Marco José Molina Pradillo at [marco.molina@uv.es].

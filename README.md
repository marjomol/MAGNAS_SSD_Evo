# MAGNAS Small Scale Dynamo Evolution

## Overview

The MAGNAS SSD Evo (MAGnetic field Non-linear Amplification in the large scale Structure for the study of the Small Sacale Dynamo Evolution) repository contains a set of Python scripts and functions to analyse simulated cosmological magnetic field induction. The repository also includes tools to plot and analyze this magnetic field.

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
│   └── ... (generated magnetic field files and live animations)
│
├── scripts/
│   ├── __init__.py
│   ├── diff.py
│   ├── plot_fields.py
│   ├── induction_evo.py
│   ├── spectral.py
│   ├── units.py
│   └── utils.py
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/marjomol/MAGNAS_SSD_Evo.git
    cd MAGNAS_SSD_Evo
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

1. Choose and prepare your AMR simulation data for different epochs.


2. Run the main script to analyze the magnetic field induction evolution and plot the results:
    ```bash
    python main.py
    ```

3. The analics and results will be saved with the desired format in the data directory.

4. The generated plots and animations will be saved in the specified `image_folder` directory.

5. For debugging purposes, run the script in debugging mode with a small resolution (recommended 32 cell size arrays) to save intermediary arrays.

## Scripts

- **diff.py**: Contains functions to compute the general differential calculus.
- **plot_fields.py**: Contains functions to plot the magnetic field seeds and create animations.
- **induction_evo.py**: Contains functions to calculate the magnetic field induction components and their evolution.
- **spectral.py**: Contains functions to perform spectral analysis on the generated magnetic field seeds.
- **units.py**: Provides unit conversion utilities for various physical quantities used in the simulations.
- **utils.py**: Includes general utility functions that support the main operations of the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/marjomol/MAGNAS_SSD_Evo/blob/master/LICENSE.md) file for details.

## Acknowledgments

This project uses tools developed in the [MASCLET Framework](https://github.com/dvallesp/masclet_framework.git) to analyze and operate with the simulation data.

## Contact

For any questions or issues, please contact Marco José Molina Pradillo at [marco.molina@uv.es].

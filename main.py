import numpy as np
import time
import gc
import scripts.utils as utils
from config import SEED_PARAMS as seed_params
from config import OUTPUT_PARAMS as out_params
from scripts.seed_generator import power_spectrum_amplitude, generate_fourier_space, generate_random_seed_amplitudes, seed_transverse_projector, generate_magnetic_field_seed, generate_seed_properties
from scripts.plot_fields import plot_seed_spectrum, scan_animation_3D, zoom_animation_3D
from scripts.units import *
npatch = np.array([0]) # We only want the zero patch for the seed
# np.random.seed(out_params["random_seed"]) # Set the random seed for reproducibility


# ============================
# Only edit the section below
# ============================

def bones():
    temp_files_list = []

    k_grid = np.memmap('k_grid.dat', dtype='float32', mode='w+', shape=(seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"], 3))
    k_magnitude = np.memmap('k_magnitude.dat', dtype='float32', mode='w+', shape=(seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]))
    k_grid, k_magnitude = generate_fourier_space(
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        seed_params["epsilon"], verbose=out_params["verbose"], debug=out_params["debug"]
        )
    PB = np.memmap('PB.dat', dtype='float64', mode='w+', shape=(seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]))
    PB = power_spectrum_amplitude(
        k_magnitude, seed_params["alpha"], seed_params["lambda_comoving"], seed_params["B0"], h, 
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]], 
        gauss_rad_factor=seed_params["smothing"], filtering=seed_params["filtering"], verbose=out_params["verbose"]
        )
    B_random_kx = np.memmap('B_random_kx.dat', dtype='float64', mode='w+', shape=(seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]))
    B_random_kx = generate_random_seed_amplitudes(
        'x', k_grid, k_magnitude, PB,
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    B_random_ky = np.memmap('B_random_ky.dat', dtype='float64', mode='w+', shape=(seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]))
    B_random_ky = generate_random_seed_amplitudes(
        'y', k_grid, k_magnitude, PB,
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    B_random_kz = np.memmap('B_random_kz.dat', dtype='float64', mode='w+', shape=(seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]))
    B_random_kz = generate_random_seed_amplitudes(
        'z', k_grid, k_magnitude, PB,
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    
    del PB, k_magnitude
    gc.collect()
    utils.delete_temp_files(['k_magnitude.dat', 'PB.dat'])
    
    k_dot_B = np.memmap('k_dot_B.dat', dtype='float64', mode='w+', shape=(seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]))
    k_dot_B = seed_transverse_projector(
        k_grid, [B_random_kx, B_random_ky, B_random_kz],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    
    temp_files_list.extend(['B_random_kx.dat', 'B_random_ky.dat', 'B_random_kz.dat', 'k_dot_B.dat'])
    
    return B_random_kx, B_random_ky, B_random_kz, k_dot_B, k_grid, temp_files_list

def mainx(B_random_kx, k_dot_B, k_grid):
    
    generate_magnetic_field_seed(
        'x', k_grid, B_random_kx, k_dot_B, seed_params["alpha"], seed_params["size"],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        gauss_rad_factor=seed_params["smothing"],
        verbose=out_params["verbose"], debug=out_params["debug"], format=out_params["format"], run=out_params["run"]
        )

def mainy(B_random_ky, k_dot_B, k_grid):
    
    generate_magnetic_field_seed(
        'y', k_grid, B_random_ky, k_dot_B, seed_params["alpha"], seed_params["size"],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        gauss_rad_factor=seed_params["smothing"],
        verbose=out_params["verbose"], debug=out_params["debug"], format=out_params["format"], run=out_params["run"]
        )

def mainz(B_random_kz, k_dot_B, k_grid):
    
    generate_magnetic_field_seed(
        'z', k_grid, B_random_kz, k_dot_B, seed_params["alpha"], seed_params["size"],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        gauss_rad_factor=seed_params["smothing"],
        verbose=out_params["verbose"], debug=out_params["debug"], format=out_params["format"], run=out_params["run"]
        )

def plot(Bx, By, Bz):               
    out_params["run"] = f'{out_params["run"]}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    
    plot_seed_spectrum(seed_params["alpha"], Bx, By, Bz, seed_params["dx"], mode=1, 
                        epsilon=seed_params["epsilon"], Save=out_params["save"], DPI=out_params["dpi"], 
                        run=out_params["run"], folder=out_params["image_folder"])
    
    BM = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
    
    scan_animation_3D(BM, seed_params["dx"], study_box=1, arrow_scale=10, units='Mpc', 
                    title=f'Gaussian Filtered Magnetic Field Scan, $\\alpha$ = {seed_params["alpha"]}', 
                    Save=out_params["save"], DPI=out_params["dpi"], run=out_params["run"], folder=out_params["image_folder"])
    zoom_animation_3D(BM, seed_params["dx"], arrow_scale=1, units='Mpc', 
                    title=f'Gaussian Filtered Magnetic Field Zoom, $\\alpha$ = {seed_params["alpha"]}', 
                    Save=out_params["save"], DPI=out_params["dpi"], run=out_params["run"], folder=out_params["image_folder"])

def load_and_plot():
    xelements = utils.get_fortran_file_size('x', out_params["run"], dtype=np.float64)
    yelements = utils.get_fortran_file_size('y', out_params["run"], dtype=np.float64)
    zelements = utils.get_fortran_file_size('z', out_params["run"], dtype=np.float64)
    print(f"X-Elements: {xelements}")
    print(f"Y-Elements: {yelements}")
    print(f"Z-Elements: {zelements}")
    print(f"Expected Elements: {seed_params['nmax']*seed_params['nmay']*seed_params['nmaz']}")
    # Load Bx, By, Bz from binary Fortran format files
    Bx = [utils.load_magnetic_field('x', out_params["run"], format='fortran')]
    By = [utils.load_magnetic_field('y', out_params["run"], format='fortran')]
    Bz = [utils.load_magnetic_field('z', out_params["run"], format='fortran')]
    
    # Call the plot function
    plot(Bx, By, Bz)


if __name__ == "__main__":
    start_time = time.time()
    B_random_kx, B_random_ky, B_random_kz, k_dot_B, k_grid, temp_files_list = bones()
    mainx(B_random_kx, k_dot_B, k_grid)
    mainy(B_random_ky, k_dot_B, k_grid)
    mainz(B_random_kz, k_dot_B, k_grid)
    load_and_plot()
    del B_random_kx, B_random_ky, B_random_kz, k_dot_B, k_grid
    gc.collect()
    utils.delete_temp_files(temp_files_list)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.5f}s")
    print(f"Box Cell Side: {seed_params['nmax'], seed_params['nmay'], seed_params['nmaz']}")
    print(f"Cells: {seed_params['nmax']*seed_params['nmay']*seed_params['nmaz']}")

# if __name__ == "__main__":
#     load_and_plot()
    
# ============================
# Only edit the section above
# ============================

if out_params["save"] == True:
    print(f"Plots saved in {out_params['image_folder']}")
    print(f"Results saved in /data")
else:
    print("Results not saved")
import numpy as np
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

Run = f'PRIMAL_Seed_Gen_test'

def bones():
    k_grid, k_magnitude = generate_fourier_space(
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        seed_params["epsilon"], verbose=out_params["verbose"], debug=out_params["debug"]
        )
    PB = power_spectrum_amplitude(
        k_magnitude, seed_params["alpha"], seed_params["lambda_comoving"], seed_params["B0"], h, 
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]], 
        gauss_rad_factor=seed_params["smothing"], filtering=seed_params["filtering"], verbose=out_params["verbose"]
        )
    B_random_kx = generate_random_seed_amplitudes(
        'x', k_grid, k_magnitude, PB,
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    B_random_ky = generate_random_seed_amplitudes(
        'y', k_grid, k_magnitude, PB,
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    B_random_kz = generate_random_seed_amplitudes(
        'z', k_grid, k_magnitude, PB,
        seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    k_dot_B = seed_transverse_projector(
        k_grid, [B_random_kx, B_random_ky, B_random_kz],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        verbose=out_params["verbose"], debug=out_params["debug"]
        )
    return B_random_kx, B_random_ky, B_random_kz, k_dot_B, k_grid

def main():
    B_random_kx, B_random_ky, B_random_kz, k_dot_B, k_grid = bones()
    
    Bx = generate_magnetic_field_seed(
        'x', k_grid, B_random_kx, k_dot_B, seed_params["alpha"], seed_params["size"],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        gauss_rad_factor=seed_params["smothing"],
        verbose=out_params["verbose"], debug=out_params["debug"], format=out_params["format"], run=Run
        )
    
    By = generate_magnetic_field_seed(
        'y', k_grid, B_random_ky, k_dot_B, seed_params["alpha"], seed_params["size"],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        gauss_rad_factor=seed_params["smothing"],
        verbose=out_params["verbose"], debug=out_params["debug"], format=out_params["format"], run=Run
        )
    
    Bz = generate_magnetic_field_seed(
        'z', k_grid, B_random_kz, k_dot_B, seed_params["alpha"], seed_params["size"],
        [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]],
        gauss_rad_factor=seed_params["smothing"],
        verbose=out_params["verbose"], debug=out_params["debug"], format=out_params["format"], run=Run
        )
    
    return Bx, By, Bz

def plot(Bx, By, Bz, Run):               
    run = f'{Run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    
    plot_seed_spectrum(seed_params["alpha"], Bx, By, Bz, seed_params["dx"], mode=1, 
                        epsilon=seed_params["epsilon"], Save=out_params["save"], DPI=out_params["dpi"], 
                        run=run, folder=out_params["image_folder"])
    
    BM = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
    
    scan_animation_3D(BM, seed_params["dx"], study_box=1, arrow_scale=10, units='Mpc', 
                    title=f'Gaussian Filtered Magnetic Field Scan, $\\alpha$ = {seed_params["alpha"]}', 
                    Save=out_params["save"], DPI=out_params["dpi"], run=run, folder=out_params["image_folder"])
    zoom_animation_3D(BM, seed_params["dx"], arrow_scale=1, units='Mpc', 
                    title=f'Gaussian Filtered Magnetic Field, $\\alpha$ = {seed_params["alpha"]}', 
                    Save=out_params["save"], DPI=out_params["dpi"], run=run, folder=out_params["image_folder"])

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     plot(*main(), Run)
    
# ============================
# Only edit the section above
# ============================

if out_params["save"] == True:
    print(f"Plots saved in {out_params['image_folder']}")
    print(f"Results saved in /data")
else:
    print("Results not saved")
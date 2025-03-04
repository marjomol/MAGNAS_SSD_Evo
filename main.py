import numpy as np
from config import SEED_PARAMS as seed_params
from config import OUTPUT_PARAMS as out_params
from scripts.seed_generator import generate_magnetic_field_seed
from scripts.plot_fields import plot_seed_spectrum, scan_animation_3D, zoom_animation_3D
from scripts.units import *

np.random.seed(out_params["random_seed"]) # Set the random seed for reproducibility
npatch = np.array([0]) # We only want the zero patch for the seed

# ============================
# Only edit the section below
# ============================

Run = f'PRIMAL_Seed_Gen_test'

def main():
    for index in range(len(seed_params["alpha"])):
        Bx, By, Bz = generate_magnetic_field_seed(
            seed_params["alpha"][index], seed_params["lambda_comoving"], seed_params["B0"][index], h, 
            seed_params["size"], [seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]], 
            gauss_rad_factor=seed_params["smothing"], epsilon=seed_params["epsilon"], 
            filtering=True, verbose=True, debug=False
        )
    return Bx, By, Bz

def plot(Bx, By, Bz, Run):
    for index in range(len(seed_params["alpha"])):
        run = f'{Run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"][index]}'
        plot_seed_spectrum(seed_params["alpha"][index], Bx, By, Bz, seed_params["dx"], mode=1, 
                            epsilon=seed_params["epsilon"], Save=out_params["save"], DPI=out_params["dpi"], 
                            run=run, folder=out_params["image_folder"])
        
        BM = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
        
        scan_animation_3D(BM, seed_params["dx"], study_box=1, arrow_scale=10, units='Mpc', 
                        title=f'Gaussian Filtered Magnetic Field Scan, $\\alpha$ = {seed_params["alpha"][index]}', 
                        Save=out_params["save"], DPI=out_params["dpi"], run=run, folder=out_params["image_folder"])
        zoom_animation_3D(BM, seed_params["dx"], arrow_scale=1, units='Mpc', 
                        title=f'Gaussian Filtered Magnetic Field, $\\alpha$ = {seed_params["alpha"][index]}', 
                        Save=out_params["save"], DPI=out_params["dpi"], run=run, folder=out_params["image_folder"])

if __name__ == "__main__":
    main

# if __name__ == "__main__":
#     plot(*main(), Run)
    
# ============================
# Only edit the section above
# ============================

if out_params["save"] == True:
    print(f"Results saved in {out_params['image_folder']}")
else:
    print("Results not saved")
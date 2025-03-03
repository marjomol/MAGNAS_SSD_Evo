import numpy as np
from config import SEED_PARAMS as seed
from config import OUTPUT_PARAMS as out
from scripts.seed_generator import generate_magnetic_field_seed
from scripts.plot_fields import plot_seed_spectrum, scan_animation_3D, zoom_animation_3D

np.random.seed(seed["random_seed"]) # Set the random seed for reproducibility
npatch = np.array([0]) # We only want the zero patch for the seed

# ============================
# Only edit the section below
# ============================

Run = f'try_3'

def main():
    for index in range(len(seed["alpha"])):
        Bx, By, Bz = generate_magnetic_field_seed(
            seed["alpha"][index], seed["lambda_comoving"], seed["B0"][index], seed["h"], 
            seed["size"], [seed["nmax"], seed["nmay"], seed["nmaz"]], 
            gauss_rad_factor=seed["smothing"], epsilon=seed["epsilon"], 
            filtering=True, verbose=True, debug=False
        )
    return Bx, By, Bz

def plot(Bx, By, Bz, Run):
    for index in range(len(seed["alpha"])):
        run = f'{Run}_{seed["nmax"]}_{seed["alpha"][index]}'
        plot_seed_spectrum(seed["alpha"][index], Bx, By, Bz, seed["dx"], mode=1, 
                            epsilon=seed["epsilon"], Save=out["save"], DPI=out["dpi"], 
                            run=run, folder=out["image_folder"])
        
        BM = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
        
        scan_animation_3D(BM, seed["dx"], study_box=1, arrow_scale=10, units='Mpc', 
                        title=f'Gaussian Filtered Magnetic Field Scan, $\\alpha$ = {seed["alpha"][index]}', 
                        Save=out["save"], DPI=out["dpi"], run=run, folder=out["image_folder"])
        zoom_animation_3D(BM, seed["dx"], arrow_scale=1, units='Mpc', 
                        title=f'Gaussian Filtered Magnetic Field, $\\alpha$ = {seed["alpha"][index]}', 
                        Save=out["save"], DPI=out["dpi"], run=run, folder=out["image_folder"])
        
if __name__ == "__main__":
    plot(*main(), Run)
    
# ============================
# Only edit the section above
# ============================

if out["save"]:
    print(f"Results saved in {out['image_folder']}")
else:
    print("Results not saved")
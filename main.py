import numpy as np
import time
import gc
import scripts.utils as utils
from config import SEED_PARAMS as seed_params
from config import OUTPUT_PARAMS as out_params
from scripts.seed_generator import generate_seed, generate_seed_properties
from scripts.plot_fields import plot_seed_spectrum, scan_animation_3D, zoom_animation_3D
from scripts.units import *
npatch = np.array([0]) # We only want the zero patch for the seed
np.random.seed(out_params["random_seed"]) # Set the random seed for reproducibility

start_time = time.time()

# ============================
# Only edit the section below
# ============================

def gen():
    
    generate_seed(out_params["chunk_factor"], seed_params, out_params)

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
    
    xelements = utils.get_fortran_file_size('x', out_params["run"], dtype=out_params["bitformat"])
    yelements = utils.get_fortran_file_size('y', out_params["run"], dtype=out_params["bitformat"])
    zelements = utils.get_fortran_file_size('z', out_params["run"], dtype=out_params["bitformat"])
    
    print(f'***********************************************************')
    print(f"Bx Elements: {xelements-2}") # Fortran files have 2 extra elements (header and footer)
    print(f"By Elements: {yelements-2}")
    print(f"Bz Elements: {zelements-2}")
    print(f"Expected Elements: {seed_params['nmax']*seed_params['nmay']*seed_params['nmaz']}")
    print(f'***********************************************************')
    
    # Load Bx, By, Bz from binary Fortran format files
    Bx = [utils.load_magnetic_field('x', out_params["run"], format='fortran')]
    By = [utils.load_magnetic_field('y', out_params["run"], format='fortran')]
    Bz = [utils.load_magnetic_field('z', out_params["run"], format='fortran')]
    
    # Call the plot function
    plot(Bx, By, Bz)


if __name__ == "__main__":
    
    gen()
    # load_and_plot()
    
# ============================
# Only edit the section above
# ============================

if out_params["save"] == True:
    print(f'***********************************************************')
    print(f"Plots saved in {out_params['image_folder']}")
    print(f"Results saved in {out_params['data_folder']}")
else:
    print("Results not saved")
    
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f'***********************************************************')
print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.5f}s")
print(f"Box Cell Side: {seed_params['nmax'], seed_params['nmay'], seed_params['nmaz']}")
print(f"Nº Cells: {seed_params['nmax']*seed_params['nmay']*seed_params['nmaz']}")
print(f'***********************************************************')
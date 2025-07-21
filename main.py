import numpy as np
import time
import scripts.utils as utils
from config import IND_PARAMS as ind_params
from config import OUTPUT_PARAMS as out_params
from scripts.induction_evo import find_most_massive_halo, create_region, process_iteration
from scripts.plot_fields import plot_seed_spectrum, scan_animation_3D, zoom_animation_3D
from scripts.units import *
np.random.seed(out_params["random_seed"]) # Set the random seed for reproducibility

start_time = time.time()

# ============================
# Only edit the section below
# ============================

if __name__ == "__main__":
    
    if out_params["parallel"]:
        print(f'**************************************************************')
        print(f"Running in parallel mode with {out_params['ncores']} cores")
        print(f'**************************************************************')
        # Find the most massive halo in each snapshot
        Coords, Rad = find_most_massive_halo(out_params["sims"], out_params["it"], 
                                            ind_params["a0"], 
                                            out_params["dir_halos"], 
                                            out_params["dir_grids"], 
                                            out_params["data_folder"], 
                                            vir_kind=ind_params["vir_kind"], 
                                            rad_kind=ind_params["rad_kind"], 
                                            verbose=out_params["verbose"])
        # Create the regions of interest in the grid
        Region = create_region(out_params["sims"], out_params["it"], Coords, Rad, 
                            F=out_params["F"], reg=ind_params["region"], 
                            verbose=out_params["verbose"])
        
        # Process each iteration in parallel

    else:
        print(f'*************************')
        print("Running in serial mode")
        print(f'*************************')
        # Find the most massive halo in each snapshot
        Coords, Rad = find_most_massive_halo(out_params["sims"], out_params["it"], 
                                            ind_params["a0"], 
                                            out_params["dir_halos"], 
                                            out_params["dir_grids"], 
                                            out_params["data_folder"], 
                                            vir_kind=ind_params["vir_kind"], 
                                            rad_kind=ind_params["rad_kind"], 
                                            verbose=out_params["verbose"])
        # Create the regions of interest in the grid
        Region_Coord = create_region(out_params["sims"], out_params["it"], Coords, Rad, 
                            F=ind_params["F"], reg=ind_params["region"], 
                            verbose=out_params["verbose"])
        
        # Process each iteration in serial
        for it, sims, i in zip(out_params["it"], out_params["sims"], range(len(out_params["sims"]))):
            # Process the iteration
            vectorial, induction, magnitudes, induction_energy, induction_energy_integral, induction_energy_profiles, induction_uniform = process_iteration(
                ind_params["components"], 
                out_params["dir_grids"], 
                out_params["dir_gas"],
                out_params["dir_params"][i], 
                sims, it, Coords[i], Region_Coord[i],
                Rad[i], ind_params["rmin"][i], 
                ind_params["level"], ind_params["up_to_level"],
                ind_params["rho_b"], ind_params["nmax"][i],
                ind_params["size"][i], ind_params["H"], ind_params["a"],
                units=ind_params["units"][i],
                nbins=ind_params["nbins"][i],
                logbins=ind_params["logbins"],
                stencil=out_params["stencil"],
                A2U=ind_params["A2U"],
                mag=ind_params["mag"],
                energy_evolution=ind_params["energy_evolution"],
                profiles=ind_params["profiles"],
                projection=ind_params["projection"],
                verbose=out_params["verbose"])
            
            field = np.abs(np.sqrt(induction_uniform['uniform_MIE_compres_x']**2 + 
                            induction_uniform['uniform_MIE_compres_y']**2 + 
                            induction_uniform['uniform_MIE_compres_z']**2))
            
            print(field.shape)
            
            scan_animation_3D(field, ind_params['dx'][i], study_box=1, depth=ind_params['up_to_level'], arrow_scale=10, units='Mpc', 
                    title=f'Magnetic Field_Induction_Compression Scan', verbose=out_params["verbose"], 
                    Save=True, DPI=out_params["dpi"], run=out_params["run"], folder=out_params["image_folder"])
            
        
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
for i in range(len(out_params["sims"])):
    print(f"Simulation: {out_params['sims'][i]}")
    print(f"Box Cell Size: {ind_params['nmax'][i], ind_params['nmay'][i], ind_params['nmaz'][i]}")
    print(f"Nº Cells: {ind_params['nmax'][i]*ind_params['nmay'][i]*ind_params['nmaz'][i]}")
print(f'***********************************************************')
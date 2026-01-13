import numpy as np
import time
import scripts.utils as utils
from config import IND_PARAMS as ind_params
from config import OUTPUT_PARAMS as out_params
from config import EVO_PLOT_PARAMS as evo_plot_params
from config import PROFILE_PLOT_PARAMS as prof_plot_params
from config import DEBUG_PARAMS as debug_params
from scripts.induction_evo import find_most_massive_halo, create_region, process_iteration, induction_energy_integral_evolution
from scripts.plot_fields import plot_integral_evolution, plot_radial_profiles, distribution_check, scan_animation_3D, zoom_animation_3D
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
        for L in range(len(ind_params["level"])):
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
            Region_Coord, Region_Size = create_region(out_params["sims"], out_params["it"], Coords, Rad, 
                                F=ind_params["F"], reg=ind_params["region"], 
                                verbose=out_params["verbose"])
            
            # Initialize result dictionaries before the loop
            all_data = {}
            all_vectorial = {}
            all_induction = {}
            all_magnitudes = {}
            all_induction_energy = {}
            all_induction_energy_integral = {}
            all_induction_test_energy_integral = {}
            all_induction_energy_profiles = {}
            all_induction_uniform = {}
            all_debug_fields = {}

            # Flag to track first iteration for dictionary initialization
            first_iteration = True
            
            # Process each iteration in serial
            for sims, i in zip(out_params["sims"], range(len(out_params["sims"]))):
                for it, j in zip(out_params["it"], range(len(out_params["it"]))):
                    
                    data, _, _, _, induction_energy, induction_energy_integral, induction_test_energy_integral, induction_energy_profiles, _, debug_fields = process_iteration(
                        ind_params["components"], 
                        out_params["dir_grids"], 
                        out_params["dir_gas"],
                        out_params["dir_params"][i], 
                        sims, it, Coords[i+j], Region_Coord[i+j],
                        Rad[i+j], ind_params["rmin"][i], 
                        ind_params["level"][L], ind_params["level"][L],
                        ind_params["nmax"][i], ind_params["size"][i],
                        ind_params["H0"], ind_params["a0"],
                        test=ind_params["test_params"],
                        units=ind_params["units"],
                        nbins=ind_params["nbins"][i],
                        logbins=ind_params["logbins"],
                        stencil=ind_params["stencil"],
                        buffer=ind_params["buffer"],
                        interpol=ind_params["interpol"],
                        nghost=ind_params["nghost"],
                        bitformat=out_params["bitformat"],
                        mag=ind_params["mag"],
                        energy_evolution=ind_params["energy_evolution"],
                        profiles=ind_params["profiles"] if it in [out_params["it"][k] for k in prof_plot_params["it_indx"]] else False,
                        # profiles=ind_params["profiles"],
                        projection=ind_params["projection"],
                        debug=out_params["debug"] if it in [out_params["it"][k] for k in debug_params["it_indx"]] else [False, None],                        
                        # debug=out_params["debug"]
                        verbose=out_params["verbose"])
                    
                    # Initialize dictionaries on first iteration
                    if first_iteration:
                        # Initialize all result dictionaries with empty lists
                        for key in data.keys():
                            all_data[key] = []
                        
                        # for key in vectorial.keys():
                        #     all_vectorial[key] = []
                        
                        # for key in induction.keys():
                        #     all_induction[key] = []
                        
                        # if magnitudes is not None:
                        #     for key in magnitudes.keys():
                        #         all_magnitudes[key] = []
                        
                        # for key in induction_energy.keys():
                        #     all_induction_energy[key] = []
                        
                        if induction_energy_integral is not None:
                            for key in induction_energy_integral.keys():
                                all_induction_energy_integral[key] = []
                                
                        if induction_test_energy_integral is not None:
                            for key in induction_test_energy_integral.keys():
                                all_induction_test_energy_integral[key] = []
                        
                        if induction_energy_profiles is not None:
                            for key in induction_energy_profiles.keys():
                                all_induction_energy_profiles[key] = []
                        
                        # if induction_uniform is not None:
                        #     for key in induction_uniform.keys():
                        #         all_induction_uniform[key] = []
                        
                        if debug_fields is not None:
                            for key in debug_fields.keys():
                                all_debug_fields[key] = []
                        
                        first_iteration = False
                    
                    # Append results from current iteration to accumulated results
                    for key in data.keys():
                        all_data[key].append(data[key])
                    
                    # for key in vectorial.keys():
                    #     all_vectorial[key].append(vectorial[key])
                    
                    # for key in induction.keys():
                    #     all_induction[key].append(induction[key])
                    
                    # if magnitudes is not None:
                    #     for key in magnitudes.keys():
                    #         all_magnitudes[key].append(magnitudes[key])
                    
                    # for key in induction_energy.keys():
                    #     all_induction_energy[key].append(induction_energy[key])
                    
                    if induction_energy_integral is not None:
                        for key in induction_energy_integral.keys():
                            all_induction_energy_integral[key].append(induction_energy_integral[key])
                            
                    if induction_test_energy_integral is not None:
                        for key in induction_test_energy_integral.keys():
                            all_induction_test_energy_integral[key].append(induction_test_energy_integral[key])
                    
                    if induction_energy_profiles is not None:
                        for key in induction_energy_profiles.keys():
                            all_induction_energy_profiles[key].append(induction_energy_profiles[key])
                    
                    # if induction_uniform is not None:
                    #     for key in induction_uniform.keys():
                    #         all_induction_uniform[key].append(induction_uniform[key])
                    
                    if debug_fields is not None:
                        for key in debug_fields.keys():
                            all_debug_fields[key].append(debug_fields[key])

            # field = np.abs(np.sqrt(induction_uniform['uniform_MIE_compres_x']**2 + 
            #                 induction_uniform['uniform_MIE_compres_y']**2 + 
            #                 induction_uniform['uniform_MIE_compres_z']**2))
            
            # print(field.shape)
            
            # scan_animation_3D(field, Region_Size[i+j], study_box=1, depth=ind_params["up_to_level"], arrow_scale=1, units='Mpc', 
            #         title=f'Magnetic Field Induction Compression Scan - Level {ind_params["up_to_level"]}', verbose=out_params["verbose"], 
            #         Save=True, DPI=out_params["dpi"], run=out_params["run"] + f'_Level_{ind_params["up_to_level"]}', folder=out_params["image_folder"])
            
            # Actual evolution calculation
            if ind_params["energy_evolution"] == False:
                print("Energy evolution calculation is disabled in the configuration. Skipping evolution plots.")
                
                ind_params["up_to_level"] = ind_params["level"][L]
                
            else:
                induction_energy_integral_evo = induction_energy_integral_evolution(
                    ind_params["components"], all_induction_energy_integral,
                    ind_params['evolution_type'], ind_params['derivative'],
                    all_data['rho_b'], all_data['grid_time'], all_data['grid_zeta'],
                    verbose=out_params["verbose"])
                
                ind_params["up_to_level"] = ind_params["level"][L]
                
                plot_integral_evolution(
                    induction_energy_integral_evo,
                    evo_plot_params, ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    Rad[-1], verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )

                print(f"Ploting " + evo_plot_params["title"] + " completed.")
            
            if ind_params["profiles"] == False:
                print("Profiles calculation is disabled in the configuration. Skipping profile plots.")
            else:
                plot_radial_profiles(
                    all_induction_energy_profiles,
                    prof_plot_params, ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    Rad[-1], verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']                
                )
                
                print(f"Ploting " + prof_plot_params["title"] + " completed.")
                
            if out_params["debug"][0] and ind_params["components"]["divergence"]:
                
                inv_resolution = [1.0 / np.array(all_data['resolution'][i]) for i in range(len(all_data['resolution']))]
                
                print("DEBUG - inv resolution length:", len(inv_resolution), len(inv_resolution[0]), len(all_data['resolution'][0]), "values:", all_data['resolution'], "inverse values:", inv_resolution)
                
                for field_key, ref_key, ref_scale_val, quantity in zip(['clus_B2', 'diver_B', 'MIE_diver_x', 'MIE_diver_y', 'MIE_diver_z', 'MIE_diver_B2'],
                                                        ['clus_B2', 'clus_B', 'clus_Bx', 'clus_By', 'clus_Bz', 'clus_B2'],
                                                        [1.0, inv_resolution, 1.0, 1.0, 1.0, 1.0],
                                                        debug_params['quantities']):
                    distribution_check(all_debug_fields[field_key], quantity, debug_params, ind_params,
                                    all_data['grid_time'], all_data['grid_zeta'],
                                    Rad[-1], ref_field=all_debug_fields[ref_key], ref_scale=ref_scale_val,
                                    verbose=out_params["verbose"], save=out_params["save"],
                                    folder=out_params["image_folder"])
                    print(f"Ploting distribution check for " + quantity + " completed.")
                    
            
            # Test evolution (using the test parameters)
            # induction_test_energy_integral_evo = induction_energy_integral_evolution(
            #     ind_params["components"], all_induction_test_energy_integral,
            #     ind_params['evolution_type'], ind_params['derivative'],
            #     all_data['rho_b'], all_data['grid_time'], all_data['grid_zeta'],
            #     verbose=out_params["verbose"])
            
            # ind_params["up_to_level"] = ind_params["level"][L]

            # plot_integral_evolution(
            #     induction_test_energy_integral_evo,
            #     ind_params["test_params"]['evo_plot_params'], ind_params,
            #     all_data['grid_time'], all_data['grid_zeta'],
            #     Rad[-1], verbose=out_params['verbose'], save=out_params['save'],
            #     folder=out_params['image_folder']
            # )

            # print(f"Ploting " + ind_params["test_params"]['evo_plot_params']["title"] + " completed.")
            
        
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
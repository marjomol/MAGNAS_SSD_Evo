import numpy as np
import time
import scripts.utils as utils
from config import IND_PARAMS as ind_params
from config import OUTPUT_PARAMS as out_params
from config import EVO_PLOT_PARAMS as evo_plot_params
from config import PROFILE_PLOT_PARAMS as prof_plot_params
from config import DEBUG_PARAMS as debug_params
from config import PERCENTILE_PLOT_PARAMS as percentile_plot_params
from scripts.induction_evo import find_most_massive_halo, create_region, process_iteration, induction_energy_integral_evolution
from scripts.plot_fields import plot_integral_evolution, plot_radial_profiles, plot_percentile_evolution, distribution_check, scan_animation_3D, zoom_animation_3D
from scripts.parallel_utils import process_iteration_with_logging
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
        Region_Coord, Region_Size = create_region(out_params["sims"], out_params["it"], Coords, Rad, 
                    F=ind_params["F"], reg=ind_params["region"], 
                            verbose=out_params["verbose"])
        
        # Prepare per-snapshot flags from config indices
        profile_idx_set = set([out_params["it"][k] for k in prof_plot_params["it_indx"]]) if len(out_params["it"]) else set()
        debug_idx_set = set([out_params["it"][k] for k in debug_params["it_indx"]]) if len(out_params["it"]) else set()

        # Process all configured levels, one parallel batch per level
        for L, lvl in enumerate(ind_params["level"]):
            # Initialize result dictionaries for this level based on what's enabled
            all_data = {}
            all_vectorial = {} if ind_params["return_vectorial"] else None
            all_induction = {} if ind_params["return_induction"] else None
            all_magnitudes = {} if ind_params["mag"] else None
            all_induction_energy = {} if ind_params["return_induction_energy"] else None
            all_induction_energy_integral = {} if ind_params["energy_evolution"] else None
            all_induction_test_energy_integral = {} if ind_params["energy_evolution"] else None
            all_induction_energy_profiles = {} if ind_params["profiles"] else None
            all_induction_uniform = {} if ind_params["projection"] else None
            all_diver_B_percentiles = {} if ind_params["percentiles"] else None
            all_debug_fields = {} if out_params["debug"][0] else None
            first_iteration = True

            from concurrent.futures import ProcessPoolExecutor
            futures = []
            with ProcessPoolExecutor(max_workers=out_params["ncores"]) as executor:
                for i, sims in enumerate(out_params["sims"]):
                    for j, it in enumerate(out_params["it"]):
                        profiles_flag = bool(ind_params["profiles"]) and (it in profile_idx_set)
                        debug_flag = out_params["debug"] if it in debug_idx_set else [False, None]

                        fut = executor.submit(
                            process_iteration_with_logging,
                            ind_params["components"],
                            out_params["dir_grids"],
                            out_params["dir_gas"],
                            out_params["dir_params"][i],
                            sims,
                            it,
                            Coords[i + j],
                            Region_Coord[i + j],
                            Rad[i + j],
                            ind_params["rmin"][i],
                            lvl,
                            lvl,
                            ind_params["nmax"][i],
                            ind_params["size"][i],
                            ind_params["H0"],
                            ind_params["a0"],
                            test=ind_params["test_params"],
                            units=ind_params["units"],
                            nbins=ind_params["nbins"][i],
                            logbins=ind_params["logbins"],
                            stencil=ind_params["stencil"],
                            buffer=ind_params["buffer"],
                            use_siblings=ind_params["use_siblings"],
                            interpol=ind_params["interpol"],
                            use_parent_diff=ind_params["use_parent_diff"],
                            nghost=ind_params["nghost"],
                            bitformat=out_params["bitformat"],
                            mag=ind_params["mag"],
                            energy_evolution=ind_params["energy_evolution"],
                            profiles=profiles_flag,
                            projection=ind_params["projection"],
                            percentiles=ind_params["percentiles"],
                            percentile_levels=ind_params["percentile_levels"],
                            debug=debug_flag,
                            return_vectorial=ind_params["return_vectorial"],
                            return_induction=ind_params["return_induction"],
                            return_induction_energy=ind_params["return_induction_energy"],
                            verbose=out_params["verbose"],
                        )
                        futures.append(fut)

                for fut in futures:
                    # Unpack results with logging information
                    (data, vectorial, induction, magnitudes, induction_energy, induction_energy_integral, induction_test_energy_integral, induction_energy_profiles, induction_uniform, diver_B_percentiles, debug_fields), log_output, (sim_name, iteration) = fut.result()
                    
                    # Print the captured logs with a header showing which iteration this is
                    if log_output.strip():  # Only print if there's actual output
                        print(f"\n{'*'*80}")
                        print(f"Output from Simulation: {sim_name}, Iteration: {iteration}")
                        print(f"{'*'*80}")
                        print(log_output, end='')
                        print(f"{'*'*80}\n")

                    if first_iteration:
                        for key in data.keys():
                            all_data[key] = []
                        if all_vectorial is not None and vectorial is not None:
                            for key in vectorial.keys():
                                all_vectorial[key] = []
                        if all_induction is not None and induction is not None:
                            for key in induction.keys():
                                all_induction[key] = []
                        if all_magnitudes is not None and magnitudes is not None:
                            for key in magnitudes.keys():
                                all_magnitudes[key] = []
                        if all_induction_energy is not None and induction_energy is not None:
                            for key in induction_energy.keys():
                                all_induction_energy[key] = []
                        if all_induction_energy_integral is not None and induction_energy_integral is not None:
                            for key in induction_energy_integral.keys():
                                all_induction_energy_integral[key] = []
                        if all_induction_test_energy_integral is not None and induction_test_energy_integral is not None:
                            for key in induction_test_energy_integral.keys():
                                all_induction_test_energy_integral[key] = []
                        if all_induction_energy_profiles is not None and induction_energy_profiles is not None:
                            for key in induction_energy_profiles.keys():
                                all_induction_energy_profiles[key] = []
                        if all_induction_uniform is not None and induction_uniform is not None:
                            for key in induction_uniform.keys():
                                all_induction_uniform[key] = []
                        if all_diver_B_percentiles is not None and diver_B_percentiles is not None:
                            for key in diver_B_percentiles.keys():
                                all_diver_B_percentiles[key] = []
                        if all_debug_fields is not None and debug_fields is not None:
                            for key in debug_fields.keys():
                                all_debug_fields[key] = []
                        first_iteration = False

                    for key in data.keys():
                        all_data[key].append(data[key])
                    if all_vectorial is not None and vectorial is not None:
                        for key in vectorial.keys():
                            all_vectorial[key].append(vectorial[key])
                    if all_induction is not None and induction is not None:
                        for key in induction.keys():
                            all_induction[key].append(induction[key])
                    if all_magnitudes is not None and magnitudes is not None:
                        for key in magnitudes.keys():
                            all_magnitudes[key].append(magnitudes[key])
                    if all_induction_energy is not None and induction_energy is not None:
                        for key in induction_energy.keys():
                            all_induction_energy[key].append(induction_energy[key])
                    if all_induction_energy_integral is not None and induction_energy_integral is not None:
                        for key in induction_energy_integral.keys():
                            all_induction_energy_integral[key].append(induction_energy_integral[key])
                    if all_induction_test_energy_integral is not None and induction_test_energy_integral is not None:
                        for key in induction_test_energy_integral.keys():
                            all_induction_test_energy_integral[key].append(induction_test_energy_integral[key])
                    if all_induction_energy_profiles is not None and induction_energy_profiles is not None:
                        for key in induction_energy_profiles.keys():
                            all_induction_energy_profiles[key].append(induction_energy_profiles[key])
                    if all_induction_uniform is not None and induction_uniform is not None:
                        for key in induction_uniform.keys():
                            all_induction_uniform[key].append(induction_uniform[key])
                    if all_diver_B_percentiles is not None and diver_B_percentiles is not None:
                        for key in diver_B_percentiles.keys():
                            all_diver_B_percentiles[key].append(diver_B_percentiles[key])
                    if all_debug_fields is not None and debug_fields is not None:
                        for key in debug_fields.keys():
                            all_debug_fields[key].append(debug_fields[key])

            # Plotting and post-processing driven by config for this level
            if ind_params["energy_evolution"] and all_induction_energy_integral is not None:
                induction_energy_integral_evo = induction_energy_integral_evolution(
                    ind_params["components"], all_induction_energy_integral,
                    ind_params['evolution_type'], ind_params['derivative'],
                    all_data['rho_b'], all_data['grid_time'], all_data['grid_zeta'],
                    verbose=out_params["verbose"])
                ind_params["up_to_level"] = lvl
                plot_integral_evolution(
                    induction_energy_integral_evo,
                    evo_plot_params, ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    Rad[-1], verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )
                print(f"Ploting " + evo_plot_params["title"] + f" completed (level {lvl}).")

            if ind_params["profiles"] and all_induction_energy_profiles is not None:
                ind_params["up_to_level"] = lvl
                plot_radial_profiles(
                    all_induction_energy_profiles,
                    prof_plot_params, ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    Rad[-1], verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']                
                )
                print(f"Ploting " + prof_plot_params["title"] + f" completed (level {lvl}).")

            if ind_params["percentiles"] and all_diver_B_percentiles is not None:
                ind_params["up_to_level"] = lvl
                plot_percentile_evolution(
                    all_diver_B_percentiles,
                    percentile_plot_params, ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )
                print(f"Ploting " + percentile_plot_params["title"] + f" completed (level {lvl}).")

            if out_params["debug"][0] and ind_params["components"]["divergence"] and all_debug_fields:
                inv_resolution = [1.0 / np.array(all_data['resolution'][i]) for i in range(len(all_data['resolution']))]
                ind_params["up_to_level"] = lvl
                for field_key, ref_key, ref_scale_val, quantity in zip(['clus_B2', 'diver_B', 'MIE_diver_x', 'MIE_diver_y', 'MIE_diver_z', 'MIE_diver_B2'],
                                                        ['clus_B2', 'clus_B', 'clus_Bx', 'clus_By', 'clus_Bz', 'clus_B2'],
                                                        [1.0, inv_resolution, 1.0, 1.0, 1.0, 1.0],
                                                        debug_params['quantities']):
                    distribution_check(all_debug_fields[field_key], quantity, debug_params, ind_params,
                                    all_data['grid_time'], all_data['grid_zeta'],
                                    Rad[-1], ref_field=all_debug_fields[ref_key], ref_scale=ref_scale_val,
                                    clean=out_params["debug"][2], verbose=out_params["verbose"], save=out_params["save"],
                                    folder=out_params["image_folder"])
                    print(f"Ploting distribution check for " + quantity + f" completed (level {lvl}).")

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
            
            # Initialize result dictionaries before the loop (conditional based on config)
            all_data = {}
            all_vectorial = {} if ind_params["return_vectorial"] else None
            all_induction = {} if ind_params["return_induction"] else None
            all_magnitudes = {} if ind_params["mag"] else None
            all_induction_energy = {} if ind_params["return_induction_energy"] else None
            all_induction_energy_integral = {} if ind_params["energy_evolution"] else None
            all_induction_test_energy_integral = {} if ind_params["energy_evolution"] else None
            all_induction_energy_profiles = {} if ind_params["profiles"] else None
            all_induction_uniform = {} if ind_params["projection"] else None
            all_diver_B_percentiles = {} if ind_params["percentiles"] else None
            all_debug_fields = {} if out_params["debug"][0] else None

            # Flag to track first iteration for dictionary initialization
            first_iteration = True
            
            # Process each iteration in serial
            for sims, i in zip(out_params["sims"], range(len(out_params["sims"]))):
                for it, j in zip(out_params["it"], range(len(out_params["it"]))):
                    
                    (data, vectorial, induction, magnitudes, induction_energy, induction_energy_integral, induction_test_energy_integral, induction_energy_profiles, induction_uniform, diver_B_percentiles, debug_fields), log_output, _ = process_iteration_with_logging(
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
                        use_siblings=ind_params["use_siblings"],
                        interpol=ind_params["interpol"],
                        use_parent_diff=ind_params["use_parent_diff"],
                        nghost=ind_params["nghost"],
                        bitformat=out_params["bitformat"],
                        mag=ind_params["mag"],
                        energy_evolution=ind_params["energy_evolution"],
                        profiles=ind_params["profiles"] if it in [out_params["it"][k] for k in prof_plot_params["it_indx"]] else False,
                        projection=ind_params["projection"],
                        percentiles=ind_params["percentiles"],
                        percentile_levels=ind_params["percentile_levels"],
                        debug=out_params["debug"] if it in [out_params["it"][k] for k in debug_params["it_indx"]] else [False, None],
                        return_vectorial=ind_params["return_vectorial"],
                        return_induction=ind_params["return_induction"],
                        return_induction_energy=ind_params["return_induction_energy"],
                        verbose=out_params["verbose"])
                    
                    # Print the captured logs with a header showing which iteration this is
                    if log_output.strip():  # Only print if there's actual output
                        print(f"\n{'*'*80}")
                        print(f"Output from Simulation: {sims}, Iteration: {it}")
                        print(f"{'*'*80}")
                        print(log_output, end='')
                        print(f"{'*'*80}\n")
                    
                    # Initialize dictionaries on first iteration
                    if first_iteration:
                        # Initialize all result dictionaries with empty lists
                        for key in data.keys():
                            all_data[key] = []
                        
                        if vectorial is not None:
                            for key in vectorial.keys():
                                all_vectorial[key] = []
                        
                        if induction is not None:
                            for key in induction.keys():
                                all_induction[key] = []
                        
                        if magnitudes is not None:
                            for key in magnitudes.keys():
                                all_magnitudes[key] = []
                        
                        if induction_energy is not None:
                            for key in induction_energy.keys():
                                all_induction_energy[key] = []
                        
                        if induction_energy_integral is not None:
                            for key in induction_energy_integral.keys():
                                all_induction_energy_integral[key] = []
                                
                        if induction_test_energy_integral is not None:
                            for key in induction_test_energy_integral.keys():
                                all_induction_test_energy_integral[key] = []
                        
                        if induction_energy_profiles is not None:
                            for key in induction_energy_profiles.keys():
                                all_induction_energy_profiles[key] = []
                        
                        if diver_B_percentiles is not None:
                            for key in diver_B_percentiles.keys():
                                all_diver_B_percentiles[key] = []
                        
                        if induction_uniform is not None:
                            for key in induction_uniform.keys():
                                all_induction_uniform[key] = []
                        
                        if debug_fields is not None:
                            for key in debug_fields.keys():
                                all_debug_fields[key] = []
                        
                        first_iteration = False
                    
                    for key in data.keys():
                        all_data[key].append(data[key])
                    
                    if all_vectorial is not None and vectorial is not None:
                        for key in vectorial.keys():
                            all_vectorial[key].append(vectorial[key])
                    
                    if all_induction is not None and induction is not None:
                        for key in induction.keys():
                            all_induction[key].append(induction[key])
                    
                    if all_magnitudes is not None and magnitudes is not None:
                        for key in magnitudes.keys():
                            all_magnitudes[key].append(magnitudes[key])
                    
                    if all_induction_energy is not None and induction_energy is not None:
                        for key in induction_energy.keys():
                            all_induction_energy[key].append(induction_energy[key])
                    
                    if induction_energy_integral is not None:
                        for key in induction_energy_integral.keys():
                            all_induction_energy_integral[key].append(induction_energy_integral[key])
                            
                    if induction_test_energy_integral is not None:
                        for key in induction_test_energy_integral.keys():
                            all_induction_test_energy_integral[key].append(induction_test_energy_integral[key])
                    
                    if induction_energy_profiles is not None:
                        for key in induction_energy_profiles.keys():
                            all_induction_energy_profiles[key].append(induction_energy_profiles[key])
                    
                    if diver_B_percentiles is not None:
                        for key in diver_B_percentiles.keys():
                            all_diver_B_percentiles[key].append(diver_B_percentiles[key])
                    
                    if induction_uniform is not None:
                        for key in induction_uniform.keys():
                            all_induction_uniform[key].append(induction_uniform[key])
                    
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
                ind_params["up_to_level"] = ind_params["level"][L]
                plot_radial_profiles(
                    all_induction_energy_profiles,
                    prof_plot_params, ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    Rad[-1], verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']                
                )
                
                print(f"Ploting " + prof_plot_params["title"] + " completed.")
            
            if ind_params["percentiles"] == False:
                print("Percentiles calculation is disabled in the configuration. Skipping percentile evolution plots.")
            else:
                ind_params["up_to_level"] = ind_params["level"][L]
                plot_percentile_evolution(
                    all_diver_B_percentiles,
                    percentile_plot_params, ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )
                
                print(f"Ploting " + percentile_plot_params["title"] + " completed.")
                
            if out_params["debug"][0] and ind_params["components"]["divergence"]:
                ind_params["up_to_level"] = ind_params["level"][L]
                inv_resolution = [1.0 / np.array(all_data['resolution'][i]) for i in range(len(all_data['resolution']))]
                
                for field_key, ref_key, ref_scale_val, quantity in zip(['clus_B2', 'diver_B', 'MIE_diver_x', 'MIE_diver_y', 'MIE_diver_z', 'MIE_diver_B2'],
                                                        ['clus_B2', 'clus_B', 'clus_Bx', 'clus_By', 'clus_Bz', 'clus_B2'],
                                                        [1.0, inv_resolution, 1.0, 1.0, 1.0, 1.0],
                                                        debug_params['quantities']):
                    distribution_check(all_debug_fields[field_key], quantity, debug_params, ind_params,
                                    all_data['grid_time'], all_data['grid_zeta'],
                                    Rad[-1], ref_field=all_debug_fields[ref_key], ref_scale=ref_scale_val,
                                    clean=out_params["debug"][2], verbose=out_params["verbose"], save=out_params["save"],
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
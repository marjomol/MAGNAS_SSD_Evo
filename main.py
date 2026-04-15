import numpy as np
import time
import atexit
import scripts.utils as utils
import scripts.debug as debug_module
from config import IND_PARAMS as ind_params
from config import OUTPUT_PARAMS as out_params
from config import EVO_PLOT_PARAMS as evo_plot_params
from config import PROD_DISS_PLOT_PARAMS as prod_diss_plot_params
from config import PROFILE_PLOT_PARAMS as prof_plot_params
from config import DEBUG_PARAMS as debug_params
from config import PERCENTILE_PLOT_PARAMS as percentile_plot_params
from config import SCAN_PLOT_PARAMS as scan_plot_params
from config import get_sim_characteristics
from scripts.induction_evo import find_most_massive_halo, create_region, process_iteration, induction_energy_integral_evolution
from scripts.plot_fields import plot_integral_evolution, plot_production_dissipation_evolution, plot_radial_profiles, plot_percentile_evolution, distribution_check, scan_animation_3D, zoom_animation_3D
from scripts.memory_utils import process_iteration_with_logging, MemoryMonitor, build_executor_kwargs
from scripts.units import *
np.random.seed(out_params["random_seed"]) # Set the random seed for reproducibility
import gc
memory_monitor = MemoryMonitor(out_params)

_log_cm = None
if out_params.get("save_terminal", False):
    log_filename = utils.build_terminal_log_filename("all", "all", ind_params, out_params)
    terminal_folder = out_params.get("terminal_folder", "terminal_output/")
    if not terminal_folder.endswith('/'):
        terminal_folder += '/'
    log_filepath = terminal_folder + log_filename
    _log_cm = utils.redirect_output_to_file(log_filepath, verbose_console=True)
    _log_cm.__enter__()
    print(f"\n[INFO] Terminal output will be saved to: {log_filepath}\n")

    def _close_log():
        _log_cm.__exit__(None, None, None)

    atexit.register(_close_log)

start_time = time.time()

# ============================
# Only edit the section below
# ============================

if __name__ == "__main__":
    memory_monitor.start()
    memory_monitor.log("startup", force=True)
    diff_params = ind_params["differentiation"]
    return_params = ind_params["return"]
    percentile_params_cfg = ind_params["percentiles"]
    if out_params["parallel"]:
        print(f'**************************************************************')
        print(f"Running in parallel mode with {out_params['ncores']} cores")
        print(f'**************************************************************')
        # Find the most massive halo and region per simulation
        Coords = []
        Rad = []
        Region_Coord = []
        Region_Size = []
        for i, sim_name in enumerate(out_params["sims"]):
            it_list = out_params["it"][i]
            coords_i, rad_i = find_most_massive_halo(
                sim_name, it_list,
                ind_params["a0"],
                out_params["dir_halos"],
                out_params["dir_grids"],
                out_params["data_folder"],
                vir_kind=ind_params["vir_kind"],
                rad_kind=ind_params["rad_kind"],
                verbose=out_params["verbose"]
            )
            region_coord_i, region_size_i = create_region(
                sim_name, it_list, coords_i, rad_i,
                size=ind_params["size"][i], F=ind_params["F"], reg=ind_params["region"],
                verbose=out_params["verbose"]
            )
            Coords.append(coords_i)
            Rad.append(rad_i)
            Region_Coord.append(region_coord_i)
            Region_Size.append(region_size_i)
        memory_monitor.log("after region build (parallel)", force=True)
        
        # Process all configured levels, one parallel batch per level
        for L, lvl in enumerate(ind_params["level"]):
            # Initialize result dictionaries for this level based on what's enabled
            all_data = {}
            all_vectorial = {} if return_params["return_vectorial"] else None
            all_induction = {} if return_params["return_induction"] else None
            all_magnitudes = {} if return_params["mag"] else None
            all_induction_energy = {} if return_params["return_induction_energy"] else None
            need_integrals = ind_params["energy_evolution"]["enabled"] or ind_params.get("production_dissipation", {}).get("enabled", False)
            all_induction_energy_integral = {} if need_integrals else None
            all_induction_test_energy_integral = {} if need_integrals else None
            all_induction_energy_profiles = {} if return_params["profiles"] else None
            all_induction_uniform = {} if return_params["projection"] else None
            all_diver_B_percentiles = {} if percentile_params_cfg["enabled"] else None
            any_debug = debug_params.get("field_analysis", {}).get("enabled", False) or debug_params.get("scan_animation", {}).get("enabled", False)
            all_debug_fields = {} if any_debug else None
            scan_meta_sim = []
            scan_meta_it = []
            scan_meta_idx = []
            profile_indices = []  # Track which indices in grid_zeta have profiles
            first_iteration = True
            iteration_counter = 0

            from concurrent.futures import ProcessPoolExecutor
            futures = []
            executor_kwargs = build_executor_kwargs(
                out_params["ncores"],
                out_params.get("max_tasks_per_child", None)
            )
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                for i, sims in enumerate(out_params["sims"]):
                    it_list = out_params["it"][i]
                    profile_it_indx = prof_plot_params.get("it_indx", [-1])
                    debug_it_indx = debug_params.get("it_indx", [-1])
                    profile_idx_set = set([it_list[k] for k in profile_it_indx if -len(it_list) <= k < len(it_list)])
                    debug_idx_set = set([it_list[k] for k in debug_it_indx if -len(it_list) <= k < len(it_list)])
                    for j, it in enumerate(it_list):
                        profiles_flag = bool(return_params["profiles"]) and (it in profile_idx_set)

                        # Always apply percentile_params to all snapshots, but keep other debug
                        # settings only for selected iterations
                        percentile_params = debug_params.get("percentile_params", {})
                        if it in debug_idx_set:
                            debug_params_flag = {**debug_params}
                            debug_params_flag["percentile_params"] = percentile_params
                        else:
                            debug_params_flag = {"percentile_params": percentile_params}

                        # Get simulation characteristics for this simulation
                        sim_characteristics = get_sim_characteristics(sims)

                        fut = executor.submit(
                            process_iteration_with_logging,
                            ind_params["components"],
                            out_params["dir_grids"],
                            out_params["dir_gas"],
                            out_params["dir_params"][i],
                            sims,
                            it,
                            Coords[i][j],
                            Region_Coord[i][j],
                            Rad[i][j],
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
                            stencil=diff_params["stencil"],
                            buffer=diff_params["buffer"],
                            use_siblings=diff_params["use_siblings"],
                            interpol=diff_params["interpol"],
                            nghost=diff_params["nghost"],
                            blend=diff_params.get("blend", False),
                            parent=diff_params.get("parent", False),
                            parent_interpol=diff_params.get("parent_interpol", diff_params["interpol"]),
                            bitformat=out_params["bitformat"],
                            mag=return_params["mag"],
                            sim_characteristics=sim_characteristics,
                            energy_evolution_config=ind_params["energy_evolution"],
                            energy_evolution=ind_params["energy_evolution"]["enabled"],
                            profiles=profiles_flag,
                            projection=return_params["projection"],
                            percentiles=percentile_params_cfg["enabled"],
                            percentile_levels=percentile_params_cfg["percentile_levels"],
                            divergence_filter=ind_params.get("divergence_filter"),
                            debug_params=debug_params_flag,
                            production_dissipation=ind_params.get("production_dissipation"),
                            return_vectorial=return_params["return_vectorial"],
                            return_induction=return_params["return_induction"],
                            return_induction_energy=return_params["return_induction_energy"],
                            gc_worker_end=out_params.get("memory_profiling", {}).get("gc_worker_end", False),
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

                    # Current index for this iteration in aggregated arrays
                    iter_idx = len(all_data.get('grid_time', []))

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
                        # Track that this iteration index has profiles
                        profile_indices.append(iter_idx)
                        for key in induction_energy_profiles.keys():
                            if key not in all_induction_energy_profiles:
                                all_induction_energy_profiles[key] = []
                            all_induction_energy_profiles[key].append(induction_energy_profiles[key])
                    if all_induction_uniform is not None and induction_uniform is not None:
                        for key in induction_uniform.keys():
                            all_induction_uniform[key].append(induction_uniform[key])
                    if all_diver_B_percentiles is not None and diver_B_percentiles is not None:
                        for key in diver_B_percentiles.keys():
                            all_diver_B_percentiles[key].append(diver_B_percentiles[key])
                    if all_debug_fields is not None and debug_fields is not None:
                        for key in debug_fields.keys():
                            # Initialize key if it doesn't exist yet (for debug-specific fields)
                            if key not in all_debug_fields:
                                all_debug_fields[key] = []
                            all_debug_fields[key].append(debug_fields[key])
                        if '_scan_volume' in debug_fields:
                            scan_meta_sim.append(sim_name)
                            scan_meta_it.append(iteration)
                            scan_meta_idx.append(iter_idx)

                    iteration_counter += 1
                    if memory_monitor.gc_main_each_iteration:
                        gc.collect()
                    memory_monitor.log(
                        f"parallel level={lvl} sim={sim_name} it={iteration} processed={iteration_counter}",
                        force=memory_monitor.should_log_iteration(iteration_counter)
                    )
                    del data, vectorial, induction, magnitudes, induction_energy
                    del induction_energy_integral, induction_test_energy_integral
                    del induction_energy_profiles, induction_uniform, diver_B_percentiles, debug_fields

            # Enforce consistent temporal ordering before any evolution/plot step.
            debug_module.ensure_temporal_order(
                all_data,
                series_dicts=[
                    all_vectorial,
                    all_induction,
                    all_magnitudes,
                    all_induction_energy,
                    all_induction_energy_integral,
                    all_induction_test_energy_integral,
                    all_induction_energy_profiles,
                    all_induction_uniform,
                    all_diver_B_percentiles,
                    all_debug_fields,
                ],
                index_lists=[profile_indices, scan_meta_idx],
                verbose=out_params["verbose"],
            )

            # Plotting and post-processing driven by config for this level
            if ind_params["energy_evolution"]["enabled"] and all_induction_energy_integral is not None:
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = lvl
                induction_energy_integral_evo = induction_energy_integral_evolution(
                    ind_params["components"], all_induction_energy_integral,
                    ind_params['energy_evolution']['derivative'],
                    all_data['rho_b'], all_data['grid_time'], all_data['grid_zeta'],
                    normalized=ind_params['energy_evolution'].get('normalized', False),
                    verbose=out_params["verbose"])
                plot_integral_evolution(
                    induction_energy_integral_evo,
                    evo_plot_params, plot_ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    utils.get_last_rad(Rad), verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )
                print(f"Ploting " + evo_plot_params["title"] + f" completed (level {lvl}).")

            if ind_params.get("production_dissipation", {}).get("enabled", False) and all_induction_energy_integral is not None:
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = lvl
                pd_figs = plot_production_dissipation_evolution(
                    all_induction_energy_integral,
                    prod_diss_plot_params,
                    plot_ind_params,
                    all_data['grid_time'],
                    all_data['grid_zeta'],
                    verbose=out_params['verbose'],
                    save=out_params['save'],
                    folder=out_params['image_folder']
                )
                if pd_figs:
                    print(f"Ploting " + prod_diss_plot_params["title"] + f" completed (level {lvl}).")
                else:
                    print(f"Ploting " + prod_diss_plot_params["title"] + f" skipped (level {lvl}, no valid integrated P/D data).")

            if return_params["profiles"] and all_induction_energy_profiles is not None:
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = lvl
                # Determine how many snapshots have profiles calculated
                num_snapshots_with_profiles = len(all_induction_energy_profiles.get('clus_b2_profile', []))
                if num_snapshots_with_profiles > 0:
                    # Use the tracked profile indices to correctly map to grid_zeta
                    prof_plot_params_adjusted = prof_plot_params.copy()
                    prof_plot_params_adjusted['it_indx'] = profile_indices
                    plot_radial_profiles(
                        all_induction_energy_profiles,
                        prof_plot_params_adjusted, plot_ind_params,
                        all_data['grid_time'], all_data['grid_zeta'],
                        utils.get_last_rad(Rad), verbose=out_params['verbose'], save=out_params['save'],
                        folder=out_params['image_folder']                
                    )
                    print(f"Ploting " + prof_plot_params["title"] + f" completed (level {lvl}).")

            if percentile_params_cfg["enabled"] and all_diver_B_percentiles is not None:
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = lvl
                plot_percentile_evolution(
                    all_diver_B_percentiles,
                    percentile_plot_params, plot_ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )
                print(f"Ploting " + percentile_plot_params["title"] + f" completed (level {lvl}).")

            if debug_params.get("field_analysis", {}).get("enabled", False) and ind_params["components"]["divergence"] and all_debug_fields:
                inv_resolution = [1.0 / np.array(all_data['resolution'][i]) for i in range(len(all_data['resolution']))]
                # Don't overwrite ind_params["up_to_level"] - create a copy for local use
                local_ind_params = ind_params.copy()
                local_ind_params["up_to_level"] = lvl
                for field_key, ref_key, ref_scale_val, quantity in zip(['clus_B2', 'diver_B', 'MIE_diver_x', 'MIE_diver_y', 'MIE_diver_z', 'MIE_diver_B2'],
                                                        ['clus_B2', 'clus_B', 'clus_Bx', 'clus_By', 'clus_Bz', 'clus_B2'],
                                                        [1.0, inv_resolution, 1.0, 1.0, 1.0, 1.0],
                                                        debug_params.get("field_analysis", {}).get("quantities", [])):
                    if field_key not in all_debug_fields or ref_key not in all_debug_fields:
                        if out_params["verbose"]:
                            print(f"Skipping " + quantity + f": field {field_key} or {ref_key} not available")
                        continue
                    distribution_check(all_debug_fields[field_key], quantity, debug_params, local_ind_params,
                                    all_data['grid_time'], all_data['grid_zeta'],
                                    utils.get_last_rad(Rad), ref_field=all_debug_fields[ref_key], ref_scale=ref_scale_val,
                                    clean=debug_params.get("field_analysis", {}).get("clean_field", False), verbose=out_params["verbose"], save=out_params["save"],
                                    folder=out_params["image_folder"])
                    print(f"Ploting distribution check for " + quantity + f" completed (level {lvl}).")

            # Render scan animations after data processing (parallel mode)
            if debug_params.get("scan_animation", {}).get("enabled", False) and all_debug_fields and '_scan_volume' in all_debug_fields:
                scan_volumes = all_debug_fields['_scan_volume']
                scan_volumes_no_buffer = all_debug_fields.get('_scan_volume_no_buffer', None)
                scan_region_sizes = all_debug_fields['_scan_region_size']
                scan_volume_levels = all_debug_fields.get('_scan_volume_levels', [])
                
                for vol_idx, volume in enumerate(scan_volumes):
                    ind_meta = ind_params.copy()
                    sim = scan_meta_sim[vol_idx] if vol_idx < len(scan_meta_sim) else out_params["sims"][0]
                    fallback_it = out_params["it"][0][0] if out_params["it"] and out_params["it"][0] else 0
                    it_val = scan_meta_it[vol_idx] if vol_idx < len(scan_meta_it) else fallback_it
                    idx = scan_meta_idx[vol_idx] if vol_idx < len(scan_meta_idx) else vol_idx
                    zeta_val = all_data['grid_zeta'][idx]
                    time_val = all_data['grid_time'][idx]
                    region_size = scan_region_sizes[vol_idx]
                
                    ind_meta['sim'] = sim
                    ind_meta['it'] = it_val
                    ind_meta['zeta'] = zeta_val
                    ind_meta['time'] = time_val
                    ind_meta['level'] = lvl
                    ind_meta['up_to_level'] = ind_params["level"][L]                    
                    
                    # Prepare volume parameters
                    vol_meta = {
                        'vol_idx': vol_idx,
                        'reg_idx': 0
                    }
                    
                    # Prepare scan plot parameters with optional AMR levels for debug visualization
                    scan_plot_params_with_amr = scan_plot_params.copy()
                    amr_levels = scan_volume_levels[vol_idx] if vol_idx < len(scan_volume_levels) else np.array([])
                    scan_plot_params_with_amr['amr_levels'] = amr_levels
                    
                    # Call scan animation with config parameters (WITH buffer)
                    scan_animation_3D(
                        volume, region_size,
                        scan_plot_params_with_amr,
                        ind_meta,
                        volume_params=vol_meta,
                        verbose=debug_params["scan_animation"]["verbose"],
                        save=debug_params["scan_animation"]["save"],
                        folder=out_params["image_folder"]
                    )

                    # Also render without buffer for comparison if available
                    if scan_volumes_no_buffer is not None and vol_idx < len(scan_volumes_no_buffer):
                        ind_meta['buffer'] = False  # Mark as no-buffer version
                        scan_plot_params_with_amr['title'] = f"{scan_plot_params.get('title', 'Field Scan')} No Buffer"

                        scan_animation_3D(
                            scan_volumes_no_buffer[vol_idx], region_size,
                            scan_plot_params_with_amr,
                            ind_meta,
                            volume_params=vol_meta,
                            verbose=debug_params["scan_animation"]["verbose"],
                            save=debug_params["scan_animation"]["save"],
                            folder=out_params["image_folder"]
                        )

            if memory_monitor.gc_main_each_iteration:
                gc.collect()
            memory_monitor.log(f"after parallel level={lvl} post-processing", force=True)

    else:
        for L in range(len(ind_params["level"])):
            print(f'*************************')
            print("Running in serial mode")
            print(f'*************************')
            # Find the most massive halo and region per simulation
            Coords = []
            Rad = []
            Region_Coord = []
            Region_Size = []
            for i, sim_name in enumerate(out_params["sims"]):
                it_list = out_params["it"][i]
                coords_i, rad_i = find_most_massive_halo(
                    sim_name, it_list,
                    ind_params["a0"],
                    out_params["dir_halos"],
                    out_params["dir_grids"],
                    out_params["data_folder"],
                    vir_kind=ind_params["vir_kind"],
                    rad_kind=ind_params["rad_kind"],
                    verbose=out_params["verbose"]
                )
                region_coord_i, region_size_i = create_region(
                    sim_name, it_list, coords_i, rad_i,
                    size=ind_params["size"][i], F=ind_params["F"], reg=ind_params["region"],
                    verbose=out_params["verbose"]
                )
                Coords.append(coords_i)
                Rad.append(rad_i)
                Region_Coord.append(region_coord_i)
                Region_Size.append(region_size_i)
            memory_monitor.log(f"after region build (serial, level={ind_params['level'][L]})", force=True)
            
            # Initialize result dictionaries before the loop (conditional based on config)
            all_data = {}
            all_vectorial = {} if return_params["return_vectorial"] else None
            all_induction = {} if return_params["return_induction"] else None
            all_magnitudes = {} if return_params["mag"] else None
            all_induction_energy = {} if return_params["return_induction_energy"] else None
            need_integrals = ind_params["energy_evolution"]["enabled"] or ind_params.get("production_dissipation", {}).get("enabled", False)
            all_induction_energy_integral = {} if need_integrals else None
            all_induction_test_energy_integral = {} if need_integrals else None
            all_induction_energy_profiles = {} if return_params["profiles"] else None
            all_induction_uniform = {} if return_params["projection"] else None
            all_diver_B_percentiles = {} if percentile_params_cfg["enabled"] else None
            any_debug = debug_params.get("field_analysis", {}).get("enabled", False) or debug_params.get("scan_animation", {}).get("enabled", False)
            all_debug_fields = {} if any_debug else None
            scan_meta_sim = []
            scan_meta_it = []
            scan_meta_idx = []
            profile_indices = []

            # Flag to track first iteration for dictionary initialization
            first_iteration = True
            iteration_counter = 0
            
            # Process each iteration in serial
            for sims, i in zip(out_params["sims"], range(len(out_params["sims"]))):
                it_list = out_params["it"][i]
                profile_it_indx = prof_plot_params.get("it_indx", [-1])
                debug_it_indx = debug_params.get("it_indx", [-1])
                profile_idx_set = set([it_list[k] for k in profile_it_indx if -len(it_list) <= k < len(it_list)])
                debug_idx_set = set([it_list[k] for k in debug_it_indx if -len(it_list) <= k < len(it_list)])
                for j, it in enumerate(it_list):
                    
                    # Print header before processing
                    print(f"\n{'*'*80}")
                    print(f"Output from Simulation: {sims}, Iteration: {it}")
                    print(f"{'*'*80}")
                    
                    # In serial mode, call the iteration directly to stream stdout live
                    # Always apply percentile_params to all snapshots, but keep other debug
                    # settings only for selected iterations
                    percentile_params = debug_params.get("percentile_params", {})
                    if it in debug_idx_set:
                        debug_params_flag = {**debug_params}
                        debug_params_flag["percentile_params"] = percentile_params
                    else:
                        debug_params_flag = {"percentile_params": percentile_params}

                    # Get simulation characteristics for this simulation
                    sim_characteristics = get_sim_characteristics(sims)

                    (data, vectorial, induction, magnitudes, induction_energy, induction_energy_integral,
                    induction_test_energy_integral, induction_energy_profiles, induction_uniform,
                    diver_B_percentiles, debug_fields) = process_iteration(
                        components=ind_params["components"],
                        dir_grids=out_params["dir_grids"],
                        dir_gas=out_params["dir_gas"],
                        dir_params=out_params["dir_params"][i],
                        sims=sims,
                        it=it,
                        coords=Coords[i][j],
                        region_coords=Region_Coord[i][j],
                        rad=Rad[i][j],
                        rmin=ind_params["rmin"][i],
                        level=ind_params["level"][L],
                        up_to_level=ind_params["level"][L],
                        nmax=ind_params["nmax"][i],
                        size=ind_params["size"][i],
                        H0=ind_params["H0"],
                        a0=ind_params["a0"],
                        test=ind_params["test_params"],
                        units=ind_params["units"],
                        nbins=ind_params["nbins"][i],
                        logbins=ind_params["logbins"],
                        stencil=diff_params["stencil"],
                        buffer=diff_params["buffer"],
                        use_siblings=diff_params["use_siblings"],
                        interpol=diff_params["interpol"],
                        nghost=diff_params["nghost"],
                        blend=diff_params.get("blend", False),
                        parent=diff_params.get("parent", False),
                        parent_interpol=diff_params.get("parent_interpol", diff_params["interpol"]),
                        bitformat=out_params["bitformat"],
                        mag=return_params["mag"],
                        sim_characteristics=sim_characteristics,
                        energy_evolution_config=ind_params["energy_evolution"],
                        energy_evolution=ind_params["energy_evolution"]["enabled"],
                        profiles=return_params["profiles"] if it in profile_idx_set else False,
                        projection=return_params["projection"],
                        percentiles=percentile_params_cfg["enabled"],
                        percentile_levels=percentile_params_cfg["percentile_levels"],
                        divergence_filter=ind_params.get("divergence_filter"),
                        debug_params=debug_params_flag,
                        production_dissipation=ind_params.get("production_dissipation"),
                        return_vectorial=return_params["return_vectorial"],
                        return_induction=return_params["return_induction"],
                        return_induction_energy=return_params["return_induction_energy"],
                        gc_worker_end=out_params.get("memory_profiling", {}).get("gc_worker_end", False),
                        verbose=out_params["verbose"])
                    
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

                    # Current index for this iteration in aggregated arrays
                    iter_idx = len(all_data.get('grid_time', []))

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
                            if key not in all_induction_energy_profiles:
                                all_induction_energy_profiles[key] = []
                            all_induction_energy_profiles[key].append(induction_energy_profiles[key])
                        profile_indices.append(iter_idx)
                    
                    if diver_B_percentiles is not None:
                        for key in diver_B_percentiles.keys():
                            all_diver_B_percentiles[key].append(diver_B_percentiles[key])
                    
                    if induction_uniform is not None:
                        for key in induction_uniform.keys():
                            all_induction_uniform[key].append(induction_uniform[key])
                    
                    if debug_fields is not None:
                        for key in debug_fields.keys():
                            if key not in all_debug_fields:
                                all_debug_fields[key] = []
                            all_debug_fields[key].append(debug_fields[key])
                        if '_scan_volume' in debug_fields:
                            scan_meta_sim.append(sims)
                            scan_meta_it.append(it)
                            scan_meta_idx.append(iter_idx)

                    iteration_counter += 1
                    if memory_monitor.gc_main_each_iteration:
                        gc.collect()
                    memory_monitor.log(
                        f"serial level={ind_params['level'][L]} sim={sims} it={it} processed={iteration_counter}",
                        force=memory_monitor.should_log_iteration(iteration_counter)
                    )
                    del data, vectorial, induction, magnitudes, induction_energy
                    del induction_energy_integral, induction_test_energy_integral
                    del induction_energy_profiles, induction_uniform, diver_B_percentiles, debug_fields

            # Enforce consistent temporal ordering before any evolution/plot step.
            debug_module.ensure_temporal_order(
                all_data,
                series_dicts=[
                    all_vectorial,
                    all_induction,
                    all_magnitudes,
                    all_induction_energy,
                    all_induction_energy_integral,
                    all_induction_test_energy_integral,
                    all_induction_energy_profiles,
                    all_induction_uniform,
                    all_diver_B_percentiles,
                    all_debug_fields,
                ],
                index_lists=[profile_indices, scan_meta_idx],
                verbose=out_params["verbose"],
            )

            # field = np.abs(np.sqrt(induction_uniform['uniform_MIE_compres_x']**2 + 
            #                 induction_uniform['uniform_MIE_compres_y']**2 + 
            #                 induction_uniform['uniform_MIE_compres_z']**2))
            
            # print(field.shape)
            
            # scan_animation_3D(field, Region_Size[i+j], study_box=1, depth=ind_params["up_to_level"], arrow_scale=1, units='Mpc', 
            #         title=f'Magnetic Field Induction Compression Scan - Level {ind_params["up_to_level"]}', verbose=out_params["verbose"], 
            #         Save=True, DPI=out_params["dpi"], run=out_params["run"] + f'_Level_{ind_params["up_to_level"]}', folder=out_params["image_folder"])
            
            # Actual evolution calculation
            if ind_params["energy_evolution"]["enabled"] == False:
                print("Energy evolution calculation is disabled in the configuration. Skipping evolution plots.")
            else:
                induction_energy_integral_evo = induction_energy_integral_evolution(
                    ind_params["components"], all_induction_energy_integral,
                    ind_params['energy_evolution']['derivative'],
                    all_data['rho_b'], all_data['grid_time'], all_data['grid_zeta'],
                    normalized=ind_params['energy_evolution'].get('normalized', False),
                    verbose=out_params["verbose"])
                
                # Create local params with current level for plotting
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = ind_params["level"][L]
                
                plot_integral_evolution(
                    induction_energy_integral_evo,
                    evo_plot_params, plot_ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    utils.get_last_rad(Rad), verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )

                print(f"Ploting " + evo_plot_params["title"] + " completed.")

            if ind_params.get("production_dissipation", {}).get("enabled", False) == False:
                print("Production/dissipation analysis is disabled in the configuration. Skipping P/D plots.")
            else:
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = ind_params["level"][L]
                pd_figs = plot_production_dissipation_evolution(
                    all_induction_energy_integral,
                    prod_diss_plot_params,
                    plot_ind_params,
                    all_data['grid_time'],
                    all_data['grid_zeta'],
                    verbose=out_params['verbose'],
                    save=out_params['save'],
                    folder=out_params['image_folder']
                )
                if pd_figs:
                    print(f"Ploting " + prod_diss_plot_params["title"] + " completed.")
                else:
                    print(f"Ploting " + prod_diss_plot_params["title"] + " skipped (no valid integrated P/D data).")
            
            if return_params["profiles"] == False:
                print("Profiles calculation is disabled in the configuration. Skipping profile plots.")
            else:
                # Create local params with current level for plotting
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = ind_params["level"][L]
                # Determine how many snapshots have profiles calculated
                num_snapshots_with_profiles = len(all_induction_energy_profiles.get('clus_b2_profile', []))
                if num_snapshots_with_profiles > 0:
                    # Adjust it_indx to plot all snapshots with profiles
                    prof_plot_params_adjusted = prof_plot_params.copy()
                    prof_plot_params_adjusted['it_indx'] = profile_indices
                    plot_radial_profiles(
                        all_induction_energy_profiles,
                        prof_plot_params_adjusted, plot_ind_params,
                        all_data['grid_time'], all_data['grid_zeta'],
                        utils.get_last_rad(Rad), verbose=out_params['verbose'], save=out_params['save'],
                        folder=out_params['image_folder']                
                    )
                    
                    print(f"Ploting " + prof_plot_params["title"] + " completed.")
            
            if percentile_params_cfg["enabled"] == False:
                print("Percentiles calculation is disabled in the configuration. Skipping percentile evolution plots.")
            else:
                # Create local params with current level for plotting
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = ind_params["level"][L]
                plot_percentile_evolution(
                    all_diver_B_percentiles,
                    percentile_plot_params, plot_ind_params,
                    all_data['grid_time'], all_data['grid_zeta'],
                    verbose=out_params['verbose'], save=out_params['save'],
                    folder=out_params['image_folder']
                )
                
                print(f"Ploting " + percentile_plot_params["title"] + " completed.")
                
            if debug_params.get("field_analysis", {}).get("enabled", False) and ind_params["components"]["divergence"]:
                # Create local params with current level for plotting
                plot_ind_params = ind_params.copy()
                plot_ind_params["up_to_level"] = ind_params["level"][L]
                inv_resolution = [1.0 / np.array(all_data['resolution'][i]) for i in range(len(all_data['resolution']))]
                
                for field_key, ref_key, ref_scale_val, quantity in zip(['clus_B2', 'diver_B', 'MIE_diver_x', 'MIE_diver_y', 'MIE_diver_z', 'MIE_diver_B2'],
                                                        ['clus_B2', 'clus_B', 'clus_Bx', 'clus_By', 'clus_Bz', 'clus_B2'],
                                                        [1.0, inv_resolution, 1.0, 1.0, 1.0, 1.0],
                                                        debug_params.get("field_analysis", {}).get("quantities", [])):
                    if field_key not in all_debug_fields or ref_key not in all_debug_fields:
                        if out_params["verbose"]:
                            print(f"Skipping " + quantity + f": field {field_key} or {ref_key} not available")
                        continue
                    distribution_check(all_debug_fields[field_key], quantity, debug_params, plot_ind_params,
                                    all_data['grid_time'], all_data['grid_zeta'],
                                    utils.get_last_rad(Rad), ref_field=all_debug_fields[ref_key], ref_scale=ref_scale_val,
                                    clean=debug_params.get("field_analysis", {}).get("clean_field", False), verbose=out_params["verbose"], save=out_params["save"],
                                    folder=out_params["image_folder"])
                    print(f"Ploting distribution check for " + quantity + " completed.")

            # Render scan animations after data processing (serial mode)
            if debug_params.get("scan_animation", {}).get("enabled", False) and all_debug_fields and '_scan_volume' in all_debug_fields:
                scan_volumes = all_debug_fields['_scan_volume']
                scan_volumes_no_buffer = all_debug_fields.get('_scan_volume_no_buffer', None)
                scan_region_sizes = all_debug_fields['_scan_region_size']
                
                for vol_idx, volume in enumerate(scan_volumes):
                    ind_meta = ind_params.copy()
                    
                    sim = scan_meta_sim[vol_idx] if vol_idx < len(scan_meta_sim) else out_params["sims"][0]
                    fallback_it = out_params["it"][0][0] if out_params["it"] and out_params["it"][0] else 0
                    it_val = scan_meta_it[vol_idx] if vol_idx < len(scan_meta_it) else fallback_it
                    idx = scan_meta_idx[vol_idx] if vol_idx < len(scan_meta_idx) else vol_idx
                    # In serial mode, all_data stores lists not dicts by sim
                    zeta_val = all_data['grid_zeta'][idx]
                    time_val = all_data['grid_time'][idx]
                    region_size = scan_region_sizes[vol_idx]
                    
                    ind_meta['sim'] = sim
                    ind_meta['it'] = it_val
                    ind_meta['zeta'] = zeta_val
                    ind_meta['time'] = time_val
                    ind_meta['level'] = ind_params["level"][L]
                    ind_meta['up_to_level'] = ind_params["up_to_level"][L]
                    
                    # Prepare volume parameters
                    vol_meta = {
                        'vol_idx': vol_idx,
                        'reg_idx': 0
                    }
                    
                    # Prepare scan plot parameters with optional AMR levels for debug visualization
                    scan_plot_params_with_amr = scan_plot_params.copy()
                    scan_plot_params_with_amr['amr_levels'] = all_debug_fields.get('_scan_volume_levels', np.array([]))
                    
                    # Call scan animation with config parameters (WITH buffer)
                    scan_animation_3D(
                        volume, region_size,
                        scan_plot_params_with_amr,
                        ind_meta,
                        volume_params=vol_meta,
                        verbose=debug_params["scan_animation"]["verbose"],
                        save=debug_params["scan_animation"]["save"],
                        folder=out_params["image_folder"]
                    )

                    # Also render without buffer for comparison if available
                    if scan_volumes_no_buffer is not None and vol_idx < len(scan_volumes_no_buffer):
                        ind_meta['buffer'] = False  # Mark as no-buffer version
                        scan_plot_params_with_amr['title'] = f"{scan_plot_params.get('title', 'Field Scan')} No Buffer"

                        scan_animation_3D(
                            scan_volumes_no_buffer[vol_idx], region_size,
                            scan_plot_params_with_amr,
                            ind_meta,
                            volume_params=vol_meta,
                            verbose=debug_params["scan_animation"]["verbose"],
                            save=debug_params["scan_animation"]["save"],
                            folder=out_params["image_folder"]
                        )

            if memory_monitor.gc_main_each_iteration:
                gc.collect()
            memory_monitor.log(f"after serial level={ind_params['level'][L]} post-processing", force=True)
                    
            
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

memory_monitor.log("final", force=True)
memory_monitor.stop()
memory_monitor.summary(out_params)
    
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
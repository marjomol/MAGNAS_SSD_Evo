"""
MAGNAS SSD Evolution
parallel_utils module
Utilities for parallel processing with organized output logging.

Created by Marco Molina Pradillo
"""

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from scripts.induction_evo import process_iteration


def process_iteration_with_logging(components, dir_grids, dir_gas, dir_params,
                                    sims, it, coords, region_coords, rad, rmin, level, up_to_level,
                                    nmax, size, H0, a0, test, units=1, nbins=25, logbins=True,
                                    stencil=3, buffer=True, use_siblings=True, interpol='TSC', use_parent_diff=True, nghost=1, bitformat=None, mag=False,
                                    energy_evolution=True, profiles=True, projection=True, percentiles=True, 
                                    percentile_levels=(100, 90, 75, 50, 25), debug=None,
                                    return_vectorial=False, return_induction=False, return_induction_energy=False,
                                    verbose=False):
    '''
    Wrapper around process_iteration that captures stdout and stderr,
    returning them along with the computation results.
    
    Args:
        Same as process_iteration
        
    Returns:
        Tuple of (results_tuple, log_output)
        where results_tuple contains all the original return values
        and log_output is a string with all captured stdout
        
    Author: Marco Molina Pradillo
    '''
    
    if debug is None:
        debug = [False, 0]
    
    # Create string buffers to capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Redirect stdout and stderr to capture all print statements
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            results = process_iteration(
                components=components,
                dir_grids=dir_grids,
                dir_gas=dir_gas,
                dir_params=dir_params,
                sims=sims,
                it=it,
                coords=coords,
                region_coords=region_coords,
                rad=rad,
                rmin=rmin,
                level=level,
                up_to_level=up_to_level,
                nmax=nmax,
                size=size,
                H0=H0,
                a0=a0,
                test=test,
                units=units,
                nbins=nbins,
                logbins=logbins,
                stencil=stencil,
                buffer=buffer,
                use_siblings=use_siblings,
                interpol=interpol,
                use_parent_diff=use_parent_diff,
                nghost=nghost,
                bitformat=bitformat,
                mag=mag,
                energy_evolution=energy_evolution,
                profiles=profiles,
                projection=projection,
                percentiles=percentiles,
                percentile_levels=percentile_levels,
                debug=debug,
                return_vectorial=return_vectorial,
                return_induction=return_induction,
                return_induction_energy=return_induction_energy,
                verbose=verbose
            )
    except Exception as e:
        log_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        sys.stderr.write(f"\n[ERROR in iteration {it} of simulation {sims}]\n")
        sys.stderr.write(f"Exception: {str(e)}\n")
        sys.stderr.write(f"Captured stdout:\n{log_output}\n")
        sys.stderr.write(f"Captured stderr:\n{stderr_output}\n")
        raise
    
    # Get the captured output
    log_output = stdout_capture.getvalue()
    
    # Return results along with the captured logs
    return (results, log_output, (sims, it))

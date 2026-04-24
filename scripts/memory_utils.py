"""
MAGNAS SSD Evolution
memory_utils module
Utilities for memory monitoring and parallel iteration wrappers.

Created by Marco Molina Pradillo
"""

import io
import os
import sys
import threading
import inspect
import concurrent.futures as cf
from contextlib import redirect_stdout, redirect_stderr

import psutil

from scripts.induction_evo import process_iteration


class MemoryMonitor:
    def __init__(self, out_params):
        self._cfg = out_params.get("memory_profiling", {})
        self.enabled = bool(self._cfg.get("enabled", False))
        self.log_interval = max(1, int(self._cfg.get("log_interval", 1)))
        self.gc_main_each_iteration = bool(self._cfg.get("gc_main_each_iteration", True))
        self.include_children = bool(self._cfg.get("include_children", True))
        self.sample_seconds = max(0.1, float(self._cfg.get("sample_seconds", 0.5)))

        self._proc = psutil.Process(os.getpid()) if self.enabled else None
        self.peak_main_gb = 0.0
        self.peak_children_gb = 0.0
        self.peak_total_gb = 0.0
        self._sampler_stop = threading.Event() if self.enabled else None
        self._sampler_thread = None

    def _rss_gb(self):
        if self._proc is None:
            return 0.0
        return self._proc.memory_info().rss / (1024 ** 3)

    def _children_rss(self):
        if self._proc is None or not self.include_children:
            return 0.0, 0, 0.0, 0.0

        total_children_rss = 0
        n_children = 0
        max_child_gb = 0.0

        try:
            children = self._proc.children(recursive=True)
        except Exception:
            return 0.0, 0, 0.0, 0.0

        for child in children:
            try:
                child_rss = child.memory_info().rss
                total_children_rss += child_rss
                n_children += 1
                max_child_gb = max(max_child_gb, child_rss / (1024 ** 3))
            except Exception:
                continue

        total_children_gb = total_children_rss / (1024 ** 3)
        avg_child_gb = (total_children_gb / n_children) if n_children > 0 else 0.0
        return total_children_gb, n_children, avg_child_gb, max_child_gb

    def _update_peaks(self):
        main_gb = self._rss_gb()
        children_gb, n_children, avg_child_gb, max_child_gb = self._children_rss()
        total_gb = main_gb + children_gb

        self.peak_main_gb = max(self.peak_main_gb, main_gb)
        self.peak_children_gb = max(self.peak_children_gb, children_gb)
        self.peak_total_gb = max(self.peak_total_gb, total_gb)

        return main_gb, children_gb, total_gb, n_children, avg_child_gb, max_child_gb

    def _sampler_loop(self):
        while self._sampler_stop is not None and not self._sampler_stop.is_set():
            if self.enabled:
                self._update_peaks()
            self._sampler_stop.wait(self.sample_seconds)

    def start(self):
        if not self.enabled:
            return
        if self._sampler_thread is not None:
            return
        self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._sampler_thread.start()

    def stop(self):
        if not self.enabled:
            return
        if self._sampler_stop is not None:
            self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=1.0)

    def log(self, stage, force=False):
        if not self.enabled or not force:
            return

        main_gb, children_gb, total_gb, n_children, avg_child_gb, max_child_gb = self._update_peaks()
        print(
            f"[MEM] {stage}: main={main_gb:.2f} GB, children={children_gb:.2f} GB "
            f"(n={n_children}, avg={avg_child_gb:.2f} GB, max={max_child_gb:.2f} GB), "
            f"total={total_gb:.2f} GB (peak_total={self.peak_total_gb:.2f} GB)"
        )

    def should_log_iteration(self, iteration_counter):
        return iteration_counter % self.log_interval == 0

    def summary(self, out_params):
        if not self.enabled:
            return

        print(
            f"[MEM] Peak summary: main={self.peak_main_gb:.2f} GB, "
            f"children={self.peak_children_gb:.2f} GB, total={self.peak_total_gb:.2f} GB"
        )

        estimated_peak_parallel_gb = float(out_params.get("estimated_peak_parallel_gb", 0.0) or 0.0)
        estimated_workers = int(out_params.get("estimated_parallel_workers", out_params.get("ncores", 1)) or 1)
        current_safety_factor = float(out_params.get("memory_safety_factor", 1.0) or 1.0)

        if estimated_peak_parallel_gb > 0.0 and self.peak_total_gb > 0.0:
            observed_to_estimated_ratio = self.peak_total_gb / estimated_peak_parallel_gb
            suggested_safety_factor = max(1.0, current_safety_factor * observed_to_estimated_ratio)
            print(
                f"[MEM] Auto-calibration: estimated_peak_parallel={estimated_peak_parallel_gb:.2f} GB "
                f"(workers={estimated_workers}), observed_peak_total={self.peak_total_gb:.2f} GB"
            )
            print(
                f"[MEM] Suggested memory_safety_factor={suggested_safety_factor:.2f} "
                f"(current={current_safety_factor:.2f}, observed/estimated={observed_to_estimated_ratio:.2f}x)"
            )
            print(
                "[MEM] Tip: set OUTPUT_PARAMS['memory_safety_factor'] to this value for better core recommendations"
            )


def build_executor_kwargs(ncores, max_tasks_per_child=None):
    kwargs = {"max_workers": ncores}
    if max_tasks_per_child is None:
        return kwargs

    if "max_tasks_per_child" in inspect.signature(cf.ProcessPoolExecutor).parameters:
        kwargs["max_tasks_per_child"] = int(max_tasks_per_child)
    else:
        print("[MEM] Warning: max_tasks_per_child is not supported by this Python version")

    return kwargs


def process_iteration_with_logging(components, dir_grids, dir_gas, dir_params,
                                    sims, it, coords, region_coords, rad, rmin, level, up_to_level,
                                    nmax, size, H0, a0, test, units=1, nbins=25, logbins=True,
                                    stencil=3, buffer=True, use_siblings=True, interpol='TSC', nghost=1, blend=False,
                                    parent=False, parent_interpol=None,
                                    bitformat=None, mag=False, sim_characteristics=None,
                                    energy_evolution_config=None,
                                    energy_evolution=True, profiles=True, induction_profiles=None, pd_profiles=False,
                                    projection=True, percentiles=True,
                                    percentile_levels=(100, 90, 75, 50, 25), divergence_filter=None, debug_params=None,
                                    production_dissipation=None,
                                    return_vectorial=False, return_induction=False, return_induction_energy=False,
                                    gc_worker_end=False, verbose=False):
    if debug_params is None:
        debug_params = {}

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
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
                nghost=nghost,
                blend=blend,
                parent=parent,
                parent_interpol=parent_interpol,
                bitformat=bitformat,
                mag=mag,
                sim_characteristics=sim_characteristics,
                energy_evolution_config=energy_evolution_config,
                energy_evolution=energy_evolution,
                profiles=profiles,
                induction_profiles=induction_profiles,
                pd_profiles=pd_profiles,
                projection=projection,
                percentiles=percentiles,
                percentile_levels=percentile_levels,
                divergence_filter=divergence_filter,
                debug_params=debug_params,
                production_dissipation=production_dissipation,
                return_vectorial=return_vectorial,
                return_induction=return_induction,
                return_induction_energy=return_induction_energy,
                gc_worker_end=gc_worker_end,
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

    log_output = stdout_capture.getvalue()
    return (results, log_output, (sims, it))

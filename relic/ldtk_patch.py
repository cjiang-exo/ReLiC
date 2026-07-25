"""
Monkey-patch for ldtk.LDPSetCreator.init_filters to use multiprocessing.

The original init_filters loops over filters sequentially, which is slow when
there are thousands of bandpass filters. 
 
from relic.ldtk_patch import apply_ldtk_patch
apply_ldtk_patch()
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import List

import astropy.io.fits as pf
import numpy as np
from numpy import array, zeros
from scipy.interpolate import LinearNDInterpolator as NDI

# ---------------------------------------------------------------------------
# Module-level globals set before forking the pool – workers inherit them via
# fork() copy-on-write on Linux.
# ---------------------------------------------------------------------------
_worker_wl: np.ndarray | None = None
_worker_integrand: np.ndarray | None = None
_worker_filters: list | None = None


# --- Fork-compatible worker (Linux) -------------------------------------------
def _integrate_range_fork(args: tuple[int, int]) -> list[np.ndarray]:
    """Integrate [start, end) using globals inherited via fork."""
    start, end = args
    wl = _worker_wl
    integrand = _worker_integrand
    filters = _worker_filters
    return [filters[i].integrate(wl, integrand) for i in range(start, end)]


# --- Spawn-compatible worker (macOS / Windows) --------------------------------
def _integrate_range_data(args: tuple) -> list[np.ndarray]:
    """Integrate [start, end) receiving wl, integrand, filters per task."""
    start, end, wl, integrand, filters = args
    return [filters[i].integrate(wl, integrand) for i in range(start, end)]

# --- Rewrite LDPSetCreator.init_filters ---------------------------------------
def _parallel_init_filters(
    self,
    filters: List,
    n_workers: int | None = None,
    chunk_size: int | None = None,
):
    """Parallel replacement for LDPSetCreator.init_filters.

    Parameters
    ----------
    filters : list
        List of ldtk Filter instances.
    n_workers : int, optional
        Number of worker processes.  Defaults to ``os.cpu_count()``.
    chunk_size : int, optional
        Filters per task.  Larger chunks reduce IPC overhead.
    """
    self.filters = filters
    self.nfilters = len(filters)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, self.nfilters)
    if chunk_size is None:
        chunk_size = max(1, self.nfilters // (n_workers * 4))
 
    if self.photon_counting:
        weight = self.wl * self.qe(self.wl)
    else:
        weight = self.qe(self.wl)

    self.fluxes = zeros([self.nfilters, self.nfiles, self.nmu])
 
    _ctx = mp.get_context()
    _use_fork = _ctx.get_start_method() == "fork"

    # --- Outer loop over stellar-model files --------
    for did, df in enumerate(self.files):
        if self.save_memory:
            data = pf.getdata(df)  # shape (nwl, nmu)
        else:
            data = self.raw_spectra[did]

        integrand = data * weight  # (nwl, nmu)

        ranges = [
            (i, min(i + chunk_size, self.nfilters))
            for i in range(0, self.nfilters, chunk_size)
        ]

        if _use_fork:
            # --- Linux ---
            global _worker_wl, _worker_integrand, _worker_filters
            _worker_wl = self.wl
            _worker_integrand = integrand
            _worker_filters = filters

            with _ctx.Pool(n_workers) as pool:
                chunk_results = pool.map(_integrate_range_fork, ranges,
                                         chunksize=1)
        else:
            # --- macOS / Windows ---
            tasks = [(s, e, self.wl, integrand, filters) for s, e in ranges]
            with _ctx.Pool(n_workers) as pool:
                chunk_results = pool.map(_integrate_range_data, tasks,
                                         chunksize=1)

        # Unpack results into the flux array.
        fid = 0
        for chunk in chunk_results:
            for flux in chunk:
                # Normalise at the limb 
                self.fluxes[fid, did, :] = flux / flux[-1]
                fid += 1

    # --- Rebuild interpolators (original code) --------------------------------
    points = array([[f.teff, f.logg, f.z] for f in self.client.files])
    self.itps = [NDI(points, self.fluxes[i, :, :]) for i in range(self.nfilters)]


def apply_ldtk_patch():
    """Replace ``LDPSetCreator.init_filters`` with the parallel version."""
    from ldtk.ldtk import LDPSetCreator

    LDPSetCreator.init_filters = _parallel_init_filters

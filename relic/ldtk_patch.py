"""
Monkey-patch for ldtk.LDPSetCreator.init_filters.

The original ``init_filters`` loops over filters and stellar-model files
sequentially. It takes too long when there are thousands of filters.

This replacement:

1. Pre-computes each filter's transmission on the stellar wavelength grid, rather than once per file.
2. Computes all filter integrals for a file as a single matrix multiplication ``data @ weighted_responses.T``, which is faster than the original implementation.

The runtime is reduced from ~30 minutes to ~20 seconds for ~3000 filters.

from relic.ldtk_patch import apply_ldtk_patch
apply_ldtk_patch()
"""

from __future__ import annotations

from typing import List

import astropy.io.fits as pf
import numpy as np
from numpy import array, zeros
from scipy.interpolate import LinearNDInterpolator as NDI


def _compute_flux(data: np.ndarray, weighted_responses: np.ndarray) -> np.ndarray:
    """Return fluxes of shape (nfilters, nmu) for one spectrum file.

    ``data`` has shape (nmu, nwl); ``weighted_responses`` has shape
    (nfilters, nwl).
    """
    flux = data @ weighted_responses.T  # (nmu, nfilters)
    return flux.T  # (nfilters, nmu)


def _parallel_init_filters(self, filters: List):
    """Vectorised replacement for LDPSetCreator.init_filters.

    Parameters
    ----------
    filters : list
        List of ldtk Filter instances. 
    """ 

    self.filters = filters
    self.nfilters = len(filters)

    if self.photon_counting:
        weight = self.wl * self.qe(self.wl)
    else:
        weight = self.qe(self.wl)

    self.fluxes = zeros([self.nfilters, self.nfiles, self.nmu])

    # Pre-compute filter transmission curves on the stellar wavelength grid.
    # In the original implementation this cubic interpolation was repeated for
    # every file, which dominates the runtime when many files are used.
    filter_responses = np.array([f(self.wl) for f in filters])  # (nfilters, nwl)
    weighted_responses = filter_responses * weight  # (nfilters, nwl)

    # Compute fluxes for all filters and all files with matrix multiplication.
    for did, df in enumerate(self.files):
        if self.save_memory:
            data = pf.getdata(df)
        else:
            data = self.raw_spectra[did]

        # Make sure the wavelength dimension matches
        if data.shape[1] != self.wl.size:
            data = data.T

        flux = _compute_flux(data, weighted_responses)
        self.fluxes[:, did, :] = flux / flux[:, -1][:, None]

    # --- Rebuild interpolators (original code) --------------------------------
    points = array([[f.teff, f.logg, f.z] for f in self.client.files])
    self.itps = [NDI(points, self.fluxes[i, :, :]) for i in range(self.nfilters)]

def apply_ldtk_patch():
    """Replace ``LDPSetCreator.init_filters`` with the vectorised version."""
    from ldtk.ldtk import LDPSetCreator

    LDPSetCreator.init_filters = _parallel_init_filters

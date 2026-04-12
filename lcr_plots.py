import corner 
import numpy as np
import matplotlib.pyplot as pl
import os
from exoiris import TSData

def plot_2dfluxes(exoiris_data: list[TSData], outputdir=""): 
    for i, d in enumerate(exoiris_data):
        fig, ax = pl.subplots(2,1, figsize=(7.2,7.2))
        _t = d.time
        _w = d.wavelength
        ax[0].imshow(d.fluxes, aspect='auto', origin='lower', 
                    extent=[_t.min(), _t.max(), _w.min(), _w.max()],
                    interpolation="none")
        ax[0].set_title('Fluxes')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Wavelength (micron)')
        fig.colorbar(ax[0].images[0], ax=ax[0])

        ax[1].imshow(d.errors, aspect='auto', origin='lower', 
                    extent=[_t.min(), _t.max(), _w.min(), _w.max()],
                    interpolation="none")
        ax[1].set_title('Errors')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Wavelength (micron)')
        fig.colorbar(ax[1].images[0], ax=ax[1])

        transit_limits = d.ephemeris.transit_limits(d.time.mean())
        [ax[0].axvline(tl, ls='--', color='w') for tl in transit_limits]
        [ax[1].axvline(tl, ls='--', color='w') for tl in transit_limits]

        fig.tight_layout()  

        outname = os.path.join(outputdir, f'fluxes_d{i}.png')
        fig.savefig(outname, dpi=100)
        print(f"A preview of 2D fluxes is saved as {outname}.") 

def plot_residuals(exoiris_data: list[TSData], fmod: list[np.ndarray], outputdir=""):
    for i, d in enumerate(exoiris_data):
        fres = d.fluxes - fmod[i][0]
        zres = fres/d.errors

        fig, ax = pl.subplots(2,1, figsize=(6, 6))
        im_f = ax[0].pcolormesh(d.time, d.wavelength, d.fluxes, shading='auto')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Wavelength (micron)') 

        im_z = ax[1].pcolormesh(d.time, d.wavelength, zres, shading='auto', 
                                vmin=-5, vmax=5, cmap='PuOr_r')
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Wavelength (micron)') 
    
        fig.colorbar(im_f, ax=ax[0], label='Fluxes')
        fig.colorbar(im_z, ax=ax[1], label='Residuals (sigma)')
        fig.tight_layout() 

        outname = os.path.join(outputdir, f'residuals_d{i}.png')
        fig.savefig(outname, dpi=100)
        print(f"A preview of residuals is saved as {outname}.")

def plot_corners(samples, labels, outputdir=""):
    fig = corner.corner(
        samples, 
        labels=labels,   
        show_titles=True, title_fmt='.4g',
        plot_datapoints=False, plot_density=True,
        range=0.999*np.ones(samples.shape[1]),
        levels=[0.3935, 0.8647, 0.9889], 
        quantiles=[0.16, 0.5, 0.84])
    outname = os.path.join(outputdir, f'corners.pdf')
    fig.savefig(outname) 
    print(f"A preview of posterior distributions is saved as {outname}.") 
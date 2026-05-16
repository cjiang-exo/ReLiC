import corner 
import numpy as np
import matplotlib.pyplot as pl
import os
# from exoiris import TSDataGroup
from relic_core import ReLic
from numpy import ndarray, savetxt, where, diff, median

def plot_white(relic: ReLic, figname="white_fit.png", dpi=100):
    fig = relic.exoiris.plot_white(figsize=(3 * relic.exoiris.data.size, 7.2))
    outname = os.path.join(relic.cfg['PATH']['output_dir'], figname)
    fig.tight_layout()
    fig.savefig(outname, dpi=dpi)
    print(f"A preview of white light curve fit saved as {outname}.")

def plot_2dfluxes(relic: ReLic, figname="fluxes.png", dpi=100, save:bool=True): 
    for i, d in enumerate(relic.exoiris.data):
        fig, ax = pl.subplots(2,1, figsize=(7.2,7.2))
        _t = d.time
        _w = d.wavelength

        _im0 = ax[0].pcolormesh(_t, _w, d.fluxes, shading='auto', zorder=0)
        _im1 = ax[1].pcolormesh(_t, _w, d.errors, shading='auto', zorder=0)

        disc_cols = where(diff(_t) > 2*median(diff(_t)))[0]
        for _c in disc_cols:
            ax[0].axvspan(_t[_c], _t[_c+1], facecolor='white', lw=0, alpha=1, zorder=1) 
            ax[1].axvspan(_t[_c], _t[_c+1], facecolor='white', lw=0, alpha=1, zorder=1) 

        ax[0].set_title('Fluxes')
        ax[1].set_title('Errors') 
        [a.set_xlabel('Time') for a in ax]
        [a.set_ylabel('Wavelength [micron]') for a in ax]

        fig.colorbar(_im0, ax=ax[0])
        fig.colorbar(_im1, ax=ax[1])

        transit_limits = d.ephemeris.transit_limits(d.time.mean())
        [ax[0].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]
        [ax[1].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]

        fig.tight_layout()  
        if save:
            outname = figname.split('.')[0] + f'_d{i}.' + figname.split('.')[1]
            outname = os.path.join(relic.cfg["PATH"]["output_dir"], outname)
            fig.savefig(outname, dpi=dpi)
            print(f"A preview of 2D fluxes is saved as {outname}.")

def plot_residuals(relic: ReLic, maxlike_params: ndarray, figname:str = "residuals.png", dpi:int=100, save:bool=True):

    fmod = relic.exoiris._tsa.flux_model(maxlike_params, include_baseline=True)

    for i, d in enumerate(relic.exoiris.data):
        fres = d.fluxes - fmod[i]
        zres = fres/d.errors

        fig, ax = pl.subplots(2,1, figsize=(6, 6))

        im_f = ax[0].pcolormesh(d.time, d.wavelength, d.fluxes, shading='auto')
        im_z = ax[1].pcolormesh(d.time, d.wavelength, zres, shading='auto', 
                                vmin=-5, vmax=5, cmap='PuOr_r')
        
        _t = d.time
        disc_cols = where(diff(_t) > 2*median(diff(_t)))[0]
        for _c in disc_cols:
            ax[0].axvspan(_t[_c], _t[_c+1], facecolor='white', lw=0, alpha=1, zorder=1) 
            ax[1].axvspan(_t[_c], _t[_c+1], facecolor='white', lw=0, alpha=1, zorder=1) 

        [a.set_xlabel('Time [BJD]') for a in ax]
        [a.set_ylabel('Wavelength [micron]') for a in ax]

        transit_limits = d.ephemeris.transit_limits(d.time.mean())
        [ax[0].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]
        [ax[1].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]

        fig.colorbar(im_f, ax=ax[0], label='Fluxes')
        fig.colorbar(im_z, ax=ax[1], label='Residuals (sigma)')
        fig.tight_layout() 

        if save:
            outname = figname.split('.')[0] + f'_d{i}.' + figname.split('.')[1]
            outname = os.path.join(relic.cfg["PATH"]["output_dir"], outname)
            fig.savefig(outname, dpi=dpi)
            print(f"A preview of residuals is saved as {outname}.")

def plot_corners(relic: ReLic, truths=None, figname="corners.pdf", save:bool=True):

    postsamples = relic.exoiris._tsa.sampler.flatchain 

    fig = corner.corner(
        postsamples, 
        labels=[p.name for p in relic.exoiris.ps],
        truths=truths,
        show_titles=True, title_fmt='.4g',
        plot_datapoints=False, plot_density=True,
        range=0.999*np.ones(postsamples.shape[1]),
        levels=[0.3935, 0.8647, 0.9889], 
        quantiles=[0.16, 0.5, 0.84])
    
    if save:
        outname = os.path.join(relic.cfg['PATH']['output_dir'], figname)
        fig.savefig(outname) 
        print(f"A preview of posterior distributions is saved as {outname}.") 


def plot_ldprofiles(relic: ReLic, teff:float=None, logg:float=None, 
    metal:float=None, figname:str="ldprofiles.png", dpi:int=100, save:bool=True):
    """ 
    Plot limb darkening profiles for the given stellar parameters. 
    If any parameter is None, it will be taken from the configuration file. 
    """

    _t = teff if teff is not None else relic.cfg['STAR']['teff'][0]
    _g = logg if logg is not None else relic.cfg['STAR']['logg'][0]
    _m = metal if metal is not None else relic.cfg['STAR']['metal'][0]
    _title = r'$T_{{\rm eff}}$={:.0f} K, $\log g$={:.2f}, [Fe/H]={:.2f}'.format(_t, _g, _m)
    outname = os.path.join(relic.cfg['PATH']['output_dir'], figname)

    fig = relic.ldmodel.plot_profiles(_t, _g, _m) 
    fig.axes[0].set_title(_title)
    fig.tight_layout()
    if save:
        fig.savefig(outname, dpi=dpi)
        print(f"A preview of LD profiles is saved as {outname}.")

    return fig

def plot_lnprob_evolution(relic: ReLic, figname: str = "lnprob.png", dpi:int=100, save:bool=True):

    lnp: ndarray = relic.exoiris.sampler.get_log_prob() 
    outputname = os.path.join(relic.cfg['PATH']['output_dir'], 'lnprob.txt')
    savetxt(outputname, lnp)
    print(f"Evolution of posterior probabilities saved as {outputname}.")

    outputname = os.path.join(relic.cfg['PATH']['output_dir'], figname)

    fig, ax = pl.subplots(1,1,figsize=(6,4))
    ax.plot(lnp, c='k', lw=0.5, alpha=0.5) 
    ax.set_xlabel(f'Iterations ({lnp.shape[0]} walkers)')
    ax.set_ylabel('Posterior probability')
    fig.tight_layout() 
    if save:
        fig.savefig(outputname, dpi=dpi)
        print(f"Evolution of posterior probability saved as {outputname}.")
    return fig
 
import corner 
import numpy as np
import matplotlib.pyplot as pl
import os   
from numpy import ndarray, savetxt, where, diff, median, sqrt
from astropy.visualization import ZScaleInterval 
from relic.core import Relic
from relic.utils import SpectrumDownsampler 

NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3


class RelicVisualization:
    """A collection of plotting methods for ReLic results."""

    def __init__(self, relic: Relic, dpi: int = 100, save: bool = True):
        self.relic = relic
        self.dpi = dpi
        self.save = save

    def plot_white(self, figname="white_fit.png"):
        ncols = self.relic.exoiris.data.size
        fig = self.relic.exoiris.plot_white(figsize=(3 * ncols, 7.2))
        fig.tight_layout()
        outname = os.path.join(self.relic.cfg['PATH']['output_dir'], figname)
        if self.save:
            fig.savefig(outname, dpi=self.dpi)
            print(f"A preview of white light curve fit saved as {outname}.")
        return fig

    def plot_2dfluxes(self, figname="fluxes.png"):
        figs = []
        for i, d in enumerate(self.relic.exoiris.data):
            fig, ax = pl.subplots(2, 1, figsize=(4.8, 4.8))
            _t = d.time
            _w = d.wavelength

            zscale = ZScaleInterval(contrast=0.5)
            vmin, vmax = zscale.get_limits(d.fluxes)

            _im0 = ax[0].pcolormesh(_t, _w, d.fluxes, shading='auto', vmin=vmin, vmax=vmax, zorder=0)
            _im1 = ax[1].pcolormesh(_t, _w, d.errors*100, shading='auto', zorder=0)

            disc_cols = where(diff(_t) > 5 * median(diff(_t)))[0]
            for _c in disc_cols:
                ax[0].axvspan(_t[_c], _t[_c + 1], facecolor='white', lw=0, alpha=1, zorder=1)
                ax[1].axvspan(_t[_c], _t[_c + 1], facecolor='white', lw=0, alpha=1, zorder=1)

            ax[0].set_title('Fluxes', fontsize='medium')
            ax[1].set_title('Errors [%]', fontsize='medium')
            [a.set_xlabel('Time') for a in ax]
            [a.set_ylabel('Wavelength [micron]') for a in ax]

            fig.colorbar(_im0, ax=ax[0])
            fig.colorbar(_im1, ax=ax[1])

            transit_limits = d.ephemeris.transit_limits(d.time.mean())
            [ax[0].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]
            [ax[1].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]

            fig.tight_layout()
            figs.append(fig)
            if self.save:
                outname = figname.split('.')[0] + f'_d{i}.' + figname.split('.')[1]
                outname = os.path.join(self.relic.cfg["PATH"]["output_dir"], outname)
                fig.savefig(outname, dpi=self.dpi)
                print(f"A preview of 2D fluxes is saved as {outname}.")
        return figs

    def plot_residuals(self, maxlike_params: ndarray, figname: str = "residuals.png"):

        tsa = self.relic.exoiris._tsa

        fmod = tsa.flux_model(maxlike_params, include_baseline=False)
        ffit = tsa.flux_model(maxlike_params, include_baseline=True)

        figs = []
        for i, d in enumerate(self.relic.exoiris.data):
            if tsa._nm in (NM_WHITE_MARGINALIZED, NM_WHITE_PROFILED):
                wn_multiplier = maxlike_params[tsa._sl_wnm][d.noise_group]
                fresidual = d.fluxes - ffit[i]
                # fdetrend = fresidual + fmod[i]
                zres = fresidual / (d.errors * wn_multiplier)
            elif tsa._nm in (NM_GP_FIXED, NM_GP_FREE):  # using GP
                if tsa._nm == NM_GP_FREE:
                    gp_pv = 10 ** maxlike_params[tsa._sl_gp]
                    for n in self.relic.exoiris.data.noise_groups:
                        tsa.set_gp_hyperparameters(*gp_pv[n * 3:(n + 1) * 3], idata=n)
                fresidual = tsa._gp_flux[i] - fmod[i][tsa.data[i].mask]
                gp_trend = tsa._gp[i].predict(fresidual, tsa._gp_time[i])
                fresidual = (fresidual - gp_trend).reshape(fmod[i].shape)
                # fdetrend = fresidual + fmod[i]
                zres = fresidual / sqrt(tsa._gp[i]._diag.reshape(fmod[i].shape))

            fig, ax = pl.subplots(3, 1, figsize=(4.8, 7.2))

            zscale = ZScaleInterval(contrast=0.5)
            vmin, vmax = zscale.get_limits(d.fluxes)
            im_f = ax[0].pcolormesh(d.time, d.wavelength, d.fluxes, shading='auto', 
                                    vmin=vmin, vmax=vmax)
            vmin, vmax = zscale.get_limits(ffit[i])
            im_m = ax[1].pcolormesh(d.time, d.wavelength, ffit[i], shading='auto', 
                                    vmin=vmin, vmax=vmax)
            im_z = ax[2].pcolormesh(d.time, d.wavelength, zres, shading='auto',
                                    vmin=-5, vmax=5, cmap='RdYlBu_r')

            _t = d.time
            disc_cols = where(diff(_t) > 5 * median(diff(_t)))[0]
            for _c in disc_cols:
                ax[0].axvspan(_t[_c], _t[_c + 1], facecolor='white', lw=0, alpha=1, zorder=1)
                ax[1].axvspan(_t[_c], _t[_c + 1], facecolor='white', lw=0, alpha=1, zorder=1)
                ax[2].axvspan(_t[_c], _t[_c + 1], facecolor='white', lw=0, alpha=1, zorder=1)

            [a.set_xlabel('Time [BJD]') for a in ax]
            [a.set_ylabel('Wavelength [$\mu$m]') for a in ax]

            transit_limits = d.ephemeris.transit_limits(d.time.mean())
            [ax[0].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]
            [ax[1].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]
            [ax[2].axvline(tl, ls='--', color='k', alpha=0.8) for tl in transit_limits]

            fig.colorbar(im_f, ax=ax[0], label='Observed fluxes')
            fig.colorbar(im_m, ax=ax[1], label='Model fluxes')
            fig.colorbar(im_z, ax=ax[2], label='Residuals (sigma)')
            fig.tight_layout()
            figs.append(fig)

            if self.save:
                outname = figname.split('.')[0] + f'_d{i}.' + figname.split('.')[1]
                outname = os.path.join(self.relic.cfg["PATH"]["output_dir"], outname)
                fig.savefig(outname, dpi=self.dpi)
                print(f"A preview of residuals is saved as {outname}.")

        return figs
    
    def plot_corners(self, samples=None, weights=None, truths=None, figname="corners.pdf"):

        if samples is None:
            samples = self.relic.exoiris._tsa.sampler.flatchain

        fig = corner.corner(
            samples,
            labels=[p.name for p in self.relic.exoiris.ps],
            weights=weights,
            truths=truths,
            show_titles=True, title_fmt='.4g',
            plot_datapoints=False, plot_density=True,
            range=0.999 * np.ones(samples.shape[1]),
            levels=[0.3935, 0.8647, 0.9889],
            quantiles=[0.16, 0.5, 0.84])

        if self.save:
            outname = os.path.join(self.relic.cfg['PATH']['output_dir'], figname)
            fig.savefig(outname, dpi=self.dpi)
            print(f"A preview of posterior distributions is saved as {outname}.")
        return fig

    def plot_ldprofiles(self, teff:float=None, logg:float=None,
                        metal:float=None, figname:str="ldprofiles.png"):
        """
        Plot limb darkening profiles for the given stellar parameters.
        If any parameter is None, it will be taken from the configuration file.
        """

        _t = teff if teff is not None else self.relic.cfg['STAR']['teff'][0]
        _g = logg if logg is not None else self.relic.cfg['STAR']['logg'][0]
        _m = metal if metal is not None else self.relic.cfg['STAR']['metal'][0]
        _title = r'$T_{{\rm eff}}$={:.0f} K, $\log g$={:.2f}, [Fe/H]={:.2f}'.format(_t, _g, _m)
        outname = os.path.join(self.relic.cfg['PATH']['output_dir'], figname)

        fig = self.relic.ldmodel.plot_profiles(_t, _g, _m)
        fig.axes[0].set_title(_title)
        fig.tight_layout()
        if self.save:
            fig.savefig(outname, dpi=self.dpi)
            print(f"A preview of limb-darkening profiles is saved as {outname}.")
        return fig

    def plot_mcmc_lnprob(self, figname: str = "lnprob.png"):

        lnp: ndarray = self.relic.exoiris.sampler.get_log_prob()
        outputname = os.path.join(self.relic.cfg['PATH']['output_dir'], 'lnprob.txt')
        savetxt(outputname, lnp)
        print(f"Evolution of posterior probabilities saved as {outputname}.")

        outputname = os.path.join(self.relic.cfg['PATH']['output_dir'], figname)

        fig, ax = pl.subplots(1, 1, figsize=(6, 4))
        ax.plot(lnp, c='k', lw=0.5, alpha=0.5)
        ax.set_xlabel(f'Iterations ({lnp.shape[0]} walkers)')
        ax.set_ylabel('Posterior probability')
        fig.tight_layout()
        if self.save:
            fig.savefig(outputname, dpi=self.dpi)
            print(f"Evolution of posterior probability saved as {outputname}.")
        return fig

    def plot_transmission_spectra(self, maxlike_param:ndarray, 
                                  figname:str="transmission_spectrum.png" ):
        
        wl_model = self.relic.atmos_model.wavelengths
        ts_modelmaxlike = 100 * self.relic.atmos_model(maxlike_param)
 
        wavelengths = [d.wavelength for d in self.relic.tsdata]
        binwidths = [d._wl_r_edges - d._wl_l_edges for d in self.relic.tsdata]
        rpfiles = self.relic.cfg["PATH"]["spec_resolving_power_files"] 
        downsamplers = [
            SpectrumDownsampler(self.relic.atmos_model.wavelengths, wl, bw, rpf) \
                for wl, bw, rpf in zip(
                    wavelengths, 
                    binwidths, 
                    rpfiles
                )
        ]

        wl_lowres = [ds.wl_data for ds in downsamplers]
        ts_lowres = [ds.convolve_and_rebin(ts_modelmaxlike).copy() for ds in downsamplers] 

        fig, ax = pl.subplots(1, 1, figsize=(6, 4))

        _a_highres, = ax.plot(wl_model, ts_modelmaxlike, c='C1', lw=0.5, alpha=0.5, label=f'original model')
        for wl, ts in zip(wl_lowres, ts_lowres): 
            _a_lowres, = ax.plot(wl, ts, lw=1, label='data-resolution', color='#5e3c99')

        ax.set_xlabel('Wavelength [micron]')
        ax.set_ylabel('Transit depth (%)')
        ax.legend([_a_highres, _a_lowres], ['original model', 'data-resolution'], loc='best', fontsize=8)
        if wl_model[-1] / wl_model[0] > 10:
            ax.set_xscale('log')
        fig.tight_layout()
        if self.save:
            outname = os.path.join(self.relic.cfg['PATH']['output_dir'], figname)
            fig.savefig(outname, dpi=self.dpi)
            print(f"A preview of transmission spectra is saved as {outname}.")
        return fig
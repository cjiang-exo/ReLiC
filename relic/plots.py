import corner 
import numpy as np
import matplotlib.pyplot as pl
import os   
from typing import TYPE_CHECKING
from numpy import ndarray, savetxt, where, diff, median, sqrt
from astropy.visualization import ZScaleInterval 
from exoiris import TSData
from pytransit.orbits import fold
from relic.utils import SpectrumDownsampler 

if TYPE_CHECKING:
    from .core import Relic

# Noise-model identifiers 
NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3


class RelicVisualization:
    """Plot ReLic results and return figures after closing them."""

    def __init__(self, relic: 'Relic', dpi: int = 150, savefig: bool = True):
        self.relic = relic
        self.dpi = dpi
        self.save = savefig

    def _save_figure(self, fig, figname, message, suffix=''):
        try:
            if self.save:
                stem, extension = os.path.splitext(figname)
                outname = os.path.join(self.relic.cfg['PATH']['output_dir'], f'{stem}{suffix}{extension}')
                fig.savefig(outname, dpi=self.dpi)
                print(f'{message} {outname}.')
        finally:
            pl.close(fig)

    def plot_white(self, figname="white_fit.png"):
        """Plot the white-light curve fit against folded time in days."""
        white = self.relic.exoiris._wa
        ncols = self.relic.exoiris.data.size
        fig, axs = pl.subplots(3, ncols, figsize=(3 * ncols, 7.2),
                               sharey='row', sharex='col', squeeze=False)
        pv = white.de.minimum_location
        period = pv[0]
        transit_centers = pv[white._sl_tc]
        transit = np.asarray(white.transit_model(pv)).reshape(-1)
        model = np.asarray(white.flux_model(pv)).reshape(-1)
        residuals = white.ofluxa - model
        detrended = transit + residuals
        duration = white.transit_duration

        for index, sl in enumerate(white.lcslices):
            center = transit_centers[self.relic.exoiris.data.epoch_groups[index]]
            time = fold(white.timea[sl], period, center)
            time_diff = np.diff(time)
            breaks = time_diff < 0
            if time_diff.size:
                breaks |= time_diff > 2 * np.median(time_diff)
            segments = np.split(np.arange(time.size), np.where(breaks)[0] + 1)

            axs[0, index].plot(time, white.ofluxa[sl], '.k', alpha=0.25)
            axs[1, index].plot(time, detrended[sl], '.k', alpha=0.25)
            axs[2, index].plot(time, residuals[sl], '.k', alpha=0.25)
            for segment in segments:
                axs[0, index].plot(time[segment], model[sl][segment], 'r', ms=1, zorder=9)
                axs[1, index].plot(time[segment], transit[sl][segment], 'r', ms=1, zorder=9)
            axs[2, index].axhline(0, ls='--', c='r', zorder=9)
            for ax in axs[:, index]: 
                for marker in (0, -0.5 * duration, 0.5 * duration):
                    ax.axvline(marker, ls='--', c='0.5')
            axs[2, index].set(xlabel='Time from mid-transit [d]', xlim=(-2*duration, 2*duration))

        for ax in axs[:2, 0]:
            ax.set_ylabel('Normalized flux')
        axs[2, 0].set_ylabel('Residuals')
        fig.tight_layout()
        self._save_figure(fig, figname, 'A preview of white light curve fit saved as')
        return fig

    def plot_2dfluxes(self, figname="fluxes.png"):
        """Plot the 2D (time & wavelength) fluxes for spectroscopic datasets."""
        figs = []
        for i, d in enumerate(self.relic.exoiris.data):
            if hasattr(d, 'response_function'):
                continue
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
            self._save_figure(fig, figname, 'A preview of 2D fluxes is saved as', suffix=f'_d{i}')
        return figs

    def plot_residuals(self, maxlike_params: ndarray, figname: str = "residuals.png"):
        """Plot best-fit 2D fluxes.
        ax[0]: Observed fluxes.
        ax[1]: Best-fit model fluxes.
        ax[2]: Normalized residuals.
        """

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

            self._save_figure(fig, figname, 'A preview of residuals is saved as', suffix=f'_d{i}')

        return figs
    
    def plot_corners(self, samples=None, weights=None, truths=None, figname="corners.pdf"):
        """ Make a posterior corner plot. """

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

        self._save_figure(fig, figname, 'A preview of posterior distributions is saved as')
        return fig

    def plot_ldprofiles(self, teff:float=None, logg:float=None,
                        metal:float=None, figname:str="ldprofiles.pdf"):
        """
        Plot each spectroscopic dataset and all photometric profiles separately.
        Parameters default to the configuration values. Return the figures as a
        list, with the combined photometry figure last when present.
        """

        _t = teff if teff is not None else self.relic.cfg['STAR']['teff'][0]
        _g = logg if logg is not None else self.relic.cfg['STAR']['logg'][0]
        _m = metal if metal is not None else self.relic.cfg['STAR']['metal'][0]
        _title = r'$T_{{\rm eff}}$={:.0f} K, $\log g$={:.2f}, [Fe/H]={:.2f}'.format(_t, _g, _m)
        model = self.relic.ldmodel
        profiles, _ = model(model.mu, np.array([_t, _g, _m]))
        datasets = self.relic.tsdata
        if isinstance(datasets, TSData):
            datasets = [datasets]

        figs = []
        photometry = []
        for index, (dataset, wlslice) in enumerate(zip(datasets, model.wlslices)):
            dataset_profiles = profiles[0, wlslice, :]
            if hasattr(dataset, 'response_function'):
                photometry.append((dataset.name, dataset_profiles[0]))
                continue

            fig, ax = pl.subplots()
            image = ax.pcolormesh(model.mu, model.wavelength[wlslice], dataset_profiles, shading='auto')
            fig.colorbar(image, ax=ax, label='Surface brightness')
            ax.set(xlabel=r'$\mu$', ylabel=r'Wavelength [$\mu$m]', title=f'{dataset.name}\n{_title}')
            fig.tight_layout()
            figs.append(fig)
            self._save_figure(fig, figname, 'A preview of limb-darkening profiles is saved as', suffix=f'_d{index}')

        if photometry:
            fig, ax = pl.subplots()
            for name, profile in photometry:
                ax.plot(model.mu, profile, label=name)
            ax.set(xlabel=r'$\mu$', ylabel='Surface brightness', title=_title)
            ax.legend()
            fig.tight_layout()
            figs.append(fig)
            self._save_figure(fig, figname, 'A preview of limb-darkening profiles is saved as', suffix='_photometry')
        return figs

    def plot_mcmc_lnprob(self, figname: str = "lnprob.png"):
        """ Plot the MCMC log-probability chains."""

        lnp: ndarray = self.relic.exoiris.sampler.get_log_prob()
        outputname = os.path.join(self.relic.cfg['PATH']['output_dir'], 'lnprob.txt')
        savetxt(outputname, lnp)
        print(f"Evolution of posterior probabilities saved as {outputname}.")

        fig, ax = pl.subplots(1, 1, figsize=(6, 4))
        ax.plot(lnp, c='k', lw=0.5, alpha=0.5)
        ax.set_xlabel(f'Iterations ({lnp.shape[0]} walkers)')
        ax.set_ylabel('Posterior probability')
        fig.tight_layout()
        self._save_figure(fig, figname, 'Evolution of posterior probability saved as')
        return fig

    def plot_transmission_spectra(self, maxlike_param:ndarray, 
                                  figname:str="transmission_spectrum.png" ):
        """ Plot the best-fit transmission spectrum. """
        
        wl_model = self.relic.atmos_model.wavelengths
        ts_modelmaxlike = 100 * self.relic.atmos_model(maxlike_param)  # transit depth [%]
 
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
        self._save_figure(fig, figname, 'A preview of transmission spectra is saved as')
        return fig
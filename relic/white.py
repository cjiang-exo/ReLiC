# import numpy as np
 
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, setp 
from numpy.linalg import lstsq, LinAlgError 
from numpy import ( array, average, atleast_2d, diff, floor, isfinite, isscalar, inf, 
    log10, nan, repeat, sqrt, squeeze, unique, where, ndarray, zeros_like, ones_like
) 
from scipy.optimize import minimize
from pytransit import BaseLPF
from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch
from pytransit.param import GParameter, NormalPrior as NP, UniformPrior as UP, PParameter
from pytransit.lpf.lpf import map_ldc

from .tslpf import NewTSLPF

class NewWhiteLPF(BaseLPF):
    def __init__(self, tsa: NewTSLPF, covariates: list[ndarray] = [None, ...]):
 
        fluxes, times, errors = [], [], []
        for i, (t, f, e, cov) in enumerate(zip(tsa.data.times, tsa.data.fluxes, tsa.data.errors, covariates)):
            weights = where(isfinite(f) & isfinite(e), 1/e**2, 0.0)
            mf = average(where(isfinite(f), f, 0), axis=0, weights=weights)
            me = sqrt(1 / weights.sum(0))
            m = isfinite(mf) & isfinite(me)
            times.append(t[m])
            fluxes.append(mf[m])
            errors.append(me[m])  
            covariates[i] = None if cov is None else cov[m]
        self.std_errors = errors
        self.neps = max(tsa.data.epoch_groups) + 1
        self.covariates = covariates

        pbs = unique(tsa.data.noise_groups).astype('<U21')
        super().__init__('white', pbs, times, fluxes, covariates=self.covariates, 
            wnids=tsa.data.noise_groups, pbids=tsa.data.noise_groups)

        self.tm.epids = array(tsa.data.epoch_groups)

        for i in range(self.neps):
            self.set_prior(f'tc_{i:02d}', tsa.ps[tsa.ps.find_pid(f'tc_{i:02d}')].prior)
        self.set_prior('p', tsa.ps[tsa.ps.find_pid('p')].prior)
        self.set_prior('rho', tsa.ps[tsa.ps.find_pid('rho')].prior)
        self.set_prior('b', tsa.ps[tsa.ps.find_pid('b')].prior) 
        ngids = tsa.data.noise_groups[self.lcids]
        for i in range(tsa.data.n_noise_groups):
            self.set_prior(f'wn_loge_{i}', 'NP', log10(diff(self.ofluxa[ngids==i]).std() / sqrt(2)), 0.1)

        self._baseline_models = ones_like(self.ofluxa)

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, 1))]
        self.ps.add_global_block('orbit', porbit)

        ptc = [GParameter(f'tc_{i:02d}', f'transit_center_{i:02d}', '-', NP(0.0, 0.1), (-inf, inf)) for i in
               range(self.neps)]
        self.ps.add_global_block('tc', ptc)
        self._pid_tc = repeat(self.ps.blocks[-1].start, self.nlc)
        self._start_tc = self.ps.blocks[-1].start
        self._sl_tc = self.ps.blocks[-1].slice

    def _init_p_planet(self):
        """Planet parameter initialisation.
        """ 
        pk2 = [PParameter(f'k2_{pb}', 'area_ratio', 'A_s', UP(0.001, 0.1), (0, inf)) for pb in self.passbands]
        self.ps.add_passband_block('k2', 1, self.npb, pk2)
        self._pid_k2 = repeat(self.ps.blocks[-1].start, self.npb)
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice

    def transit_model(self, pv):
        pv = atleast_2d(pv)
        ldc = map_ldc(pv[:, self._sl_ld])
        zero_epoch = pv[:, self._sl_tc] - self._tref
        period = pv[:, 0]
        smaxis = as_from_rhop(pv[:, 1], period)
        inclination = i_from_ba(pv[:, 2], smaxis)
        radius_ratio = sqrt(pv[:, self._sl_k2])
        return self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)

    def baseline(self, mtransit):
        mtransit = atleast_2d(mtransit)
        npv = mtransit.shape[0] # number of parameter vectors
        if (self._baseline_models is None) or (self._baseline_models.shape[0] != npv):
            self._baseline_models = zeros_like(mtransit)
        for ipv in range(npv):
            for ilc, f in enumerate(self.fluxes):
                res = f / mtransit[ipv, self.lcids == ilc]
                _cov = self.covariates[ilc]
                try:
                    coeffs = lstsq(_cov, res)[0]
                    self._baseline_models[ipv, self.lcids == ilc] = (_cov @ coeffs).T
                except LinAlgError:
                    self._baseline_models[ipv, self.lcids == ilc] = nan

        return self._baseline_models 
    
    def flux_model(self, pv, add_baseline:bool=True):
        mtransit = self.transit_model(pv)
        if add_baseline:
            baseline = self.baseline(mtransit)
        else:
            baseline = ones_like(mtransit)
        return squeeze(baseline * mtransit)

    def lnposterior(self, pv):
        lnp = self.lnprior(pv)

        if isscalar(lnp):
            if isfinite(lnp):
                lnp += self.lnlikelihood(pv)
            return lnp if isfinite(lnp) else -inf 

        mask = isfinite(lnp) 
        lnp[mask] += self.lnlikelihood(pv[mask]) 
        return where(isfinite(lnp), lnp, -inf)

    def optimize(self, pv0=None, method='powell', maxfev: int = 5000):
        if pv0 is None:
            if self.de is not None:
                pv0 = self.de.minimum_location
            else:
                pv0 = self.ps.mean_pv
        res = minimize(lambda pv: -self.lnposterior(pv), pv0, method=method, options={'maxfev':maxfev})
        self._local_minimization = res

    @property
    def transit_center(self):
        pv = self._local_minimization.x
        # pv = self.de.minimum_location
        return pv[3] + pv[0]*epoch(self.times[0].mean(), pv[3], pv[0])

    @property
    def transit_duration(self):
        pv = self._local_minimization.x
        # pv = self.de.minimum_location
        a = as_from_rhop(pv[1], pv[0])
        i = i_from_ba(pv[2], a)
        t14 = d_from_pkaiews(pv[0], sqrt(pv[self._start_k2]), a, i, 0., 0., 1, 14)
        return t14

    def plot(self, axs=None, figsize=None, ncols=2) -> Figure:
        if axs is None:
            nrows, ncols = 3, self.nlc 
            fig, axs = subplots(nrows, ncols, figsize=figsize, sharey='row', sharex='col', squeeze=False, constrained_layout=True)
        else:
            fig = axs[0].get_figure()

        # pv = self._local_minimization.x
        pv = self.de.minimum_location
        mtransit = squeeze(self.transit_model(pv))
        mflux = squeeze(self.flux_model(pv))
        whitenoise = self.ofluxa - mflux
        detrendedflux = mtransit + whitenoise
        t14 = self.transit_duration
        
        for i, sl in enumerate(self.lcslices):
            
            tref = floor(self.timea[sl].min()) 
            tc = pv[3] + pv[0]*epoch(self.times[i].mean(), pv[3], pv[0])

            ax_raw = axs[0][i]
            ax_raw.plot(self.timea[sl] - tref, self.ofluxa[sl], '.k', alpha=0.25)
            ax_raw.plot(self.timea[sl] - tref, mflux[sl], 'r', zorder=9)  

            ax_detrended = axs[1][i]
            ax_detrended.plot(self.timea[sl] - tref, detrendedflux[sl], '.k', alpha=0.25)
            ax_detrended.plot(self.timea[sl] - tref, mtransit[sl], 'r', zorder=9)  

            ax_noise = axs[2][i]
            ax_noise.plot(self.timea[sl] - tref, whitenoise[sl], '.k', alpha=0.25)
            ax_noise.axhline(0, ls='--', c='r', zorder=9)

            setp(ax_noise, xlabel=f'Time - {tref:.0f} [BJD]', xlim=(self.times[i].min()-tref, self.times[i].max()-tref))

            for irow in range(3):
                axs[irow][i].axvline(tc - tref, ls='--', c='0.5')
                axs[irow][i].axvline(tc - tref - 0.5*t14, ls='--', c='0.5')
                axs[irow][i].axvline(tc - tref + 0.5*t14, ls='--', c='0.5')

        setp(axs[:, 0], ylabel='Normalized flux')
        return fig
    
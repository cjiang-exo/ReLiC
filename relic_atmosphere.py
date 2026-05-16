""" 
This is where users define their own atmospheric models.
"""

from numpy import ones, ones_like, full_like, logspace, ndarray, where, ndarray, log, empty_like, convolve, pad, zeros_like
from petitRADTRANS.physical_constants import m_jup, r_jup_mean, r_sun, G as g_const
from petitRADTRANS.radtrans import Radtrans 
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import compute_mean_molar_masses
from petitRADTRANS.physics import temperature_profile_function_guillot_global as get_tprofile  

class BaseAtmosphere:
    def __init__(self, config: dict):
        self.cfg                    = config
        self._sl_atm: slice         = None # slice for atmospheric parameters in the parameter vector
        self.pressures_bar: ndarray = None # pressure in bar
        self.temperatures: ndarray  = None # temperature in K
        self.wavelengths: ndarray   = None # wavelengths in micron 

    def __call__(self, pv: ndarray) -> ndarray:
        """ Given a parameter vector, return a transmission or emission spectra."""
        raise NotImplementedError()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class IsothermalEqChem(BaseAtmosphere):
    def __init__(self, cfg):
        super().__init__(cfg) 

        self.pressures_bar = logspace(
            *cfg["ATMOSPHERE"]["pressure_bounds_log10bar"], 
            cfg["ATMOSPHERE"]["pressure_layers"]
        )
        self.radtrans = Radtrans(
            pressures                  = self.pressures_bar,
            wavelength_boundaries      = cfg["ATMOSPHERE"]["wavelength_bounds_micron"],
            line_species               = cfg["ATMOSPHERE"]["chemical_species"], 
            rayleigh_species           = cfg["ATMOSPHERE"]["rayleigh_species"],
            gas_continuum_contributors = cfg["ATMOSPHERE"]["continuum_species"],
            line_opacity_mode          = cfg["ATMOSPHERE"]["opacity_mode"], 
        )
        self.chem = PreCalculatedEquilibriumChemistryTable() 

        self.mass_fractions = self.chem.interpolate_mass_fractions(
            co_ratios           = full_like(self.pressures_bar, 0.5),
            log10_metallicities = full_like(self.pressures_bar, 0.0),
            temperatures        = full_like(self.pressures_bar, 1000),
            pressures           = self.pressures_bar,
            full                = False
        )

        self.wavelengths      = self.radtrans.get_wavelengths() * 1e4 # micron
        self.planet_radius_cm = cfg["PLANET"]["radius_rjup"][0] * r_jup_mean
        self.star_radius_cm   = cfg["STAR"]["radius_rsun"][0] * r_sun
        self.quench_id        = where(self.pressures_bar >= 5E-8)[0][0]
        self._cgravity        = g_const * m_jup / self.planet_radius_cm**2
        self.temperatures     = zeros_like(self.pressures_bar)

    def __call__(self, pv: ndarray, return_contribution: bool = False) -> ndarray:

        atm_params      = pv[self._sl_atm]
        ref_gravity     = atm_params[0] * self._cgravity # cgs
        ref_pressure    = 10**atm_params[1] # bar
        cloudtop_pbar   = 10**atm_params[2] # bar
        cloud_fraction  = atm_params[3] 

        temperatures = full_like(self.pressures_bar, atm_params[4]) 

        # Assume equilibrium chemistry
        metallicities = full_like(self.pressures_bar, atm_params[5])
        co_ratios = full_like(self.pressures_bar, atm_params[6])
        self.mass_fractions = self.chem.interpolate_mass_fractions(
            co_ratios           = co_ratios,
            log10_metallicities = metallicities,
            temperatures        = temperatures,
            pressures           = self.pressures_bar, 
            full                = False,
        ) 

        # quenching the chemistry above 5e-8 bar
        for sp in self.radtrans._line_species: 
            self.mass_fractions[sp][:self.quench_id] = self.mass_fractions[sp][self.quench_id]

        mmw = compute_mean_molar_masses(self.mass_fractions)

        # allow for partial cloud coverage
        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = self.mass_fractions,
            mean_molar_masses           = mmw,
            reference_gravity           = ref_gravity,
            planet_radius               = self.planet_radius_cm,
            reference_pressure          = ref_pressure,
            opaque_cloud_top_pressure   = cloudtop_pbar,
            cloud_fraction              = cloud_fraction,
            return_contribution         = return_contribution,
        ) 
        transit_depths = (transit_radius_cm / self.star_radius_cm)**2

        if not return_contribution:
            return transit_depths
        return transit_depths, _add
    
class TP6EqChem(BaseAtmosphere):    
    def __init__(self, cfg):
        super().__init__(cfg) 

        self.pressures_bar = logspace(
            *cfg["ATMOSPHERE"]["pressure_bounds_log10bar"], 
            cfg["ATMOSPHERE"]["pressure_layers"]
        )
        self.radtrans = Radtrans(
            pressures                  = self.pressures_bar,
            wavelength_boundaries      = cfg["ATMOSPHERE"]["wavelength_bounds_micron"],
            line_species               = cfg["ATMOSPHERE"]["chemical_species"], 
            rayleigh_species           = cfg["ATMOSPHERE"]["rayleigh_species"],
            gas_continuum_contributors = cfg["ATMOSPHERE"]["continuum_species"],
            line_opacity_mode          = cfg["ATMOSPHERE"]["opacity_mode"], 
        )
        self.chem = PreCalculatedEquilibriumChemistryTable() 

        self.mass_fractions = self.chem.interpolate_mass_fractions(
            co_ratios           = full_like(self.pressures_bar, 0.5),
            log10_metallicities = full_like(self.pressures_bar, 0.0),
            temperatures        = full_like(self.pressures_bar, 1000),
            pressures           = self.pressures_bar,
            full                = False
        )

        self.wavelengths      = self.radtrans.get_wavelengths() * 1e4 # micron
        self.planet_radius_cm = cfg["PLANET"]["radius_rjup"][0] * r_jup_mean
        self.star_radius_cm   = cfg["STAR"]["radius_rsun"][0] * r_sun 
        self._cgravity        = g_const * m_jup / self.planet_radius_cm**2
        self.temperatures     = zeros_like(self.pressures_bar)
        self.boxcar10         = ones(10) / 10.0
        

    def __call__(self, pv: ndarray, return_contribution: bool = False) -> ndarray:

        atm_params      = pv[self._sl_atm]
        ref_gravity     = atm_params[0] * self._cgravity # cgs
        ref_pressure    = 10**atm_params[1] # bar
        cloudtop_pbar   = 10**atm_params[2] # bar
        cloud_fraction  = atm_params[3] 

        temperatures = self.tp6madhu(self.pressures_bar, *atm_params[4:10])

        # Assume equilibrium chemistry
        metallicities = full_like(self.pressures_bar, atm_params[10])
        co_ratios = full_like(self.pressures_bar, atm_params[11])
        self.mass_fractions = self.chem.interpolate_mass_fractions(
            co_ratios           = co_ratios,
            log10_metallicities = metallicities,
            temperatures        = temperatures,
            pressures           = self.pressures_bar, 
            full                = False,
        ) 

        mmw = compute_mean_molar_masses(self.mass_fractions)

        # allow for partial cloud coverage
        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = self.mass_fractions,
            mean_molar_masses           = mmw,
            reference_gravity           = ref_gravity,
            planet_radius               = self.planet_radius_cm,
            reference_pressure          = ref_pressure,
            opaque_cloud_top_pressure   = cloudtop_pbar,
            cloud_fraction              = cloud_fraction,
            return_contribution         = return_contribution,
        ) 
        transit_depths = (transit_radius_cm / self.star_radius_cm)**2

        if not return_contribution:
            return transit_depths
        return transit_depths, _add
    
    def tp6madhu(self, pbar: ndarray, t0: float, a1: float, a2: float, 
            logp1: float, logp2: float, logp3: float) -> ndarray:
        """
        Parametric T-P profile from Madhusudhan & Seager 2009 (2009ApJ...707...24M).

        `pbar` should be in ascending order.
        """

        # t = empty_like(pbar)

        # _ln10 = log(10)
        p1 = 10**logp1
        p2 = 10**logp2
        p3 = 10**logp3

        t1 = t0 + (log(p1 / pbar[0]) / a1)**2 
        t2 = t1 - (log(p1 / p2) / a2)**2
        t3 = t2 + (log(p3 / p2) / a2)**2

        _layer1 = pbar <= p1
        _layer2 = (pbar > p1) & (pbar <= p3)
        _layer3 = pbar > p3

        self.temperatures[_layer1] = t0 + (log(pbar[_layer1] / pbar[0]) / a1)**2
        self.temperatures[_layer2] = t2 + (log(pbar[_layer2] / p2) / a2)**2
        self.temperatures[_layer3] = t3

        tpad = pad(self.temperatures, [5,4], mode='edge')
        self.temperatures = convolve(tpad, self.boxcar10, mode='valid')

        return self.temperatures

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class GuillotEQChem(BaseAtmosphere):
    def __init__(self, cfg, radtrans: Radtrans, chem: PreCalculatedEquilibriumChemistryTable):
        super().__init__(cfg)
        self.radtrans = radtrans
        self.chem = chem

    def __call__(self, pv: ndarray) -> ndarray: 
        raise NotImplementedError()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def calc_ts_prt_guillot(atm_params, atmosphere: Radtrans, 
#     chem: PreCalculatedEquilibriumChemistryTable, 
#     planet_radius_cm: float, star_radius_cm: float,
#     equilibrium_temperature: float, quench_id: int=1,
#     return_contribution=False):

#     planet_mass     = atm_params[0]*m_jup # g
#     ref_pressure    = 10**atm_params[1] # bar
#     cloudtop_pbar   = 10**atm_params[2] # bar
#     cloud_fraction  = atm_params[-1]
#     pres_bar        = atmosphere.pressures*1e-6 # cgs to bar

#     # Assume Guillot's T-P model
#     ref_gravity = g_const * planet_mass / planet_radius_cm**2 
#     temperatures = get_tprofile(
#         pressures               = pres_bar, 
#         infrared_mean_opacity   = 10**atm_params[3],
#         gamma                   = 10**atm_params[4], 
#         gravities               = ref_gravity,
#         intrinsic_temperature   = atm_params[5],
#         equilibrium_temperature = equilibrium_temperature,
#     )
    
#     # Assume equilibrium chemistry
#     metallicities = atm_params[7] * ones_like(pres_bar)
#     co_ratios = atm_params[8] * ones_like(pres_bar)
#     mass_fractions = chem.interpolate_mass_fractions(
#         co_ratios               = co_ratios,
#         log10_metallicities     = metallicities,
#         temperatures            = temperatures,
#         pressures               = pres_bar, 
#         full                    = False,
#     ) 

#     # quenching the chemistry above 1E-7 bar
#     for sp in atmosphere._line_species: 
#         mass_fractions[sp][:quench_id] = mass_fractions[sp][quench_id]

#     mmw = compute_mean_molar_masses(mass_fractions)

#     # allow for partial cloud coverage
#     _, transit_radius_cm, _add = atmosphere.calculate_transit_radii(
#         temperatures                = temperatures,
#         mass_fractions              = mass_fractions,
#         mean_molar_masses           = mmw,
#         reference_gravity           = ref_gravity,
#         planet_radius               = planet_radius_cm,
#         reference_pressure          = ref_pressure,
#         opaque_cloud_top_pressure   = cloudtop_pbar,
#         cloud_fraction              = cloud_fraction,
#         return_contribution         = return_contribution,
#     ) 

#     transit_depths = (transit_radius_cm / star_radius_cm)**2
#     if not return_contribution:
#         return transit_depths
#     return transit_depths, _add

# from astropy.convolution import convolve, Box1DKernel


 

""" 
This is where users define their own atmospheric models.
"""

import os

from astropy import constants as const
from numpy import array, ones, full_like, logspace, ndarray, where, log, empty_like, convolve, pad, zeros_like, zeros
from petitRADTRANS.physical_constants import m_jup, r_jup_mean, r_sun, G as g_const
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import compute_mean_molar_masses
from petitRADTRANS.physics import temperature_profile_function_guillot_global as get_tprofile  
import pyfastchem as fc

KB = const.k_B.cgs.value

class BaseAtmosphere:
    def __init__(self, config: dict):
        self.cfg                    = config
        self._sl_atm: slice         = None # slice for atmospheric parameters in the parameter vector
        self.pressures_bar: ndarray = None # pressure in bar 
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

        _ = self.chem.interpolate_mass_fractions(
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
        mass_fractions = self.chem.interpolate_mass_fractions(
            co_ratios           = co_ratios,
            log10_metallicities = metallicities,
            temperatures        = temperatures,
            pressures           = self.pressures_bar, 
            full                = False,
        ) 

        # quenching the chemistry above 5e-8 bar
        for sp in self.radtrans._line_species: 
            mass_fractions[sp][:self.quench_id] = mass_fractions[sp][self.quench_id]

        mmw = compute_mean_molar_masses(mass_fractions)

        # allow for partial cloud coverage
        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = mass_fractions,
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

class IsothermalFreeChem(BaseAtmosphere):
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

        self.mass_fractions = {
            "H2": full_like(self.pressures_bar, 0.74),
            "He": full_like(self.pressures_bar, 0.25),
        }
        self.mass_fractions.update({
            sp: full_like(self.pressures_bar, 1e-3) for sp in self.radtrans._line_species
        })

        self.wavelengths      = self.radtrans.get_wavelengths() * 1e4 # micron
        self.planet_radius_cm = cfg["PLANET"]["radius_rjup"][0] * r_jup_mean
        self.star_radius_cm   = cfg["STAR"]["radius_rsun"][0] * r_sun 
        self._cgravity        = g_const * m_jup / self.planet_radius_cm**2
        # self.temperatures     = zeros_like(self.pressures_bar)

    def __call__(self, pv: ndarray, return_contribution: bool = False) -> ndarray:

        atm_params      = pv[self._sl_atm]
        ref_gravity     = atm_params[0] * self._cgravity # cgs
        ref_pressure    = 10**atm_params[1] # bar
        cloudtop_pbar   = 10**atm_params[2] # bar
        cloud_fraction  = atm_params[3] 
        haze_factor     = 10**atm_params[4]

        temperatures = full_like(self.pressures_bar, atm_params[5]) 

        for i, sp in enumerate(self.radtrans._line_species):
            self.mass_fractions[sp][:] = full_like(self.pressures_bar, 10**atm_params[6+i])

        _msum = sum([self.mass_fractions[sp][0] for sp in self.radtrans._line_species])
        if _msum < 1.0:
            self.mass_fractions["H2"][:] = full_like(self.pressures_bar, 0.74 * (1 - _msum))
            self.mass_fractions["He"][:] = full_like(self.pressures_bar, 0.25 * (1 - _msum))
        else:
            self.mass_fractions["H2"][:] = full_like(self.pressures_bar, 0.0)
            self.mass_fractions["He"][:] = full_like(self.pressures_bar, 0.0)
            for sp in self.radtrans._line_species:
                self.mass_fractions[sp] /= _msum

        mmw = compute_mean_molar_masses(self.mass_fractions)

        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = self.mass_fractions,
            mean_molar_masses           = mmw,
            reference_gravity           = ref_gravity,
            planet_radius               = self.planet_radius_cm,
            reference_pressure          = ref_pressure,
            opaque_cloud_top_pressure   = cloudtop_pbar,
            cloud_fraction              = cloud_fraction,
            haze_factor                 = haze_factor,
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

        _ = self.chem.interpolate_mass_fractions(
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
        
    def __call__(self, pv: ndarray, return_contribution: bool = False) -> ndarray:

        atm_params      = pv[self._sl_atm]
        ref_gravity     = atm_params[0] * self._cgravity # cgs
        ref_pressure    = 10**atm_params[1] # bar
        cloudtop_pbar   = 10**atm_params[2] # bar
        cloud_fraction  = atm_params[3] 
        haze_factor     = 10**atm_params[4]

        temperatures = tp6madhu(self.pressures_bar, *atm_params[5:10])

        # Assume equilibrium chemistry
        metallicities = full_like(self.pressures_bar, atm_params[10])
        co_ratios = full_like(self.pressures_bar, atm_params[11])
        mass_fractions = self.chem.interpolate_mass_fractions(
            co_ratios           = co_ratios,
            log10_metallicities = metallicities,
            temperatures        = temperatures,
            pressures           = self.pressures_bar, 
            full                = False,
        ) 

        mmw = compute_mean_molar_masses(mass_fractions)

        # allow for partial cloud coverage
        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = mass_fractions,
            mean_molar_masses           = mmw,
            reference_gravity           = ref_gravity,
            planet_radius               = self.planet_radius_cm,
            reference_pressure          = ref_pressure,
            opaque_cloud_top_pressure   = cloudtop_pbar,
            haze_factor                 = haze_factor,
            cloud_fraction              = cloud_fraction,
            return_contribution         = return_contribution,
        ) 
        transit_depths = (transit_radius_cm / self.star_radius_cm)**2

        if not return_contribution:
            return transit_depths
        return transit_depths, _add

class TP6FreeChem(TP6EqChem):
    def __init__(self, cfg):
        super().__init__(cfg) 

        self.mass_fractions = {
            "H2": full_like(self.pressures_bar, 0.74),
            "He": full_like(self.pressures_bar, 0.25),
        }
        self.mass_fractions.update({
            sp: full_like(self.pressures_bar, 1e-3) for sp in self.radtrans._line_species
        })

    def __call__(self, pv: ndarray, return_contribution: bool = False) -> ndarray:

        atm_params      = pv[self._sl_atm]
        ref_gravity     = atm_params[0] * self._cgravity # cgs
        ref_pressure    = 10**atm_params[1] # bar
        cloudtop_pbar   = 10**atm_params[2] # bar
        cloud_fraction  = atm_params[3] 
        haze_factor     = 10**atm_params[4]

        temperatures = tp6madhu(self.pressures_bar, *atm_params[5:10])

        for i, sp in enumerate(self.radtrans._line_species):
            self.mass_fractions[sp][:] = full_like(self.pressures_bar, 10**atm_params[10+i])

        _msum = sum([self.mass_fractions[sp][0] for sp in self.radtrans._line_species])
        if _msum < 1.0:
            self.mass_fractions["H2"][:] = full_like(self.pressures_bar, 0.74 * (1 - _msum))
            self.mass_fractions["He"][:] = full_like(self.pressures_bar, 0.25 * (1 - _msum))
        else:
            self.mass_fractions["H2"][:] = full_like(self.pressures_bar, 0.0)
            self.mass_fractions["He"][:] = full_like(self.pressures_bar, 0.0)
            for sp in self.radtrans._line_species:
                self.mass_fractions[sp] /= _msum

        mmw = compute_mean_molar_masses(self.mass_fractions)

        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = self.mass_fractions,
            mean_molar_masses           = mmw,
            reference_gravity           = ref_gravity,
            planet_radius               = self.planet_radius_cm,
            reference_pressure          = ref_pressure,
            opaque_cloud_top_pressure   = cloudtop_pbar,
            haze_factor                 = haze_factor,
            cloud_fraction              = cloud_fraction,
            return_contribution         = return_contribution,
        ) 
        transit_depths = (transit_radius_cm / self.star_radius_cm)**2

        if not return_contribution:
            return transit_depths
        return transit_depths, _add
    
class TP6FastChem(BaseAtmosphere):    
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

        self.wavelengths      = self.radtrans.get_wavelengths() * 1e4 # micron
        self.planet_radius_cm = cfg["PLANET"]["radius_rjup"][0] * r_jup_mean 
        self.star_radius_cm   = cfg["STAR"]["radius_rsun"][0] * r_sun 
        self._cgravity        = g_const * m_jup / self.planet_radius_cm**2   
        
        if os.path.exists(cfg["FASTCHEM"]["logk"]) and os.path.exists(cfg["FASTCHEM"]["element_abundances"]):
            self.fastchem = fc.FastChem(
                cfg["FASTCHEM"]["element_abundances"],
                cfg["FASTCHEM"]["logk"],
                cfg["FASTCHEM"].get("logk_condensates", "none"),
                0
            )
        else:
            raise FileNotFoundError("FASTCHEM input files not found. Please check the paths in the configuration file.")
        self.fastchem_output = fc.FastChemOutput() 
        self.fastchem_input = fc.FastChemInput()
        self.fastchem_input.pressure = sorted(self.pressures_bar)[::-1] # descending 
        self.fastchem_input.temperature = full_like(self.pressures_bar, 1000) 

        if cfg["FASTCHEM"]["condensation_mode"] == "equilibrium":
            self.fastchem_input.equilibrium_condensation = True
            self.fastchem_input.rainout_condensation = False
        elif cfg["FASTCHEM"]["condensation_mode"] == "rainout":
            self.fastchem_input.equilibrium_condensation = False
            self.fastchem_input.rainout_condensation = True
        else:
            self.fastchem_input.equilibrium_condensation = False
            self.fastchem_input.rainout_condensation = False 

        self.init_abundances = array(self.fastchem.getElementAbundances())
        self.index_C = self.fastchem.getElementIndex('C')
        self.index_O = self.fastchem.getElementIndex('O')
        self.index_M = array([ # indices of heavy elements
            n for n in range(self.fastchem.getElementNumber())
            if self.fastchem.getElementSymbol(n) not in ('H', 'He')
        ])
        self.sum_CO = self.init_abundances[self.index_C] + self.init_abundances[self.index_O]

        _ = self.fastchem.calcDensities(self.fastchem_input, self.fastchem_output)

        hillnotations = ['H2', 'He'] + [self.fastchem.convertToHillNotation(n) for n in cfg["ATMOSPHERE"]["chemical_species"]] 
        self.species_indices = [self.fastchem.getGasSpeciesIndex(n) for n in hillnotations]
        self.species_weights = array([self.fastchem.getGasSpeciesWeight(i) for i in self.species_indices])
 
        n_layers = len(self.pressures_bar)
        n_spec   = len(self.species_indices)
        self._p_over_kb       = self.pressures_bar * 1e6 / KB # ascending pressure
        self._new_abundances = zeros_like(self.init_abundances) # element abundance

        self.mmw              = zeros_like(self.pressures_bar)
        self._tot_mass_density = zeros_like(self.pressures_bar) # total mass density  
        self._gas_num_density = zeros((n_layers, n_spec)) 
        # self._gas_mole_frac   = zeros((n_layers, n_spec)) 
        self._gas_mass_frac   = zeros((n_layers, n_spec)) 
        self._mass_frac_dict  = { 
            n: zeros_like(self.pressures_bar) for n in ['H2', 'He'] + self.radtrans._line_species
        }

    def __call__(self, pv: ndarray, return_contribution: bool = False):

        atm_params      = pv[self._sl_atm]
        ref_gravity     = atm_params[0] * self._cgravity # cgs
        ref_pressure    = 10**atm_params[1] # bar
        cloudtop_pbar   = 10**atm_params[2] # bar
        cloud_fraction  = atm_params[3] 
        haze_factor     = 10**atm_params[4] 

        temperatures = tp6madhu(self.pressures_bar, *atm_params[5:10]) # ascending
        temperatures = temperatures.clip(100, 3400)  
 
        metallicity = 10**atm_params[10]
        co_ratios = atm_params[11] 
 
        mass_fractions = self.get_mass_fractions(metallicity, co_ratios, temperatures)
        if mass_fractions == -1:
            return zeros_like(self.wavelengths) # capture and return null values
 
        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = mass_fractions,
            mean_molar_masses           = self.mmw,
            reference_gravity           = ref_gravity,
            planet_radius               = self.planet_radius_cm,
            reference_pressure          = ref_pressure,
            opaque_cloud_top_pressure   = cloudtop_pbar,
            haze_factor                 = haze_factor,
            cloud_fraction              = cloud_fraction,
            return_contribution         = return_contribution,
        ) 
        transit_depths = (transit_radius_cm / self.star_radius_cm)**2 

        if not return_contribution:
            return transit_depths
        return transit_depths, _add

    def get_mass_fractions(self, metallicity: float, co_ratios: float, temperatures: ndarray) -> dict:

        self._new_abundances[:] = self.init_abundances
        self._new_abundances[self.index_M] *= metallicity
        _c1 = self.sum_CO * metallicity / (1 + co_ratios)
        self._new_abundances[self.index_C] = _c1 * co_ratios 
        self._new_abundances[self.index_O] = _c1 
        
        self.fastchem_input.temperature = temperatures[::-1] # descending
        self.fastchem.setElementAbundances(self._new_abundances)
        _flag = self.fastchem.calcDensities(self.fastchem_input, self.fastchem_output) 
        if _flag != 0:
            return -1 # capture and reject

        self._gas_num_density[:] = array(self.fastchem_output.number_densities)[::-1, self.species_indices] # ascending
        self.mmw[:] = array(self.fastchem_output.mean_molecular_weight)[::-1] # ascending
        
        # temperatures = self.fastchem_input.temperature[::-1] # ascending
        self._tot_mass_density[:] = self._p_over_kb / temperatures * self.mmw # ascending 
        self._gas_mass_frac[:] = self._gas_num_density * self.species_weights[None, :] / self._tot_mass_density[:, None]  # ascending
         
        for i, n in enumerate(self._mass_frac_dict.keys()):
            self._mass_frac_dict[n][:] = self._gas_mass_frac[:, i] + 1e-99
        
        return self._mass_frac_dict

class TP6FastChem_SO2(TP6FastChem):
    def __call__(self, pv: ndarray, return_contribution: bool = False):

        atm_params      = pv[self._sl_atm]
        ref_gravity     = atm_params[0] * self._cgravity # cgs
        ref_pressure    = 10**atm_params[1] # bar
        cloudtop_pbar   = 10**atm_params[2] # bar
        cloud_fraction  = atm_params[3] 
        haze_factor     = 10**atm_params[4] 

        temperatures = tp6madhu(self.pressures_bar, *atm_params[5:10]) # ascending
        temperatures = temperatures.clip(100, 3400)  
 
        metallicity = 10**atm_params[10]
        co_ratios = atm_params[11] 
        x_so2 = 10**atm_params[12]
 
        mass_fractions = self.get_mass_fractions(metallicity, co_ratios, temperatures)
        if mass_fractions == -1:
            return zeros_like(self.wavelengths) # capture and return null values
 
        mass_fractions['SO2'][:] = x_so2
 
        _, transit_radius_cm, _add = self.radtrans.calculate_transit_radii(
            temperatures                = temperatures,
            mass_fractions              = mass_fractions,
            mean_molar_masses           = self.mmw,
            reference_gravity           = ref_gravity,
            planet_radius               = self.planet_radius_cm,
            reference_pressure          = ref_pressure,
            opaque_cloud_top_pressure   = cloudtop_pbar,
            haze_factor                 = haze_factor,
            cloud_fraction              = cloud_fraction,
            return_contribution         = return_contribution,
        ) 
        transit_depths = (transit_radius_cm / self.star_radius_cm)**2 

        if not return_contribution:
            return transit_depths
        return transit_depths, _add
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def tp6madhu(pbar: ndarray, t0: float, lga1: float, lga2: float, 
        lgp1: float, lgp2: float, lgp3: float=0) -> ndarray:
    """
    Parametric T-P profile from Madhusudhan & Seager 2009 (2009ApJ...707...24M).

    `pbar` should be in ascending order.
    """

    temperatures = empty_like(pbar)

    a1 = 10**lga1
    a2 = 10**lga2
    p1 = 10**lgp1
    p2 = 10**lgp2
    p3 = 10**lgp3

    t1 = t0 + (log(p1 / pbar[0]) / a1)**2 
    t2 = t1 - (log(p1 / p2) / a2)**2
    t3 = t2 + (log(p3 / p2) / a2)**2

    _layer1 = pbar <= p1
    _layer2 = (pbar > p1) & (pbar <= p3)
    _layer3 = pbar > p3

    temperatures[_layer1] = t0 + (log(pbar[_layer1] / pbar[0]) / a1)**2
    temperatures[_layer2] = t2 + (log(pbar[_layer2] / p2) / a2)**2
    temperatures[_layer3] = t3

    tpad = pad(temperatures, [5,4], mode='edge')
    temperatures = convolve(tpad, ones(10) / 10.0, mode='valid')

    return temperatures
 

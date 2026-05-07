import numpy as np
from numpy import ones_like, inf
from petitRADTRANS.physical_constants import m_jup, m_sun, r_jup_mean, r_sun, G as grav_const
from petitRADTRANS.radtrans import Radtrans 
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import compute_mean_molar_masses
from petitRADTRANS.physics import temperature_profile_function_guillot_global as get_tprofile
from pytransit.param import UniformPrior as UP, NormalPrior as NP, GParameter  

""" 
Design a base class for general atmospheric models. 
Subclass it for a specific atmospheric model.
"""

atm_ps = [
    GParameter('mp', 'planet_mass', 'M_jup', NP(1.0, 1e-2), (1e-4, inf)),
    GParameter('ref_p', 'reference pressure', 'log10 bar', UP(-10, 2), (-inf, inf)),
    GParameter('cloud_p', 'cloud-top pressure', 'log10 bar', UP(-10, 2), (-inf, inf)), 
    GParameter('kir', 'infrared opacity', 'log10 cm^2/g', UP(-5, 2), (-inf, inf)),
    GParameter('gamma', 'kv/kir', 'log10', UP(-3, 3), (-inf, inf)),
    GParameter('tint', 'intrinsic temperature', 'K', UP(10, 500), (1, inf)),
    GParameter('ab', 'Bond albedo', '', UP(0, 0.9), (0, 1)),
    GParameter('m2h', 'metallicity', 'log10 solar', UP(-1, 3), (-inf, inf)),
    GParameter('c2o', 'C/O ratio', '', UP(0.1, 1.6), (0, inf)),
    GParameter('cloud_f', 'cloud fraction', '', UP(0.0, 1.0), (0, 1)),
    ]

def calc_ts_prt(atm_params, atmosphere: Radtrans, 
    chem: PreCalculatedEquilibriumChemistryTable, 
    planet_radius_cm: float, star_radius_cm: float,
    equilibrium_temperature: float, return_contribution=False):

    planet_mass     = atm_params[0]*m_jup # g
    ref_pressure    = 10**atm_params[1] # bar
    cloudtop_pbar   = 10**atm_params[2] # bar
    cloud_fraction  = atm_params[-1]
    pres_bar        = atmosphere.pressures*1e-6 # cgs to bar

    # Assume Guillot's T-P model
    ref_gravity = grav_const * planet_mass / planet_radius_cm**2 
    temperatures = get_tprofile(
        pressures               = pres_bar, 
        infrared_mean_opacity   = 10**atm_params[3],
        gamma                   = 10**atm_params[4], 
        gravities               = ref_gravity,
        intrinsic_temperature   = atm_params[5],
        equilibrium_temperature = equilibrium_temperature,
    )
    
    # Assume equilibrium chemistry
    metallicities = atm_params[7] * ones_like(pres_bar)
    co_ratios = atm_params[8] * ones_like(pres_bar)
    mass_fractions = chem.interpolate_mass_fractions(
        co_ratios               = co_ratios,
        log10_metallicities     = metallicities,
        temperatures            = temperatures,
        pressures               = pres_bar, 
        full                    = False,
    ) 
    mmw = compute_mean_molar_masses(mass_fractions)

    # allow for partial cloud coverage
    _, transit_radius_cm, _add = atmosphere.calculate_transit_radii(
        temperatures                = temperatures,
        mass_fractions              = mass_fractions,
        mean_molar_masses           = mmw,
        reference_gravity           = ref_gravity,
        planet_radius               = planet_radius_cm,
        reference_pressure          = ref_pressure,
        opaque_cloud_top_pressure   = cloudtop_pbar,
        cloud_fraction              = cloud_fraction,
        return_contribution         = return_contribution,
    ) 

    transit_depths = (transit_radius_cm / star_radius_cm)**2
    if not return_contribution:
        return transit_depths
    return transit_depths, _add
    
def calc_teq(teff, a_rs, albedo=0):
    """Calculate the equilibrium temperature of a planet.
    
    Parameters
    ----------
    teff : float
        Effective temperature of the star (K).
    a_rs : float
        Semi-major axis in units of stellar radii.
    albedo : float, optional
        Bond albedo of the planet, by default 0.
    
    Returns
    -------
    float
        Equilibrium temperature of the planet (K).
    """
    return teff * (0.5 / a_rs)**0.5 * (1 - albedo)**0.25
import numpy as np
from numpy import ones_like
from petitRADTRANS.physical_constants import m_jup, m_sun, r_jup_mean, r_sun, G as grav_const
from petitRADTRANS.radtrans import Radtrans 
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import compute_mean_molar_masses
from petitRADTRANS.physics import temperature_profile_function_guillot_global as get_tprofile

# SMALL_MASS = 1e-6 * m_jup

def calc_ts_prt(atm_params, atmosphere: Radtrans, 
    chem: PreCalculatedEquilibriumChemistryTable, 
    planet_radius_cm: float, star_radius_cm: float,
    equilibrium_temperature: float, ):

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
    metallicities = atm_params[6] * ones_like(pres_bar)
    co_ratios = atm_params[7] * ones_like(pres_bar)
    mass_fractions = chem.interpolate_mass_fractions(
        co_ratios               = co_ratios,
        log10_metallicities     = metallicities,
        temperatures            = temperatures,
        pressures               = pres_bar, 
        full                    = False,
    ) 
    mmw = compute_mean_molar_masses(mass_fractions)

    # allow for partial cloud coverage
    _, transit_radius_cm, _ = atmosphere.calculate_transit_radii(
        temperatures                = temperatures,
        mass_fractions              = mass_fractions,
        mean_molar_masses           = mmw,
        reference_gravity           = ref_gravity,
        planet_radius               = planet_radius_cm,
        reference_pressure          = ref_pressure,
        opaque_cloud_top_pressure   = cloudtop_pbar,
        cloud_fraction              = cloud_fraction,
    ) 

    transit_depths = (transit_radius_cm / star_radius_cm)**2
    return transit_depths
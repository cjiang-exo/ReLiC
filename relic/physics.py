

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
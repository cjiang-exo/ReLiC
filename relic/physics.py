from numpy import empty_like, log, ndarray, pad, convolve, ones, exp

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

def get_temperatures_g10(pbar, lg_kir=-2, lg_gamma=0, beta=0.25, t_int=300, t_star=6000, sma=8.0, gravity=1000.0):
    """
    Guillot (2010) T-P profile (Eq. 29)

    Parameters
    ----------
    pbar : array-like
        Pressure in bar.
    lg_kir : float
        Log10 of the infrared opacity (cm^2/g)
    lg_gamma : float
        Log10 of the ratio of visible to infrared opacity.
    beta : float, optional
        f * (1 - A_bond), default is 0.25
    t_int : float, optional
        Intrinsic temperature in Kelvin, default is 300
    t_star : float, optional
        Stellar effective temperature in Kelvin, default is 6000
    sma : float, optional
        Semi-major axis scaled to stellar radius (a/R_star), default is 8.0
    gravity : float, optional
        Surface gravity in cm/s^2, default is 1000.0

    3**-0.5 = 0.577350 
    """
    gamma = 10**lg_gamma 
    tau = 10**lg_kir * (pbar * 1e6) / gravity 
    t_irr =  t_star * sma**-0.5 

    component1 = 0.75 * t_int**4 * (2.0/3 + tau)
    component2 = 0.75 * beta * t_irr**4 * (2.0/3 + 0.577350 / gamma + (0.577350 * gamma - 0.577350 / gamma) * exp(-gamma * tau / 0.577350))
    temperatures = (component1 + component2)**0.25  
    return temperatures
    
def get_temperatures_m09(pbar: ndarray, t0: float, lga1: float, lga2: float, 
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
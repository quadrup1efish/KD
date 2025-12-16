from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot.simsnap import SimSnap

"""
Units Setting
"""

UnitLength = units.kpc / units.h
UnitMass = 1e10 * units.Msol / units.h
UnitVel = units.km / units.s
UnitComvingLength = units.a * UnitLength
UnitNo = units.no_unit

"""
field units
"""

def TNG_field_units(field: str) -> units.Unit:
    field_units = {
        'Coordinates': UnitComvingLength,
        'Velocities': units.km * units.a ** (1, 2) / units.s,
        'Masses': UnitMass,
        'GFM_Metallicity': UnitNo,
        'ParticleIDs': UnitNo,
        'Potential': (UnitVel) ** 2 / units.a,
        'StarFormationRate': units.Msol / units.yr,
        'BirthPos': UnitComvingLength,
        'BirthVel': units.km * units.a ** (1, 2) / units.s,
        'GFM_StellarFormationTime': UnitNo,
        'GFM_StellarPhotometrics': UnitNo,
        'BH_Mass': UnitMass,
    }
    return field_units[field]

"""
Name Mapping
"""

def NameMapping(field: str) -> str:
    Mapping = {
        'Coordinates': 'pos',
        'Density': 'rho',
        'ParticleIDs': 'iord',
        'Potential': 'phi',
        'Masses': 'mass',
        'Velocities': 'vel',
        'GFM_StellarFormationTime': 'aform',
        'GFM_Metallicity': 'metals',
        'InternalEnergy': 'u',
        'StarFormationRate': 'sfr',
        'BH_Mass': 'mass'
    }
    return Mapping[field]

"""
Eps and mDM
"""

def get_eps_mDM(Subhalo: SimSnap) -> tuple[SimArray, SimArray]:
    """
    Retrieves the gravitational softenings for stars and dark matter (DM) based on the simulation run and redshift.

    Parameters:
    -----------
    Subhalo : object
        An object containing subhalo properties, including `z` (redshift) and `run` (simulation run).

    Returns:
    --------
    eps_star : SimArray
        The gravitational softening length for stars.
    eps_dm : SimArray
        The gravitational softening length for dark matter.
    ------
    'Gravitational softenings for stars and DM are in comoving kpc until z=1,
    after which they are fixed to their z=1 values.' -- Dylan Nelson.
    Data is sourced from https://www.tng-project.org/data/docs/background/.
    """
    MatchRun = {
        'TNG50-1': [0.39, 3.1e5 / 1e10],
        'TNG50-2': [0.78, 2.5e6 / 1e10],
        'TNG50-3': [1.56, 2e7 / 1e10],
        'TNG50-4': [3.12, 1.6e8 / 1e10],
        'TNG100-1': [1., 5.1e6 / 1e10],
        'TNG100-2': [2., 4e7 / 1e10],
        'TNG100-3': [4., 3.2e8 / 1e10],
        'TNG300-1': [2., 4e7 / 1e10],
        'TNG300-2': [4., 3.2e8 / 1e10],
        'TNG300-3': [8., 2.5e9 / 1e10],
        'TNG-Cluster':[2, 6.1e7 / 1e10]
    }

    if Subhalo.properties['Redshift'] > 1:
        return SimArray(
            MatchRun[Subhalo.properties['run']][0], units.a * units.kpc / units.h
        ), SimArray(
            MatchRun[Subhalo.properties['run']][1], 1e10 * units.Msol / units.h
        )
    else:
        return SimArray(
            MatchRun[Subhalo.properties['run']][0] / 2., units.kpc / units.h
        ), SimArray(
            MatchRun[Subhalo.properties['run']][1], 1e10 * units.Msol / units.h
        )

def create_fields(**kwargs) -> dict:

    base = {
        'star': ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs', 'Potential', 'GFM_StellarFormationTime', 'GFM_Metallicity'],
        'gas': ['Coordinates', 'Velocities', 'Masses'],
        'dm': ['Coordinates', 'Velocities', 'Masses'],
    }

    if kwargs:
        base.update(kwargs)

    return base

if __name__ == '__main__':
    default = create_fields()
    #print("default", default)
    partTypes = default.keys()
    for _ in partTypes:
        print("partType:", _)
    custom = create_fields(
        gas=default['gas'] + ['Density'],
        bh=['BH_Mass', 'Potential'],
    )
    print("custom", custom)


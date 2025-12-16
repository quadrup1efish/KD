import numpy as np
from typing import Tuple

import illustris_python as il

import pynbody
from pynbody.array import SimArray
from pynbody.snapshot.simsnap import SimSnap

from TNGsetup import TNG_field_units, NameMapping, get_eps_mDM, create_fields, UnitComvingLength, UnitMass 

def loadGalaxy(basePath: str, run: str, snapNum: int, subID: int, fields: dict | None = None) -> SimSnap:

    # Construct a subhalo using pynbody's SimSnap Class

    header = il.groupcat.loadHeader(basePath, snapNum)
    subset = il.snapshot.getSnapOffsets(basePath=basePath, snapNum=snapNum, id=subID, type='Subhalo')
    lenType = subset['lenType']
    
    if fields is None:
        fields = create_fields()
    partTypes = fields.keys()

    type_map = {'gas': 0, 'dm': 1, 'star': 4, 'bh': 5}
    new_args = {ptype: int(lenType[type_map.get(ptype, 0)]) for ptype in fields}
    subhalo = pynbody.new(**new_args)     

    subhalo.properties['run'] = run
    subhalo.properties['a'] = header['Time']
    subhalo.properties['h'] = header['HubbleParam']
    subhalo.properties['Redshift']= header['Redshift'] 
    subhalo.properties['omegaM0'] = header['Omega0']
    subhalo.properties['omegaL0'] = header['OmegaLambda'] 
    subhalo.properties['boxsize'] = SimArray(header['BoxSize'], UnitComvingLength)

    eps, mDM = get_eps_mDM(subhalo)
    subhalo.properties['mDM'] = SimArray(mDM, UnitMass)
    subhalo.properties['eps'] = SimArray(eps, UnitComvingLength)

    for i in subhalo.properties:
        if isinstance(subhalo.properties[i], SimArray):
            subhalo.properties[i].sim = subhalo

    # Load particles
     
    for partType in partTypes:
        if   partType == 'star' and lenType[4] > 0:
            local_fields = fields[partType]
            loadSubset_data = il.snapshot.loadSubset(basePath, snapNum, partType, local_fields, subset, float32=True)
            for field in local_fields:
                Name = NameMapping(field)
                field_unit = TNG_field_units(field)
                subhalo.s[Name] = SimArray(data=loadSubset_data[field], dtype=np.float32, units=field_unit)
        elif partType == 'gas' and lenType[0] > 0:
            local_fields = fields[partType]
            loadSubset_data = il.snapshot.loadSubset(basePath, snapNum, partType, local_fields, subset, float32=True)
            subhalo.g['phi']  = SimArray(data=np.zeros(len(subhalo.g)), dtype=np.float32, units=TNG_field_units('Potential'))
            for field in local_fields:
                Name = NameMapping(field)
                field_unit = TNG_field_units(field) 
                subhalo.g[Name] = SimArray(data=loadSubset_data[field], dtype=np.float32, units=field_unit) 
        elif partType == 'dm' and lenType[1] > 0:
            local_fields = fields[partType]
            local_fields.remove('Masses')
            loadSubset_data = il.snapshot.loadSubset(basePath, snapNum, partType, local_fields, subset, float32=True)
            local_fields.append('Masses')
            subhalo.dm['phi']  = SimArray(data=np.zeros(len(subhalo.dm)), dtype=np.float32, units=TNG_field_units('Potential'))
            for field in local_fields:
                Name = NameMapping(field)
                field_unit = TNG_field_units(field)
                if field == 'Masses':
                    subhalo.dm[Name] = SimArray(data=np.full(len(subhalo.dm['pos']), mDM), dtype=np.float32, units=field_unit)
                else:
                    subhalo.dm[Name] = SimArray(data=loadSubset_data[field], dtype=np.float32, units=field_unit)
        elif partType == 'bh' and lenType[5] > 0:
            local_fields = fields[partType]
            loadSubset_data = il.snapshot.loadSubset(basePath, snapNum, partType, local_fields, subset, float32=True)
            for field in local_fields:
                Name = NameMapping(field)
                field_unit = TNG_field_units(field)
                subhalo.bh[Name] = SimArray(data=loadSubset_data[field], dtype=np.float32, units=field_unit)    # Preprocessing: physical units
    
    subhalo.physical_units()

    return subhalo

def center(subhalo: SimSnap, mode: str = 'pot') -> SimSnap:
    pynbody.analysis.center(subhalo, mode=mode, wrap=True, with_velocity=True, return_cen=False, cen_size="2 kpc")
    return subhalo

def faceon(subhalo: SimSnap, range: Tuple[float, float] = (2,30)) -> SimSnap:
    rmin, rmax = range[0], range[1]
    selected_region = subhalo.s[(subhalo.s['r'] >= rmin) & (subhalo.s['r'] <= rmax)]

    angmom = pynbody.analysis.angmom.ang_mom_vec(selected_region)
    trans = pynbody.analysis.angmom.calc_faceon_matrix(angmom)
    subhalo.rotate(trans)
    return subhalo

def FindMainProgentiors(basePath, snapNum, subID, fields=None):
    if fields is None:
        fields = ['SnapNum','SubfindID','SubhaloMassType']
    tree = il.sublink.loadTree(basePath, snapNum, subID, fields=fields, onlyMPB=True, treeName="SubLink") 
    return tree

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    run = 'TNG100-3'
    basePath = f"/Users/yuwa/Tools/KD/{run}/output/"
    subID = 1
    snapNum = 99
    #galaxy = loadGalaxy(basePath, run, snapNum, subID)
    #center(galaxy)
    #faceon(galaxy)
    #print(galaxy.all_keys())
    #print(galaxy.s['tform'])
    MP = FindMainProgentiors(basePath, snapNum, subID)
    print(MP.keys())
    #plt.plot(MP['Mass'])
    #plt.show()


    
    

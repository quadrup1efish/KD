import numpy as np
from scipy.interpolate import interp1d
from pynbody.snapshot.simsnap import SimSnap

from derived_array import *


def GravitySolver(galaxy: SimSnap, Solver: str = 'Agama') -> SimSnap:
    pos = galaxy['pos']
    mass= galaxy['mass']
    eps = galaxy.properties['eps']
    if Solver == 'Agama':
        import agama
        agama.setUnits(length=1, mass=1, velocity=1)

        if galaxy['r'].min() == 0: pos += 1e-10
        pot = agama.Potential(type='Multipole', particles=(pos, mass), symmetry='s', smoothing=eps)

        galaxy['phi'] = pot.potential(galaxy['pos'])
        r = galaxy.s['r']    
        pos = galaxy.s['pos']
        
        r_bins = np.logspace(np.log10(0.9*np.min(r[r>0])), np.log10(1.1*np.max(r)), 100)
        r_mid  = 0.5*(r_bins[:-1] + r_bins[1:])
        points = np.column_stack([r_mid, np.zeros_like(r_mid), np.zeros_like(r_mid)])

        mid_pots = pot.potential(points)
        F_R = np.linalg.norm(pot.force(points)[:, :2], axis=1)
        mid_vels = np.sqrt(r_mid * F_R)
        mid_e = 0.5*mid_vels**2 + mid_pots
        j_circ = r_mid * mid_vels

        j_from_E = interp1d(np.log10(-mid_e)[::-1], np.log10(j_circ)[::-1], 
                            fill_value='extrapolate', bounds_error=False)
        jc = 10**j_from_E(np.log10(-galaxy.s['e']))

        jc[galaxy.s['e'] > mid_e.max()] = np.inf
        jc[galaxy.s['e'] < mid_e.min()] = j_circ[0]
        
        galaxy.s['jc'] = SimArray(jc, units=galaxy.s['pos'].units * galaxy.s['vel'].units)
        galaxy.s['e/emin'] = SimArray(galaxy.s['e']/galaxy.s['e'].min().abs(), units="")
        galaxy.s['jz/jc'] = SimArray(galaxy.s['jz']/galaxy.s['jc'], units="")
        galaxy.s['jp/jc'] = SimArray(galaxy.s['jp']/galaxy.s['jc'], units="")
    elif Solver == 'Tree':
        from pytreegrav.frontend import ConstructTree,PotentialTarget,AccelTarget
        G = 4.3009e-6
        eps    = np.repeat(eps, len(mass)).view(np.ndarray)

        kdtree=ConstructTree(pos, mass, softening=eps)
        phi   =PotentialTarget(pos,pos,mass,softening_target=eps,softening_source=eps,G=G,theta=0.5,tree=kdtree,parallel=True,method='tree')

        galaxy['phi'] = phi

        r = galaxy.s['r']    
        pos = galaxy.s['pos']
        
        r_bins = np.logspace(np.log10(0.99*np.min(r[r>0])), np.log10(1.01*np.max(r)), 100)
        rxy_points=0.5 * (r_bins[:-1] + r_bins[1:])
        rs = np.array([position for r in rxy_points for position in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]], dtype=float)
        
        potential = PotentialTarget(rs,pos,mass,eps,eps,method='tree',G=4.302e-6,tree=kdtree,parallel=True)
        accel     = AccelTarget(rs,pos,mass,eps,eps,method='tree',G=4.302e-6,tree=kdtree,parallel=True)

        pots = []
        i = 0
        for r in rxy_points:
            pot = []
            for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
                pot.append(potential[i])
                i = i + 1
            pots.append(np.mean(pot))  
        
        vels = []
        i = 0
        for r in rxy_points:
            r_acc_r = []
            for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
                r_acc_r.append(np.dot(-accel[i, :], pos))
                i = i + 1

            vel2 = np.mean(r_acc_r)
            if vel2 > 0:
                vel = vel2**0.5
            else:
                vel = 0
            vels.append(vel)

        mid_vels = np.array(vels)
        mid_pots = np.array(pots)
        j_circ=rxy_points*mid_vels
        mid_e = 0.5*mid_vels**2 + mid_pots

        j_from_E = interp1d(np.log10(-mid_e)[::-1], np.log10(j_circ)[::-1], 
                            fill_value='extrapolate', bounds_error=False)
        jc = 10**j_from_E(np.log10(-galaxy.s['e']))
        
        jc[galaxy.s['e'] > mid_e.max()] = np.inf
        jc[galaxy.s['e'] < mid_e.min()] = j_circ[0]
        
        galaxy.s['jc'] = SimArray(jc, units=galaxy.s['pos'].units * galaxy.s['vel'].units)
        galaxy.s['e/emin'] = SimArray(galaxy.s['e']/galaxy.s['e'].min().abs(), units="")
        galaxy.s['jz/jc'] = SimArray(galaxy.s['jz']/galaxy.s['jc'], units="")
        galaxy.s['jp/jc'] = SimArray(galaxy.s['jp']/galaxy.s['jc'], units="")
    
    elif Solver == 'direct':
        from pynbody import units
        from pynbody.gravity import direct
        units.G = 4.30091e-6 * units.Unit('kpc Msol**-1 km**2 s**-2')
        
        phi, accel = direct(galaxy, galaxy['pos'].view(np.ndarray),  np.repeat(eps, len(mass)))
        phi = phi.in_units('km**2 s**-2')
        accel = accel.in_units('km s**-2')

        galaxy['phi'] = phi
        galaxy['acc'] = accel

        r = galaxy.s['r']    
        pos = galaxy.s['pos']
        
        r_bins = np.logspace(np.log10(0.99*np.min(r[r>0])), np.log10(1.01*np.max(r)), 100)
        rxy_points=0.5 * (r_bins[:-1] + r_bins[1:])
        rs = np.array([position for r in rxy_points for position in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]], dtype=float)
        potential, accel = direct(galaxy, np.array(rs, dtype=galaxy['pos'].dtype), eps=np.repeat(eps, len(mass)).view(np.ndarray))
        potential = potential.in_units('km**2 s**-2')
        accel = accel.in_units('km**2 kpc**-1 s**-2')
        pots = []
        i = 0
        for r in rxy_points:
            pot = []
            for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
                pot.append(potential[i])
                i = i + 1
            pots.append(np.mean(pot))  
        
        vels = []
        i = 0
        for r in rxy_points:
            r_acc_r = []
            for pos in [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0)]:
                r_acc_r.append(np.dot(-accel[i, :], pos))
                i = i + 1

            vel2 = np.mean(r_acc_r)
            if vel2 > 0:
                vel = vel2**0.5
            else:
                vel = 0
            vels.append(vel)

        mid_vels = np.array(vels)
        mid_pots = np.array(pots)
        j_circ=rxy_points*mid_vels
        mid_e = 0.5*mid_vels**2 + mid_pots
        
        j_from_E = interp1d(np.log10(-mid_e)[::-1], np.log10(j_circ)[::-1], 
                            fill_value='extrapolate', bounds_error=False)
        jc = 10**j_from_E(np.log10(-galaxy.s['e']))
        
        jc[galaxy.s['e'] > mid_e.max()] = np.inf
        jc[galaxy.s['e'] < mid_e.min()] = j_circ[0]
        
        galaxy.s['jc'] = SimArray(jc, units=galaxy.s['pos'].units * galaxy.s['vel'].units)
        galaxy.s['e/emin'] = SimArray(galaxy.s['e']/galaxy.s['e'].min().abs(), units="")
        galaxy.s['jz/jc'] = SimArray(galaxy.s['jz']/galaxy.s['jc'], units="")
        galaxy.s['jp/jc'] = SimArray(galaxy.s['jp']/galaxy.s['jc'], units="")
    return galaxy


if __name__ == '__main__':
    from TNGloading import loadGalaxy, center, faceon
    from Visualize import PhaseSpace
    import matplotlib.pyplot as plt
    run = 'TNG100-3'
    basePath = f"/Users/yuwa/Tools/KD/{run}/output/"
    subID = 1
    snapNum = 99
    galaxy1 = loadGalaxy(basePath, run, snapNum, subID) 
    center(galaxy1)
    faceon(galaxy1)
    galaxy2 = loadGalaxy(basePath, run, snapNum, subID) 
    center(galaxy2)
    faceon(galaxy2)
    galaxy3 = loadGalaxy(basePath, run, snapNum, subID) 
    center(galaxy3)
    faceon(galaxy3)
    
    galaxy1 = GravitySolver(galaxy1, Solver='direct') 
    plt.scatter(galaxy1['r'], galaxy1['phi'], color='Navy')
    galaxy2 = GravitySolver(galaxy2, Solver='Tree') 
    plt.scatter(galaxy2['r'], galaxy2['phi'], color='Red', s=0.5)
    galaxy3 = GravitySolver(galaxy3, Solver='Agama') 
    plt.scatter(galaxy3['r'], galaxy3['phi'], color='Orange', s=0.5)
    plt.show()
    #PhaseSpace(X=np.column_stack((galaxy.s['e/emin'], galaxy.s['jz/jc'])))
    
 

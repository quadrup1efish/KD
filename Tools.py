import scipy
import numpy as np
import illustris_python as il
from dataclasses import dataclass, fields

G = 4.3009e-6
"""
Save the GMM model data
"""
@dataclass
class GMMcomponent:
    ncs: np.int64 = None
    weights: np.ndarray = None
    means: np.ndarray = None
    covariances: np.ndarray = None
    ecut: np.float32 | None = None
    radius: np.ndarray = None
    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self):
        return [field.name for field in fields(self)]

@dataclass
class GMMData:      
    total: GMMcomponent = None
    disk: GMMcomponent = None
    colddisk: GMMcomponent = None
    warmdisk: GMMcomponent = None
    spheroid: GMMcomponent = None
    bulge: GMMcomponent = None
    halo: GMMcomponent = None
    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self):
        return [field.name for field in fields(self)]

"""
Save the structure properties data
"""
@dataclass
class Properties:
    re: np.float64 = None
    Re: np.float64 = None
    r50: np.float64 = None
    R50: np.float64 = None
    Lum: np.float64 = None
    Age: np.float64 = None
    Metals: np.float64 = None
    Mass: np.float64 = None
    vcxy: np.float64 = None
    krot: np.float64 = None
    Ke: np.float64 | None = None
    Ke_re: np.float64 | None = None
    VelDisp: np.float64 = None
    VelDisp_re: np.float64 = None
    Sigma: np.float64 = None
    Mu: np.float64 = None
    Mdm_re: np.float64 = None
    Mdm_Re: np.float64 = None
    Mass_frac: np.float64 = None
    Mdyn_re: np.float64 = None
    """
    Shape properties
    """
    axes: np.array = None
    q_axial_ratio: np.float32 = None
    p_axial_ratio: np.float32 = None
    s_axial_ratio: np.float32 = None
    elongation: np.float32 = None
    flattening: np.float32 = None
    triaxiality: np.float32 = None
    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self):
        return [field.name for field in fields(self)]

@dataclass
class DMProperties:
    v_vir: np.float64 = None
    r_vir: np.float64 = None
    m_vir: np.float64 = None
    j_vir: np.float64 = None
    VelDisp: np.float64= None
    Lambda: np.float64 = None
    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self):
        return [field.name for field in fields(self)]

@dataclass
class StructureData:
    total: Properties = None
    disk: Properties = None
    colddisk: Properties = None
    warmdisk: Properties = None
    spheroid: Properties = None
    bulge: Properties = None
    halo: Properties = None
    DM: DMProperties = None
    BH_mass: np.float64 = None
    def __getitem__(self, key): return getattr(self, key)
    def keys(self):
        return [field.name for field in fields(self)]

"""
Save the particles data
"""
@dataclass
class Component:
    # Particles data
    age: np.ndarray = None
    pos: np.ndarray = None
    vel: np.ndarray = None
    lum: np.ndarray = None
    iord: np.ndarray = None 
    mass: np.ndarray = None 
    metals: np.ndarray = None

    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self): return [field.name for field in fields(self)]
    def __add__(self, other):
        result = Component()
        for field in self.keys():
            a = getattr(self, field)
            b = getattr(other, field)
            if a is not None and b is not None:
                setattr(result, field, np.concatenate([a, b]))
            elif a is not None:
                setattr(result, field, a)
            elif b is not None:
                setattr(result, field, b)
        return result
    
@dataclass
class GalaxyData:
    BH: np.float64 = None
    Mass: np.float64 = None
    colddisk: Component = None
    warmdisk: Component = None
    bulge: Component = None
    halo: Component = None

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return [field.name for field in fields(self)] + ['disk', 'spheroid', 'total']

    @property
    def disk(self):
        cold = self.colddisk or Component()
        warm = self.warmdisk or Component()
        combined = cold + warm
        if all(getattr(combined, field) is None for field in combined.keys()):
            return None
        return combined

    @property
    def spheroid(self):
        bulge = self.bulge or Component()
        halo = self.halo or Component()
        combined = bulge + halo
        if all(getattr(combined, field) is None for field in combined.keys()):
            return None
        return combined

    @property
    def total(self):
        parts = [self.colddisk, self.warmdisk, self.bulge, self.halo]
        total = Component()
        has_data = False
        for part in parts:
            if part is not None:
                total += part
                has_data = True
        return total if has_data else None

"""
Calculate the r percent like re, Re, R50...
"""
def r_percent(radius, weight, percent=0.50, rlim=None):
    if rlim is None: rlim = np.max(radius)
    if weight is None: return None
    mask = radius <= rlim
    r_filtered = radius[mask]
    w_filtered = weight[mask]
    sort_idx = np.argsort(r_filtered)
    r_sorted = r_filtered[sort_idx]
    w_sorted = w_filtered[sort_idx]
    cum_mass = np.cumsum(w_sorted)
    total_mass = cum_mass[-1]
    re_idx = np.searchsorted(cum_mass, total_mass * percent)
    re = r_sorted[re_idx]
    return re

"""
Calculate the vcxy
"""
def cal_vcxy(pos, vel):
    vcxy = (pos[:,0] * vel[:,1] - pos[:,1] * vel[:,0]) / np.sqrt((pos[:,0]**2 + pos[:,1]**2))
    return vcxy

"""
Calculate the velocity dispersion
"""
def cal_VelDisp(vel):
    return np.linalg.norm(np.std(vel, axis=0))

"""
Calculate the krot
"""
def cal_krot(pos, vel, mass):
    vcxy = cal_vcxy(pos, vel)
    krot = np.array(np.sum((0.5*mass*(vcxy**2))) / np.sum(mass*0.5*(vel**2).sum(axis=1)))
    return krot

def cal_angmom(pos, vel, mass):
    angmom = (mass.reshape(-1, 1)*np.cross(pos, vel)).sum(axis=0)
    return angmom

"""
halo spin: lambda from Bullock et al. (2001)
lambda = jvir/(sqrt(2)*Rvir*Vvir)
"""
def cal_lambda(pos, vel, mass, r_vir):
    r = np.linalg.norm(pos, axis=1)
    idx = r<=r_vir
    m_vir = np.sum(mass[idx])
    angmom= cal_angmom(pos[idx], vel[idx], mass[idx])
    J = np.sqrt((angmom ** 2).sum())
    j_vir = J/m_vir
    v_vir = np.sqrt(G*m_vir/r_vir)
    lamb = j_vir/(np.sqrt(2)*v_vir*r_vir)
    return lamb, j_vir, m_vir, v_vir

def face_on(pos, vel, mass):
    up=[0.0, 1.0, 0.0]
    angmom_vec = cal_angmom(pos, vel, mass)

    vec_in = np.asarray(angmom_vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross(up, vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)
    
    matr = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))
    
    pos_new = np.dot(pos, matr.T)
    vel_new = np.dot(vel, matr.T)
    return pos_new, vel_new, matr

def center(pos, mass):
    """
    Only support the center of mass
    """
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    
    x_cen = np.sum(mass*x)/np.sum(mass)
    y_cen = np.sum(mass*y)/np.sum(mass)
    z_cen = np.sum(mass*z)/np.sum(mass)
    return x_cen, y_cen, z_cen

def check_boundary(pos, boxsize):
    x,y,z = pos[:,0],pos[:,1],pos[:,2]
    cen_pos = pos[0]
    index = np.where((x-cen_pos[0])>(boxsize/2))
    pos[index,0]-=boxsize
    index = np.where((cen_pos[0]-x)>(boxsize/2))
    pos[index,0]+=boxsize
    
    index = np.where((y-cen_pos[0])>(boxsize/2))
    pos[index,1]-=boxsize
    index = np.where((cen_pos[0]-y)>(boxsize/2))
    pos[index,1]+=boxsize
    
    index = np.where((z-cen_pos[0])>(boxsize/2))
    pos[index,2]-=boxsize
    index = np.where((cen_pos[0]-z)>(boxsize/2))
    pos[index,2]+=boxsize
    
    return pos

def getaxes(pos, mass, radius=np.inf, max_iter=100, tol=1e-2):
    '''
    Measure the principal axes of the moment of inertia tensor,
    using only particles within the elliptical radius smaller than the provided
    value (iteratively adjusting the axes of the bounding ellipse to match the
    axes of the moment of inertia).
    Arguments:
      pos: Nx3 array of particle positions;
      mass: array of particle masses of length N;
      radius: maximum elliptical radius of particles included in the computation,
        defined to be the geometric average of the three principal axes of the
        ellipse (default: infinity).
    Return:
      a tuple of three arrays:
      - three principal axes of the moment of inertia tensor;
      - a boolean array of length N indicating which particles were used in
        computing the moment of inertia;
      - the rotation matrix defined so that the positions and velocities in the rotated reference frame aligned with the principal axes are given by pos_aligned = pos.dot(matrix)
        vel_aligned = vel.dot(matrix)
    '''
    evec = np.eye(3)   # initial guess for axes orientation
    axes = np.ones(3)  # and axes ratios; these are updated at each iteration
    for _ in range(max_iter):
        # use particles within the elliptical radius less than the provided value
        ellpos  = pos.dot(evec) / axes
        filter  = np.sum(ellpos**2, axis=1) < radius**2
        inertia = pos[filter].T.dot(pos[filter] * mass[filter,None])
        val,vec = np.linalg.eigh(inertia)
        order   = np.argsort(-val)  # sort axes in decreasing order
        evec    = vec[:,order]         # updated axes directions
        axesnew = (val[order] / np.prod(val)**(1./3))**0.5  # updated axes ratios, normalized so that ax*ay*az=1
        if sum(abs(axesnew-axes))<tol: break
        axes    = axesnew
    # evec is almost equivalent to the rotation matrix, just need to ensure that it is right-handed
    # and preferrably has positive values on the diagonal
    if np.linalg.det(evec)<0: evec *= -1
    if evec[2,2]<0: evec[:,1:3] *= -1
    if evec[1,1]<0: evec[:,0:2] *= -1
    return axes, filter, evec

"""
Calculate the axial ratios
"""
def cal_axial_ratios(axes):
    a = axes[0]
    b = axes[1]
    c = axes[2]
    q = b/a
    p = c/b
    s = c/a
    e = np.sqrt(1-q**2)
    f = np.sqrt(1-p**2)
    T = (1-q**2)/(1-s**2)
    return q,p,s,e,f,T

"""
Calculate the information of structures receiving the particles infomation
"""
def cal_structure_info(galaxy, Particles_info):
    
    total_stellar_mass = np.sum(Particles_info.total.mass)
    
    def _cal_dm(galaxy):
        r_vir=None#galaxy.dm['r_vir']
        if r_vir != 0 and r_vir is not None:
            lamb, j_vir, m_vir, v_vir = cal_lambda(galaxy.dm['pos'], galaxy.dm['vel'], galaxy.dm['mass'], r_vir)
            return DMProperties(
                v_vir=v_vir, r_vir=r_vir, 
                m_vir=m_vir, 
                j_vir=j_vir, 
                VelDisp=np.linalg.norm(np.std(galaxy.dm['vel'], axis=0)), 
                Lambda=lamb)
        else: return None
    
    def _cal(galaxy, component):
        if component is not None:
            age = component.age
            pos = component.pos
            mass= component.mass
            r   = np.sqrt(component['pos'][:,0]**2+component['pos'][:,1]**2+component['pos'][:,2]**2)
            lum = component['lum']
            vel = component.vel
            vcxy= np.mean(cal_vcxy(pos, vel))
            krot= cal_krot(pos, vel, mass)
            re  = r_percent(r, mass)
            Re  = r_percent(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2), mass)
            r50 = r_percent(r, lum)
            R50 = r_percent(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2), lum)
            Lum = np.sum(lum)
            Mass= np.sum(mass)

            metals     = component.metals
            Ke         = np.sqrt(np.mean(np.linalg.norm(vel, axis=1)**2))
            Ke_re      = np.sqrt(np.mean(np.linalg.norm(vel, axis=1)[r<re]**2))
            VelDisp    = np.linalg.norm(np.std(vel, axis=0))
            VelDisp_re = np.linalg.norm(np.std(vel[r<re], axis=0))
            Sigma      = Mass / re**2
            Mu         = Lum / r50**2 if r50 is not None else None

            Mdm_re = np.sum(galaxy.dm['mass'][galaxy.dm['r'] <= re])
            Mdm_Re = np.sum(galaxy.dm['mass'][galaxy.dm['r'] <= Re])

            Mdyn_re = np.sum(galaxy['mass'][galaxy['r'] <= re])
            Mass_frac  = Mass / total_stellar_mass

            axes, _, _ = getaxes(pos, mass)
            q,p,s,e,f,T= cal_axial_ratios(axes)  

            return Properties(
                    re=re, 
                    Re=Re, 
                    r50=r50, 
                    R50=R50, 
                    Age=np.mean(age), 
                    Metals=np.mean(metals),
                    Lum=Lum, 
                    Mass=Mass,
                    Ke  = Ke,
                    Ke_re = Ke_re,
                    VelDisp=VelDisp,
                    VelDisp_re=VelDisp_re, 
                    vcxy=vcxy, 
                    krot=krot,
                    Sigma=Sigma, 
                    Mu=Mu,
                    Mdm_re=Mdm_re, 
                    Mdm_Re=Mdm_Re,
                    Mass_frac=Mass_frac,
                    Mdyn_re=Mdyn_re,
                    axes = axes,
                    q_axial_ratio = q,
                    p_axial_ratio = p,
                    s_axial_ratio = s,
                    elongation = e,
                    flattening = f,
                    triaxiality = T)
        else: return None
    
    Structure_info = StructureData(
    total=_cal(galaxy, Particles_info.total),
    disk=_cal(galaxy, Particles_info.disk),
    colddisk=_cal(galaxy, Particles_info.colddisk),
    warmdisk=_cal(galaxy, Particles_info.warmdisk),
    spheroid=_cal(galaxy, Particles_info.spheroid),
    bulge=_cal(galaxy, Particles_info.bulge),
    halo=_cal(galaxy, Particles_info.halo),
    DM=_cal_dm(galaxy),
    BH_mass=Particles_info['BH']
    )
    return Structure_info
            
            

"""
Transform the units
"""
def units(Snapshot, run=None, BasePath='/home/tnguser/sims.TNG/TNG100-1/output'):
    grav_const = 4.3009e-6   # For galaxy unit Msun/
    H0 = 67.8 # km/s/Mpc
    Munit = 1e10    # Msun/h

    # The cosmological parameters of IllustrisTNG 
    # Planck 2015 XIII Table 4, last column (http://arxiv.org/abs/1502.01589)
    OmegaM = 0.3089
    OmegaB = 0.0486
    OmegaL = 0.6911
    sigma8 = 0.8159
    h0 = 0.6774
    
    Header = il.groupcat.loadHeader(BasePath, Snapshot)
    h = Header['HubbleParam']   # hubble parameter
    z = Header['Redshift']
    Munith = Munit / h
    a = 1/(1 + z) # scale factor
    lunith = a / h 
    vunith = 1. * np.sqrt(a)    # The unit of velocity is km sqrt(a)/s
    BoxSize = Header['BoxSize'] * lunith
    if run == 'TNG100': 
        Mdm = 7.4634529 * 1e6     # Mass of dark matter particles, unit Msun
        softr = 0.7         # Softening radius of stellar particles
    if run == 'TNG50': 
        Mdm = 4.53746 * 1e5     # Mass of dark matter particles, unit Msun
        softr = 0.3

    if run is None: 
        return Munith, lunith, vunith
    else:
        return Munith, lunith, vunith, Mdm, softr

def _a_dot(a, h0, om_m, om_l):
    om_k = 1.0 - om_m - om_l
    return h0 * a * np.sqrt(om_m * (a ** -3) + om_k * (a ** -2) + om_l)

def _a_dot_recip(*args):
    return 1. / _a_dot(*args)

"""
from snapshot to scale factor.
"""
def snapnum_to_scale_factor(snap):
    data = np.loadtxt('/home/tnguser/gsf/output/Redshift_snapshot.txt', skiprows=1, usecols=(0, 1))
    snaps = data[:, 0]
    scales = data[:, 1]

    if snap in snaps:
        snap_idx = np.where(snaps == snap)[0][0]
        return scales[snap_idx]
    else:
        None

def snapnum_to_redshift(snap):
    data = np.loadtxt('/home/tnguser/gsf/output/Redshift_snapshot.txt', skiprows=1, usecols=(0, 1))
    snaps = data[:, 0]
    scales = data[:, 1]

    if snap in snaps:
        snap_idx = np.where(snaps == snap)[0][0]
        return 1-1/scales[snap_idx]
    else:
        None
        
"""
from snapshot to age.
"""
def redshift_to_age(redshift, h0=0.6774, OmegaM=0.3089, OmegaL=0.6911):
    from pynbody import units
    conv = units.Unit("0.01 s Mpc km^-1").ratio('Gyr')
    a = 1./(1. + redshift)
    ns = np.size(a)
    if ns > 1:
        FT = np.zeros(ns)
        for ii in range(ns): FT[ii] = scipy.integrate.quad(_a_dot_recip, 0, a[ii], (h0, OmegaM, OmegaL))[0] * conv
    else:
        FT = scipy.integrate.quad(_a_dot_recip, 0, a, (h0, OmegaM, OmegaL))[0] * conv
    return FT

"""
from scale factor to age.
"""
def scale_factor_to_age(a, h0=0.6774, OmegaM=0.3089, OmegaL=0.6911):
    from pynbody import units
    conv = units.Unit("0.01 s Mpc km^-1").ratio('Gyr')
    ns = np.size(a)
    if ns > 1:
        FT = np.zeros(ns)
        for ii in range(ns): FT[ii] = scipy.integrate.quad(_a_dot_recip, 0, a[ii], (h0, OmegaM, OmegaL))[0] * conv
    else:
        FT = scipy.integrate.quad(_a_dot_recip, 0, a, (h0, OmegaM, OmegaL))[0] * conv
    return FT

"""
from snapshot to cosmological age (Gyr)
"""
def snapshot_to_age(snap):
    data = np.loadtxt('/home/tnguser/gsf/output/Redshift_snapshot.txt', skiprows=1, usecols=(0, 1))
    snaps = data[:, 0]
    scales = data[:, 1]
    ns = np.size(snap)
    if ns > 1:
        age = np.zeros(ns)
        for ii in range(ns): 
            snap_idx = np.where(snaps == snap[ii])[0][0]
            a = scales[snap_idx]
            age[ii] = scale_factor_to_age(a, h0=0.6774, OmegaM=0.3089, OmegaL=0.6911)
    else:
        snap_idx = np.where(snaps == snap)[0][0]
        a = scales[snap_idx]
        age = scale_factor_to_age(a, h0=0.6774, OmegaM=0.3089, OmegaL=0.6911)
    return age

"""
from cosmological age (Gyr) to nearest snapshot
"""
def age_to_snapshot(age):
    data = np.loadtxt('/home/tnguser/gsf/output/Redshift_snapshot.txt', skiprows=1, usecols=(0, 1))
    snaps = data[:, 0]
    scales = data[:, 1]

    all_ages = np.array([scale_factor_to_age(a, h0=0.6774, OmegaM=0.3089, OmegaL=0.6911) for a in scales])
    
    if isinstance(age, (list, np.ndarray)):
        ns = len(age)
        for ii in range(ns):
            idx = np.argmin(np.abs(all_ages - age[ii]))
            snapshots[ii] = snaps[idx]
    else:
        idx = np.argmin(np.abs(all_ages - age))
        snapshots = int(snaps[idx])
    
    return snapshots

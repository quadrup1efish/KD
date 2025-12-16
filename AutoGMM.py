import scipy
import numpy as np
import scipy.stats
from GMM import GMM
from joblib import Parallel, delayed
from scipy.interpolate import PchipInterpolator
from Preprocessing import RobustScaler
from Tools import *

class AutoGMM:
    def __init__(self, galaxy, n_components=None, ini_method='kmeans', fit_method='mini-batch', criterion=0.1, morphology=True, model=None, morphology_type=None):
        self.galaxy = galaxy
        self.ini_method = ini_method
        self.fit_method = fit_method
        self.criterion  = criterion
        self.ncs        = np.arange(4, 16)
        self.mBICs      = None
        self.mBIC       = None
        self.BIC        = None
        self.n_components = n_components
        self.morphology   = morphology
        self.morphology_model= None
        self.morphology_type = morphology_type
        self.best_model = model
        self.ecut_train = None
        self.X = np.column_stack((galaxy.s['e/emin'], galaxy.s['jz/jc'], galaxy.s['jp/jc']))
        self.remove_particles = (galaxy.s['e/emin']<0)&(np.abs(galaxy.s['jz/jc'])<1.5)&(galaxy.s['jp/jc']<1.5)
        self.scaler = RobustScaler(self.X[self.remove_particles])
        self.X_train= self.scaler.transform()
        self.number = len(self.X_train)
        self.f_e= None # this is a map from e/emin to r
        self.f_r= None # this is a map from r to e/emin
        
    def _morphology_class(self):
        if self.fit_method == 'mini-batch': 
            model= GMM(n_components=3, n_init=1, ini_method='kmeans', max_iter=500, min_iter=100, tol=1e-6).partial_fit(self.X_train[:,[0,1]])
        elif self.fit_method == 'batch': 
            model= GMM(n_components=3, n_init=1, ini_method='kmeans', max_iter=500, min_iter=10, tol=1e-3).fit(self.X_train[:,[0,1]])
        if np.max(model.means_[:,1])>self._scale(0.5,self.galaxy.s['jz/jc']):
            self.morphology_type='disk'
        if np.max(model.means_[:,1])>self._scale(0.5,self.galaxy.s['jz/jc']): 
            self.morphology_type='spheroid-disk'
        if np.max(model.means_[:,1])<self._scale(0.5,self.galaxy.s['jz/jc']): 
            self.morphology_type='spheroid'
        print(f"Morphology type is {self.morphology_type}")
        self.morphology_model=model
        return model
    
    def _2D_or_3D(self):
        if self.morphology_type=='spheroid' or self.morphology_type=='spheroid-disk':
            print("This is the spheroid-dominated galaxy, using two-dim(kinematic) phase space!")
            self.X = np.column_stack((self.galaxy.s['e/emin'], self.galaxy.s['jz/jc']))
            self.remove_particles = (self.galaxy.s['e/emin']<0)&(np.abs(self.galaxy.s['jz/jc'])<1.5)&(self.galaxy.s['jp/jc']<1.5)
            self.scaler = RobustScaler(self.X[self.remove_particles])
            self.X_train= self.scaler.transform()
        else: print("This is disk-dominated, using three-dim(kinematic) phase space!")

    def _check_decreasing_mBICs(self):
        for i in range(1, len(self.mBICs)):
            if self.mBICs[i] > self.mBICs[i - 1]:
                for j in range(i + 1, len(self.mBICs)):
                    if self.mBICs[j] <= self.mBICs[i - 1]:
                        self.mBICs[i] = (self.mBICs[i - 1] + self.mBICs[j])/2
                        break
                else:
                    self.mBICs[i] = self.mBICs[i - 1]

    def _fit_mBIC(self, nc):
        if self.fit_method == 'mini-batch': 
            model= GMM(n_components=nc, n_init=1, ini_method='kmeans', batch_size=10240, max_iter=500, min_iter=100, tol=1e-3).partial_fit(self.X_train, n_jobs=1)
        elif self.fit_method == 'batch': 
            model= GMM(n_components=nc, n_init=1, ini_method='kmeans', max_iter=500, min_iter=0, tol=1e-3).fit(self.X_train)
        mBIC = model.mbic(self.X_train)
        return mBIC
    
    def _fit_model(self, nc):
        if self.fit_method == 'mini-batch': 
            model= GMM(n_components=nc, n_init=10, ini_method=self.ini_method, batch_size=10240, max_iter=1000, min_iter=500, tol=1e-6, PRINT=False).partial_fit(self.X_train,n_jobs=1)
        elif self.fit_method == 'batch': 
            model= GMM(n_components=nc, n_init=1, ini_method=self.ini_method, PRINT=True).fit(self.X_train)
        return model
    
    def _scale(self, x, X):
        x_scale = (x-np.median(X))/(np.percentile(X,84)-np.percentile(X,16))
        return x_scale

    def _Ecut(self):
        e = self.galaxy.s['e/emin']
        r = self.galaxy.s['r']
        m = self.galaxy.s['mass']

        rmax, bin_edges, _ = scipy.stats.binned_statistic(e, r, statistic='max', bins=100, range=[-1, -0.])
        ebin = 0.5*(bin_edges[:-1] + bin_edges[1:])
        valid_mask = np.isfinite(rmax)
        rmax, ebin = rmax[valid_mask], ebin[valid_mask]
        emin = ebin[np.argmin(np.abs(rmax-1))]
        r95  = r_percent(r, m, 0.95)
        mask = (rmax < min(30, r95)) & (rmax > 0)
        if np.sum(mask)<10: mask = (rmax>0)
        x, y = ebin[mask], rmax[mask]   # x=e, y=r
        
        y  = np.maximum.accumulate(y)
        y, idx = np.unique(y, return_index=True)
        x = x[idx]
        
        f_r= PchipInterpolator(y, x, extrapolate=True)
        f_e= PchipInterpolator(x, y, extrapolate=True)
        
        self.f_r = f_r
        self.f_e = f_e
        
        y = np.arange(y.min(), y.max(), 0.1)
        x = f_r(y)
    
        inner_mask = (y >= 0.5) & (y <= 3.5)
        if np.any(inner_mask):
            x_inner = x[inner_mask]
            y_inner = y[inner_mask]
        else:
            rcut = 5
            ecut = f_r(rcut)
            self.ecut=ecut
            self.ecut_train = self._scale(ecut, e)
            return self._scale(ecut, e), ecut
        coeff1 = np.polyfit(x_inner, y_inner, 1)
       
        end_num = 30
        x_outer = x[-end_num:-2]
        y_outer = y[-end_num:-2]
        if y_outer.min() < y_inner.max():
            outer_mask = (y > y_inner.max())
            x_outer = x[outer_mask]
            y_outer = y[outer_mask]
        coeff2 = np.polyfit(x_outer, y_outer, 1)
        
        k1,b1 = coeff1
        k2,b2 = coeff2
        
        if k2 > 0 and k1 > 0 and k2 > k1:
            x_int = (b2-b1)/(k1-k2)
            y_int = k1*x_int + b1
            v1    = np.array([1,k1])/np.sqrt(1+k1**2)
            v2    = np.array([1,k2])/np.sqrt(1+k2**2)
            slopes= [v[1]/v[0] if abs(v[0])>1e-8 else np.inf for v in (v1+v2, v1-v2)]
            k_bis = [s for s in slopes if s<0][0]
            y_bis = lambda x_: k_bis*(x_-x_int)+y_int

            try:
                diff = y - y_bis(x)
                sign_changes = np.where(np.diff(np.signbit(diff)))[0]
                if len(sign_changes)>0:
                    mid_idx = len(x)//2
                    ecut = max(x[sign_changes[np.argmin(np.abs(sign_changes-mid_idx))]], emin)
                else:
                    ecut = emin
                rcut = float(f_e(ecut))
            except:
                rcut = 5
                ecut = f_r(rcut)
            print(f"ecut={ecut:.4f}, rcut={rcut:.4f}, r95={r95:.4f}")
        else:
            rcut = 5
            ecut = f_r(rcut)
            print(f"ecut={ecut:.4f}, rcut={rcut:.4f}, r95={r95:.4f}")
        if ecut > emin: 
            self.ecut=ecut
            self.ecut_train = self._scale(ecut, e)
            return self._scale(ecut, e), ecut
        else: 
            rcut = 5
            ecut = f_r(rcut)
            self.ecut=ecut
            self.ecut_train = self._scale(ecut, e)
            return self._scale(ecut, e), ecut
    
    def _estimate_covariances(self, X, type='full'):
        full_cov = np.cov(X, rowvar=False)
        if type =='full': return full_cov
        if type =='diag': return np.diag(np.diag(full_cov))
    
    def _estimate_weights(self, X):
        return len(X)/self.number

    def _estimate_means(self, X):
        return np.mean(X, axis=0)

    def _ini_physics(self): 
        if self.morphology_type == 'disk':
            Ecut, rcut = self._Ecut()
            means = self.morphology_model.means_.copy()
            labels = self.morphology_model.predict(self.X_train[:,[0,1]]) # due to the morphology_model is two-dim
            disk_components = np.where(means[:, 1] > self._scale(0.5,self.galaxy.s['jz/jc']))[0]
            spheroid_components = np.where(means[:, 1] <= self._scale(0.5,self.galaxy.s['jz/jc']))[0]
            disk_mask = np.isin(labels, disk_components)
            spheroid_mask = np.isin(labels, spheroid_components)
            
            disk_X = self.X_train[disk_mask]
            spheroid_X = self.X_train[spheroid_mask]
            bulge_X = spheroid_X[spheroid_X[:,0]<self.ecut_train]
            halo_X  = spheroid_X[spheroid_X[:,0]>=self.ecut_train]

            disk_nc  = max(int(np.round(self._estimate_weights(disk_X)*self.n_components)),1)
            bulge_nc = max(int(np.round(self._estimate_weights(bulge_X)*self.n_components)),1)
            halo_nc  = max(self.n_components - disk_nc - bulge_nc,1)
            if disk_nc + bulge_nc + halo_nc > self.n_components:
                max_val = max(disk_nc, bulge_nc, halo_nc)
                disk_nc = disk_nc - 1 if disk_nc == max_val else disk_nc
                bulge_nc = bulge_nc - 1 if bulge_nc == max_val else bulge_nc
                halo_nc = halo_nc - 1 if halo_nc == max_val else halo_nc

            disk_gmm = GMM(n_components=disk_nc, max_iter=1, min_iter=0)
            disk_model = disk_gmm.fit(disk_X[:,[0,1]])
            disk_labels = disk_model.predict(disk_X[:,[0,1]])
            disk_weights = []
            disk_means = []
            disk_covariances = []
            for i in range(disk_nc):
                x = disk_X[disk_labels==i]
                disk_weights.append(self._estimate_weights(x))
                disk_means.append(self._estimate_means(x))
                disk_covariances.append(self._estimate_covariances(x, type='diag'))
            
            bulge_gmm = GMM(n_components=bulge_nc, max_iter=1, min_iter=0)
            bulge_model = bulge_gmm.fit(bulge_X[:,0].reshape(-1, 1))
            bulge_labels = bulge_model.predict(bulge_X[:,0].reshape(-1, 1))
            bulge_weights = []
            bulge_means = []
            bulge_covariances = []
            for i in range(bulge_nc):
                x = bulge_X[bulge_labels==i]
                bulge_weights.append(self._estimate_weights(x))
                bulge_means.append(self._estimate_means(x))
                bulge_covariances.append(self._estimate_covariances(x, type='diag'))
            
            halo_gmm = GMM(n_components=halo_nc, max_iter=1, min_iter=0)
            halo_model = halo_gmm.fit(halo_X[:,0].reshape(-1, 1))
            halo_labels = halo_model.predict(halo_X[:,0].reshape(-1, 1))
            halo_weights = []
            halo_means = []
            halo_covariances = []

            for i in range(halo_nc):
                x = halo_X[halo_labels==i]
                halo_weights.append(self._estimate_weights(x))
                halo_means.append(self._estimate_means(x))
                halo_covariances.append(self._estimate_covariances(x, type='diag'))
            
            weights = np.concatenate([disk_weights, bulge_weights, halo_weights])
            means = np.vstack([disk_means, bulge_means, halo_means])
            covariances = np.vstack([np.array(disk_covariances), np.array(bulge_covariances), np.array(halo_covariances)])
            means[disk_nc:, 1] = self._scale(0.0,self.galaxy.s['jz/jc'])
        if self.morphology_type == 'spheroid-disk':
            Ecut, rcut = self._Ecut()
            means = self.morphology_model.means_.copy()
            labels = self.morphology_model.predict(self.X_train)
            disk_components = np.where(means[:, 1] > self._scale(0.5,self.galaxy.s['jz/jc']))[0]
            spheroid_components = np.where(means[:, 1] <= self._scale(0.5,self.galaxy.s['jz/jc']))[0]
            disk_mask = np.isin(labels, disk_components)
            spheroid_mask = np.isin(labels, spheroid_components)
            
            disk_X = self.X_train[disk_mask]
            spheroid_X = self.X_train[spheroid_mask]
            bulge_X = spheroid_X[spheroid_X[:,0]<self.ecut_train]
            halo_X  = spheroid_X[spheroid_X[:,0]>=self.ecut_train]
            
            disk_nc  = max(int(np.round(self._estimate_weights(disk_X)*self.n_components)),1)
            bulge_nc = max(int(np.round(self._estimate_weights(bulge_X)*self.n_components)),1)
            halo_nc  = max(self.n_components - disk_nc - bulge_nc,1)
            if disk_nc + bulge_nc + halo_nc > self.n_components:
                max_val = max(disk_nc, bulge_nc, halo_nc)
                disk_nc = disk_nc - 1 if disk_nc == max_val else disk_nc
                bulge_nc = bulge_nc - 1 if bulge_nc == max_val else bulge_nc
                halo_nc = halo_nc - 1 if halo_nc == max_val else halo_nc

            disk_gmm = GMM(n_components=disk_nc, max_iter=1, min_iter=0)
            disk_model = disk_gmm.fit(disk_X[:,[0,1]])
            disk_labels = disk_model.predict(disk_X[:,[0,1]])
            disk_weights = []
            disk_means = []
            disk_covariances = []
            for i in range(disk_nc):
                x = disk_X[disk_labels==i]
                disk_weights.append(self._estimate_weights(x))
                disk_means.append(self._estimate_means(x))
                disk_covariances.append(self._estimate_covariances(x, type='diag'))
            
            bulge_gmm = GMM(n_components=bulge_nc, max_iter=1, min_iter=0)
            bulge_model = bulge_gmm.fit(bulge_X[:,0].reshape(-1, 1))
            bulge_labels = bulge_model.predict(bulge_X[:,0].reshape(-1, 1))
            bulge_weights = []
            bulge_means = []
            bulge_covariances = []
            for i in range(bulge_nc):
                x = bulge_X[bulge_labels==i]
                bulge_weights.append(self._estimate_weights(x))
                bulge_means.append(self._estimate_means(x))
                bulge_covariances.append(self._estimate_covariances(x, type='diag'))
            
            halo_gmm = GMM(n_components=halo_nc, max_iter=1, min_iter=0)
            halo_model = halo_gmm.fit(halo_X[:,0].reshape(-1, 1))
            halo_labels = halo_model.predict(halo_X[:,0].reshape(-1, 1))
            halo_weights = []
            halo_means = []
            halo_covariances = []
            
            for i in range(halo_nc):
                x = halo_X[halo_labels==i]
                halo_weights.append(self._estimate_weights(x))
                halo_means.append(self._estimate_means(x))
                halo_covariances.append(self._estimate_covariances(x, type='diag'))
            
            weights = np.concatenate([disk_weights, bulge_weights, halo_weights])
            means = np.vstack([disk_means, bulge_means, halo_means])
            covariances = np.vstack([np.array(disk_covariances), np.array(bulge_covariances), np.array(halo_covariances)])
            means[disk_nc:, 1] = self._scale(0.0,self.galaxy.s['jz/jc'])
        if self.morphology_type == 'spheroid':
            Ecut, rcut = self._Ecut()
            spheroid_X = self.X_train
            bulge_X = spheroid_X[spheroid_X[:,0]<self.ecut_train]
            halo_X  = spheroid_X[spheroid_X[:,0]>=self.ecut_train]
            
            bulge_nc = max(int(np.round(self._estimate_weights(bulge_X)*self.n_components)),1)
            halo_nc  = max(self.n_components - bulge_nc,1)
            if bulge_nc + halo_nc > self.n_components:
                if bulge_nc > halo_nc: bulge_nc -= 1
                else: halo_nc-=1
            bulge_gmm = GMM(n_components=bulge_nc, max_iter=1, min_iter=0)
            bulge_model = bulge_gmm.fit(bulge_X[:,0].reshape(-1, 1))
            bulge_labels = bulge_model.predict(bulge_X[:,0].reshape(-1, 1))
            bulge_weights = []
            bulge_means = []
            bulge_covariances = []
            for i in range(bulge_nc):
                x = bulge_X[bulge_labels==i]
                bulge_weights.append(self._estimate_weights(x))
                bulge_means.append(self._estimate_means(x))
                bulge_covariances.append(self._estimate_covariances(x, type='diag'))
            
            halo_gmm = GMM(n_components=halo_nc, max_iter=1, min_iter=0)
            halo_model = halo_gmm.fit(halo_X[:,0].reshape(-1, 1))
            halo_labels = halo_model.predict(halo_X[:,0].reshape(-1, 1))
            halo_weights = []
            halo_means = []
            halo_covariances = []
            
            for i in range(halo_nc):
                x = halo_X[halo_labels==i]
                halo_weights.append(self._estimate_weights(x))
                halo_means.append(self._estimate_means(x))
                halo_covariances.append(self._estimate_covariances(x, type='diag'))
            
            weights = np.concatenate([bulge_weights, halo_weights])
            means = np.vstack([bulge_means, halo_means])
            covariances = np.vstack([np.array(bulge_covariances), np.array(halo_covariances)])
            means[:,1] = self._scale(0.0,self.galaxy.s['jz/jc'])
        return weights, means, covariances
    
    def fit(self, max_iter=1000, min_iter=500, dims='auto'):
        if self.morphology == True:
            if dims=='auto':
                self.morphology_model = self._morphology_class()
                self._2D_or_3D()
            if dims==2:
                self.morphology_model = self._morphology_class()
                self.morphology_type  = 'spheroid'
                self._2D_or_3D()
            if dims==3:
                self.morphology_model = self._morphology_class()
                self.morphology_type  = 'disk'
                self._2D_or_3D()
            if self.n_components is None:
                print("Finding the best n_components using 1 cores")
                with Parallel(n_jobs=1, verbose=0) as parallel:
                    results = parallel(delayed(self._fit_mBIC)(n) for n in np.arange(4, 16))
                self.mBICs = np.array(results)
                self._check_decreasing_mBICs()
                min_mBIC = np.mean(self.mBICs[-5:])
                Delta_mBICs = self.mBICs - min_mBIC
                min_index = np.min(np.where(Delta_mBICs < self.criterion)[0])
                self.n_components = self.ncs[min_index]
            weights, means, covariances = self._ini_physics()
            gmm = GMM(n_components=self.n_components, n_init=1,ini_weights=weights,ini_means=means,ini_covariances=covariances, max_iter=max_iter, min_iter=min_iter).partial_fit(self.X_train, do_init=False)
            self.BIC = gmm.bic(self.X_train)
            self.mBIC= self.BIC/self.X_train.shape[0]
            mask = gmm.weights_ != 0
            gmm.weights_ =  gmm.weights_[mask]
            gmm.means_ =  gmm.means_[mask]
            gmm.covariances_ =  gmm.covariances_[mask]
            nc = len(gmm.weights_)
            print(f"The best number of components is {nc}.")
            self.best_model = self.scaler.inverse_transform_GMM(gmm)
            print("We return the best GMM model instead of train model.")
            self.labels = self.best_model.predict(self.X)
            return self.best_model
        elif self.morphology == False:
            print("Using Du2019 method")
            if self.n_components is None:
                print("Finding the best n_components using 8 cores")
                with Parallel(n_jobs=8, verbose=0) as parallel:
                    results = parallel(delayed(self._fit_mBIC)(n) for n in np.arange(4, 16))
                self.mBICs = np.array(results)
                self._check_decreasing_mBICs()
                min_mBIC = np.mean(self.mBICs[-5:])
                Delta_mBICs = self.mBICs - min_mBIC
                min_index = np.min(np.where(Delta_mBICs < self.criterion)[0])
                self.n_components = self.ncs[min_index]
                
            gmm = GMM(n_components=self.n_components, n_init=1, ini_method='kmeans', max_iter=500, min_iter=0, tol=1e-3).fit(self.X_train)
            mask = gmm.weights_ != 0
            gmm.weights_ =  gmm.weights_[mask]
            gmm.means_ =  gmm.means_[mask]
            gmm.covariances_ =  gmm.covariances_[mask]
            nc = len(gmm.weights_)
            print(f"The best number of components is {nc}.")
            self.best_model = self.scaler.inverse_transform_GMM(gmm) 
            self.labels = self.best_model.predict(self.X)
            return self.best_model
            
            
    def _make_positive_definite(self, cov, epsilon=1e-6):
        eigvals = np.linalg.eigvals(cov)
        if np.any(eigvals <= 0):
             cov += (np.abs(np.min(eigvals)) + epsilon) * np.eye(cov.shape[0])
        return cov
    
    def decompose(self):
        labels = self.labels
        weights=self.best_model.weights_
        means  =self.best_model.means_
        covariances=self.best_model.covariances_
        radius = self.f_e(means[:,0])
        if self.morphology_type=='spheroid':
            spheroid_idx = np.arange(len(means))
            halo_mask = means[:, 0] >= self.ecut
            bulge_mask = means[:, 0] < self.ecut
            
            bulge_weights= weights[bulge_mask]
            bulge_means= means[bulge_mask]
            bulge_covariances= covariances[bulge_mask]
            bulge_radius = None#self.f_e(bulge_means[:,0])
            
            halo_weights= weights[halo_mask]
            halo_means= means[halo_mask]
            halo_covariances= covariances[halo_mask]
            halo_radius = None#self.f_e(halo_means[:,0])
            
            GMM_info = GMMData(
            total=GMMcomponent(
                ncs=len(weights),
                weights=weights,
                means=means,
                covariances=covariances,
                radius=radius
            ),
            spheroid=GMMcomponent(
                ncs=len(weights),
                weights=weights,
                means=means,
                covariances=covariances,
                radius=radius
            ),
            bulge=GMMcomponent(
                ncs=len(bulge_weights),
                weights=bulge_weights,
                means=bulge_means,
                covariances=bulge_covariances,
                radius=bulge_radius
            ),
            halo=GMMcomponent(
                ncs=len(halo_weights),
                weights=halo_weights,
                means=halo_means,
                covariances=halo_covariances,
                radius=halo_radius
            )
            )
            
            halo_idx = np.where(halo_mask)[0]
            bulge_idx = np.where(bulge_mask)[0]
            Particles_data = GalaxyData(
                Mass=np.sum(self.galaxy.s['mass']),
                BH=np.sum(self.galaxy.bh['mass']),
                bulge=None,  # Default to None
                halo=None    # Default to None
            )

            # Only create bulge component if there are bulge particles
            if len(bulge_idx) > 0:
                mask = np.isin(labels, bulge_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.bulge = Component(
                age=np.array(self.galaxy.s['age'][mask]),
                pos=np.array(self.galaxy.s['pos'][mask]),
                vel=np.array(self.galaxy.s['vel'][mask]),
                lum=lum,
                iord=np.array(self.galaxy.s['iord'][mask]),
                mass=np.array(self.galaxy.s['mass'][mask]),
                metals=np.array(self.galaxy.s['metals'][mask]),
                )


            # Only create halo component if there are halo particles
            if len(halo_idx) > 0:
                mask = np.isin(labels, halo_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.halo = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask]), 
                )
        if self.morphology_type=='disk' or self.morphology_type=='spheroid-disk':
            disk_mask = means[:,1] > 0.5
            cold_mask = means[:,1] > 0.85
            warm_mask = (means[:,1] > 0.5)&(means[:,1] < 0.85)
            spheroid_mask = means[:,1] <= 0.5
            bulge_mask = (means[:,1] <= 0.5)&(means[:, 0] < self.ecut)
            halo_mask = (means[:,1] <= 0.5)&(means[:, 0] >= self.ecut)
            
            disk_weights= weights[disk_mask]
            disk_means= means[disk_mask]
            disk_covariances= covariances[disk_mask]
            colddisk_weights= weights[cold_mask]
            colddisk_means= means[cold_mask]
            colddisk_covariances= covariances[cold_mask]
            warmdisk_weights= weights[warm_mask]
            warmdisk_means= means[warm_mask]
            warmdisk_covariances= covariances[warm_mask]
            spheroid_weights= weights[spheroid_mask]
            spheroid_means= means[spheroid_mask]
            spheroid_covariances= covariances[spheroid_mask]
            bulge_weights= weights[bulge_mask]
            bulge_means= means[bulge_mask]
            bulge_covariances= covariances[bulge_mask]
            halo_weights= weights[halo_mask]
            halo_means= means[halo_mask]
            halo_covariances= covariances[halo_mask]
            
            GMM_info = GMMData(
            total=GMMcomponent(
                ncs=len(weights),
                weights=weights,
                means=means,
                covariances=covariances,
                radius=radius
            ),
            disk=GMMcomponent(
                ncs=len(disk_weights),
                weights=disk_weights,
                means=disk_means,
                covariances=disk_covariances
            ),
            colddisk=GMMcomponent(
                ncs=len(colddisk_weights),
                weights=colddisk_weights,
                means=colddisk_means,
                covariances=colddisk_covariances
            ),
            warmdisk=GMMcomponent(
                ncs=len(warmdisk_weights),
                weights=warmdisk_weights,
                means=warmdisk_means,
                covariances=warmdisk_covariances
            ),
            spheroid=GMMcomponent(
                ncs=len(spheroid_weights),
                weights=spheroid_weights,
                means=spheroid_means,
                covariances=spheroid_covariances
            ),
            bulge=GMMcomponent(
                ncs=len(bulge_weights),
                weights=bulge_weights,
                means=bulge_means,
                covariances=bulge_covariances
            ),
            halo=GMMcomponent(
                ncs=len(halo_weights),
                weights=halo_weights,
                means=halo_means,
                covariances=halo_covariances
            ),
            )
            
            disk_idx = np.where(disk_mask)[0] 
            cold_idx = np.where(cold_mask)[0] 
            warm_idx = np.where(warm_mask)[0] 
            spheroid_idx = np.where(spheroid_mask)[0] 
            # Split spheroid components into halo and bulge
            spheroid_means = means[spheroid_idx]
            spheroid_covs = covariances[spheroid_idx]
            halo_mask = spheroid_means[:, 0] >= self.ecut
            bulge_mask = spheroid_means[:, 0] < self.ecut
            halo_idx = spheroid_idx[np.where(halo_mask)[0]]
            bulge_idx = spheroid_idx[np.where(bulge_mask)[0]]
            
            # Initialize with required fields and None for components
            Particles_data = GalaxyData(
                Mass=np.sum(self.galaxy.s['mass']),
                BH=np.sum(self.galaxy.bh['mass']),
                colddisk=None,
                warmdisk=None,
                bulge=None,
                halo=None
            )

            # Only create components if they have particles
            if len(cold_idx) > 0:
                mask = np.isin(labels, cold_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.colddisk = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask]), 
                )

            if len(warm_idx) > 0:
                mask = np.isin(labels, warm_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.warmdisk = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask]),
                )

            if len(bulge_idx) > 0:
                mask = np.isin(labels, bulge_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.bulge = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask]),
                )

            if len(halo_idx) > 0:
                mask = np.isin(labels, halo_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.halo = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask]),
                )
        if self.morphology_type==None:
            print("Using Du2019 threshold")
            disk_mask = means[:,1] > 0.5
            cold_mask = means[:,1] > 0.85
            warm_mask = (means[:,1] > 0.5)&(means[:,1] < 0.85)
            spheroid_mask = means[:,1] <= 0.5
            bulge_mask = (means[:,1] <= 0.5)&(means[:, 0] < -0.75)
            halo_mask = (means[:,1] <= 0.5)&(means[:, 0] >= -0.75)
            
            disk_weights= weights[disk_mask]
            disk_means= means[disk_mask]
            disk_covariances= covariances[disk_mask]
            colddisk_weights= weights[cold_mask]
            colddisk_means= means[cold_mask]
            colddisk_covariances= covariances[cold_mask]
            warmdisk_weights= weights[warm_mask]
            warmdisk_means= means[warm_mask]
            warmdisk_covariances= covariances[warm_mask]
            spheroid_weights= weights[spheroid_mask]
            spheroid_means= means[spheroid_mask]
            spheroid_covariances= covariances[spheroid_mask]
            bulge_weights= weights[bulge_mask]
            bulge_means= means[bulge_mask]
            bulge_covariances= covariances[bulge_mask]
            halo_weights= weights[halo_mask]
            halo_means= means[halo_mask]
            halo_covariances= covariances[halo_mask]
            
            GMM_info = GMMData(
            total=GMMcomponent(
                ncs=len(weights),
                weights=weights,
                means=means,
                covariances=covariances,
            ),
            disk=GMMcomponent(
                ncs=len(disk_weights),
                weights=disk_weights,
                means=disk_means,
                covariances=disk_covariances
            ),
            colddisk=GMMcomponent(
                ncs=len(colddisk_weights),
                weights=colddisk_weights,
                means=colddisk_means,
                covariances=colddisk_covariances
            ),
            warmdisk=GMMcomponent(
                ncs=len(warmdisk_weights),
                weights=warmdisk_weights,
                means=warmdisk_means,
                covariances=warmdisk_covariances
            ),
            spheroid=GMMcomponent(
                ncs=len(spheroid_weights),
                weights=spheroid_weights,
                means=spheroid_means,
                covariances=spheroid_covariances
            ),
            bulge=GMMcomponent(
                ncs=len(bulge_weights),
                weights=bulge_weights,
                means=bulge_means,
                covariances=bulge_covariances
            ),
            halo=GMMcomponent(
                ncs=len(halo_weights),
                weights=halo_weights,
                means=halo_means,
                covariances=halo_covariances
            ),
            )
            
            disk_idx = np.where(disk_mask)[0] 
            cold_idx = np.where(cold_mask)[0] 
            warm_idx = np.where(warm_mask)[0] 
            spheroid_idx = np.where(spheroid_mask)[0] 
            # Split spheroid components into halo and bulge
            spheroid_means = means[spheroid_idx]
            spheroid_covs = covariances[spheroid_idx]
            halo_mask = spheroid_means[:, 0] >= -0.75
            bulge_mask = spheroid_means[:, 0] < -0.75
            halo_idx = spheroid_idx[np.where(halo_mask)[0]]
            bulge_idx = spheroid_idx[np.where(bulge_mask)[0]]
            
            # Initialize with required fields and None for components
            Particles_data = GalaxyData(
                Mass=np.sum(self.galaxy.s['mass']),
                BH=np.sum(self.galaxy.bh['mass']),
                colddisk=None,
                warmdisk=None,
                bulge=None,
                halo=None
            )

            # Only create components if they have particles
            if len(cold_idx) > 0:
                mask = np.isin(labels, cold_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.colddisk = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask])
                )

            if len(warm_idx) > 0:
                mask = np.isin(labels, warm_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.warmdisk = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask])
                )

            if len(bulge_idx) > 0:
                mask = np.isin(labels, bulge_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.bulge = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask])
                )

            if len(halo_idx) > 0:
                mask = np.isin(labels, halo_idx)
                try:
                    lum = np.array(self.galaxy.s['r_lum'][mask])
                except:
                    lum = None
                Particles_data.halo = Component(
                    age=np.array(self.galaxy.s['age'][mask]),
                    pos=np.array(self.galaxy.s['pos'][mask]),
                    vel=np.array(self.galaxy.s['vel'][mask]),
                    lum=lum,
                    iord=np.array(self.galaxy.s['iord'][mask]),
                    mass=np.array(self.galaxy.s['mass'][mask]),
                    metals=np.array(self.galaxy.s['metals'][mask])
                )

        return Particles_data, GMM_info

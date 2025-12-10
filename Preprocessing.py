import numpy as np
from copy import deepcopy

class RobustScaler():
    def __init__(self, X, quantile_range=[16, 84]):
        
        self.X     = X
        self.mean  = np.mean(X, axis=0)
        self.std   = np.std(X, axis=0)
        self.median= np.median(X, axis=0)
        self.IQR   = np.percentile(X,quantile_range[1], axis=0)-np.percentile(X,quantile_range[0], axis=0)
        self.max   = np.percentile(np.abs(X), 99, axis=0)
        
    def _fresh(self):
        lower_bound = self.mean - 3 * self.std
        upper_bound = self.mean + 3 * self.std
        mask = np.all((self.X >= np.maximum(lower_bound, np.percentile(self.X,0.3))) & (self.X <= np.minimum(upper_bound, np.percentile(self.X,99.7))), axis=1)
        print(f"Remove particles = {(len(self.X) - np.sum(mask))/len(self.X)}")
        return self.X[mask]
        
    def transform(self, fresh=False):
        if fresh: x = self._fresh()
        else: x = self.X
        X_scale = (x-self.median)/self.IQR
        return X_scale
    
    def inverse_transform_GMM(self, gmm):
        transformed_gmm = deepcopy(gmm)
        try:
            means=gmm.means_.copy()
            means=means*self.IQR+self.median
            transformed_gmm.means_ = means

            covariances=gmm.covariances_.copy()
            scale_matrix = np.outer(self.IQR, self.IQR)
            covariances = covariances * scale_matrix[np.newaxis, :, :]
            transformed_gmm.covariances_ = covariances
        except:
            means=gmm.means_.copy()
            means=means*self.IQR[:2]+self.median[:2]
            transformed_gmm.means_ = means

            covariances=gmm.covariances_.copy()
            scale_matrix = np.outer(self.IQR[:2], self.IQR[:2])
            covariances = covariances * scale_matrix[np.newaxis, :, :]
            transformed_gmm.covariances_ = covariances
        return transformed_gmm

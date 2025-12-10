import numpy as np
import scipy.stats
import scipy.special
from time import time
from tqdm import trange
from copy import deepcopy
from dataclasses import dataclass
from sklearn.cluster import KMeans, kmeans_plusplus
from joblib import Parallel, delayed, parallel_backend
from typing import Optional, Union, Tuple, List, Dict, Any

@dataclass
class Config:
    """Configuration parameters for the GMM model."""
    n_components: int = 3
    n_init: int = 1
    ini_method: str = 'kmeans'
    ini_weights: np.ndarray | None = None
    ini_means: np.ndarray | None = None
    ini_covariances: np.ndarray | None = None
    covariance_type: str = 'full'
    batch_size: int = 10240
    decay: float = 1.0
    tol: float = 1e-6
    max_iter: int = 1000
    min_iter: int = 500
    reg_covar: float = 1e-6
    convergence: bool = False
    seed: int = 42
    Print: bool = False

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")
        if self.ini_method not in ['kmeans', 'k-means++', 'random']:
            raise ValueError("ini_method must be one of 'kmeans', 'k-means++', 'random'")
        if self.covariance_type not in ['full', 'spherical']:
            raise ValueError('Only support the full or spherical type')
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.max_iter < 0:
            raise ValueError("max_iter must be positive")
        if self.min_iter < 0 or self.min_iter > self.max_iter:
            raise ValueError("min_iter must be between 0 and max_iter")
        if self.reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class GMM:
    def __init__(self, config: Optional[Config] = None, **kwargs):
        """Gaussian Mixture Model for GALAXY kinematic decomposition.

        Parameters
        ----------

        n_components : int, default=3
            The number of clusters to form.

        ini_method= : {'kmeans', k-means++', 'random'}, callable or array-like of shape \
                (n_clusters, n_features), default='k-means++'
            Method for initialization:.

        n_init : 'auto' or int, default='1'
            Number of times the algorithm is run with different initial condition.

        ini_weights : ndarray of shape (n_components,), default=None
            Initial weights for each components.
        
        ini_means : default=None
            Initial means for each components.

        ini_covariances : default=None
            Initial covariances for each components.
        
        covariance_type : default='full'
            The type of covariance.

        batch_size : default=10240
            The size of the batch only useful while using partial_fit.
        
        decay : float, default=1.0
            The decay rate while using partial_fit.
            
        max_iter : int, default=1000
            Maximum number of iterations of the algorithm for a
            single run.

        min_iter : int, default=500
            minimum number of iterations of the algorithm for a
            single run.

        tol : float, default=1e-6
            Relative tolerance with regards to Frobenius norm of the difference
            in the cluster centers of two consecutive iterations to declare
            convergence.

        reg_covar : float, default=1e-6
            Non-negative regularization added to the diagonal of covariance.
        """
        if config is None:
            config = Config()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.validate()

        self.n_components = config.n_components
        self.covariance_type = config.covariance_type
        self.min_iter = config.min_iter
        self.max_iter = config.max_iter
        self.reg_covar = config.reg_covar
        self.batch_size = config.batch_size
        self.decay = config.decay
        self.tol = config.tol
        self.n_init = config.n_init
        self.ini_method = config.ini_method
        self.weights_ = config.ini_weights
        self.means_ = config.ini_means
        self.covariances_ = config.ini_covariances
        self.PRINT = config.Print
        self.convergence = config.convergence
        self.Print = config.Print
        self.convergence = config.convergence
        self.seed = config.seed

        self.lower_bounds = []
        self.lower_bounds_smooth = []
        self.random_state = np.random.default_rng(seed=self.seed)
        
        # Some information about the model
        self.time  = 0
        self.n_iter= 0
        self.time_per_iter = 0

    def _estimate_complexity(self, X):
        """Estimate the computational complexity of the FULL model."""
        n_samples, n_features = X.shape
        E_cost = n_samples * self.n_components * (n_features ** 2)
        M_cost = self.n_components * (n_features ** 3)
        total_ops = E_cost + M_cost
        return total_ops

    def _n_parameters(self):
        """Return the number of free parameters in the FULL model.""" 
        assert self.means_ is not None
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)
    
    def _ini_kmeans(self, X):
        """Initialize the responsibilities using KMeans clustering."""
        n_samples = len(X)
        resp = np.zeros((n_samples, self.n_components))
        label = (KMeans(n_clusters=self.n_components, n_init=1, random_state=None).fit(X).labels_)
        resp[np.arange(n_samples), label] = 1
        return resp
    
    def _ini_k_means_plusplus(self, X):
        """Initialize the responsibilities using KMeans++ clustering."""
        n_samples = len(X)
        resp = np.zeros((n_samples, self.n_components))
        _, indices = kmeans_plusplus(X, self.n_components,random_state=None)
        resp[indices, np.arange(self.n_components)] = 1
        return resp
    
    def _ini_random(self, X):
        """Initialize the responsibilities randomly."""
        n_samples = len(X)
        resp = self.random_state.uniform(size=(n_samples, self.n_components))
        resp /= resp.sum(axis=1)[:, np.newaxis] 
        return resp
    def is_pos_def(self, cov):
        """Check if a covariance matrix is positive definite."""
        if self.covariance_type == 'spherical':
            return cov > 0
        try:
            scipy.linalg.eigh(cov, check_finite=False)
            return True
        except np.linalg.LinAlgError:
            return False

    def _initialization_(self, X):
        """Initialize the GMM parameters."""
        if (self.ini_method=='kmeans'): 
            resp = self._ini_kmeans(X)
        if (self.ini_method=='k-means++'): 
            resp = self._ini_k_means_plusplus(X)
        if (self.ini_method=='random' or self.ini_method==None): 
            resp = self._ini_random(X)

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        n_components, n_features = means.shape
        if self.covariance_type == 'full': covariances = np.empty((n_components, n_features, n_features))
        elif self.covariance_type =='spherical':
            covariances = np.empty(n_components)
        for k in range(n_components):
            diff = X - means[k]
            if self.covariance_type == 'full':
                covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                covariances[k].flat[:: n_features + 1] += self.reg_covar
            elif self.covariance_type == 'spherical':
                full_cov = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                covariances[k] = np.trace(full_cov) / n_features + self.reg_covar
            if not self.is_pos_def(covariances[k]): 
                covariances[k] = np.diag(np.diag(covariances[k]))
        
        self.weights_     = nk/len(X)
        self.means_       = means
        self.covariances_ = covariances
        return self.weights_, self.means_, self.covariances_

    def _estimate_log_prob(self, X):
        """Estimate the log probabilities for each component."""
        log_prob = np.zeros((len(X), len(self.means_)))
        for k in range(len(self.means_)):
            if self.covariance_type =='full':
                log_prob[:, k] = scipy.stats.multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=False)
            elif self.covariance_type == 'spherical':
                cov_matrix = np.eye(self.means_.shape[1]) * self.covariances_[k]
                log_prob[:, k] = scipy.stats.multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=cov_matrix, allow_singular=False)
        return log_prob

    
    def _estimate_log_prob_norm(self, weighted_log_prob):
        """Estimate the log probability normalization."""
        log_prob_norm = scipy.special.logsumexp(weighted_log_prob, axis=1)
        return log_prob_norm

    def _e_step(self, X):
        log_prob = self._estimate_log_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_)
        log_prob_norm = self._estimate_log_prob_norm(weighted_log_prob)
        log_responsibilities = weighted_log_prob - log_prob_norm[:, np.newaxis]
        responsibilities = np.exp(log_responsibilities)
        return responsibilities, log_prob_norm
    
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        n_components = responsibilities.shape[1]

        nk = np.sum(responsibilities, axis=0) + 1e-10
        weights = nk / n_samples
        means = (responsibilities.T @ X) / nk[:, np.newaxis]

        if self.covariance_type == 'full':
            covariances = np.empty((n_components, n_features, n_features))
        elif self.covariance_type =='spherical':
            covariances = np.empty(n_components)
        else:
            covariances = np.empty((n_components, n_features, n_features))

        for k in range(n_components):
            diff = X - means[k]
            weighted_diff = diff * responsibilities[:, k][:, np.newaxis]
            if self.covariance_type == 'full':
                covariances[k] = (weighted_diff.T @ diff) / nk[k]
                covariances[k] += self.reg_covar * np.eye(n_features)
            elif self.covariance_type == 'spherical':
                covariances[k] = np.trace(weighted_diff.T @ diff) / (nk[k] * n_features) + self.reg_covar
        return weights, means, covariances
    def _partial_m_step(self, X, responsibilities):
        n_samples, n_features = X.shape

        nk = np.sum(responsibilities, axis=0) + self.reg_covar
        xk = responsibilities.T @ X
        
        diff = X[:, np.newaxis, :] - self.means_  # (n_samples, n_components, n_features)
        weighted_diff = diff * responsibilities[:, :, np.newaxis]  # (n_samples, n_components, n_features)
        sk = np.einsum('sij,sik->ijk', weighted_diff, diff)  # (n_components, n_features, n_features)

        new_weights = (1 - self.decay) * self.weights_ + self.decay * nk / n_samples
        new_weights /= np.sum(new_weights)
        new_means = (1 - self.decay) * self.means_ + self.decay * xk / nk[:, np.newaxis]

        new_covariances = np.empty_like(self.covariances_)
        for k in range(len(self.means_)):
            if self.covariance_type == 'full':
                new_covariances[k] = (1 - self.decay) * self.covariances_[k] + self.decay * sk[k] / nk[k]
                new_covariances[k] += self.reg_covar * np.eye(n_features)
            elif self.covariance_type == 'spherical':
                new_covariances[k] = (1 - self.decay) * self.covariances_[k] + self.decay * np.trace(sk[k]) / (nk[k] * n_features)
                new_covariances[k] += self.reg_covar
        #new_covariances = (1 - self.decay) * self.covariances_ + self.decay * sk / nk[:, np.newaxis, np.newaxis]
        #new_covariances += self.reg_covar * np.eye(n_features)
        return new_weights, new_means, new_covariances

    
    def fit(self, X, min_iter=None, max_iter=None):
        start_time = time()
        if max_iter is not None: self.max_iter = max_iter
        if min_iter is not None: self.min_iter = min_iter
        if self.weights_ is None or self.means_ is None or self.covariances_ is None: self._initialization_(X)

        prev_lower_bound = -np.inf
        self.lower_bounds = []
        iterator = trange(self.max_iter) if self.Print else range(self.max_iter)
        for iteration in iterator:
            # E-step
            responsibilities, log_prob_norm = self._e_step(X)
            # M-step 
            self.weights_, self.means_, self.covariances_ = self._m_step(X, responsibilities)
            
            current_lower_bound = np.mean(log_prob_norm)
            self.lower_bounds.append(current_lower_bound)
            if iteration>=self.min_iter and np.abs(current_lower_bound - prev_lower_bound) < self.tol:
                break
            prev_lower_bound = current_lower_bound
        end_time = time()
        self.time = end_time - start_time
        Ncomplex = self._estimate_complexity(X)
        try: self.lower_bound = current_lower_bound
        except: self.lower_bound = None
        if self.Print==True:
            print(f"Batch-size={self.batch_size}")
            print(f"Ncomplex is between [{Ncomplex*self.min_iter:.0e},{Ncomplex*self.max_iter:.0e}], per iterate is {Ncomplex:.0e} ")
            print(f"Total fitting time: {self.time:.2f} s in {iteration} iterations")
            print(f"Average time per iterate is {self.time/iteration:.4f}")
            print(f"lower bound = {self.lower_bound}")
        return self

    def _single_partial_fit(self, X, do_init):
        """Perform a single partial fit on the data X."""
        if self.weights_ is None or self.means_ is None or self.covariances_ is None or do_init:
            print("Init...")
            self._initialization_(X)

        n_samples = X.shape[0]
        prev_lower_bound_smooth = -np.inf
        
        iterator = trange(self.max_iter) if self.Print else range(self.max_iter)
        
        use_batch = n_samples >= self.batch_size
        if use_batch:
            permutation = np.random.permutation(n_samples)
        
        for iteration in iterator:
            if use_batch:
                X_batch = X[self._get_batch_indices(self, permutation, n_samples)]
            else:
                X_batch = X

            current_lower_bound, current_lower_bound_smooth = self._single_iteration(
                X_batch, iteration, prev_lower_bound_smooth)
            
            if iteration >= self.min_iter and np.abs(current_lower_bound_smooth - prev_lower_bound_smooth) < self.tol:
                break
                
            prev_lower_bound_smooth = current_lower_bound_smooth
        
        self._record_training_results(iteration)
        return self

    def _single_iteration(self, X_batch, iteration, prev_lower_bound_smooth):
        """Perform a single iteration of the EM algorithm on a batch of data."""
        # E-step
        responsibilities, log_prob_norm = self._e_step(X_batch)
        mean_ll = np.mean(log_prob_norm)

        if iteration == 0:
            self.initial_log_likelihood = mean_ll
            current_lower_bound = mean_ll
        else:
            if mean_ll >= self.initial_log_likelihood:
                self.weights_, self.means_, self.covariances_ = self._partial_m_step(X_batch, responsibilities)
                current_lower_bound = mean_ll
            else:
                current_lower_bound = prev_lower_bound_smooth

        self.lower_bounds.append(current_lower_bound)
        
        current_lower_bound_smooth = self._compute_smooth_lower_bound(
            current_lower_bound, prev_lower_bound_smooth, iteration)
        self.lower_bounds_smooth.append(current_lower_bound_smooth)
        
        return current_lower_bound, current_lower_bound_smooth

    def _compute_smooth_lower_bound(self, current_lower_bound, prev_lower_bound_smooth, iteration):
        """compute the smoothed lower bound."""
        if iteration > self.min_iter:
            beta = 0.99
        elif 100 < iteration <= self.min_iter:
            beta = 0.95
        else:
            beta = 0.90
        if iteration == 0:
            return current_lower_bound
        else:
            return prev_lower_bound_smooth * beta + (1 - beta) * current_lower_bound

    def _record_training_results(self, iteration):
        try:
            self.n_iter = iteration + 1
        except:
            self.n_iter = 0
        
        try:
            start_idx = max(0, iteration - self.min_iter)
            self.lower_bound = np.mean(self.lower_bounds[start_idx:])
        except:
            self.lower_bound = None

    @staticmethod
    def _run_single_partial_fit(model, X, do_init):
        model._single_partial_fit(X, do_init)
        return {
            'lower_bound': model.lower_bound,
            'weights': model.weights_,
            'means': model.means_,
            'covariances': model.covariances_,
            'n_iter': model.n_iter,
            'lower_bounds': model.lower_bounds,
            'lower_bounds_smooth': model.lower_bounds_smooth
        }

    def partial_fit(self, X, do_init=False, n_jobs=10, max_iter=None, min_iter=None):
        if self.max_iter == 0:
            print("Do not iterate, only create a GMM model.")
            return self
        start_time = time()
        if max_iter is not None:
            self.max_iter = max_iter
        if min_iter is not None:
            self.min_iter = min_iter

        if self.n_init == 1 or (not do_init and (self.weights_ is None or self.means_ is None or self.covariances_ is None)):
            self._single_partial_fit(X, do_init)
        else:
            print(f"Using {self.n_init} cores to find best!")
            models = [deepcopy(self) for _ in range(self.n_init)]
            with parallel_backend('loky', inner_max_num_threads=1):
                results = Parallel(n_jobs=min(n_jobs, self.n_init),
                )(delayed(self._run_single_partial_fit)(model, X, do_init) for model in models)

            lower_bounds = [r['lower_bound'] for r in results]
            best_idx = np.argmax(lower_bounds)
            best = results[best_idx]

            self.weights_ = best['weights']
            self.means_ = best['means']
            self.covariances_ = best['covariances']
            self.lower_bound = best['lower_bound']
            self.lower_bounds = best['lower_bounds']
            self.lower_bounds_smooth = best['lower_bounds_smooth']
            self.n_iter = best['n_iter']

        end_time = time()
        self.time = end_time - start_time
        Ncomplex = self._estimate_complexity(X)/(len(X)/self.batch_size)
        if self.Print==True:
            print(f"Batch-size={self.batch_size}")
            print(f"Ncomplex is between [{Ncomplex*self.min_iter:.0e},{Ncomplex*self.max_iter:.0e}], per iterate is {Ncomplex:.0e} ")
            print(f"Total fitting time: {self.time:.2f} s in {self.n_init} inits")
            print(f"Average time per iterate is {self.time/self.n_iter:.4f}")
            print(f"lower bound = {self.lower_bound}")
        return self

    def predict(self, X):
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        responsibilities, _ = self._e_step(X)
        return responsibilities

    def score_samples(self, X):
        log_prob = self._estimate_log_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_)
        return self._estimate_log_prob_norm(weighted_log_prob)
    
    def score(self, X):
        return np.mean(self.score_samples(X))
    
    def bic(self, X):
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(X.shape[0])
    
    def mbic(self, X):
        return self.bic(X)/X.shape[0]
    
    @staticmethod
    def _get_batch_indices(self, permutation, n_samples):
        start = np.random.randint(0, n_samples)
        end = start + self.batch_size
        if end <= n_samples:
            return permutation[start:end]
        else:
            return np.concatenate((permutation[start:], permutation[:end - n_samples]))

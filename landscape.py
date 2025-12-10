"""
This function supply visualize the landscape of GMM model 
using PCA.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

from GMM import GMM, Config
from Visualize import PhaseSpace

def get_flat_params(model):
    return np.concatenate([model.weights_, 
                           model.means_.flatten(), 
                           model.covariances_.flatten()])

def scale_params(params, scalers=None, n_components=2, n_dim=2):

    w_end = n_components
    m_end = w_end + n_components * n_dim
    c_end = m_end + n_components * n_dim * n_dim
    
    weights = params[:, :w_end]
    means = params[:, w_end:m_end]
    covs = params[:, m_end:c_end]
    if scalers is None:
        scaler_w = StandardScaler()
        scaler_m = StandardScaler()
        scaler_c = StandardScaler()
        
        weights_scaled = scaler_w.fit_transform(weights)
        means_scaled = scaler_m.fit_transform(means)
        covs_scaled = scaler_c.fit_transform(covs)
    else:
        scaler_w, scaler_m, scaler_c = scalers
        weights_scaled = scaler_w.transform(weights)
        means_scaled = scaler_m.transform(means)
        covs_scaled = scaler_c.transform(covs)
 
    scaled_params = np.hstack([weights_scaled, means_scaled, covs_scaled])
    
    return scaled_params, (scaler_w, scaler_m, scaler_c)

def inverse_scale_params(scaled_params, scalers, n_components=2, n_dim=2):
    w_end = n_components
    m_end = w_end + n_components * n_dim
    c_end = m_end + n_components * n_dim * n_dim
    
    scaler_w, scaler_m, scaler_c = scalers
    
    weights = scaler_w.inverse_transform(scaled_params[:, :w_end])
    means = scaler_m.inverse_transform(scaled_params[:, w_end:m_end])
    covs = scaler_c.inverse_transform(scaled_params[:, m_end:c_end])
    
    return np.hstack([weights, means, covs])

def construct_model(model, flat_params, epsilon=1e-3):
    n_comp = model.n_components
    n_dim = model.means_.shape[1] 
    
    w_end = n_comp
    m_end = w_end + n_comp * n_dim
    c_end = m_end + n_comp * n_dim * n_dim

    weights = flat_params[:w_end].copy()
    means = flat_params[w_end:m_end].reshape(n_comp, n_dim).copy()
    covs = flat_params[m_end:c_end].reshape(n_comp, n_dim, n_dim).copy()
     
    weights = np.clip(weights, 1e-10, 1.0)
    weights = weights / np.sum(weights)
    
    for i in range(n_comp):
        covs[i] = (covs[i] + covs[i].T) / 2
        
        try:
            np.linalg.cholesky(covs[i])
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(covs[i])
            if np.any(eigvals <= 0):
                eigvals = np.maximum(eigvals, epsilon)
                covs[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T
                covs[i] = (covs[i] + covs[i].T) / 2
    
    model.weights_ = weights
    model.means_ = means
    model.covariances_ = covs
    
    return model

def label_alignment(models_solutions, n_components=2, n_dim=2):
    standardized_models = []
    standardized_models.append(models_solutions[0])
    
    for model_idx in range(1, len(models_solutions)):
        model_solutions = models_solutions[model_idx]
        
        reference_final = models_solutions[0][-1]
        current_final = model_solutions[-1]
        
        w_end = n_components
        m_end = w_end + n_components * n_dim
        
        reference_means = reference_final[w_end:m_end].reshape(n_components, n_dim)
        current_means = current_final[w_end:m_end].reshape(n_components, n_dim)
        
        cost_matrix = np.zeros((n_components, n_components))
        for i in range(n_components):
            for j in range(n_components):
                cost_matrix[i, j] = np.linalg.norm(reference_means[i] - current_means[j])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapping = np.zeros(n_components, dtype=int)
        for ref_idx, curr_idx in zip(row_ind, col_ind):
            mapping[curr_idx] = ref_idx
                
        standardized_model = []
        for solution in model_solutions:
            w_end = n_components
            m_end = w_end + n_components * n_dim
            c_end = m_end + n_components * n_dim * n_dim
            
            weights = solution[:w_end]
            means = solution[w_end:m_end].reshape(n_components, n_dim)
            covs = solution[m_end:c_end].reshape(n_components, n_dim, n_dim)
            
            standardized_weights = np.zeros_like(weights)
            standardized_means = np.zeros_like(means)
            standardized_covs = np.zeros_like(covs)
            
            for curr_idx, ref_idx in enumerate(mapping):
                standardized_weights[ref_idx] = weights[curr_idx]
                standardized_means[ref_idx] = means[curr_idx]
                standardized_covs[ref_idx] = covs[curr_idx]
            
            standardized_flat = np.concatenate([
                standardized_weights,
                standardized_means.flatten(),
                standardized_covs.flatten()
            ])
            standardized_model.append(standardized_flat)
        
        standardized_models.append(np.array(standardized_model))
    
    return standardized_models
    

def main():
    import matplotlib.pyplot as plt
    from TNGloading import loadGalaxy, center, faceon
    from KinematicSolver import GravitySolver
    from sklearn.decomposition import PCA
    from AutoGMM import AutoGMM
    run = 'TNG100-3'
    basePath = f"/Users/yuwa/Tools/KD/{run}/output/"
    subID = 2
    snapNum = 99
    galaxy = loadGalaxy(basePath, run, snapNum, subID) 
    center(galaxy)
    faceon(galaxy)
    galaxy = GravitySolver(galaxy, Solver='Agama')
    X = np.column_stack((galaxy.s['e/emin'], galaxy.s['jz/jc']))
    Auto_GMM_model = AutoGMM(galaxy, n_components=2)
    Auto_GMM_model._morphology_class()
    Auto_GMM_model._2D_or_3D()
    X = Auto_GMM_model.X_train

    max_iter = 100

    config = Config(n_components=2, max_iter=1, min_iter=0, seed=42) 
    model = GMM(config=config)
    
    solutions = []
    losses = []
    for i in range(max_iter):
        model.fit(X)
        #if i == 0: PhaseSpace(X, model.means_, model.covariances_)
        solutions.append(get_flat_params(model))
        losses.append(-model.score(X))
    solutions = np.array(solutions)
    
    weights, means, covariances = Auto_GMM_model._ini_physics()
    config1 = Config(n_components=2, ini_means=means, ini_weights=weights, ini_covariances=covariances, max_iter=1, min_iter=0, seed=42) 
    model1 = GMM(config=config1)
    #PhaseSpace(X, means, covariances) 
    solutions1 = []
    losses1 = []
    for i in range(max_iter):
        model1.fit(X)
        solutions1.append(get_flat_params(model1))
        losses1.append(-model1.score(X))
    #PhaseSpace(X, model1.means_, model1.covariances_) 
    #PhaseSpace(X, model.means_, model.covariances_) 

    standardized_models = label_alignment([solutions, solutions1])
    solutions = standardized_models[0]
    solutions1= standardized_models[1]  

    ref_model = GMM(n_components=2, max_iter=1, min_iter=0)
    ref_model.fit(X)

    #scaler = StandardScaler()
    #solutions_scaled = scaler.fit_transform(solutions)
    #solutions1_scaled= scaler.transform(solutions1)
    all_solutions = np.vstack([solutions, solutions1])
    all_solutions_scaled, scalers = scale_params(params=all_solutions)
    index = np.int_(all_solutions_scaled.shape[0]/2)
    solutions_scaled = all_solutions_scaled[:index]
    solutions1_scaled= all_solutions_scaled[index:]

    all_solutions_scaled = np.vstack([solutions_scaled, solutions1_scaled])
    pca = PCA(n_components=2)
    pca.fit(all_solutions_scaled)

    solutions_pca = pca.transform(solutions_scaled)
    solutions1_pca = pca.transform(solutions1_scaled)

    first_pc_weights = pca.components_[0]

    print(first_pc_weights)
    print(f"explained_variance_ratio_: {pca.explained_variance_ratio_}")
    
    res = 50

    x_min = min(solutions_pca[:, 0].min(), solutions1_pca[:, 0].min())-1
    x_max = max(solutions_pca[:, 0].max(), solutions1_pca[:, 0].max())+1
    y_min = min(solutions_pca[:, 1].min(), solutions1_pca[:, 1].min())-1
    y_max = max(solutions_pca[:, 1].max(), solutions1_pca[:, 1].max())+1


    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X_grid, Y_grid = np.meshgrid(x, y)
    points_grid = np.c_[X_grid.ravel(), Y_grid.ravel()]
    
    loss_vals = []
    
    for point in points_grid:
        point_scaled = point.reshape(1, -1)
        theta_recon_scaled = pca.inverse_transform(point_scaled)[0]
        #theta_recon = scaler.inverse_transform(theta_recon_scaled.reshape(1, -1))[0]
        theta_recon = inverse_scale_params(theta_recon_scaled.reshape(1, -1), scalers)[0]
        construct_model(ref_model, theta_recon)
        loss = -(ref_model.score(X))
        loss_vals.append(loss)

    loss_vals = np.array(loss_vals).reshape(X_grid.shape)

    pca_errors = []

    for i, (original_param, original_loss) in enumerate(zip(solutions, losses)):
        if not np.isnan(original_loss) and original_loss < 1e10:              
            original_param_scaled, _ = scale_params(original_param.reshape(1, -1), scalers)
            pca_proj = pca.transform(original_param_scaled.reshape(1, -1))[0]
            pca_recon_scaled = pca.inverse_transform(pca_proj.reshape(1, -1))[0]
            pca_recon = inverse_scale_params(pca_recon_scaled.reshape(1, -1), scalers)[0]
            
            temp_model = GMM(n_components=2, max_iter=1, min_iter=0)
            temp_model.fit(X)            
            construct_model(temp_model, pca_recon)
            
            try:
                pca_recon_loss = -temp_model.score(X)
                if not np.isnan(pca_recon_loss) and pca_recon_loss < 1e10:
                    relative_error = abs(pca_recon_loss - original_loss)
                    pca_errors.append(relative_error)
            except:
                continue

    if pca_errors:   
        print(f"mean error: {np.mean(pca_errors):.4f}")
        print(f"max error: {np.max(pca_errors):.4f}")
    
    vmin = np.min(losses)
    vmax = np.max(losses)
    loss_vals = np.where(loss_vals > vmax, vmax, loss_vals)

    plt.figure(figsize=(7, 5))
    contour_final = plt.contourf(X_grid, Y_grid, loss_vals, levels=200, cmap='coolwarm_r', vmin=vmin, vmax=vmax)
    plt.colorbar(contour_final, label='|Log-Likelihood|', extend='neither')

    plt.scatter(solutions_pca[:,0], solutions_pca[:, 1], c=np.log10(np.arange(0,max_iter,1)) ,cmap='plasma')
    plt.scatter(solutions1_pca[:,0], solutions1_pca[:, 1], c=np.log10(np.arange(0,max_iter,1)) ,cmap='Grays')



    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

if __name__ == '__main__':
    main()

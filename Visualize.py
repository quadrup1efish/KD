"""
You should add : plotting the surface density image.
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d

def loss_profile(GMM, ax=None, show=False, **kwargs):
    if ax is None: fig, ax = plt.subplots()
    ax.plot(GMM.lower_bounds, **kwargs)
    ax.set_xlabel('Loss')
    ax.set_ylabel('Iteration')
    if show == True: plt.show()

def gaussian_ell(ax, mean, covariance, color):
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 0)
    widths = 2 * np.sqrt(2) * np.sqrt(eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    ellipse = matplotlib.patches.Ellipse(
        xy=mean,
        width=widths[0],
        height=widths[1],
        angle=angle,
        edgecolor=color,
        facecolor='none',
        linewidth=3,
        linestyle='solid',
        alpha=0.7,
    )
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], marker='x', color='k')

def PhaseSpace(X, means=None, covariances=None, ecut=-0.75, etacut=0.50, dims=2):
    try: ncs, dims = means.shape
    except: dims = dims
    axis_labels = ['$e/|e|_\mathregular{max}$', '$j_z/j_c$', '$j_p/j_c$']

    colors_colddisk = ['darkviolet', 'indigo', 'blue', 'cornflowerblue', 'lightskyblue', 'aliceblue']
    colors_warmdisk = ['green', 'yellowgreen', 'yellow', 'lightyellow']
    colors_bulge = ['darkred', 'firebrick', 'red', 'salmon', 'lightcoral', 'lightsalmon', 'mistyrose']
    colors_halo = ['darkgoldenrod','darkorange', 'orange', 'gold', 'yellow', 'lightyellow','whitesmoke']
    
    if dims == 3:
        projects = [[1,0],[1,2],[2,0]]
        ranges = [[X[:,0].min(), X[:,0].max()],[X[:,1].min(),X[:,1].max()],[X[:,2].min(),X[:,2].max()]]#[[-1,0],[-1,1],[0,1]]
        fig, axes = plt.subplots(1, 3, figsize=(8,2))
    if dims == 2:
        projects = [[1,0]]
        ranges = [[X[:,0].min(), X[:,0].max()],[X[:,1].min(),X[:,1].max()]]##[[-1,0],[-1,1]]
        fig, axes = plt.subplots(1, 1, figsize=(6,4))
        axes=[axes]
    
    N = len(X)
    bins_total = int(N/100)
    xrange = np.ptp(X[:, 0])
    yrange = np.ptp(X[:, 1])
    area_ratio = (xrange * yrange)
    bin_length = np.sqrt(area_ratio / bins_total)
    bins = max(int(np.sqrt(area_ratio)/bin_length),20)
    hist_params = {
        'bins': bins,
        'cmap': 'Spectral',
        'cmin': 1,
        'norm': LogNorm(),
    }
    
    if means is not None and covariances is not None:
        bulge_index = (means[:, 0] < ecut) & (means[:, 1] < etacut)
        halo_index  = (means[:, 0] > ecut) & (means[:, 1] < etacut)
        warmdisk_index = (means[:, 1] > etacut) & (means[:, 1] < 0.85)
        colddisk_index = (means[:, 1] > 0.85)

        bulge_means = means[bulge_index]
        bulge_covariances = covariances[bulge_index]

        halo_means = means[halo_index]
        halo_covariances = covariances[halo_index]

        warmdisk_means = means[warmdisk_index]
        warmdisk_covariances = covariances[warmdisk_index]

        colddisk_means = means[colddisk_index]
        colddisk_covariances = covariances[colddisk_index]

        bulge_sort_idx = bulge_means[:, 0].argsort()
        bulge_means = bulge_means[bulge_sort_idx]
        bulge_covariances = bulge_covariances[bulge_sort_idx]

        halo_sort_idx = halo_means[:, 0].argsort()
        halo_means = halo_means[halo_sort_idx]
        halo_covariances = halo_covariances[halo_sort_idx]

        warmdisk_sort_idx = warmdisk_means[:, 1].argsort()[::-1]
        warmdisk_means = warmdisk_means[warmdisk_sort_idx]
        warmdisk_covariances = warmdisk_covariances[warmdisk_sort_idx]

        colddisk_sort_idx = colddisk_means[:, 1].argsort()[::-1]
        colddisk_means = colddisk_means[colddisk_sort_idx]
        colddisk_covariances = colddisk_covariances[colddisk_sort_idx]
    
    for i, proj in enumerate(projects):
        im = axes[i].hist2d(X[:, proj[0]], X[:, proj[1]], range=[ranges[proj[0]],ranges[proj[1]]],**hist_params)
        axes[i].set_xlabel(f"{axis_labels[proj[0]]}", fontsize=15)
        axes[i].set_ylabel(f"{axis_labels[proj[1]]}", fontsize=15)
        axes[i].tick_params(labelsize=12)
        if means is not None and covariances is not None:
            for j, (mean, covariance) in enumerate(zip(bulge_means, bulge_covariances)):
                gaussian_ell(axes[i], mean[proj], covariance[np.ix_(proj, proj)], colors_bulge[j])
            for j, (mean, covariance) in enumerate(zip(halo_means, halo_covariances)):
                gaussian_ell(axes[i], mean[proj], covariance[np.ix_(proj, proj)], colors_halo[j])
            for j, (mean, covariance) in enumerate(zip(colddisk_means, colddisk_covariances)):
                gaussian_ell(axes[i], mean[proj], covariance[np.ix_(proj, proj)], colors_colddisk[j])
            for j, (mean, covariance) in enumerate(zip(warmdisk_means, warmdisk_covariances)):
                gaussian_ell(axes[i], mean[proj], covariance[np.ix_(proj, proj)], colors_warmdisk[j])
    cbar = plt.colorbar(im[3])
    cbar.set_label('$N_{*}$', fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    #plt.xlabel('$j_z/j_c$', fontsize=15)
    #plt.ylabel('$e/|e|_\mathregular{max}$', fontsize=15)
    plt.show()
    plt.close()

def SurfaceDensity(Particles_info):
    pass

    

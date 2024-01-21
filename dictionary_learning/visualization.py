import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


def gmm_density1d(μ, var, weights, x):
    K = μ.shape[0]
    y = 0
    for j in range(K):
        y += weights[j] * sps.norm.pdf(x, loc=μ[j], scale=np.sqrt(var[j]))
    return y.reshape(x.shape)


def gmm_density2d(μ, Σ, weights, x):
    K = μ.shape[0]
    weights = weights.reshape(1,K)
    y = 0
    for j in range(K):
        y += weights[0,j] * sps.multivariate_normal.pdf(x, mean=μ[j], cov=Σ[j])
    return y


def gaussian_density2d(μ, Σ, x):
    return sps.multivariate_normal.pdf(x, mean=μ, cov=Σ)


def display_gaussian(μ, Σ, n=50, ax=0, bx=1, ay=0, by=1, cmap='viridis', axis=None):
    if axis is None:
        axis = plt.gca()

    x = np.linspace(ax, bx,num=n)
    y = np.linspace(ay, by,num=n)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = gaussian_density2d(μ, Σ, XX)
    Z = Z.reshape(X.shape)
    plt.axis('equal')
    return axis.contour(X, Y, Z, 8, cmap=cmap)    


def display_gmm(gmm, n=50, ax=0, bx=1, ay=0, by=1, cmap='viridis', axis=None):
    if axis is None:
        axis = plt.gca()
        
    β, μ, Σ = gmm.weights, gmm.means, gmm.covs
    
    x = np.linspace(ax, bx,num=n)
    y = np.linspace(ay, by,num=n)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = gmm_density2d(μ, Σ, β, XX)
    Z = Z.reshape(X.shape)
    plt.axis('equal')
    return axis.contour(X, Y, Z, 8, cmap=cmap)


def scatterplot_errorbars(x, y, x_unique=None, ax=None, c='steelblue', **kwargs):
    if x_unique is None:
        x_unique = np.unique(x)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    means, stds = [], []
    for xi in x_unique:
        means.append(y[np.where(x == xi)[0]].mean())
        stds.append(y[np.where(x == xi)[0]].std())
    
    ax.errorbar(x=x_unique, y=means, yerr=stds, fmt='o', mfc='w', zorder=2.5, **kwargs)
    ax.scatter(x=x_unique, y=means, c=c, zorder=5)

    return ax
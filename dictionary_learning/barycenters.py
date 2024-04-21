r"""Module for computing Wasserstein barycenters

## References

[Cuturi and Doucet, 2014] Cuturi, M., & Doucet, A. (2014, June). Fast computation of Wasserstein barycenters.
                            In International conference on machine learning (pp. 685-693). PMLR.

[Álvarez-Esteban et al., 2016] Álvarez-Esteban, P. C., Del Barrio, E., Cuesta-Albertos, J. A., & Matrán, C. (2016).
                                A fixed-point approach to barycenters in Wasserstein space. Journal of Mathematical
                                Analysis and Applications, 441(2), 744-762.

[Cuturi, 2013] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
                In Advances in neural information processing systems, 26.
"""


import ot
import time
import torch
import numpy as np

import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import linprog

from dictionary_learning.utils import unif
from dictionary_learning.utils import sqrtm
from dictionary_learning.losses import label_loss
from dictionary_learning.gmm import GaussianMixtureModel
from dictionary_learning.losses import parametric_bures_wasserstein
from dictionary_learning.mapping import LinearOptimalTransportMapping
from dictionary_learning.initialization import BarycenterInitializer


def wasserstein_barycenter_with_cost(XP, YP=None, n_samples=None, ϵ=0.0, α=None, β=None, num_iter_max=100,
                           num_iter_sinkhorn=1000, τ=1e-4, verbose=False, inner_verbose=False,
                           initialization='random', propagate_labels=False, penalize_labels=False,
                           device='cuda', log=False, covariance_type='full', label_metric='l2'):
    r"""Computes the Wasserstein Barycenter

    $$ \hat{B}( \alpha, \mathcal{P} ) = \underset{P}{ \text{argmin} }\sum_{k=1}^{K} \alpha_{k} W_{c}(P, P_{k} ),$$

    for a list of distributions $\mathcal{P}$, containing $\hat{P}_{1}, \cdots ,\hat{P}_{K}$ and weights $\alpha \in \Delta_{K}$. Each
    distribution is parametrized through their support $\mathbf{X}^{( P_{k} )}, k=1, \cdots ,K$. This consists on a
    implementation of the Free-Support Wasserstien Barycenter of [[Cuturi and Doucet, 2014]](https://proceedings.mlr.press/v32/cuturi14.html).
    Our implementation relies on the fixed-point iteration of [[Alvarez-Esteban et al., 2016]](https://arxiv.org/abs/1511.05355),

    $$ \hat{B}^{(it+1)} = \psi( \hat{B}^{(it)} ), $$

    where $\psi(\hat{P}) = T_{it,\sharp}\hat{P}$, $T_{it} = \sum_{k}\alpha_{k}T_{k,it}$, for $T_{k,it}$, the
    barycentric mapping between $\hat{P}_{k}$ and $\hat{B}^{(it)}$.

    Args:
        XP: List of numpy arrays/torch tensors containing the support of each measure
        YP: Optional. List of numpy arrays/torch tensors containing the labels of each support point.
        n_samples: Optional. Number of points in the Barycenter's support. If not given, assumes that barycenter is
                   supported on sum(nk) points, for nk being the number of points the support of the k-th measure.
        ϵ: Optional. 0 if not given. Entropic Regularization penalty. For ϵ, OT is solved using the Sinkhorn algorithm of [[Cuturi, 2013]](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).
        num_iter_max: Optional. Maximum number of iterations for computing the barycenter.
        num_iter_sinkhorn: Optional. When ϵ > 0, maximum number of iterations for the Sinkhorn algorithm. Ignored if ϵ = 0.0.
        τ: Optional. Tolerance between barycentric iterations. If $\lVert \mathbf{X}^{(B^{(it+1)})} - \mathbf{X}^{(B^{(it)})} \rVert_{2}^{2} < \tau$, the algorithm stops.
        verbose: Optional. If True, displays information about barycenter iterations
        inner_verbose: Optional. If True, displays information about Sinkhorn's iterations
        initialization: Optional. Either 'random' or 'samples'. If 'random', initializes the Barycenter support
                        randomly, i.e., $\mathbf{x}_{i}^{(B^{(0)})} \sim \mathcal{N}(\mathbf{0},\mathbf{I}_{d})$.
                        If 'samples', sub-samples the support of each measure for creating the initial barycenter
                        support
        propagate_labels: Optional. If True, uses Wasserstein propagation for generating barycenter labels.
                          Ignored if labels are not provided.
        penalize_labels: Optional.  If True, penalizes OT plans that mix classes
        device: Optional. Either 'cpu' or 'cuda'. 'cuda' should only be provided if GPUs are available.
        log: Optional. If True, saves additional information in a dictionary.
    """
    if YP is None:
        if propagate_labels:
            raise ValueError("Expected labels to be given in 'y' for 'propagate_labels' = True")
        if penalize_labels:
            raise ValueError("Expected labels to be given in 'y' for 'penalize_labels' = True")
    dtype = XP[0].dtype
    device = XP[0].device
    init = BarycenterInitializer(n_samples=n_samples, type=initialization, device=device,
                                 covariance_type=covariance_type)

    if α is None:
        α = unif(len(XP), device=device, dtype=dtype)

    if n_samples is None:
        # If number of points is not provided,
        # assume that the support of the barycenter
        # has sum(nsi) where si is the i-th source
        # domain.
        n_samples = int(np.sum([len(XPk) for XPk in XP]))

    it = 0
    comp_start = time.time()

    if YP is None:
        XB = init(torch.cat(XP, dim=0))
    else:
        XB, YB = init(torch.cat(XP, dim=0), torch.cat(YP, dim=0))

    # Displacement of points in the support
    δ = τ + 1
    old_L = np.inf
    # Create uniform weights
    u_P = [unif(len(XPk), device=device) for XPk in XP]
    u_B = unif(len(XB), device=device)

    if verbose:
        print("-" * (26 * 4 + 1))
        print("|{:^25}|{:^25}|{:^25}|{:^25}|".format('Iteration', 'Loss', 'δLoss', 'Elapsed Time'))
        print("-" * (26 * 4 + 1))

    if log:
        extra_ret = {'transport_plans': [], 'd_loss': [], 'loss_hist': []}

    while (δ > τ and it < num_iter_max):
        # NOTE: Here we solve the barycenter problem without calculating gradients at each iteration,
        # as per the envelope theorem, we only need to compute gradients at optimality.
        with torch.no_grad():
            tstart = time.time()
            C, π = [], []

            for k in range(len(XP)):
                print('XP[k]', XP[k])
                print('XB', XB)
                C_k = torch.cdist(XP[k], XB, p=2) ** 2
                _β = β if β is not None else C_k.max()
                if penalize_labels:
                    if label_metric.lower() == 'l2':
                        YPk_double = YP[k].double()
                        YB_double = YB.double()
                        C_k = C_k + _β * torch.cdist(YPk_double, YB_double, p=2) ** 2
                        #C_k = C_k + _β * torch.cdist(YP[k], YB, p=2) ** 2
                    elif label_metric.lower() == 'delta':
                        YPk_double = YP[k].double()
                        YB_double = YB.double()
                        C_k = C_k + _β * label_loss(YPk_double, YB_double).to(device)
                        #C_k = C_k + _β * label_loss(YP[k], YB).to(device)
                    else:
                        raise ValueError('Invalid label metric {}'.format(label_metric.lower()))

                C.append(C_k)
                if ϵ > 0.0:
                    π_k = ot.sinkhorn(u_P[k], u_B, C_k, ϵ * C_k.max(), numItermax=num_iter_sinkhorn,
                                      verbose=inner_verbose, warn=False)
                else:
                    π_k = ot.emd(u_P[k], u_B, C_k)
                π.append(π_k.to(dtype))
            XB = sum([α_k * n_samples * torch.mm(π_k.T, XP_k) for α_k, π_k, XP_k in zip(α, π, XP)])
            if propagate_labels:
                #YB = sum([α_k * n_samples * torch.mm(π_k.T, YPk) for α_k, π_k, YPk in zip(α, π, YP)])
                YB = sum([α_k.double() * n_samples * torch.mm(π_k.double().T, YPk.double()) for α_k, π_k, YPk in zip(α, π, YP)])
            L = sum([torch.sum(C_k * π_k) for C_k, π_k in zip(C, π)])
            δ = torch.norm(L - old_L) / n_samples
            old_L = L
            tfinish = time.time()

            if verbose:
                δt = tfinish - tstart
                print("|{:^25}|{:^25}|{:^25}|{:^25}|".format(it, L, δ, δt))

            if log:
                extra_ret['loss_hist'].append(L)
                extra_ret['d_loss'].append(δ)

            it += 1
    if verbose:
        print("-" * (26 * 4 + 1))
        print("Barycenter calculation took {} seconds".format(time.time() - comp_start))
    # Re-evaluate the support at optimality for calculating the gradients
    # NOTE: now we define the support while holding its gradients w.r.t. the weight vector
    # and eventually the support.
    XB = sum([α_k * n_samples * torch.mm(π_k.T, XPk) for α_k, π_k, XPk in zip(α, π, XP)])
    if propagate_labels:
        #YB = sum([α_k * n_samples * torch.mm(π_k.T, YPk) for α_k, π_k, YPk in zip(α, π, YP)])
        YB = sum([α_k.double() * n_samples * torch.mm(π_k.double().T, YPk.double()) for α_k, π_k, YPk in zip(α, π, YP)])
        if log:
            extra_ret['transport_plans'] = π
            return XB, YB, extra_ret
        return XB, YB
    if log:
        extra_ret['transport_plans'] = π
        return XB, extra_ret


    return XB,C , π








def legacy_wasserstein_barycenter(XP, YP, ϵ=0.0, α=None, β=None, num_iter_max=100,
                                  num_iter_sinkhorn=1000, τ=1e-4, verbose=False, inner_verbose=False,
                                  initialization='random', penalize_labels=False,
                                  device='cuda', log=False, covariance_type='full', label_metric='l2'):
    dtype = XP[0].dtype
    device = XP[0].device
    n_samples = int(np.sum([len(XPk) for XPk in XP]))
    init = BarycenterInitializer(n_samples=n_samples, type=initialization, device=device, covariance_type=covariance_type)
    
    if α is None:
        α = unif(len(XP), device=device, dtype=dtype)

    it = 0
    comp_start = time.time()

    if YP is None:
        XB = init(torch.cat(XP, dim=0))
    else:
        XB, _ = init(torch.cat(XP, dim=0), torch.cat(YP, dim=0))
    YB = torch.cat(YP, dim=0)

    # Displacement of points in the support
    δ = τ + 1
    old_L = np.inf
    # Create uniform weights
    u_P = [unif(len(XPk), device=device) for XPk in XP]
    u_B = unif(len(XB), device=device)

    if verbose:
        print("-" * (26 * 4 + 1))
        print("|{:^25}|{:^25}|{:^25}|{:^25}|".format('Iteration', 'Loss', 'δLoss', 'Elapsed Time'))
        print("-" * (26 * 4 + 1))

    if log:
        extra_ret = {'transport_plans': [], 'd_loss': [], 'loss_hist': []}

    while (δ > τ and it < num_iter_max):
        # NOTE: Here we solve the barycenter problem without calculating gradients at each iteration,
        # as per the envelope theorem, we only need to compute gradients at optimality.
        with torch.no_grad():
            tstart = time.time()
            C, π = [], []

            for k in range(len(XP)):
                C_k = torch.cdist(XP[k], XB, p=2) ** 2
                _β = β if β is not None else C_k.max()
                YPk_double = YP[k].double()
                YB_double = YB.double()
                C_k = C_k + _β * label_loss(YPk_double, YB_double).to(device)
                #C_k = C_k + _β * label_loss(YP[k], YB).to(device)
                    
                C.append(C_k)
                if ϵ > 0.0:
                    π_k = ot.sinkhorn(u_P[k], u_B, C_k, ϵ * C_k.max(), numItermax=num_iter_sinkhorn,
                                      verbose=inner_verbose, warn=False)
                else:
                    π_k = ot.emd(u_P[k], u_B, C_k)
                π.append(π_k.to(dtype))
            XB = sum([α_k * n_samples * torch.mm(π_k.T, XP_k) for α_k, π_k, XP_k in zip(α, π, XP)])
            L = sum([torch.sum(C_k * π_k) for C_k, π_k in zip(C, π)])
            δ = torch.norm(L - old_L) / n_samples
            old_L = L
            tfinish = time.time()

            if verbose:
                δt = tfinish - tstart
                print("|{:^25}|{:^25}|{:^25}|{:^25}|".format(it, L, δ, δt))
            
            if log:
                extra_ret['loss_hist'].append(L)
                extra_ret['d_loss'].append(δ)

            it += 1
    if verbose:
        print("-" * (26 * 4 + 1))
        print("Barycenter calculation took {} seconds".format(time.time() - comp_start))
    # Re-evaluate the support at optimality for calculating the gradients
    # NOTE: now we define the support while holding its gradients w.r.t. the weight vector
    # and eventually the support.
    XB = sum([α_k * n_samples * torch.mm(π_k.T, XPk) for α_k, π_k, XPk in zip(α, π, XP)])
    return XB


def wasserstein_barycenter(XP, YP=None, n_samples=None, ϵ=0.0, α=None, β=None, num_iter_max=100,
                           num_iter_sinkhorn=1000, τ=1e-4, verbose=False, inner_verbose=False,
                           initialization='random', propagate_labels=False, penalize_labels=False,
                           device='cuda', log=False, covariance_type='full', label_metric='l2'):
    r"""Computes the Wasserstein Barycenter
    
    $$ \hat{B}( \alpha, \mathcal{P} ) = \underset{P}{ \text{argmin} }\sum_{k=1}^{K} \alpha_{k} W_{c}(P, P_{k} ),$$
    
    for a list of distributions $\mathcal{P}$, containing $\hat{P}_{1}, \cdots ,\hat{P}_{K}$ and weights $\alpha \in \Delta_{K}$. Each
    distribution is parametrized through their support $\mathbf{X}^{( P_{k} )}, k=1, \cdots ,K$. This consists on a
    implementation of the Free-Support Wasserstien Barycenter of [[Cuturi and Doucet, 2014]](https://proceedings.mlr.press/v32/cuturi14.html).
    Our implementation relies on the fixed-point iteration of [[Alvarez-Esteban et al., 2016]](https://arxiv.org/abs/1511.05355),
    
    $$ \hat{B}^{(it+1)} = \psi( \hat{B}^{(it)} ), $$

    where $\psi(\hat{P}) = T_{it,\sharp}\hat{P}$, $T_{it} = \sum_{k}\alpha_{k}T_{k,it}$, for $T_{k,it}$, the 
    barycentric mapping between $\hat{P}_{k}$ and $\hat{B}^{(it)}$.

    Args:
        XP: List of numpy arrays/torch tensors containing the support of each measure
        YP: Optional. List of numpy arrays/torch tensors containing the labels of each support point.
        n_samples: Optional. Number of points in the Barycenter's support. If not given, assumes that barycenter is
                   supported on sum(nk) points, for nk being the number of points the support of the k-th measure.
        ϵ: Optional. 0 if not given. Entropic Regularization penalty. For ϵ, OT is solved using the Sinkhorn algorithm of [[Cuturi, 2013]](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).
        num_iter_max: Optional. Maximum number of iterations for computing the barycenter.
        num_iter_sinkhorn: Optional. When ϵ > 0, maximum number of iterations for the Sinkhorn algorithm. Ignored if ϵ = 0.0.
        τ: Optional. Tolerance between barycentric iterations. If $\lVert \mathbf{X}^{(B^{(it+1)})} - \mathbf{X}^{(B^{(it)})} \rVert_{2}^{2} < \tau$, the algorithm stops.
        verbose: Optional. If True, displays information about barycenter iterations
        inner_verbose: Optional. If True, displays information about Sinkhorn's iterations
        initialization: Optional. Either 'random' or 'samples'. If 'random', initializes the Barycenter support
                        randomly, i.e., $\mathbf{x}_{i}^{(B^{(0)})} \sim \mathcal{N}(\mathbf{0},\mathbf{I}_{d})$.
                        If 'samples', sub-samples the support of each measure for creating the initial barycenter
                        support
        propagate_labels: Optional. If True, uses Wasserstein propagation for generating barycenter labels.
                          Ignored if labels are not provided.
        penalize_labels: Optional.  If True, penalizes OT plans that mix classes
        device: Optional. Either 'cpu' or 'cuda'. 'cuda' should only be provided if GPUs are available.
        log: Optional. If True, saves additional information in a dictionary.
    """
    if YP is None:
        if propagate_labels:
            raise ValueError("Expected labels to be given in 'y' for 'propagate_labels' = True")
        if penalize_labels:
            raise ValueError("Expected labels to be given in 'y' for 'penalize_labels' = True")
    dtype = XP[0].dtype
    device = XP[0].device
    init = BarycenterInitializer(n_samples=n_samples, type=initialization, device=device, covariance_type=covariance_type)
    
    if α is None:
        α = unif(len(XP), device=device, dtype=dtype)

    if n_samples is None:
        # If number of points is not provided,
        # assume that the support of the barycenter
        # has sum(nsi) where si is the i-th source
        # domain.
        n_samples = int(np.sum([len(XPk) for XPk in XP]))

    it = 0
    comp_start = time.time()

    if YP is None:
        XB = init(torch.cat(XP, dim=0))
    else:
        XB, YB = init(torch.cat(XP, dim=0), torch.cat(YP, dim=0))
    
    # Displacement of points in the support
    δ = τ + 1
    old_L = np.inf
    # Create uniform weights
    u_P = [unif(len(XPk), device=device) for XPk in XP]
    u_B = unif(len(XB), device=device)

    if verbose:
        print("-" * (26 * 4 + 1))
        print("|{:^25}|{:^25}|{:^25}|{:^25}|".format('Iteration', 'Loss', 'δLoss', 'Elapsed Time'))
        print("-" * (26 * 4 + 1))

    if log:
        extra_ret = {'transport_plans': [], 'd_loss': [], 'loss_hist': []}

    while (δ > τ and it < num_iter_max):
        # NOTE: Here we solve the barycenter problem without calculating gradients at each iteration,
        # as per the envelope theorem, we only need to compute gradients at optimality.
        with torch.no_grad():
            tstart = time.time()
            C, π = [], []

            for k in range(len(XP)):
                C_k = torch.cdist(XP[k], XB, p=2) ** 2
                _β = β if β is not None else C_k.max()
                if penalize_labels:
                    if label_metric.lower() == 'l2':
                        YPk_double = YP[k].double()
                        YB_double = YB.double()
                        C_k = C_k + _β * torch.cdist(YPk_double, YB_double, p=2) ** 2
                        #C_k = C_k + _β * torch.cdist(YP[k], YB, p=2) ** 2
                    elif label_metric.lower() == 'delta':
                        YPk_double = YP[k].double()
                        YB_double = YB.double()
                        C_k = C_k + _β * label_loss(YPk_double, YB_double).to(device)
                        #C_k = C_k + _β * label_loss(YP[k], YB).to(device)
                    else:
                        raise ValueError('Invalid label metric {}'.format(label_metric.lower()))
                    
                C.append(C_k)
                if ϵ > 0.0:
                    π_k = ot.sinkhorn(u_P[k], u_B, C_k, ϵ * C_k.max(), numItermax=num_iter_sinkhorn,
                                      verbose=inner_verbose, warn=False)
                else:
                    π_k = ot.emd(u_P[k], u_B, C_k)
                π.append(π_k.to(dtype))
            XB = sum([α_k * n_samples * torch.mm(π_k.T, XP_k) for α_k, π_k, XP_k in zip(α, π, XP)])
            if propagate_labels:
                #YB = sum([α_k * n_samples * torch.mm(π_k.T, YPk) for α_k, π_k, YPk in zip(α, π, YP)])
                YB = sum([α_k.double() * n_samples * torch.mm(π_k.double().T, YPk.double()) for α_k, π_k, YPk in zip(α, π, YP)])
            L = sum([torch.sum(C_k * π_k) for C_k, π_k in zip(C, π)])
            δ = torch.norm(L - old_L) / n_samples
            old_L = L
            tfinish = time.time()

            if verbose:
                δt = tfinish - tstart
                print("|{:^25}|{:^25}|{:^25}|{:^25}|".format(it, L, δ, δt))
            
            if log:
                extra_ret['loss_hist'].append(L)
                extra_ret['d_loss'].append(δ)

            it += 1
    if verbose:
        print("-" * (26 * 4 + 1))
        print("Barycenter calculation took {} seconds".format(time.time() - comp_start))
    # Re-evaluate the support at optimality for calculating the gradients
    # NOTE: now we define the support while holding its gradients w.r.t. the weight vector
    # and eventually the support.
    XB = sum([α_k * n_samples * torch.mm(π_k.T, XPk) for α_k, π_k, XPk in zip(α, π, XP)])
    if propagate_labels:
        #YB = sum([α_k * n_samples * torch.mm(π_k.T, YPk) for α_k, π_k, YPk in zip(α, π, YP)])
        YB = sum([α_k.double() * n_samples * torch.mm(π_k.double().T, YPk.double()) for α_k, π_k, YPk in zip(α, π, YP)])
        if log:
            extra_ret['transport_plans'] = π
            return XB, YB, extra_ret
        return XB, YB
    if log:
        extra_ret['transport_plans'] = π
        return XB, extra_ret
    return XB


def linear_wasserstein_barycenter(XP, YP=None, n_samples=None, ϵ=0.0, α=None, num_iter_max=100,
                                  τ=1e-4, verbose=False, initialization='random', propagate_labels=False,
                                  penalize_labels=False, device='cuda', log=False):
    if YP is None:
        if propagate_labels:
            raise ValueError("Expected labels to be given in 'y' for 'propagate_labels' = True")
        if penalize_labels:
            raise ValueError("Expected labels to be given in 'y' for 'penalize_labels' = True")
    device = XP[0].device
    init = BarycenterInitializer(n_samples=n_samples, type=initialization, device=device)
    
    if α is None:
        α = unif(len(XP))

    if n_samples is None:
        # If number of points is not provided,
        # assume that the support of the barycenter
        # has sum(nsi) where si is the i-th source
        # domain.
        n_samples = int(np.sum([len(XPk) for XPk in XP]))

    it = 0
    comp_start = time.time()

    XB, YB = init(torch.cat(XP, dim=0), torch.cat(YP, dim=0))
    
    # Displacement of points in the support
    δ = τ + 1
    old_L = np.inf

    if verbose:
        print("-" * (26 * 4 + 1))
        print("|{:^25}|{:^25}|{:^25}|{:^25}|".format('Iteration', 'Loss', 'δLoss', 'Elapsed Time'))
        print("-" * (26 * 4 + 1))

    if log:
        extra_ret = {'transport_plans': [], 'd_loss': [], 'loss_hist': []}

    while (δ > τ and it < num_iter_max):
        with torch.no_grad():
            tstart = time.time()
            TXB, T = [], []
            for k in range(len(XP)):
                Tk = LinearOptimalTransportMapping()
                TXB.append(Tk(XB, XP[k]))
                T.append(Tk)
            XB = sum([α_k * TXB_k for α_k, TXB_k in zip(α, TXB)])
            L = 0
            for Tk in T:
                L += Tk.dist()
            δ = torch.norm(L - old_L) / n_samples
            old_L = L
            tfinish = time.time()

            if verbose:
                δt = tfinish - tstart
                print("|{:^25}|{:^25}|{:^25}|{:^25}|".format(it, L, δ, δt))
            
            if log:
                extra_ret['loss_hist'].append(L)
                extra_ret['d_loss'].append(δ)

            it += 1
    if verbose:
        print("-" * (26 * 4 + 1))
        print("Barycenter calculation took {} seconds".format(time.time() - comp_start))
    XB = sum([α_k * Tk(XB, XP_k) for α_k, Tk, XP_k in zip(α, T, XP)])
    return XB


def bures_wasserstein_barycenter(means, covs, weights, num_iter_max=10, tol=1e-6, verbose=False, extra_ret=False, return_cost=False):
    mB = torch.einsum('i,ij->j', weights, means)

    it = 0
    var = np.inf
    
    covB = torch.eye(len(mB)).to(means[0].dtype)
    _covB = covB.clone()
    
    if extra_ret:
        history = [covB]

    while var > tol and it < num_iter_max:
        std_pos = sqrtm(covB)
        std_neg = torch.linalg.inv(std_pos)

        inner = sum([
            w_k * sqrtm(torch.matmul(std_pos, torch.matmul(cov_k, std_pos))) for w_k, cov_k in zip(weights, covs)
        ])
        inner = torch.matmul(inner, inner)

        covB = torch.matmul(std_neg, torch.matmul(inner, std_neg))
        
        if extra_ret:
            history.append(covB)

        it += 1
        var = torch.linalg.norm(covB - _covB, ord='fro')

        _covB = covB.clone()

        if verbose:
            print("It {}, var {}".format(it, var))

    if return_cost:
        cost = 0
        for mP_k, covP_k in zip(means, covs):
            cost += parametric_bures_wasserstein(mP=mP_k, mQ=mB, sP=covP_k, sQ=covB)
        if extra_ret:
            return mB, covB, history, cost
        return mB, covB, cost
    if extra_ret:
        return mB, covB, history
    return mB, covB


def barycenter_gmms(gmms, weights, tol_solution=1e-9, **kwargs):
    def create_cost_gmm(gmms, weights, **kwargs):
        assert len(gmms) == len(weights), "Expected same number of gmms and weights, but got {} and {}".format(len(gmms), len(weights))
        
        n_dimensions = gmms[0].d

        tup = ()
        for gmm in gmms:
            tup += (len(gmm),)

        C = np.zeros(tup)
        meansB = np.zeros(tup + (n_dimensions,))
        covsB = np.zeros(tup + (n_dimensions, n_dimensions))
        it = np.nditer(C, ['multi_index'])
        while not it.finished:
            indices = it.multi_index
            
            means = []
            covs = []
            for i_k, gmm_k in zip(indices, gmms):
                # NOTE: i_k is an index that indicates a specific component (μk,i_k, Σk,i_k) of the
                #       k-th GMM
                _, m_k, s_k = gmm_k.weights, gmm_k.means, gmm_k.covs

                # Selects the appropriate mean and covariance
                means.append(m_k[i_k])
                covs.append(s_k[i_k])
            means = torch.stack(means)
            covs = torch.stack(covs)
            mB, covB, cost = bures_wasserstein_barycenter(means, covs, weights, return_cost=True, **kwargs)
            C[indices] = cost
            meansB[indices] = mB
            covsB[indices] = covB
        
            it.iternext()

        return meansB, covsB, C
    n_distributions = len(gmms)
    meansB, covsB, C = create_cost_gmm(gmms, weights, **kwargs)

    β = [gmm.weights for gmm in gmms]

    n_parameters = 1
    n_constraints = 0
    
    for nc_k in C.shape:
        n_constraints += nc_k
        n_parameters *= nc_k

    index = 0
    A = np.zeros([n_constraints, n_parameters])
    b = np.zeros(n_constraints)

    for k1 in range(n_distributions):
        nc_k = C.shape[k1]
        b[index: index + nc_k] = β[k1]

        for ik in range(nc_k):
            Ap = np.zeros(C.shape)
            tup = ()
            for k2 in range(n_distributions):
                if k2 == k1:
                    tup += (ik,)
                else:
                    tup += (slice(0, C.shape[k2]),)
            Ap[tup] = 1
            A[index + ik, :] = Ap.flatten()
        index += C.shape[k1]
    A = A.tolist()
    b = b.tolist()
    Cflat = C.flatten().tolist()

    solution = linprog(Cflat, A_eq=A, b_eq=b).x.reshape(C.shape)
    solution[solution < tol_solution] = 0.0

    β_B, μ_B, Σ_B = [], [], []
    it = np.nditer(solution, ['multi_index'])

    while not it.finished:
        indices = it.multi_index

        if solution[indices] > 0.0:
            β_B.append(solution[indices])
            μ_B.append(meansB[indices])
            Σ_B.append(covsB[indices])

        it.iternext()

    β_B = torch.from_numpy(np.array(β_B))
    μ_B = torch.from_numpy(np.array(μ_B))
    Σ_B = torch.from_numpy(np.array(Σ_B))

    gmmB = GaussianMixtureModel()
    gmmB.set_gmm(β_B, μ_B, Σ_B)

    return solution, gmmB
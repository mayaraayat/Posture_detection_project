r"""Module for losses between probability distributions: $\mathcal{L}(P,Q)$."""

import ot
import torch

from functools import partial
from dictionary_learning.utils import unif
from dictionary_learning.utils import sqrtm


def compute_pairwise_wasserstein(Xs, Ys):
    uniques = Ys[0].argmax(dim=1).unique()

    M = [torch.zeros([len(uniques), len(uniques)]) for _ in Xs]

    for i, (Xsi, Ysi) in enumerate(zip(Xs, Ys)):
        for yu1 in uniques:
            ind1 = torch.where(Ysi.argmax(dim=1) == yu1)[0]
            for yu2 in uniques:
                ind2 = torch.where(Ysi.argmax(dim=1) == yu2)[0]

                x1, x2 = Xsi[ind1], Xsi[ind2]
                a = unif(len(x1), device=Xsi.device, dtype=Xsi.dtype)
                b = unif(len(x2), device=Xsi.device, dtype=Xsi.dtype)
                C = torch.cdist(x1, x2, p=2) ** 2
                
                M[i][yu1.item(), yu2.item()] = ot.emd2(a, b, C)

    return M


def label_loss(YA, YB):
    r"""Computes the 0-1 label loss between one-hot encoded label vectors,

    $$d_{\mathcal{Y}}(\mathbf{Y}^{(P)},\mathbf{Y}^{(Q)}) = \delta(\mathbf{Y}^{(P)} - \mathbf{Y}^{(Q)})$$

    __NOTE:__ this function is not differentiable w.r.t. YA nor YB.

    Args:
        YA: labels for samples in P.
        YB: labels for samples in Q.
        device: device to allocate the matrix. Either 'cpu' or 'cuda'.
    """
    n_classes_A, n_classes_B = YA.shape[1], YB.shape[1]
    labels_A, labels_B = YA.cpu().argmax(dim=1), YB.cpu().argmax(dim=1)
    M = torch.ones(n_classes_A, n_classes_B) - torch.eye(n_classes_A, n_classes_B)

    return M[labels_A, :][:, labels_B]


class DifferentiableDeltaLabelLoss(torch.nn.Module):
    def __init__(self, precomputed_M=None, n_classes_P=None, n_classes_Q=None):
        super(DifferentiableDeltaLabelLoss, self).__init__()

        if precomputed_M is not None:
            self.M = precomputed_M
        else:
            assert n_classes_P is not None and n_classes_Q is not None
            self.M = torch.ones(n_classes_P, n_classes_Q) - torch.eye(n_classes_P, n_classes_Q)


    def forward(self, YP, YQ):
        r"""Computes the 0-1 label loss between one-hot encoded label vectors,

        $$d_{\mathcal{Y}}(\mathbf{Y}^{(P)},\mathbf{Y}^{(Q)}) = \delta(\mathbf{Y}^{(P)} - \mathbf{Y}^{(Q)})$$

        __NOTE:__ this function is not differentiable w.r.t. YA nor YB.

        Args:
            YA: labels for samples in P.
            YB: labels for samples in Q.
            device: device to allocate the matrix. Either 'cpu' or 'cuda'.
        """

        return YP @ self.M @ YQ.T


class EnvelopeWassersteinLoss(torch.nn.Module):
    r"""Wasserstein loss using the Primal Kantorovich formulation. Gradients are computed using the Envelope Theorem."""
    def __init__(self, ϵ=0.0, num_iter_sinkhorn=20, debias=False):
        r"""Creates the loss object.
        
        Args:
            ϵ: entropic regularization penalty.
            num_iter_sinkhorn: maximum number of sinkhorn iterations. Only used for ϵ > 0.
            debias: whether or not compute the debiased sinkhorn loss. Only used when ϵ > 0.
        """
        super(EnvelopeWassersteinLoss, self).__init__()
        self.ϵ = ϵ
        self.num_iter_sinkhorn = num_iter_sinkhorn
        self.debias = debias

    def forward(self, XP, XQ):
        r"""Computes the Wasserstien loss between samples XP ~ P and XQ ~ Q,

        $$\mathcal{L}(\mathbf{X}^{(P)},\mathbf{X}^{(Q)}) = W_{2}(P,Q) = \underset{\pi\in U(\mathbf{u}_{n},\mathbf{u}_{m})}{\text{argmin}}\sum_{i=1}^{n}\sum_{j=1}^{m}\pi_{i,j}\lVert \mathbf{x}_{i}^{(P)} - \mathbf{x}_{j}^{(Q)} \rVert_{2}^{2}$$

        Args:
            XP: Tensor of shape (n, d) containing i.i.d samples from distribution P
            XQ: Tensor of shape (m, d) containing i.i.d samples from distribution Q
        """
        uP = unif(XP.shape[0], device=XP.device)
        uQ = unif(XQ.shape[0], device=XQ.device)
        
        if self.debias and self.ϵ > 0.0:
            CPP = torch.cdist(XP, XP) ** 2
            with torch.no_grad():
                πPP = ot.sinkhorn(uP, uP, CPP,
                                    reg=self.ϵ * CPP.detach().max(),
                                    numItermax=self.num_iter_sinkhorn,
                                    warn=False)

            CQQ = torch.cdist(XQ, XQ) ** 2
            with torch.no_grad():
                πQQ = ot.sinkhorn(uQ, uQ, CQQ,
                                    reg=self.ϵ * CQQ.detach().max(),
                                    numItermax=self.num_iter_sinkhorn,
                                    warn=False)

            CPQ = torch.cdist(XP, XQ) ** 2
            with torch.no_grad():
                πPQ = ot.sinkhorn(uP, uQ, CPQ,
                                    reg=self.ϵ * CPQ.detach().max(),
                                    numItermax=self.num_iter_sinkhorn,
                                    warn=False)

            loss_val = torch.sum(CPQ * πPP) - 0.5 * (torch.sum(CPP * πQQ) + torch.sum(CQQ * πPQ))
        else:
            C = torch.cdist(XP, XQ) ** 2
            with torch.no_grad():
                if self.ϵ > 0.0:
                    π = ot.sinkhorn(uP, uQ, C,
                                    reg=self.ϵ * C.detach().max(),
                                    numItermax=self.num_iter_sinkhorn,
                                    warn=False)
                else:
                    π = ot.emd(uP, uQ, C)
            loss_val = torch.sum(C * π)
        return loss_val


class JointWassersteinLoss(torch.nn.Module):
    r"""Wasserstein loss between joint distributions of labels and features, using the Primal Kantorovich formulation. Gradients are computed using the Envelope Theorem."""
    def __init__(self, ϵ=0.0, τ=0.0, β=None, label_metric=None, num_iter_sinkhorn=20, max_val=None, p=2, q=2):
        r"""Creates the loss object.
        
        Args:
            ϵ: entropic regularization penalty.
            τ: marginal OT plan relaxation. __remark:__ not used in the paper. Should be set to 0.
            num_iter_sinkhorn: maximum number of sinkhorn iterations. Only used for ϵ > 0.
        """
        super(JointWassersteinLoss, self).__init__()
        self.ϵ = ϵ
        self.τ = τ
        self.β = β
        self.p = p
        self.q = q
        self.max_val = max_val
        self.label_metric = label_metric
        self.num_iter_sinkhorn = num_iter_sinkhorn
        if label_metric is None:
            self.label_metric = lambda YP, YQ: torch.cdist(YP, YQ, p=self.p) ** self.q
    
    def forward(self, XQ, YQ, XP, YP, index=None):
        r"""Computes the Wasserstien loss between samples XP ~ P and XQ ~ Q,

        $$\mathcal{L}(\mathbf{X}^{(P)}, \mathbf{Y}^{(P)},\mathbf{X}^{(Q)}, \mathbf{Y}^{(Q)}) = W_{2}(P,Q) = \underset{\pi\in U(\mathbf{u}_{n},\mathbf{u}_{m})}{\text{argmin}}\sum_{i=1}^{n}\sum_{j=1}^{m}\pi_{i,j}(\lVert \mathbf{x}_{i}^{(P)} - \mathbf{x}_{j}^{(Q)} \rVert_{2}^{2}+\beta\lVert \mathbf{Y}_{i}^{(P)} - \mathbf{Y}_{j}^{(Q)} \rVert_{2}^{2})$$

        __Remark:__ as in the paper, we set $\beta = \text{max}_{i,j}\lVert \mathbf{x}_{i}^{(P)} - \mathbf{x}_{j}^{(Q)} \rVert_{2}^{2}$

        Args:
            XP: Tensor of shape (n, d) containing i.i.d features from distribution P
            YP: Tensor of shape (n, nc) containing i.i.d labels from distribution P
            XQ: Tensor of shape (m, d) containing i.i.d samples from distribution Q
            YQ: Tensor of shape (n, nc) containing i.i.d labels from distribution Q
        """
        a = unif(XP.shape[0], device=XP.device)
        b = unif(XQ.shape[0], device=XQ.device)
        YP_double = YP.double()
        YQ_double = YQ.double()
        device = YP_double.device
        YQ_double = YQ_double.to(device)
        CY = self.label_metric(YP_double, YQ_double)
        XP = XP.float()
        XQ = XQ.float()
        device = XP.device
        XQ = XQ.to(device)
        CX = torch.cdist(XP, XQ, p=self.p) ** self.q
        if YP is not None and YQ is not None:
            CY = self.label_metric(YP_double, YQ_double)
        else:
            CY = torch.zeros_like(CX)

        if CY.detach().max() == 0.0:
            _β = 0.0
        else:
            _β = self.β if self.β is not None else (CX.detach().max() / CY.detach().max())

        C = CX + _β * CY
        with torch.no_grad():
            if self.ϵ > 0.0:
                if self.τ > 0.0:
                    π = ot.unbalanced.sinkhorn_knopp_unbalanced(
                        a, b, C, reg=self.ϵ * C.max(), reg_m=self.τ,
                        numItermax=self.num_iter_sinkhorn
                    )
                else:
                    π = ot.sinkhorn(a, b, C,
                                    reg=self.ϵ * C.max(),
                                    numItermax=self.num_iter_sinkhorn,
                                    warn=False)
            else:
                if self.τ > 0.0:
                    π = ot.unbalanced.mm_unbalanced(
                        a, b, C, reg_m=self.τ
                    )
                else:
                    π = ot.emd(a, b, C)
        loss_val = torch.sum(C * π)
        return loss_val


class SupervisedPartialWassersteinDistance(torch.nn.Module):
    def __init__(self, n_dummies=1, m=0.9, β=None, label_metric='l2'):
        super(SupervisedPartialWassersteinDistance, self).__init__()
        self.m = m
        self.β = β
        self.n_dummies = n_dummies
        self.label_metric = label_metric

    def forward(self, XQ, YQ, XP, YP, index=None):
        a = unif(XP.shape[0], device=XP.device)
        b = unif(XQ.shape[0], device=XQ.device)
        XP = XP.float()
        XQ = XQ.float()
        CX = torch.cdist(XP, XQ) ** 2

        if self.label_metric.lower() == 'l2':
            CY = torch.cdist(YP.float(), YQ.float()) ** 2 if (YP is not None) and (YQ is not None) else torch.zeros_like(CX)
        elif self.label_metric.lower() == 'delta':
            CY = label_loss(YP.float(), YQ.float()) if (YP is not None) and (YQ is not None) else torch.zeros_like(CX)
        else:
            raise ValueError('Invalid label_metric {}'.format(self.label_metric.lower()))

        _β = self.β if self.β is not None else CX.detach().max()
        C = CX + _β * CY
        with torch.no_grad():
            π = ot.partial.partial_wasserstein(a=a.detach().numpy(),
                                               b=b.detach().numpy(),
                                               M=C.detach().numpy(), m=self.m, nb_dummies=self.n_dummies)
            π = torch.from_numpy(π).to(C.dtype).to(C.device)
        loss_val = torch.sum(C * π)
        return loss_val


class RenyiEntropy(torch.nn.Module):
    r"""Rényi Entropy regularization"""
    def __init__(self, β=1):
        r"""For a random variable $X$ assuming discrete values $1,\cdots,n$ with probabilities
        $\alpha_{i}, i=1,\cdots,n$, $\sum_{i}\alpha_{i}=1$, the Renyi entropy is,
        
        $$H_{\beta}(\alpha)=\dfrac{\beta}{1-\beta}\log \lVert \alpha \rVert_{\alpha}$$

        args:
            β: parameter for the Renyi entropy.
        """
        super(RenyiEntropy, self).__init__()
        self.β = β

    def forward(self, x):
        r"""computes the renyi entropy. __remark:__ x must be non-negative and sum to 1."""
        if self.β == 1:
            return - (x * x.log()).sum()
        else:
            return self.β / (1 - self.β) * (x.norm(p=self.β)).log()


def parametric_bures_wasserstein(mP, mQ, sP, sQ):
    sP_pos = sqrtm(sP)

    M = sqrtm(torch.mm(sP_pos, torch.mm(sQ, sP_pos)))
    bures_metric = torch.trace(sP) + torch.trace(sQ) - 2 * torch.trace(M)

    return torch.dist(mP, mQ, p=2) ** 2 + bures_metric ** 2


class JointDeltaWassersteinLoss(torch.nn.Module):
    def __init__(self, M, β=None, p=2, q=2):
        super(JointDeltaWassersteinLoss, self).__init__()
        self.M = M
        self.β = β
        self.p, self.q = p, q
    
    def forward(self, XQ, YQ, XP, YP, index=None):
        a = unif(XP.shape[0], device=XP.device)
        b = unif(XQ.shape[0], device=XQ.device)
        XP = XP.float()
        XQ = XQ.float()
        CX = torch.cdist(XP, XQ, p=self.p) ** self.q

        if YP is not None and YQ is not None:
            if index is not None:
                CY = YP @ self.M[index] @ YQ.T
            else:
                CY = YP @ self.M @ YQ.T
        else:
            CY = torch.zeros_like(CX)
            print(CY.shape)

        if CY.detach().max() == 0.0:
            _β = 0.0
        else:
            _β = self.β if self.β is not None else (CX.detach().max() / CY.detach().max())

        C = CX + _β * CY
        with torch.no_grad():
            π = ot.emd(a, b, C)
        loss_val = torch.sum(C * π)
        return loss_val
    

class SlicedWassersteinLoss(torch.nn.Module):
    def __init__(self, n_projections=50, use_max=False, p=2):
        super(SlicedWassersteinLoss, self).__init__()
        
        self.n_projections = n_projections
        self.use_max = use_max
        self.p = p
        
    def forward(self, XP, XQ):
        if self.use_max:
            return ot.sliced.max_sliced_wasserstein_distance(XP, XQ, n_projections=self.n_projections, p=self.p)
        else:
            return ot.sliced.sliced_wasserstein_distance(XP, XQ, n_projections=self.n_projections, p=self.p)
        

class MaximumMeanDiscrepancy(torch.nn.Module):
    def __init__(self, kernel='linear', bandwidth=None):
        super(MaximumMeanDiscrepancy, self).__init__()
        
        self.kernel = kernel
        self.bandwidth = bandwidth
        
    def forward(self, XP, XQ):
        if self.kernel == 'linear':
            KPP = torch.mm(XP, XP.T)
            KPQ = torch.mm(XP, XQ.T)
            KQQ = torch.mm(XQ, XQ.T)
        elif self.kernel == 'gaussian':
            _X = torch.cat([XP, XQ], dim=0)
            C = torch.cdist(_X, _X, p=2) ** 2
            if self.bandwidth == None:
                std = C.detach().mean()
            else:
                std = self.bandwidth
            _K = torch.exp(- C / (2 * std ** 2))
            
            KPP = _K[:len(XP), :][:, :len(XP)]
            KPQ = _K[:len(XP), :][:, len(XP):]
            KQQ = _K[len(XP):, :][:, len(XP):]
            
        return KPP.mean() + KQQ.mean() - 2 * KPQ.mean()
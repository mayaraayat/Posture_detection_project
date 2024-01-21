import ot
import torch
from dictionary_learning.utils import unif


def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix

    ref: https://discuss.pytorch.org/t/pytorch-square-root-of-a-positive-semi-definite-matrix/100138/5
    """
    vals, vecs = torch.linalg.eig(matrix)
    vals_pow = vals.pow(p)
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow.real


class BarycentricMapping(torch.nn.Module):
    def __init__(self, reg=0.0, num_iter_sinkhorn=50):
        super(BarycentricMapping, self).__init__()
        
        self.reg = reg
        self.num_iter_sinkhorn = num_iter_sinkhorn

    def fit(self, XP, XQ, p=None, q=None):
        # if p is not given, assume uniform
        if p is None:
            self.p = unif(len(XP))
        else:
            self.p = p

        # if q is not given, assume uniform
        if q is None:
            q = unif(len(XQ))

        # Calculates pairwise distances
        C = torch.cdist(XP, XQ, p=2) ** 2
        
        # Calculates transport plan
        with torch.no_grad():
            if self.reg > 0.0:
                self.π = ot.sinkhorn(self.p, q, C,
                                     reg=self.reg,
                                     numItermax=self.num_iter_sinkhorn,
                                     method='sinkhorn_log').detach()
            else:
                self.π = ot.emd(self.p, q, C).detach()

    def forward(self, XP, XQ, p=None, q=None):
        # Defines internal π
        self.fit(XP, XQ, p, q)

        return torch.mm((self.π / self.p[:, None]), XQ) , self.π


class SupervisedBarycentricMapping(torch.nn.Module):
    def __init__(self, reg=0.0, num_iter_sinkhorn=50, label_importance=None):
        super(SupervisedBarycentricMapping, self).__init__()
        
        self.reg = reg
        self.label_importance = label_importance
        self.num_iter_sinkhorn = num_iter_sinkhorn

    def fit(self, XP, XQ, YP, YQ, p=None, q=None):
        # if p is not given, assume uniform
        if p is None:
            self.p = unif(len(XP))
        else:
            self.p = p

        # if q is not given, assume uniform
        if q is None:
            q = unif(len(XQ))

        # Calculates pairwise distances
        C_features = torch.cdist(XP, XQ, p=2) ** 2
        C_labels = torch.cdist(YP, YQ, p=2) ** 2

        # Calculates overall ground cost
        β = C_features.detach().max() if self.label_importance is None else self.label_importance
        C = C_features + β * C_labels
        
        # Calculates transport plan
        with torch.no_grad():
            if self.reg > 0.0:
                self.π = ot.sinkhorn(self.p, q, C,
                                     reg=self.reg,
                                     numItermax=self.num_iter_sinkhorn,
                                     method='sinkhorn_log').detach()
            else:
                self.π = ot.emd(self.p, q, C).detach()

    def forward(self, XP, XQ, YP, YQ, p=None, q=None):
        # Defines internal π
        self.fit(XP, XQ, YP, YQ, p, q)

        TXP = torch.mm((self.π / self.p[:, None]), XQ)
        TYP = torch.mm((self.π / self.p[:, None]), YQ)

        return TXP, TYP


class LinearOptimalTransportMapping(torch.nn.Module):
    def __init__(self, reg=1e-6):
        super(LinearOptimalTransportMapping, self).__init__()
        self.reg = reg
        self.fitted = False

    def fit(self, XP, XQ):
        with torch.no_grad():
            self.mP = torch.mean(XP, dim=0, keepdim=True)
            self.mQ = torch.mean(XQ, dim=0, keepdim=True)

            self.sP = torch.cov(XP.T) + self.reg * torch.eye(XP.shape[1])
            self.sQ = torch.cov(XQ.T) + self.reg * torch.eye(XQ.shape[1])

            sP_pos = _matrix_pow(self.sP, p=0.5)
            sP_neg = _matrix_pow(self.sP, p=-0.5)

            self.M = torch.mm(sP_pos, torch.mm(self.sQ, sP_pos))
            self.A = torch.mm(sP_neg, torch.mm(self.M, sP_neg))
        self.fitted = True

    def dist(self):
        bures_metric = torch.trace(self.sP) + torch.trace(self.sQ) - 2 * torch.trace(self.M) ** (1 / 2)
        return torch.dist(self.mP, self.mQ, p=2) ** 2 + bures_metric

    def forward(self, XP, XQ, p=None, q=None):
        if self.fitted == False:
            self.fit(XP, XQ)

        return self.mQ + torch.mm(XP - self.mP, self.A)


class OracleMapping(torch.nn.Module):
    def __init__(self, mP, mQ, sP, sQ, reg=1e-6):
        super(OracleMapping, self).__init__()
        self.reg = reg
        
        self.mP = mP
        self.mQ = mQ

        self.sP = sP
        self.sQ = sQ

    def fit(self):
        sP_pos = _matrix_pow(self.sP, p=0.5)
        sP_neg = _matrix_pow(self.sP, p=-0.5)

        self.M = torch.mm(sP_pos, torch.mm(self.sQ, sP_pos))
        self.A = torch.mm(sP_neg, torch.mm(self.M, sP_neg))

    def forward(self, XP, XQ):
        self.fit()

        return self.mQ + torch.mm(XP - self.mP, self.A)
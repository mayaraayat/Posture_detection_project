import ot
import torch

from dictionary_learning.losses import parametric_bures_wasserstein


class OptimalTransportSolverGMM:
    def __init__(self, reg=0.0, num_iter_sinkhorn=100):
        self.reg = reg
        self.num_iter_sinkhorn = num_iter_sinkhorn
        self.ot_plan = None

    def __call__(self, gmmP, gmmQ):
        C = torch.zeros(len(gmmP), len(gmmQ))
        for k1 in range(len(gmmP)):
            for k2 in range(len(gmmQ)):
                _, μP_k1, ΣP_K1 = gmmP[k1]
                _, μQ_k2, ΣQ_K2 = gmmQ[k2]
                C[k1, k2] = parametric_bures_wasserstein(μP_k1, μQ_k2, ΣP_K1, ΣQ_K2)

        πP, πQ = gmmP.weights, gmmQ.weights
        if self.reg > 0.0:
            self.ot_plan = ot.emd(πP, πQ, C, reg=self.reg, numItermax=self.num_iter_sinkhorn, method='sinkhorn_log')
        else:
            self.ot_plan = ot.emd(πP, πQ, C)
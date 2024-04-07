import torch
import numpy as np


class AtomsInitializer:
    def __init__(self, samples_per_class, n_components, reg_cov=1e-9, estimate_cov=True):
        self.λ = reg_cov
        self.n_components = n_components
        self.samples_per_class = samples_per_class
        self.estimate_cov = estimate_cov

    def __call__(self, X, Y):
        y = Y.argmax(dim=1)
        
        XP, YP = [], []
        for _ in range(self.n_components):
            XP_k, YP_k = [], []
            for yu in y.unique():
                ind = torch.where(y == yu)[0]

                μ = X[ind].mean(dim=0)
                if self.estimate_cov:
                    Σ = X[ind].T.cov() + self.λ * torch.eye(X.shape[1])
                else:
                    Σ = torch.eye(X.shape[1])
                N_k = torch.distributions.MultivariateNormal(loc=μ, covariance_matrix=Σ)
                XP_k.append(N_k.sample_n(self.samples_per_class))
                YP_k.append(
                    torch.nn.functional.one_hot((yu * torch.ones(self.samples_per_class)).long(), num_classes=Y.shape[1]).type(μ.dtype)
                )
            XP_k = torch.cat(XP_k, dim=0)
            YP_k = torch.cat(YP_k, dim=0)
            ind = np.arange(len(XP_k))
            np.random.shuffle(ind)
            XP_k, YP_k = XP_k[ind], YP_k[ind]
            XP.append(XP_k)
            YP.append(YP_k)

        return XP, YP


class BarycenterInitializer:
    def __init__(self, n_samples, type='random', device='cuda', covariance_type='none'):
        assert type.lower() in ['random', 'class', 'samples', 'zeros']

        self.type = type.lower()
        self.n_samples = n_samples
        self.device = device
        self.covariance_type = covariance_type.lower()

    def __zeros_initializer(self, Xs, Ys=None):
        XB = torch.zeros([self.n_samples, Xs.shape[1]], dtype=Xs.dtype)
        if Ys is not None:
        #     ind = np.arange(len(Ys))
        #     sub_ind = np.random.choice(ind, size=self.n_samples)
        #     YB = Ys[sub_ind]
            YB = torch.zeros([self.n_samples, Ys.shape[1]], dtype=Ys.dtype)
            return XB.to(self.device), YB.to(self.device)
        return XB.to(self.device)      

    def __random_initializer(self, Xs, Ys=None):
        XB = torch.randn(self.n_samples, Xs.shape[1], dtype=Xs.dtype)
        if Ys is not None:
            ind = np.arange(len(Ys))
            sub_ind = np.random.choice(ind, size=self.n_samples)
            YB = Ys[sub_ind]
            return XB.to(self.device), YB.to(self.device)
        return XB.to(self.device)

    def __class_initializer(self, Xs, Ys=None):
        if Ys is None:
            raise ValueError("Expected Ys to be given when 'type' = 'class'")
        _, nd = Xs.shape
        _, nc = Ys.shape
        device = Xs.device

        XB = torch.zeros_like(Xs, dtype=Xs.dtype).to(device)
        yB = torch.zeros(len(Ys), dtype=Xs.dtype).to(device)

        for cl in torch.unique(Ys.argmax(dim=1)):
            ind = torch.where(Ys.argmax(dim=1) == cl)[0]
            μ = Xs[ind].mean(dim=0, keepdim=True)
            
            if self.covariance_type.lower() == 'none':
                L = Xs[ind].std(dim=0) * torch.eye(nd)
            elif self.covariance_type.lower() == 'diag':
                L = torch.diag(Xs[ind].std(dim=0))
            elif self.covariance_type.lower() == 'full':
                Σ = torch.cov(Xs[ind].T)
                L = torch.linalg.cholesky(Σ)
            else:
                raise ValueError("self.covariance_type == '{}'".format(self.covariance_type))

            XB[ind, :] = μ + (torch.randn(len(ind), nd) @ L).to(device)
            yB[ind] = cl.type(μ.dtype)
        YB = torch.nn.functional.one_hot(yB.long(), num_classes=nc).type(μ.dtype)

        ind = np.arange(len(XB))
        sub_ind = np.random.choice(ind, size=self.n_samples)
        XB, YB = XB[sub_ind], YB[sub_ind]

        return XB.to(self.device), YB.to(self.device)

    def __samples_initializer(self, Xs, Ys=None):
        ind = np.arange(len(Xs))
        sub_ind = np.random.choice(ind, size=self.n_samples)
        if Ys is not None:
            return Xs[sub_ind], Ys[sub_ind]
        return Xs[sub_ind].to(self.device)

    def __call__(self, Xs, Ys=None):
        if self.type == 'random':
            return self.__random_initializer(Xs, Ys)
        elif self.type == 'zeros':
            return self.__zeros_initializer(Xs, Ys)
        elif self.type == 'class':
            return self.__class_initializer(Xs, Ys)
        elif self.type == 'samples':
            return self.__samples_initializer(Xs, Ys)
        else:
            raise ValueError("Expected type to be either 'random', 'class', or 'samples' but {} was passed to the constructor.".format(self.type))
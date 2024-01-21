import torch
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel:
    def __init__(self, d=None, nc=None, reg=1e-6):
        self.d = d
        self.nc = nc
        self.reg = reg
        self.fitted = False

    def __len__(self):
        return int(self.nc)

    def __getitem__(self, index):
        return self.weights[index], self.means[index], self.covs[index]

    def set_gmm(self, weights, means, covs):
        assert len(weights) == len(means), "Expected to have the same number of weights and means."
        assert len(weights) == len(covs), "Expected to have the same number of weights and covariances."

        self.weights = weights
        self.means = means
        self.covs = covs

        self.d = means[0].shape[0]
        self.nc = len(means)
        self.reg = 1e-6

        self.fitted = True

    def fit(self, X, Y=None):
        if self.d is not None:
            assert X.shape[1] == self.d, "Expected X to have {} dimensions, but got {}".format(self.d, X.shape[1])
        else:
            self.d = X.shape[1]

        if Y is not None:
            _y = Y.argmax(dim=1)

            n = X.shape[0]
            self.nf = X.shape[1]
            self.nc = Y.shape[1]
            
            self.means, self.covs, self.weights = [], [], []
            for c in range(self.nc):
                ind = torch.where(_y == c)[0]
                Xc = X[ind]

                self.weights.append(len(ind) / n)
                self.means.append(torch.mean(Xc, dim=0))
                self.covs.append(torch.cov(Xc.T) + self.reg * torch.eye(self.nf))

            self.weights = torch.Tensor(self.weights)
            self.means = torch.stack(self.means)
            self.covs = torch.stack(self.covs)

            self.fitted = True
        elif self.nc is not None:
            gmm = GaussianMixture(n_components=self.nc)
            gmm.fit(X.cpu().numpy())

            self.weights = torch.from_numpy(gmm.weights_).to(X.dtype)
            self.means = torch.from_numpy(gmm.means_).to(X.dtype)
            self.covs = torch.from_numpy(gmm.covariances_).to(X.dtype)
        else:
            raise ValueError("When Y is not given, expected self.nc to be not None")
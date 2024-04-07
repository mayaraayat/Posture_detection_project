r"""Module for Dictionary Learning"""


import ot
import torch
import numpy as np

from tqdm.auto import tqdm
from itertools import chain
from abc import ABC, abstractmethod

from dictionary_learning.utils import check_device
from dictionary_learning.losses import RenyiEntropy
from dictionary_learning.losses import EnvelopeWassersteinLoss
from dictionary_learning.measures import DatasetMeasure
from dictionary_learning.measures import LabeledDatasetMeasure
from dictionary_learning.barycenters import wasserstein_barycenter



class AbstractDictionary(ABC):
    r"""Abstract Dictionary class"""
    def __init__(self,
                 n_samples=64,
                 n_components=2,
                 n_dim=2,
                 lr=1e-2,
                 device='cuda',
                 loss_fn=None,
                 regularizer_fn=None,
                 weight_initialization='uniform',
                 num_iter_barycenter=5,
                 num_iter_sinkhorn=20):
        r"""Initializes an abstract dictionary.
        
        Args:
            n_samples: number of samples in the barycenters support.
            n_components: number of atoms.
            n_dim: number of dimensions in the data.
            lr: learning_rate in gradient descent.
            loss_fn: loss between distributions.
            regularizer: regularizer for dictionary weights
            weight_initialization: either 'uniform' or 'random'.
            num_iter_barycenter: number of iterations used in the fixed-point algorithm.
            num_iter_sinkhorn: entropic regularization penalty.
        """
        self.device = check_device(device)
        self.n_samples = n_samples
        self.n_components = n_components
        self.n_dim = n_dim
        self.lr = lr
        self.loss_fn = EnvelopeWassersteinLoss() if loss_fn is None else loss_fn
        self.regularizer_fn = RenyiEntropy() if regularizer_fn is None else regularizer_fn
        self.weight_initialization = weight_initialization
        self.num_iter_barycenter = num_iter_barycenter
        self.num_iter_sinkhorn = num_iter_sinkhorn

        self.atoms = None
        self.weights = None
        self.fitted = False

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_atoms(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def reconstruct(self, **kwargs):
        pass

    @abstractmethod
    def transform(self):
        pass


class EmpiricalDictionary(AbstractDictionary):
    r"""Empirical Dictionary class. This represents a dictionary with empirical distributions as atoms,
    namely $\mathcal{P} = \{\hat{P}\}_{k=1}^{K}$, where each $\hat{P}_{k} = \frac{1}{n}\sum_{i=1}^{n}\delta_{\mathbf{x}_{i}^{(P_{k})}}$.
    
    __Remark:__ this class is abstract, as we do not implement the fit method.
    """
    def __init__(self,
                 n_samples=64,
                 n_components=2,
                 n_dim=2,
                 lr=1e-2,
                 η_A=0.0,
                 ϵ=1e-2,
                 device='cuda',
                 loss_fn=None,
                 regularizer_fn=None,
                 weight_initialization='uniform',
                 num_iter_barycenter=5,
                 num_iter_sinkhorn=20):
        r"""Initializes an empirical dictionary.
        
        Args:
            n_samples: number of samples in the barycenters support.
            n_components: number of atoms.
            n_dim: number of dimensions in the data.
            lr: learning_rate in gradient descent.
            ϵ: entropic regularization penalty.
            η_A: sparse regularization for the atoms weights. __remark:__ not used in our paper.
            loss_fn: loss between distributions.
            regularizer: regularizer for dictionary weights
            weight_initialization: either 'uniform' or 'random'.
            num_iter_barycenter: number of iterations used in the fixed-point algorithm.
            num_iter_sinkhorn: entropic regularization penalty.
        """
        super().__init__(n_samples=n_samples,
                         n_components=n_components,
                         n_dim=n_dim,
                         lr=lr,
                         device=device,
                         loss_fn=loss_fn,
                         regularizer_fn=regularizer_fn,
                         weight_initialization=weight_initialization,
                         num_iter_barycenter=num_iter_barycenter,
                         num_iter_sinkhorn=num_iter_sinkhorn)
        self.ϵ = ϵ
        self.η_A = η_A

    def initialize(self, n_datasets):
        atoms = [
            torch.randn(self.n_samples, self.n_dim, requires_grad=True, device=self.device).to(self.device)
            for _ in range(self.n_components)
        ]

        if self.weight_initialization == 'random':
            weights = torch.randn(n_datasets, self.n_components, requires_grad=True, device=self.device)
        elif self.weight_initialization == 'uniform':
            weights = torch.ones(n_datasets, self.n_components, requires_grad=True, device=self.device)
        else:
            raise ValueError("invalid initialization '{}'".format(self.weight_initialization))

        return atoms, weights

    def get_atoms(self):
        r"""Gets the entire support for each atom."""
        return [atom.detach().cpu().numpy() for atom in self.atoms]

    def get_weights(self):
        r"""Gets the learned weights."""
        return torch.nn.functional.softmax(self.weights.detach(), dim=1).to('cpu').numpy()

    def reconstruct(self):
        with torch.no_grad():
            A = torch.nn.functional.softmax(self.weights, dim=1)
            Q_rec = []
            for αi in A:
                XB = wasserstein_barycenter(self.atoms,
                                            α=αi,
                                            n_samples=self.n_samples,
                                            ϵ=self.ϵ,
                                            num_iter_max=self.num_iter_barycenter,
                                            num_iter_sinkhorn=self.num_iter_sinkhorn,
                                            device=self.device,
                                            verbose=False)
                Q_rec.append(XB)

        return Q_rec


class MinibatchDictionary(EmpiricalDictionary):
    r"""Minibatch Dictionary Class. This class represents a dictionary where we can sample
    from its atoms for learning. This is exactly the same as EmpiricalDictionary, except that
    we can call the fit method.
    """
    def __init__(self,
                 n_samples=1000,
                 batch_size=128,
                 n_components=2,
                 n_dim=2,
                 lr_features=1e-2,
                 lr_labels=None,
                 lr_weights=None,
                 grad_labels=True,
                 ϵ=0.0,
                 device='cuda',
                 loss_fn=None,
                 regularizer_fn=None,
                 weight_initialization='uniform',
                 num_iter_barycenter=5,
                 num_iter_sinkhorn=20):
        r"""Initializes a minibatch dictionary.
        
        Args:
            n_samples: number of samples in the barycenters support.
            batch_size: number of samples on each mini-batch. __remark:__ should be at most n_samples.
            n_components: number of atoms.
            n_dim: number of dimensions in the data.
            lr: learning_rate in gradient descent.
            ϵ: entropic regularization penalty.
            η_A: sparse regularization for the atoms weights. __remark:__ not used in our paper.
            loss_fn: loss between distributions.
            regularizer: regularizer for dictionary weights
            weight_initialization: either 'uniform' or 'random'.
            num_iter_barycenter: number of iterations used in the fixed-point algorithm.
            num_iter_sinkhorn: entropic regularization penalty.
        """
        super().__init__(n_samples=n_samples,
                         n_components=n_components,
                         n_dim=n_dim,
                         lr=0.0,
                         η_A=0.0,
                         ϵ=ϵ,
                         device=device,
                         loss_fn=loss_fn,
                         regularizer_fn=regularizer_fn,
                         weight_initialization=weight_initialization,
                         num_iter_barycenter=num_iter_barycenter,
                         num_iter_sinkhorn=num_iter_sinkhorn)

        self.lr_features = lr_features
        self.lr_labels = lr_labels
        self.lr_weights = lr_weights
        self.grad_labels = grad_labels
        self.batch_size = batch_size

        assert self.n_samples >= self.batch_size, "Expected number of samples ({}) to be at least the number of one\
                                                   batch ({}).".format(self.n_samples, self.batch_size)

    def sample_from_atoms(self, n=None, detach=False):
        batch_features, batch_labels = [], []

        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        # Sampling
        for tracker, XPk, YPk in zip(self.var_tracker, self.XP, self.YP):
            # If balanced sampling, needs to select sampler_per_class from each class
            if self.balanced_sampling:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                sampled_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    # Gets indices from current class
                    ind = torch.where(yPk == yu)[0]
                    # Randomly permutes labels
                    perm = torch.randperm(len(ind))
                    ind = ind[perm]
                    if samples_per_class is None:
                        # If n was not given, samples all samples from the said class
                        sampled_indices.append(ind[:])
                    else:
                        # Samples "samples_per_class" from given class
                        sampled_indices.append(ind[:samples_per_class])
                # Concatenates all indices
                sampled_indices = torch.cat(sampled_indices, dim=0).cpu().numpy()
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples), size=n)
            
            # Adds counter of sampling
            tracker[sampled_indices] += 1
            
            # Creates batch arrays
            features_k, labels_k = XPk[sampled_indices], YPk[sampled_indices]
            
            if self.grad_labels:
                labels_k = labels_k.softmax(dim=-1)

            if detach:
                features_k, labels_k = features_k.detach(), labels_k.detach()
            batch_features.append(features_k)
            batch_labels.append(labels_k)

        return batch_features, batch_labels


    def generate_batch_indices_without_replacement(self, batch_size=None):
        n_batches = self.n_samples // batch_size
        n_classes_per_batch = batch_size // self.n_classes

        for i in range(n_batches + 1):
            batch_indices = []
            for YPk in self.YP:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                atom_batch_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    indices = np.where(yPk == yu)[0]
                    atom_batch_indices.append(
                        indices[n_classes_per_batch * i: n_classes_per_batch * (i + 1)]
                    )
                atom_batch_indices = np.concatenate(atom_batch_indices)
                np.random.shuffle(atom_batch_indices)
                batch_indices.append(atom_batch_indices)
            yield batch_indices
            

    def fit(self,
            datasets,
            num_iter_max=100,
            batches_per_it=100):
        r"""Dictionary Learning method. In this method we minimize,

        $$L(\mathcal{P},\mathcal{A}) = \dfrac{1}{N}\sum_{\ell=1}^{N}\mathcal{L}(\hat{Q}_{\ell},\hat{B}_{\ell}),$$

        where $\hat{B}_{\ell} = \mathcal{B}(\alpha;\mathcal{P})$. Optimization is carried w.r.t. $\mathbf{x}_{j}^{(P_{k})}$ and
        $\alpha_{\ell} \in \Delta_{K}$.

        __Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf),
        we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator,

        $$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$

        Args:
            datasets: List of arrays $\mathbf{X}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}$.
            num_iter_max: maximum number of iterations for dictionary learning.
            batches_per_it: number of batches sampled per iteration.
        """
        self.atoms, self.weights = self.initialize(n_datasets=len(datasets))
        optimizer = torch.optim.Adam([atom for atom in self.atoms] + [self.weights], lr=self.lr)

        self.history = {'loss': [], 'weights': []}
        for it in range(num_iter_max):
            # Calculates the loss
            avg_it_loss = 0
            for _ in tqdm(range(batches_per_it)):
                optimizer.zero_grad()
                loss = 0

                # Change of Variables
                A = torch.nn.functional.softmax(self.weights, dim=1)
                XP = self.sample_from_atoms(n=self.batch_size)

                # Calculating Dictionary Learning Loss
                for Qℓ, α_ℓ in zip(datasets, A):
                    XQ_ℓ = Qℓ.sample()
                    XB = wasserstein_barycenter(XP=XP,
                                                 α=α_ℓ,
                                                 n_samples=self.batch_size,
                                                 ϵ=self.ϵ,
                                                 num_iter_max=self.num_iter_barycenter,
                                                 num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                 device=self.device)
                    loss += self.loss_fn(XQ_ℓ, XB)
                    loss += self.η_A * self.regularizer_fn(α_ℓ)

                loss.backward()
                optimizer.step()
                avg_it_loss += loss.item() / batches_per_it
            # Saves weights in history
            self.history['weights'].append(A.detach().cpu())
            self.history['loss'].append(avg_it_loss)
            print('It {}/{}, Loss: {}'.format(it, num_iter_max, loss.item(), avg_it_loss))
        self.fitted = True

    def transform(self, datasets, num_iter_max=10, batches_per_it=10):
        r"""Embeds distributions $\mathcal{Q} = \{\hat{Q}_{\ell}\}_{\ell=1}^{N}$ into the simplex $\Delta_{K}$. It is
        equivalent to the Barycentric Coordinate Regression method of [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf),
        
        $$\alpha^{\star}_{\ell} = \varphi(\hat{Q}) = \underset{\alpha \in \Delta_{K}}{\text{argmin}}\sum_{k=1}^{K}\alpha_{k}W_{2}(\hat{P}_{k},\hat{Q}_{\ell}).$$

        __Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf),
        we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator,

        $$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$

        Args:
            datasets: List of arrays $\mathbf{X}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}$.
            num_iter_max: maximum number of iterations for optimizing the cost function.
            batches_per_it: number of mini-batches sampled per iteration.
        """
        embeddings = torch.randn(len(datasets), self.n_components, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([embeddings], lr=self.lr)

        for step in range(num_iter_max):
            pbar = tqdm(range(batches_per_it))
            avg_it_loss = 0
            for _ in pbar:
                optimizer.zero_grad()

                loss = 0
                for Qℓ, aℓ in zip(datasets, embeddings):
                    XQℓ = Qℓ.sample()
                    XP = self.sample_from_atoms(n=self.batch_size, detach=True)
                    αℓ = torch.nn.functional.softmax(aℓ, dim=0)
                    XB = wasserstein_barycenter(XP=XP, α=αℓ,
                                                n_samples=self.batch_size,
                                                ϵ=self.ϵ,
                                                num_iter_max=self.num_iter_barycenter,
                                                num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                verbose=False)
                    XB.to(self.device)
                    loss += self.loss_fn(XB, XQℓ) + self.η_A * self.regularizer_fn(αℓ)
                loss.backward()
                avg_it_loss += loss.detach().item() / batches_per_it
                optimizer.step()
                pbar.set_description('loss: {}'.format(loss.detach().item()))
            print('Step {:<4}/{:<4} loss: {:^15}'.format(step, num_iter_max, avg_it_loss))

        return embeddings.softmax(dim=1).detach().cpu().numpy()

class LabeledMinibatchDictionary(MinibatchDictionary):
    r"""Class for dictionary learning when the support of distributions is labeled"""
    def __init__(self,
                 n_samples=1000,
                 batch_size=128,
                 n_components=2,
                 n_dim=2,
                 n_classes=None,
                 lr_features=1e-2,
                 lr_labels=None,
                 lr_weights=None,
                 ϵ=0.0,
                 device='cuda',
                 loss_fn=None,
                 regularizer_fn=None,
                 proj_grad=True,
                 grad_labels=True,
                 balanced_sampling=True,
                 sample_with_repetition=True,
                 weight_initialization='uniform',
                 barycenter_initialization='samples',
                 num_iter_barycenter=5,
                 num_iter_sinkhorn=20,
                 penalize_labels=True,
                 names=None,
                 track_atoms=False):
        r"""Initializes a labeled minibatch dictionary.
        
        Args:
            n_samples: number of samples in the barycenters support.
            batch_size: number of samples on each mini-batch. __remark:__ should be at most n_samples.
            n_components: number of atoms.
            n_dim: number of dimensions in the data.
            lr: learning_rate in gradient descent.
            ϵ: entropic regularization penalty.
            η_A: sparse regularization for the atoms weights. __remark:__ not used in our paper.
            loss_fn: loss between distributions. __remark:__ this function must accept 4 arguments in its \_\_call\_\_,
                     namely XP, YP, XQ, YQ
            regularizer: regularizer for dictionary weights
            weight_initialization: either 'uniform' or 'random'.
            num_iter_barycenter: number of iterations used in the fixed-point algorithm.
            num_iter_sinkhorn: entropic regularization penalty.
            penalize_labels: If True, includes a class-based penalty in the OT plan estimation.
        """
        super().__init__(n_samples=n_samples,
                         n_components=n_components,
                         batch_size=batch_size,
                         n_dim=n_dim,
                         lr_features=lr_features,
                         lr_labels=lr_labels,
                         lr_weights=lr_weights,
                         ϵ=ϵ,
                         device=device,
                         loss_fn=loss_fn,
                         regularizer_fn=regularizer_fn,
                         weight_initialization=weight_initialization,
                         num_iter_barycenter=num_iter_barycenter,
                         num_iter_sinkhorn=num_iter_sinkhorn)
        self.n_classes = n_classes
        self.grad_labels = grad_labels
        self.balanced_sampling = balanced_sampling
        self.barycenter_initialization = barycenter_initialization
        self.sample_with_repetition = sample_with_repetition
        self.penalize_labels = penalize_labels
        self.proj_grad = proj_grad
        self.names = names
        self.track_atoms = track_atoms
        self.var_tracker = [torch.zeros(n_samples) for _ in range(self.n_components)]

    def initialize(self, n_datasets, n_classes):
        r"""Initialization of atoms and weights. For $k=1,\cdots,K$, $j=1,\cdots,n$, $c=1,\cdots,n_{c}$, and
        $\ell=1,\cdots,N$,
        
        $$\mathbf{x}_{j}^{(P_{k})} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d}),$$
        $$\mathbf{y}_{j}^{(P_{k})} = \dfrac{e^{p_{c}}}{\sum_{c'}e^{p_{c'}}}, p_{c} \sim \mathcal{N}(0, 1).$$
        $$a_{k,\ell} \sim \mathcal{N}(0, 1).$$
        
        Args:
            n_datasets: equivalent to $\ell$. Number of datasets in Dictionary Learning.
            n_classes: number of classes in the datasets.
        """
        if self.names is None:
            self.names = ['Dataset {}'.format(ℓ) for ℓ in range(n_datasets)]
        XP_data = [torch.randn(self.n_samples, self.n_dim, requires_grad=True) for _ in range(self.n_components)]
        YP_data = [
            torch.nn.functional.one_hot(
                torch.from_numpy(np.random.randint(low=0, high=n_classes, size=self.n_samples)).long(),
                num_classes=n_classes
            ).float() for _ in range(self.n_components)
        ]

        if self.weight_initialization == 'random':
            weights_data = torch.randn(n_datasets, self.n_components, requires_grad=True, device=self.device)
        elif self.weight_initialization == 'uniform':
            weights_data = torch.ones(n_datasets, self.n_components, requires_grad=True, device=self.device)
        else:
            raise ValueError("invalid initialization '{}'".format(self.weight_initialization))
        weights_data = ot.utils.proj_simplex(weights_data.T).T
        return XP_data, YP_data, weights_data

    def sample_from_atoms(self, n=None, detach=False):
        batch_features, batch_labels = [], []

        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        # Sampling
        for tracker, XPk, YPk in zip(self.var_tracker, self.XP, self.YP):
            # If balanced sampling, needs to select sampler_per_class from each class
            if self.balanced_sampling:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                sampled_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    # Gets indices from current class
                    ind = torch.where(yPk == yu)[0]
                    # Randomly permutes labels
                    perm = torch.randperm(len(ind))
                    ind = ind[perm]
                    if samples_per_class is None:
                        # If n was not given, samples all samples from the said class
                        sampled_indices.append(ind[:])
                    else:
                        # Samples "samples_per_class" from given class
                        sampled_indices.append(ind[:samples_per_class])
                # Concatenates all indices
                sampled_indices = torch.cat(sampled_indices, dim=0).cpu().numpy()
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples), size=n)
            
            # Adds counter of sampling
            tracker[sampled_indices] += 1
            
            # Creates batch arrays
            features_k, labels_k = XPk[sampled_indices], YPk[sampled_indices]
            
            if self.grad_labels:
                labels_k = labels_k.softmax(dim=-1)

            if detach:
                features_k, labels_k = features_k.detach(), labels_k.detach()
            batch_features.append(features_k)
            batch_labels.append(labels_k)

        return batch_features, batch_labels


    def generate_batch_indices_without_replacement(self, batch_size=None):
        n_batches = self.n_samples // batch_size
        n_classes_per_batch = batch_size // self.n_classes

        for i in range(n_batches + 1):
            batch_indices = []
            for YPk in self.YP:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                atom_batch_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    indices = np.where(yPk == yu)[0]
                    atom_batch_indices.append(
                        indices[n_classes_per_batch * i: n_classes_per_batch * (i + 1)]
                    )
                atom_batch_indices = np.concatenate(atom_batch_indices)
                np.random.shuffle(atom_batch_indices)
                batch_indices.append(atom_batch_indices)
            yield batch_indices

    def get_atoms(self):
        r"""Gets the features and labels from learned atoms."""
        return [
            [X.detach().cpu().numpy().copy(), Y.detach().softmax(dim=1).cpu().numpy().copy()] for X, Y in zip(self.XP, self.YP)
        ]

    def fit_without_replacement(self, datasets, num_iter_max=100, batches_per_it=100):
        r"""Dictionary Learning method. In this method we minimize,

        $$L(\mathcal{P},\mathcal{A}) = \dfrac{1}{N}\sum_{\ell=1}^{N}\mathcal{L}(\hat{Q}_{\ell},\hat{B}_{\ell}),$$

        where $\hat{B}_{\ell} = \mathcal{B}(\alpha;\mathcal{P})$. Optimization is carried w.r.t. $\mathbf{x}_{j}^{(P_{k})}$ and
        $\alpha_{\ell} \in \Delta_{K}$.

        __Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf),
        we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator,

        $$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$

        Args:
            datasets: List of tuples containing 2 arrays, $\mathbf{X}^{(Q_{\ell})}$ and $\mathbf{Y}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}(X,Y)$.
            num_iter_max: maximum number of iterations for dictionary learning.
            batches_per_it: number of batches sampled per iteration.
        """
        if self.n_classes is None:
            self.n_classes = datasets[0].n_classes
        XP_data, YP_data, W_data = self.initialize(n_datasets=len(datasets), n_classes=datasets[0].n_classes)
        self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp, requires_grad=True) for xp in XP_data])
        self.YP = torch.nn.ParameterList([
            torch.nn.parameter.Parameter(data=yp.float(), requires_grad=self.grad_labels) for yp in YP_data
        ])
        self.W = torch.nn.parameter.Parameter(data=W_data, requires_grad=True)
        if self.grad_labels:
            optimizer = torch.optim.Adam([
                {'params': self.XP, 'lr': self.lr_features},
                {'params': self.YP, 'lr': self.lr_labels},
                {'params': self.W, 'lr': self.lr_weights}
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': self.XP, 'lr': self.lr_features},
                {'params': self.W, 'lr': self.lr_weights}
            ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        self.history = {
            'loss': [],
            'weights': [],
            'loss_per_dataset': {name: [] for name in self.names}
        }
        for it in range(num_iter_max):
            # Calculates the loss
            avg_it_loss = 0
            avg_it_loss_per_dataset = {self.names[ℓ]: 0 for ℓ in range(len(datasets))}
            for batch_indices in self.generate_batch_indices_without_replacement(batch_size=self.n_classes * self.batch_size):
                optimizer.zero_grad()

                XP = [XPk[ind_k] for XPk, ind_k in zip(self.XP, batch_indices)]
                YP = [YPk[ind_k] for YPk, ind_k in zip(self.YP, batch_indices)]

                loss = 0
                for ℓ, (Qℓ, wℓ) in enumerate(zip(datasets, self.W)):
                    XQℓ, YQℓ = Qℓ.sample()
                    XBℓ, YBℓ = wasserstein_barycenter(XP=XP, YP=YP, α=wℓ, n_samples=self.batch_size,
                                                      ϵ=self.ϵ, initialization=self.barycenter_initialization,
                                                      num_iter_max=self.num_iter_barycenter,
                                                      num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                      device=self.device, propagate_labels=True,
                                                      penalize_labels=self.penalize_labels)

                    loss_ℓ = self.loss_fn(XQ=XQℓ, YQ=YQℓ, XP=XBℓ, YP=YBℓ)
                    # penalty_ℓ = self.η_A * self.regularizer_fn(α_ℓ)
                    avg_it_loss_per_dataset[self.names[ℓ]] += loss_ℓ.detach().cpu().item() / batches_per_it
                    loss += loss_ℓ # + penalty_ℓ

                loss.backward()
                optimizer.step()

                # Projects the weights into the simplex
                if self.proj_grad:
                    with torch.no_grad():
                        self.W.data = ot.utils.proj_simplex(self.W.data.T).T

                avg_it_loss += loss.item() / batches_per_it
            # Saves weights in history
            self.history['weights'].append(ot.utils.proj_simplex(self.W.data.T).T)
            self.history['loss'].append(avg_it_loss)
            for ℓ in range(len(datasets)):
                self.history['loss_per_dataset'][self.names[ℓ]].append(
                    avg_it_loss_per_dataset[self.names[ℓ]]
                )
            print('It {}/{}, Loss: {}'.format(it, num_iter_max, avg_it_loss))
            scheduler.step(avg_it_loss)
        self.fitted = True


    def fit(self, datasets, num_iter_max=100, batches_per_it=100):
        r"""Dictionary Learning method. In this method we minimize,

        $$L(\mathcal{P},\mathcal{A}) = \dfrac{1}{N}\sum_{\ell=1}^{N}\mathcal{L}(\hat{Q}_{\ell},\hat{B}_{\ell}),$$

        where $\hat{B}_{\ell} = \mathcal{B}(\alpha;\mathcal{P})$. Optimization is carried w.r.t. $\mathbf{x}_{j}^{(P_{k})}$ and
        $\alpha_{\ell} \in \Delta_{K}$.

        __Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf),
        we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator,

        $$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$

        Args:
            datasets: List of tuples containing 2 arrays, $\mathbf{X}^{(Q_{\ell})}$ and $\mathbf{Y}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}(X,Y)$.
            num_iter_max: maximum number of iterations for dictionary learning.
            batches_per_it: number of batches sampled per iteration.
        """
        if self.n_classes is None:
            self.n_classes = datasets[0].n_classes
        XP_data, YP_data, W_data = self.initialize(n_datasets=len(datasets), n_classes=datasets[0].n_classes)
        self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp, requires_grad=True) for xp in XP_data])
        self.YP = torch.nn.ParameterList([
            torch.nn.parameter.Parameter(data=yp.float(), requires_grad=self.grad_labels) for yp in YP_data
        ])
        self.W = torch.nn.parameter.Parameter(data=W_data, requires_grad=True)
        if self.grad_labels:
            optimizer = torch.optim.Adam([
                {'params': self.XP, 'lr': self.lr_features},
                {'params': self.YP, 'lr': self.lr_labels},
                {'params': self.W, 'lr': self.lr_weights}
            ])
        else:
            optimizer = torch.optim.Adam([
                {'params': self.XP, 'lr': self.lr_features},
                {'params': self.W, 'lr': self.lr_weights}
            ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        self.history = {
            'loss': [],
            'weights': [],
            'loss_per_dataset': {self.names[ℓ]: [] for ℓ in range(len(datasets))},
            'atoms': {'Atom {}'.format(k): {'Features': [], 'Labels': []} for k in range(self.n_components)}
        }
        for it in range(num_iter_max):
            # Calculates the loss
            avg_it_loss = 0
            avg_it_loss_per_dataset = {self.names[ℓ]: 0 for ℓ in range(len(datasets))}
            for _ in tqdm(range(batches_per_it)):
                optimizer.zero_grad()
                
                # Change of Variables
                # A = torch.nn.functional.softmax(self.W, dim=1)

                # Sample minibatch from atoms
                XP, YP = self.sample_from_atoms(n=self.batch_size)

                loss = 0
                for ℓ, (Qℓ, wℓ) in enumerate(zip(datasets, self.W)):
                    XQℓ, YQℓ = Qℓ.sample()
                    XBℓ, YBℓ = wasserstein_barycenter(XP=XP, YP=YP, α=wℓ, n_samples=self.batch_size,
                                                      ϵ=self.ϵ, initialization=self.barycenter_initialization,
                                                      num_iter_max=self.num_iter_barycenter,
                                                      num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                      device=self.device, propagate_labels=True,
                                                      penalize_labels=self.penalize_labels)

                    loss_ℓ = self.loss_fn(XQ=XQℓ, YQ=YQℓ, XP=XBℓ, YP=YBℓ)
                    # penalty_ℓ = self.η_A * self.regularizer_fn(α_ℓ)
                    avg_it_loss_per_dataset[self.names[ℓ]] += loss_ℓ.detach().cpu().item() / batches_per_it
                    loss += loss_ℓ # + penalty_ℓ

                loss.backward()
                optimizer.step()

                # Projects the weights into the simplex
                if self.proj_grad:
                    with torch.no_grad():
                        self.W.data = ot.utils.proj_simplex(self.W.data.T).T

                avg_it_loss += loss.item() / batches_per_it
            # Saves weights in history
            self.history['weights'].append(ot.utils.proj_simplex(self.W.data.T).T)
            self.history['loss'].append(avg_it_loss)
            for ℓ in range(len(datasets)):
                self.history['loss_per_dataset'][self.names[ℓ]].append(
                    avg_it_loss_per_dataset[self.names[ℓ]]
                )
            if self.track_atoms:
                atoms = self.get_atoms()
                for k, (_XPk, _YPk) in enumerate(atoms):
                    self.history['atoms']['Atom {}'.format(k)]['Features'].append(_XPk)
                    self.history['atoms']['Atom {}'.format(k)]['Labels'].append(_YPk)
            print('It {}/{}, Loss: {}'.format(it, num_iter_max, avg_it_loss))
            scheduler.step(avg_it_loss)
        self.fitted = True

    def transform(self, Q, num_iter_max=100, batches_per_it=10):
        r"""Embeds distributions $\mathcal{Q} = \{\hat{Q}_{\ell}\}_{\ell=1}^{N}$ into the simplex $\Delta_{K}$. It is
        equivalent to the Barycentric Coordinate Regression method of [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf),
        
        $$\alpha^{\star}_{\ell} = \varphi(\hat{Q}) = \underset{\alpha \in \Delta_{K}}{\text{argmin}}\sum_{k=1}^{K}\alpha_{k}W_{2}(\hat{P}_{k},\hat{Q}_{\ell}).$$

        __Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf),
        we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator,

        $$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$

        Args:
            datasets: List of tuples containing 2 arrays, $\mathbf{X}^{(Q_{\ell})}$ and $\mathbf{Y}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}(X,Y)$.
            num_iter_max: maximum number of iterations for optimizing the cost function.
            batches_per_it: number of mini-batches sampled per iteration.
        """
        embeddings = torch.randn(len(Q), self.n_components, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([embeddings], lr=self.lr)

        for step in range(num_iter_max):
            pbar = tqdm(range(batches_per_it))
            avg_it_loss = 0
            for _ in pbar:
                optimizer.zero_grad()

                loss = 0
                
                for Qℓ, aℓ in zip(Q, embeddings):
                    XQℓ, YQℓ = Qℓ.sample()
                    XP, YP = self.sample_from_atoms(n=self.batch_size, detach=True)
                    αℓ = aℓ.softmax(dim=0)

                    XB, YB = wasserstein_barycenter(XP=XP, YP=YP, α=αℓ, n_samples=self.batch_size,
                                                      ϵ=self.ϵ, initialization=self.barycenter_initialization,
                                                      num_iter_max=self.num_iter_barycenter,
                                                      num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                      device=self.device, propagate_labels=True,
                                                      penalize_labels=self.penalize_labels)
                    XB.to(self.device)
                    loss = self.loss_fn(XQℓ, YQℓ, XB, YB) + self.η_A * self.regularized_fn(αℓ)
                
                loss.backward()
                avg_it_loss += loss.detach().item() / batches_per_it
                optimizer.step()
                pbar.set_description("Loss: {}".format(loss.item()))
            print('Step {:<4}/{:<4} loss: {:^15}'.format(step, num_iter_max, avg_it_loss))

        return embeddings.softmax(dim=1).detach().numpy()

    def reconstruct(self, weights=None, n_samples_atoms=None, n_samples_barycenter=None):
        r"""Uses Wasserstein barycenters for reconstructing samples given $\alpha$,

        $$\hat{B} = \mathcal{B}(\alpha;\mathcal{P}) = \underset{B}{\text{argmin}}\sum_{k=1}^{K}\alpha_{k}W_{2}(B, \hat{P}_{k}) $$
        
        Args:
            α: List of weights $\{\alpha_{\ell}\}_{\ell=1}^{N}$, $\alpha_{\ell} \in \Delta_{K}$ which will be used
               to reconstruct samples from a given distribution.
            n_samples_atoms: number of samples sampled from each atom for the reconstruction.
            n_samples_barycenter: number of samples in the barycenter's support.
        """
        n_samples_atoms = self.batch_size if n_samples_atoms is None else n_samples_atoms
        n_samples_barycenter = self.batch_size if n_samples_barycenter is None else n_samples_barycenter
        XP, YP = self.sample_from_atoms(n=n_samples_atoms, detach=True)
        with torch.no_grad():
            if weights is None:
                # A = torch.nn.functional.softmax(self.weights, dim=1).detach()
                Q_rec = []
                for wℓ in self.W:
                    XB, YB = wasserstein_barycenter(XP=XP, YP=YP, α=wℓ,
                                                      n_samples=n_samples_barycenter, ϵ=self.ϵ,
                                                      num_iter_max=self.num_iter_barycenter,
                                                      num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                      device=self.device, propagate_labels=True,
                                                      penalize_labels=self.penalize_labels)
                    Q_rec.append([XB, YB])
            else:
                XB, YB = wasserstein_barycenter(XP=XP, YP=YP, α=weights,
                                                  n_samples=n_samples_barycenter, ϵ=self.ϵ,
                                                  num_iter_max=self.num_iter_barycenter,
                                                  num_iter_sinkhorn=self.num_iter_barycenter,
                                                  device=self.device, propagate_labels=True,
                                                  penalize_labels=self.penalize_labels)
                Q_rec = [XB, YB]
        return Q_rec

r"""Module with utility functions."""


import os
import ot
import math
import json
import torch
import pickle
import numpy as np
import pandas as pd

from PIL import Image
from itertools import permutations
from torchvision.utils import make_grid


def sqrtm(A):
    D, V = torch.linalg.eig(A)
    _D = D.pow(1 / 2)

    return torch.mm(V, torch.mm(torch.diag(_D), torch.linalg.inv(V))).real


def proj_simplex(a):
    r"""Projects a non-negative vector $\mathbf{a} \in \mathbb{R}^{n}$ into the simplex $\Delta_{n}$ using,
    
    $$p_{i} = \dfrac{a_{i}}{\sum_{j}a_{j}}$$
    
    Args:
        a: numpy array of shape (n,) with non-negative entries.
    """
    a = a.astype(np.float64)
    a = a / a.sum()
    return a


def check_device(device):
    r"""Checks if using the correct device (e.g. setting gpu with gpu's available)."""
    if not torch.cuda.is_available() and device == 'gpu':
        print('Warning: trying to use gpu when not available. Setting to cpu')
        return torch.device('cpu')
    return torch.device(device)


def unif(n, device='cpu', dtype=torch.float32):
    r"""Returns uniform sample weights for a number of samples $n > 0$.
    
    Args:
        n: number of samples
        device: whether the returned tensor is on 'cpu' or 'gpu'.
    """
    return torch.ones(n, device=device, dtype=dtype) / n


def images2grid(images, grid_size=8, random=True):
    r"""Converts images into a grid using torchvision's make_grid."""
    if random:
        ind = np.random.choice(np.arange(len(images)), size=grid_size ** 2)
    else:
        ind = np.arange(grid_size ** 2)
    grid = make_grid(images[ind], nrow=grid_size)
    return grid.numpy().transpose([1, 2, 0])


def histogram2d_gradients(grads, n_bins=100):
    bins = np.linspace(grads.min(), grads.max(), n_bins)
    hists = []
    edges = []
    for g in grads:
        h, e = np.histogram(g, bins, density=True)
        hists.append(h)
        edges.append(.5 * (e[1:] + e[:-1]))
    hists = np.array(hists)
    edges = np.array(edges)

    x = np.arange(grads.shape[0])
    y = edges[0, :].copy()

    X, Y = np.meshgrid(x, y)
    Z = hists.T

    return X, Y, Z


def ot_adapt(Xs, Xt, ϵ=1e-2, τ=0.0):
    us, ut = ot.unif(Xs.shape[0]), ot.unif(Xt.shape[0])
    us, ut = torch.from_numpy(us).float(), torch.from_numpy(ut).float()
    Cst = torch.cdist(Xs, Xt, p=2) ** 2
    if ϵ > 0.0:
        if τ > 0.0:
            π = ot.unbalanced.mm_unbalanced(us, ut, Cst, reg_m=τ * Cst.max())
        else:
            π = ot.sinkhorn(us, ut, Cst, reg=ϵ * Cst.max())
    else:
        if τ > 0.0:
            π = ot.unbalanced.sinkhorn_unbalanced(us, ut, Cst, reg=ϵ * Cst.max(), reg_m=τ * Cst.max())
        else:
            π = ot.emd(us, ut, Cst)

    return Xs.shape[0] * torch.mm(π, Xt)


def preprocess_torch_dataset(dataset):
    try:
        X, y = dataset.data, dataset.targets
    except AttributeError:
        X, y = dataset.data, dataset.labels

    if len(X.shape) == 4 and X.shape[1] == 3:
        X = X.transpose([0, 2, 3, 1])

    if type(X) != torch.Tensor:
        X = torch.from_numpy(X)

    if type(y) == list:
        y = torch.Tensor(dataset.targets).long()
    elif type(y) == np.ndarray:
        y = torch.from_numpy(y).long()

    return X, y


def preprocess_gtzan_dataset(target_domain, as_list=False, fold=1):
    domains = ['original', 'factory2', 'f16', 'destroyerengine', 'buccaneer2']
    source_domains = [d for d in domains if d != target_domain]

    with open('./data/gtzan/raw/gtzan_images_dataset.pickle', 'rb') as f:
        dataset = pickle.loads(f.read())

    with open(os.path.abspath(os.path.abspath("./data/gtzan/crossval_indices.json")), 'r') as f:
        fold_dict = json.loads(f.read())

    # Create source domains
    Xs, Ys = [], []
    for source in source_domains:
        X = dataset[source]['Images']
        y = dataset[source]['Labels']
        Y = torch.nn.functional.one_hot(torch.from_numpy(y), num_classes=10).numpy()
        
        Xs.append(X)
        Ys.append(Y)
    
    # Create target domains
    ind_tr = fold_dict[target_domain]['fold {}'.format(fold)]['train']
    ind_ts = fold_dict[target_domain]['fold {}'.format(fold)]['test']

    Xt_tr, yt_tr = dataset[target_domain]['Images'][ind_tr], dataset[target_domain]['Labels'][ind_tr]
    Xt_ts, yt_ts = dataset[target_domain]['Images'][ind_ts], dataset[target_domain]['Labels'][ind_ts]
    Yt_tr = torch.nn.functional.one_hot(torch.from_numpy(yt_tr), num_classes=10).numpy()
    Yt_ts = torch.nn.functional.one_hot(torch.from_numpy(yt_ts), num_classes=10).numpy()

    print(np.unique(yt_tr, return_counts=True))
    print(np.unique(yt_ts, return_counts=True))

    if as_list:
        return Xs, Ys, Xt_tr, Yt_tr, Xt_ts, Yt_ts
    else:
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0), Xt_tr, Yt_tr, Xt_ts, Yt_ts


def probability_grid(values, n):
    values = set(values)
    # Check if we can extend the probability distribution with zeros
    with_zero = 0. in values
    values.discard(0.)
    if not values:
        raise StopIteration
    values = list(values)
    for p in _probability_grid_rec(values, n, [], 0.):
        if with_zero:
            # Add necessary zeros
            p += (0.,) * (n - len(p))
        if len(p) == n:
            yield from set(permutations(p))  # faster: more_itertools.distinct_permutations(p)


def _probability_grid_rec(values, n, current, current_sum, eps=1e-10):
    if not values or n <= 0:
        if abs(current_sum - 1.) <= eps:
            yield tuple(current)
    else:
        value, *values = values
        inv = 1. / value
        # Skip this value
        yield from _probability_grid_rec(
            values, n, current, current_sum, eps)
        # Add copies of this value
        precision = round(-math.log10(eps))
        adds = int(round((1. - current_sum) / value, precision))
        for i in range(adds):
            current.append(value)
            current_sum += value
            n -= 1
            yield from _probability_grid_rec(
                values, n, current, current_sum, eps)
        # Remove copies of this value
        if adds > 0:
            del current[-adds:]


def stratified_sampling_with_replacement(X, Y, n):
    y = Y.argmax(dim=1)
    groups = torch.unique(y)
    n_groups = len(groups)
    n_per_class = n // n_groups
    new_n = n_per_class * len(groups)
    if new_n != n:
        print("[WARNING] value of n ({}) is not divisible by number of groups ({}). Using n = {} instead.".format(n, n_groups, new_n))
    _X = []
    _Y = []
    for yu in groups:
        ind = torch.where(y == yu)[0]
        selected_ind = np.random.choice(ind, size=n_per_class, replace=True)
        np.random.shuffle(selected_ind)
        _X.append(X[selected_ind])
        _Y.append(Y[selected_ind])
    _X = torch.cat(_X, dim=0)
    _Y = torch.cat(_Y, dim=0)
        
    return _X, _Y


class MultiDomainBalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Sources:
        https://github.com/kilianFatras/JUMBOT/blob/main/Domain_Adaptation/utils.py
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, n_classes, batch_size, n_batches):
        self.labels = labels
        self.samples_per_class = batch_size // n_classes
        self.batch_size = n_classes * self.samples_per_class
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            indices = []
            for k, ys_k in enumerate(self.labels):
                selected_indices_k = []
                for c in np.unique(ys_k):
                    ind_c = np.where(ys_k == c)[0]
                    _ind_c = np.random.choice(ind_c, size=self.samples_per_class)
                    selected_indices_k.append(_ind_c)
                selected_indices_k = np.concatenate(selected_indices_k, axis=0)
                indices.append(selected_indices_k)
            indices = np.stack(indices).T
            yield indices

    def __len__(self):
        return self.n_batches


class BalancedBatchSamplerWithTargets(torch.utils.data.sampler.BatchSampler):
    """Samples balanced batches from all domains, including target domain.

    Sources:
        https://github.com/kilianFatras/JUMBOT/blob/main/Domain_Adaptation/utils.py
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, n_classes, batch_size, n_batches, debug=False):
        self.labels = labels
        self.samples_per_class = batch_size // n_classes
        self.batch_size = n_classes * self.samples_per_class
        self.n_batches = n_batches
        self.debug = debug
        
    def __iter__(self):
        for _ in range(self.n_batches):
            indices = []
            for yk in self.labels:
                selected_indices_k = []
                for c in np.unique(yk):
                    ind_c = np.where(yk == c)[0]
                    num_samples = min(self.samples_per_class, ind_c.size)
                    selected_indices_k.append(np.random.choice(ind_c, size=num_samples))
                indices.append(np.concatenate(selected_indices_k, axis=0))
            yield [idx for idx in zip(*indices)]

    def __len__(self):
        return self.n_batches


class BalancedBatchSamplerDA(torch.utils.data.sampler.BatchSampler):
    """
    Sources:
        https://github.com/kilianFatras/JUMBOT/blob/main/Domain_Adaptation/utils.py
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, source_labels, n_target, n_classes, batch_size, n_batches, debug=False):
        self.source_labels = source_labels
        self.n_target = n_target
        self.samples_per_class = batch_size // n_classes
        self.batch_size = n_classes * self.samples_per_class
        self.n_batches = n_batches
        self.debug = debug
        
    def __iter__(self):
        for _ in range(self.n_batches):
            source_indices = []
            for ys_k in self.source_labels:
                selected_indices_k = []
                for c in np.unique(ys_k):
                    ind_c = np.where(ys_k == c)[0]
                    selected_indices_k.append(np.random.choice(ind_c, size=self.samples_per_class))
                source_indices.append(np.concatenate(selected_indices_k, axis=0))
            
            target_indices = np.random.choice(np.arange(self.n_target), size=self.batch_size)

            if self.debug:
                print("[Sending]")
                for k, selected_indices_k in enumerate(source_indices):
                    print("Source {}: {}".format(k, selected_indices_k))
                print("Target: {}".format(target_indices))
                print("-" * 150)
            yield [indices for indices in zip(*source_indices, target_indices)]

    def __len__(self):
        return self.n_batches


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Sources:
        https://github.com/kilianFatras/JUMBOT/blob/main/Domain_Adaptation/utils.py
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, n_classes, samples_per_class, n_batches, debug=False):
        self.labels = labels
        self.n_classes = n_classes
        self.samples_per_class = samples_per_class
        self.batch_size = n_classes * self.samples_per_class
        self.n_batches = n_batches
        self.debug = debug
        
    def __iter__(self):
        for _ in range(self.n_batches):
            selected_indices = []
            for c in np.unique(self.labels):
                ind_c = np.where(self.labels == c)[0]
                selected_indices.append(np.random.choice(ind_c, size=self.samples_per_class))
            selected_indices = np.concatenate(selected_indices, axis=0)
            
            yield selected_indices

    def __len__(self):
        return self.n_batches


class MusicSpeechDiscriminationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root='/home/efernand/data/GTZAN/music_speech/',
                 domains=None,
                 transform=None,
                 train=True,
                 test=True):
        self.name2cat = {'music': 0, 'speech': 1}

        self.root = root
        self.domains = ['original', 'factory2', 'f16', 'destroyerengine', 'buccaneer2']
        self.domains = domains if domains is not None else self.domains[:-1]
        self.transform = transform
        self.num_classes = len(self.name2cat)

        self.filepaths = {}
        self.labels = {}

        self.filepaths, self.labels = [], []
        for domain in self.domains:
            filepaths, labels = self.get_filepaths_and_labels(domain, train=train, test=test)

            self.filepaths += filepaths
            self.labels += labels

    def get_filepaths_and_labels(self, domain, train=True, test=True):
        with open(os.path.join(self.root, 'train_test_splits.json'), 'r') as f:
            splits = json.loads(f.read())

        filepaths = []
        labels = []

        if train:
            filepaths += splits[domain]['Train']
        if test:
            filepaths += splits[domain]['Test']
        
        labels = [f.split('/')[-2] for f in filepaths]
        
        return filepaths, labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        is_list = False
        if torch.is_tensor(idx):
            idx = idx.tolist()
            is_list = True

        if is_list:
            x = []
            y = []

            ims = [Image.open(self.filepaths[i]) for i in idx]
            if self.transform:
                ims = [self.transform(im) for im in ims]
            x = torch.stack(ims)
            label = np.array([self.name2cat[self.labels[i]] for i in idx]).reshape(-1,)
            y = torch.from_numpy(label).float()
            # y = torch.nn.functional.one_hot(torch.from_numpy(label).long(), num_classes=self.num_classes).float().squeeze()
        else:
            im = Image.open(self.filepaths[idx])
            if self.transform:
                im = self.transform(im)
            x = im
            y = torch.tensor([self.name2cat[self.labels[idx]]]).float()
            # y = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float().squeeze()

        return x, y


class Office31Dataset(torch.utils.data.Dataset):
    def __init__(self, root='./data/office31', folds_path="./data/office31/folds",
                 domains=None, transform=None, train=True, test=True):
        self.name2cat = {
            'headphones': 0,
            'bike': 1,
            'mouse': 2,
            'file_cabinet': 3,
            'bottle': 4,
            'desk_lamp': 5,
            'back_pack': 6,
            'desktop_computer': 7,
            'letter_tray': 8,
            'mug': 9,
            'bookcase':10,
            'projector':11,
            'pen':12,
            'laptop_computer':13,
            'speaker':14,
            'punchers':15,
            'calculator':16,
            'tape_dispenser':17,
            'phone':18,
            'ruler':19,
            'mobile_phone':20,
            'printer':21,
            'paper_notebook':22,
            'ring_binder':23,
            'scissors':24,
            'keyboard':25,
            'trash_can':26,
            'bike_helmet':27,
            'monitor':28,
            'desk_chair':29,
            'stapler': 30
        }

        self.root = root
        self.domains = domains if domains is not None else ['amazon']
        self.transform = transform
        self.train = train

        self.filepaths, self.labels = [], []

        for domain in self.domains:
            class_and_filenames = []

            with open(os.path.join(folds_path, '{}_train_filenames.txt'.format(domain)), 'r') as f:
                train_filenames = f.read().split('\n')[:-1]

            if train: class_and_filenames += train_filenames

            with open(os.path.join(folds_path, '{}_test_filenames.txt'.format(domain)), 'r') as f:
                test_filenames = f.read().split('\n')[:-1]

            if test: class_and_filenames += test_filenames

            classes = [fname.split('/')[0] for fname in class_and_filenames]
            filenames = [fname.split('/')[1] for fname in class_and_filenames]

            for c, fname in zip(classes, filenames):
                self.filepaths.append(os.path.join(root, domain, 'images', c, fname))
                self.labels.append(self.name2cat[c])

        self.filepaths = np.array(self.filepaths)
        self.labels = torch.nn.functional.one_hot(torch.from_numpy(np.array(self.labels)).long(), num_classes=31).float()
        self.domain_names = np.array([k] * len(self.filepaths[k]) for k, domain in enumerate(self.domains))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im, label = Image.open(self.filepaths[idx]), self.labels[idx]
        
        if self.transform:
            im = self.transform(im)

        return im, label


class Office31DatasetMSDA(torch.utils.data.Dataset):
    def __init__(self, root='./data/office31', folds_path="./data/office31/folds",
                 source_domains=None, target_domain=None, transform=None):
        self.name2cat = {
            'headphones': 0,
            'bike': 1,
            'mouse': 2,
            'file_cabinet': 3,
            'bottle': 4,
            'desk_lamp': 5,
            'back_pack': 6,
            'desktop_computer': 7,
            'letter_tray': 8,
            'mug': 9,
            'bookcase':10,
            'projector':11,
            'pen':12,
            'laptop_computer':13,
            'speaker':14,
            'punchers':15,
            'calculator':16,
            'tape_dispenser':17,
            'phone':18,
            'ruler':19,
            'mobile_phone':20,
            'printer':21,
            'paper_notebook':22,
            'ring_binder':23,
            'scissors':24,
            'keyboard':25,
            'trash_can':26,
            'bike_helmet':27,
            'monitor':28,
            'desk_chair':29,
            'stapler': 30
        }

        self.root = root
        self.folds_path = folds_path
        self.source_domains = source_domains if source_domains is not None else ['amazon', 'webcam']
        self.target_domain = target_domain if target_domain is not None else 'dslr'
        self.transform = transform
        self.num_classes = len(self.name2cat)

        self.filepaths = {}
        self.labels = {}

        for domain in self.source_domains:
            filepaths, labels = self.get_filepaths_and_labels(domain)

            self.filepaths[domain] = filepaths
            self.labels[domain] = labels
        
        filepaths, labels = self.get_filepaths_and_labels(self.target_domain)
        self.filepaths[self.target_domain] = filepaths
        self.labels[self.target_domain] = labels

    def get_filepaths_and_labels(self, domain, train=True, test=True):
        class_and_filenames = []
        filepaths, labels = [], []

        with open(os.path.join(self.folds_path, '{}_train_filenames.txt'.format(domain)), 'r') as f:
            train_filenames = f.read().split('\n')[:-1]

        if train: class_and_filenames += train_filenames

        with open(os.path.join(self.folds_path, '{}_test_filenames.txt'.format(domain)), 'r') as f:
            test_filenames = f.read().split('\n')[:-1]

        if test: class_and_filenames += test_filenames

        classes = [fname.split('/')[0] for fname in class_and_filenames]
        filenames = [fname.split('/')[1] for fname in class_and_filenames]

        for c, fname in zip(classes, filenames):
            filepaths.append(os.path.join(self.root, domain, 'images', c, fname))
            labels.append(self.name2cat[c])
        
        return filepaths, labels

    def __len__(self):
        return min([len(self.filepaths[domain]) for domain in self.filepaths])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xs, ys = [], []
        for domain in self.source_domains:
            im = Image.open(self.filepaths[domain][idx])
            if self.transform:
                im = self.transform(im)
            label = np.array(self.labels[domain][idx]).reshape(-1,)
            label = torch.nn.functional.one_hot(torch.from_numpy(label).long(), num_classes=self.num_classes).float().squeeze()

            xs.append(im)
            ys.append(label)
        
        im = Image.open(self.filepaths[self.target_domain][idx])
        if self.transform:
            im = self.transform(im)
        label = np.array(self.labels[domain][idx]).reshape(-1,)
        label = torch.nn.functional.one_hot(torch.from_numpy(label).long(), num_classes=self.num_classes).float().squeeze()
        
        xt = im
        yt = label

        return xs, ys, xt, yt

class UnsupervisedDictionaryDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X
        self.n_samples = np.min([len(Xℓ) for Xℓ in X])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return [Xℓ[idx] for Xℓ in self.X]


class DictionaryDADataset(torch.utils.data.Dataset):
    def __init__(self, Xs, Ys):
        self.Xs, self.Ys = Xs, Ys
        self.n_samples = np.min([len(xs) for xs in Xs])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        source_indices = idx[:]
        indt = idx[-1]

        # Samples from source domain
        xs = [xs[inds] for xs, inds in zip(self.Xs, source_indices)]
        ys = [ys[inds] for ys, inds in zip(self.Ys, source_indices)]
        




        return xs,ys


class FullDictionaryDataset(torch.utils.data.Dataset):
    def __init__(self, Xs, Ys, Xt, Yt, Yt_hat=None):
        self.Xs = [Xsk.unsqueeze(0) for Xsk in Xs]
        self.Ys = [Ysk.unsqueeze(0) for Ysk in Ys]
        self.Xt = Xt.unsqueeze(0)
        self.Yt = Yt.unsqueeze(0)
        self.Yt_hat = Yt_hat.unsqueeze(0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return ([Xsk[idx].squeeze() for Xsk in self.Xs],
                [Ysk[idx].squeeze() for Ysk in self.Ys],
                self.Xt[idx].squeeze(),
                self.Yt[idx].squeeze(),
                self.Yt_hat[idx].squeeze())



class MultiDomainDataset(torch.utils.data.Dataset):
    def __init__(self, Xs, Ys, balanced=False):
        self.Xs, self.Ys = Xs, Ys
        self.n_samples = np.min([len(xs) for xs in Xs])
        self.balanced = balanced

    def __len__(self):
        return self.n_samples

    def __balanced_iter(self, idx):
        # Samples from source domain
        xs = [xs[inds] for xs, inds in zip(self.Xs, idx)]
        ys = [ys[inds] for ys, inds in zip(self.Ys, idx)]

        return xs, ys

    def __unbalanced_iter(self, idx):
        return [Xs_k[idx] for Xs_k in self.Xs], [Ys_k[idx] for Ys_k in self.Ys]

    def __getitem__(self, idx):
        if self.balanced:
            return self.__balanced_iter(idx)
        return self.__unbalanced_iter(idx)


class SemiSupervisedDictionaryDADataset(torch.utils.data.Dataset):
    def __init__(self, Xs, Ys, Xt, Yt, debug=False):
        self.Xs, self.Ys, self.Xt, self.Yt = Xs, Ys, Xt, Yt
        self.n_samples = np.min([len(xs) for xs in Xs] + [len(Xt)])
        self.debug = debug

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        source_indices = idx[:-1]
        indt = idx[-1]

        if self.debug:
            print("[Receiving]")
            for k, selected_indices_k in enumerate(source_indices):
                print("Source {}: {}".format(k, selected_indices_k))
            print("Target: {}".format(indt))
            print("")
            print("-" * 150)

        # Samples from source domain
        xs = [xs[inds] for xs, inds in zip(self.Xs, source_indices)]
        ys = [ys[inds] for ys, inds in zip(self.Ys, source_indices)]
        
        # Samples from the target domain
        xt = self.Xt[indt]
        yt = self.Yt[indt]

        return xs, ys, xt, yt


class SingleSourceDADataset(torch.utils.data.Dataset):
    def __init__(self, Xs, Ys, Xt, Yt=None):
        self.Xs, self.Ys, self.Xt, self.Yt = Xs, Ys, Xt, Yt
        self.n_samples = np.min([Xs.shape[0], Xt.shape[0]])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        xs, ys, xt = self.Xs[idx], self.Ys[idx], self.Xt[idx]

        if self.Yt is not None:
            yt = self.Yt[idx]
            return xs, ys, xt, yt
        return xs, ys, xt


class MultiSourceDADataset(torch.utils.data.Dataset):
    def __init__(self, Xs, Ys, Xt, Yt=None, Yt_hat=None):
        self.Xs, self.Ys, self.Xt, self.Yt, self.Yt_hat = Xs, Ys, Xt, Yt, Yt_hat
        self.n_samples = np.min([len(xs) for xs in Xs] + [len(Xt)])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        xs, ys, xt = [Xs_k[idx] for Xs_k in self.Xs], [Ys_k[idx] for Ys_k in self.Ys], self.Xt[idx]

        ret = [xs, ys, xt]
        if self.Yt is not None:
            yt = self.Yt[idx]
            ret.append(yt)

        if self.Yt_hat is not None:
            yt_hat = self.Yt_hat[idx]
            ret.append(yt_hat)

        return ret
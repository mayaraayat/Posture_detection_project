r"""Module for sampling data from datasets."""


import os
import cv2
import torch
import numpy as np
from dictionary_learning.utils import unif
from dictionary_learning.utils import check_device


class AbstractMeasure:
    r"""Abstract Measure Class"""
    def __init__(self):
        pass

    def sample(self, n):
        pass


class EmpiricalMeasure(AbstractMeasure):
    r"""Empirical Measure.
    
    Given a support $\mathbf{X}^{(P)} = [\mathbf{x}_{i}^{(P)}]_{i=1}^{n}$ with $\mathbf{x}_{i}^{(P)} \sim P$ with
    probability $0 \leq a_{i}$ \leq 1, where $a_{i}$ reflects the sample weight, samples $\mathbf{x}_{i}^{(P)} according
    to $\mathbf{a} \in \Delta_{n}$.
    """
    def __init__(self, support, weights=None, device='cuda'):
        r"""Initializes an empirical measure,

        $$\hat{P} = \sum_{i=1}^{n}a_{i}\delta_{\mathbf{x}_{i}^{(P)}}$$

        Args:
            support: tensor of shape (n, d) containing samples $\mathbf{x}_{i}^{(P)} \in \mathbb{R}^{d}$
            weights: tensor of shape (n,) of non-negative entries that sum to 1. Correspond to sample weights. If not
                     given, assumes uniform weights (i.i.d. hypothesis).
            device: either 'cpu' or 'gpu'.
        """
        super().__init__()
        self.device = check_device(device)

        self.support = support
        self.n_samples, self.n_dim = self.support.shape
        self.weights = unif(self.n_samples, device=self.device) if weights is None else weights

    def sample(self, n):
        r"""Gets $n$ samples from $\mathbf{X}^{(P)}$ according to $\mathbf{a} \in \Delta_{n}$."""
        return self.support[np.random.choice(np.arange(self.n_samples), size=n, p=self.weights)].to(self.support), None

    def to(self, device):
        r"""Moves weights and support to device."""
        self.device = check_device(device)
        self.weights = self.weights.to(self.device)
        self.support = self.support.to(self.device)

class DatasetMeasure(AbstractMeasure):
    r"""Dataset Measure class. This corresponds to an EmpiricalMeasure with i.i.d. hypothesis, namely,
    
    $$\hat{P} = \dfrac{1}{n}\sum_{i=1}^{n}\delta_{\mathbf{x}_{i}^{(P)}}$$
    """
    def __init__(self, features, transforms=None, batch_size=64, device='cuda'):
        r"""Initializes a DatasetMeasure object.
        
        Args:
            features: numpy array containing raw data.
            transforms: pre-processing steps for data samples.
            batch_size: size of batches to be sampled from the support $\mathbf{X}^{(P)}$
            device: either 'cpu' or 'gpu'.
        """
        super().__init__()
        self.device = check_device(device)

        self.features = features
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_dim = np.prod(features.shape[1:])
        self.ind = np.arange(len(features))

    def sample(self, n=None):
        r"""Samples $n$ points from the measure support.
        
        Args:
            n: if given, samples $n$ samples from the support. If $n$ is None, then samples self.batch_size samples.
        """
        n = self.batch_size if n is None else n
        minibatch_ind = np.random.choice(self.ind, size=n)
        minibatch_features = self.features[minibatch_ind]

        if self.transforms is not None:
            minibatch_features = torch.cat([self.transforms(xi)[None, ...]for xi in minibatch_features], dim=0)
        elif type(minibatch_features) == np.ndarray:
            minibatch_features = torch.from_numpy(minibatch_features)
        return minibatch_features.to(self.device), None


class LabeledDatasetMeasure(AbstractMeasure):
    r"""Labeled Dataset Measure class. This corresponds to an EmpiricalMeasure with i.i.d. hypothesis, namely,
    
    $$\hat{P} = \dfrac{1}{n}\sum_{i=1}^{n}\delta_{(\mathbf{x}_{i}^{(P)},y_{i}^{(P)})}$$
    """
    def __init__(self, features, labels, transforms=None, batch_size=64, n_classes=None, stratify=False, device='cuda'):
        r"""Initializes a LabeledDatasetMeasure object.
        
        Args:
            features: numpy array containing raw data.
            labels: numpy array containing the labels of each sample.
            transforms: pre-processing steps for data samples.
            batch_size: size of batches to be sampled from the support $\mathbf{X}^{(P)}$
            n_classes: number of classes in the labels array. If not given, infers automatically by searching the unique
                       values in labels.
            stratify: whether or not stratify mini-batches. If mini-batches are stratified, classes are balanced on
                      each mini-batch.
            device: either 'cpu' or 'gpu'.
        """        
        super().__init__()
        self.device = check_device(device)

        self.labels = labels
        self.features = features
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_dim = features.shape[1]
        self.n_classes = len(np.unique(labels)) if n_classes is None else n_classes
        self.ind = np.arange(len(features))
        self.stratify = stratify

        self.ind_per_class = [
            np.where(labels == yu)[0] for yu in np.unique(labels)
        ]

    def sample(self, n=None):
        r"""Samples $n$ points from the measure support.
        
        Args:
            n: if given, samples $n$ samples from the support. If $n$ is None, then samples self.batch_size samples.
        """
        n = self.batch_size if n is None else n
        if self.stratify:
            samples_per_class = n // self.n_classes
            minibatch_ind = np.concatenate([
                np.random.choice(indices, size=samples_per_class) for indices in self.ind_per_class
            ])
        else:
            minibatch_ind = np.random.choice(self.ind, size=n)
        minibatch_labels = self.labels[minibatch_ind]
        minibatch_features = self.features[minibatch_ind]

        if self.transforms is not None:
            minibatch_features = torch.cat([self.transforms(xi)[None, ...] for xi in minibatch_features], dim=0)
        elif type(minibatch_features) == np.ndarray:
            minibatch_features = torch.from_numpy(minibatch_features)
        
        if type(minibatch_labels) == np.ndarray:
            minibatch_labels = torch.nn.functional.one_hot(torch.from_numpy(minibatch_labels).long(), num_classes=self.n_classes).float()
        else:
            minibatch_labels = torch.nn.functional.one_hot(minibatch_labels.long(), num_classes=self.n_classes).float()

        return minibatch_features.to(self.device), minibatch_labels.to(self.device)

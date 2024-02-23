import torch
import numpy as np
import warnings
import os
import sys
from md_clustering.utils.clustering_utils import clusters
from Splitt import split

import json
import torch
import random
import warnings
import argparse
import numpy as np
import tensorflow as tf
import pytorch_lightning as pl
from main import main
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.utilss import BalancedBatchSamplerDA
from dictionary_learning.utilss import DictionaryDADataset

from dictionary_learning.barycenters import wasserstein_barycenter
from dictionary_learning.barycentric_regression import WassersteinBarycentricRegression



sys.path.append('../../')
from dictionary_learning.weighted_barycenters import compute_barycenters
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score

def eval():

    features = np.load('Data/resnet50-all--modern_office31.npy', allow_pickle=True)

    data_list = features.tolist()
    alldata = []
    for array in data_list:
        tensor = torch.from_numpy(array)

        alldata.append(tensor)

    Y1 = np.load('Results/DaDiL/MappedLabels_Domain1.npy', allow_pickle=True)
    Y2 = np.load('Results/DaDiL/MappedLabels_Domain2.npy', allow_pickle=True)
    Y3 = np.load('Results/DaDiL/MappedLabels_Domain3.npy', allow_pickle=True)
    Newlabels=[Y1,Y2,Y3]
    labels = np.load('Data/labels-resnet50-all--modern_office31.npy', allow_pickle=True)


    alllabels = [labels[len(alldata[1]):3608], labels[:len(alldata[0])],
                 labels[len(alldata[0]):(len(alldata[0]) + 793)]]

    ylabels1 = []

    # Convert each one-hot encoded array to labels and append them to the list
    for array in alllabels:
        labels = np.argmax(array, axis=1)
        ylabels1.append(labels)
    alllabels = ylabels1

    Xs=[alldata[-1], alldata[0]]
    print(Xs[0].shape,Xs[1].shape)
    ys=[alllabels[0], alllabels[1]]
    print(ys[0].shape,ys[1].shape)


    X_train, X_test, y_train, y_test=split('Amazon')

    Xs=[torch.tensor(X_train), alldata[0]]
    ys=[y_train, alllabels[1]]
    print(Xs[0].shape,Xs[1].shape)

    print(ys[0].shape,ys[1].shape)

    Ys=[torch.nn.functional.one_hot(torch.tensor(ys[0]),num_classes=31).float(),torch.nn.functional.one_hot(torch.tensor(ys[1]),num_classes=31).float()]


    Xt = alldata[1]
    Yt = torch.nn.functional.one_hot(torch.tensor(alllabels[2]), num_classes=31).float()
    print(Yt.shape)


    #Yt=torch.nn.functional.one_hot(torch.tensor(alllabels[2]),num_classes=31).float()
    #Xt=alldata[1]



    num_iter_dil=100
    n_classes=31
    batch_size=260
    n_samples=2000
    batches_per_it=n_samples // batch_size


    XAtom, yAtom = main(6)
    YAtom=[yAtom[0].argmax(dim=1),yAtom[1].argmax(dim=1),yAtom[2].argmax(dim=1)]

    train_dataset = DictionaryDADataset(XAtom, yAtom, Xt, Yt)

    S = BalancedBatchSamplerDA([ysk for ysk in YAtom],
                               n_target=len(Xt),
                               n_classes=n_classes,
                               batch_size=batch_size,
                               n_batches=batches_per_it)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=S)

    num_iter_sinkhorn=50
    ϵ=0.0

    # Creates dictionary
    loss_fn = JointWassersteinLoss(ϵ=ϵ, num_iter_sinkhorn=num_iter_sinkhorn)
    wbr = WassersteinBarycentricRegression(XAtom,
                                           yAtom,
                                           n_distributions=1,
                                           learning_rate=1e-2,
                                           sampling_with_replacement=True,
                                           weight_initialization='uniform')

    # Creates trainer object
    trainer = pl.Trainer(max_epochs=num_iter_dil, accelerator='cpu', logger=False, enable_checkpointing=False)
    trainer.fit(wbr, train_loader)

    # Gets loss at last it
    end_loss = wbr.history['loss'][-1]

    # Get dictionary weights
    weights = wbr.A.detach().squeeze()
    print(weights.squeeze())
    # Reconstruct samples




    Xr= wasserstein_barycenter(XP=XAtom, YP=yAtom, n_samples=31, ϵ=0.0, α=weights.squeeze(),
                                    β=None, num_iter_max=10, verbose=True, propagate_labels=False,
                                    penalize_labels=False)
    new_clusters = []





    np.save('Results/DaDiL/TargetBarycenters.npy',
            Xr)
    Cℓ = torch.cdist(Xr.cpu(), torch.tensor(Xt).float(), p=2) ** 2
    clusterss = Cℓ.argmin(dim=0)
    Yt=Yt.argmax(dim=1)
    print(clusterss.shape)
    print(Yt.shape)
    print(Yt.shape)

    domain_1 = clusters(Xt, clusterss, 31)
    domain_1.cluster_data()
    labels=domain_1.cluster_tensors


    ari = adjusted_rand_score(alllabels[2], clusterss)

    nmi = normalized_mutual_info_score(alllabels[2], clusterss)

    fmi = fowlkes_mallows_score(alllabels[2], clusterss)

    print(ari)






if __name__ == "__main__":
    eval()







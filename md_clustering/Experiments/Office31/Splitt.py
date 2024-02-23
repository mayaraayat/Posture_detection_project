import torch
import numpy as np
import warnings
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score

from KMeans_baseline import *
from initialize_atoms import initialize_atoms
from clustering import Dadclustering
sys.path.append('../../')
from dictionary_learning.weighted_barycenters import compute_barycenters

warnings.filterwarnings('ignore')

def split(domain):



    features=np.load('Data/resnet50-all--modern_office31.npy', allow_pickle=True)

    labels = np.load('Data/labels-resnet50-all--modern_office31.npy', allow_pickle=True)


    data_list = features.tolist()
    alldata = []
    for array in data_list:
        tensor = torch.from_numpy(array)

        alldata.append(tensor)

    alllabels = [labels[len(alldata[1]):3608], labels[:len(alldata[0])]]
    ylabels1 = []

    # Convert each one-hot encoded array to labels and append them to the list
    for array in alllabels:
        labels = np.argmax(array, axis=1)
        ylabels1.append(labels)
    alllabels = ylabels1
    alldata = [alldata[-1], alldata[0]]

    features=alldata
    labels=alllabels

    if domain=='Amazon' :
        features=features[0]
        labels=labels[0]


        features_np = features.numpy()  # Convert torch tensor to numpy array
        labels_np = np.array(labels)  # Convert list to numpy array

        X_train, X_test, y_train, y_test = train_test_split(features_np, labels_np, test_size=0.3, stratify=labels_np)

        return(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    split('Amazon')
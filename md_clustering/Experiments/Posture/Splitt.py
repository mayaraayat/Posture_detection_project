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


    with open ('Results/features_dic.pkl','rb') as file:
        dic = pickle.load(file)

    with open ('Results/labels_dic.pkl','rb') as file:
        lab = pickle.load(file)  
    features=list(dic.values())

    labels = list(lab.values())


    data_list = features
    alldata = []
    for array in data_list:
        tensor = torch.from_numpy(array)

        alldata.append(tensor)

    alllabels = [labels[-1], labels[0]]
    ylabels1 = []

    # Convert each one-hot encoded array to labels and append them to the list
    for array in alllabels:
        labels = np.argmax(array, axis=1)
        ylabels1.append(labels)
    alllabels = ylabels1
    alldata = [alldata[-1], alldata[0]]

    features=alldata
    labels=alllabels

    if domain=='Sub8' :
        features=features[0]
        labels=labels[0]


        # Assuming features is your torch tensor and labels is your list of labels
        features_np = features.numpy()  # Convert torch tensor to numpy array
        labels_np = np.array(labels)  # Convert list to numpy array
        print(features_np.shape, labels_np.shape)
        # Perform the stratified split
        X_train, X_test, y_train, y_test = train_test_split(features_np, labels_np, test_size=0.05, stratify=labels_np)

        return(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    split('Sub8')
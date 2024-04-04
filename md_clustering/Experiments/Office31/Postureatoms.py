import torch
import numpy as np
import warnings
import json
import random
import os
import pickle
import sys
sys.path.append('../../../')
from dictionary_learning.weighted_barycenters import compute_barycenters
warnings.filterwarnings('ignore')

with open ('Results/features_dic.pkl','rb') as file:
    dic = pickle.load(file)

with open ('Results/labels_dic.pkl','rb') as file:
    lab = pickle.load(file)

Y1 = np.load('Results/KMeans/MappedLabels_Domain1.npy', allow_pickle=True)
Y2 = np.load('Results/KMeans/MappedLabels_Domain2.npy', allow_pickle=True)
Y3 = np.load('Results/KMeans/MappedLabels_Domain3.npy', allow_pickle=True)

# Define hyperparameters
n_classes = 7
n_samples = 3000
batch_size = 128
ϵ = 0.01
η_A = 0.0
lr = 1e-1
num_iter_max = 20
num_iter_dil = 100


def initialize_atoms(features,Y1,Y2,Y3,n_classes,n_samples,batch_size,ϵ,η_A,lr,num_iter_max,num_iter_dil):





    # Prepare data for the barycenter computation
    print(features[1])
    flat_features = [item for sublist in features for item in sublist]
    print(len(flat_features))
    Xs=[torch.from_numpy(np.array(array)) for array in flat_features]
    if torch.is_tensor(Y1):
        Ys = [torch.nn.functional.one_hot(Y1.long(), num_classes=7).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=7).float()]
    else:

        Ys=[torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=7).float(),torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float(),torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=7).float()]

    # Compute the barycenters
    print("len(Xs[1])",len(Xs[1]))
    print("len(Ys[1])",len(Ys[1]))
    print("len(Xs[2])",len(Xs[1]))
    print("len(Ys[2])",len(Ys[1]))
    print("len(Xs[3])",len(Xs[1]))
    print("len(Ys[3])",len(Ys[1]))
    atoms=compute_barycenters(Xs,Ys, n_samples, batch_size,num_iter_dil,
                                      n_classes, ϵ, η_A, lr, num_iter_max)

    #Getting initialized atoms
    XP=[xatom[0] for xatom in atoms]
    YP=[yatom[1] for yatom in atoms]

    # Create the Results/Atoms directory if it doesn't exist
    results_directory = "Results/Atoms"
    os.makedirs(results_directory, exist_ok=True)

    # Save atoms supports as NumPy files

    for i, x_value in enumerate(XP):
        np.save(os.path.join(results_directory, f'xatom_{i}.npy'), x_value)

    for i, y_value in enumerate(YP):
        np.save(os.path.join(results_directory, f'yatom_{i}.npy'), y_value)

    return(XP, YP)
if __name__ == "__main__":
    initialize_atoms(list(dic.values()),Y1,Y2,Y3,n_classes,n_samples,batch_size,ϵ,η_A,lr,num_iter_max,num_iter_dil)
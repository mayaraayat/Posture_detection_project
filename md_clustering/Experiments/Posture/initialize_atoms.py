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

    



    #Xs=[torch.from_numpy(np.array(array)) for array in flat_features]
    if torch.is_tensor(Y1):
        Ys = [torch.nn.functional.one_hot(Y1.long(), num_classes=7).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=7).float()]
    else:

        Ys=[torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=7).float(),torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float(),torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=7).float()]

    # Compute the barycenters
    l=[y.shape for y in Ys]
    if torch.is_tensor(features[0]):
        features0 = features[0]
    else : 
        features0 = torch.from_numpy(features[0]) 
    if torch.is_tensor(features[1]):
        features1 = features[1]
    else : 
        features1 = torch.from_numpy(features[1]) 
    if torch.is_tensor(features[2]):
        features2 = features[2]
    else : 
        features2 = torch.from_numpy(features[2]) 
    Xs = [features0, features1, features2]
    print("Before fit - Xs shapes:", [x.shape for x in Xs])
    print("Before fit - Ys shapes:", [y.shape for y in Ys])
    '''atoms = []
    for i in range(0, len(Xs[0]), len(Xs[0])//100):
        X_batch = [X[i:i+batch_size] for X in Xs]
        Y_batch = [Y[i:i+batch_size] for Y in Ys]
        atoms_batch = compute_barycenters(X_batch, Y_batch, n_samples, batch_size, num_iter_dil, n_classes, ϵ, η_A, lr, num_iter_max)
    # Process the batched atoms
        atoms.extend(atoms_batch)'''
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
import torch
import numpy as np
import warnings
import os
import sys
sys.path.append('../../')
from dictionary_learning.weighted_barycenters import compute_barycenters
warnings.filterwarnings('ignore')


features = np.load('Data/resnet50-all--modern_office31.npy', allow_pickle=True)
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


def initialize_atoms(features,mapped_labels_domain,n_classes,n_samples,batch_size,ϵ,η_A,lr,num_iter_max,num_iter_dil):





    # Prepare data for the barycenter computation
    Xs=features

    Ys = []

    for Y in mapped_labels_domain:
        if torch.is_tensor(Y):
            Ys.append(torch.nn.functional.one_hot(Y.long(), num_classes=7).float())
        else:
            Ys.append(torch.nn.functional.one_hot(torch.from_numpy(Y).long(), num_classes=7).float())

    # Compute the barycenters

    atoms=compute_barycenters(Xs,Ys, n_samples, batch_size,num_iter_dil,
                                      n_classes, ϵ, η_A, lr, num_iter_max)

    #Getting initialized atoms
    XP=[xatom[0] for xatom in atoms]
    YP=[yatom[1] for yatom in atoms]
    # Create the Results/Atoms directory if it doesn't exist
    results_directory = "./md_clustering/Experiments/Office31/Results/Atoms"
    os.makedirs(results_directory, exist_ok=True)

    # Save atoms supports as NumPy files

    for i, x_value in enumerate(XP):
        np.save(os.path.join(results_directory, f'xatom_{i}.npy'), x_value)

    for i, y_value in enumerate(YP):
        np.save(os.path.join(results_directory, f'yatom_{i}.npy'), y_value)

    return(XP, YP)
if __name__ == "__main__":
    initialize_atoms()



def initialize_atomsTest(features,Y1,Y2,n_classes,n_samples,batch_size,ϵ,η_A,lr,num_iter_max,num_iter_dil):





    # Prepare data for the barycenter computation
    Xs=features
    if torch.is_tensor(Y1):
        Ys = [torch.nn.functional.one_hot(Y1.long(), num_classes=7).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float()]
    else:

        Ys=[torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=7).float(),torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float()]

    # Compute the barycenters

    atoms=compute_barycenters(Xs,Ys, n_samples, batch_size,num_iter_dil,
                                      n_classes, ϵ, η_A, lr, num_iter_max)

    #Getting initialized atoms
    XP=[xatom[0] for xatom in atoms]
    YP=[yatom[1] for yatom in atoms]
    # Create the Results/Atoms directory if it doesn't exist
    results_directory = "./md_clustering/Experiments/Office31/Results/Atoms"
    os.makedirs(results_directory, exist_ok=True)

    # Save atoms supports as NumPy files

    for i, x_value in enumerate(XP):
        np.save(os.path.join(results_directory, f'xatom_{i}.npy'), x_value)

    for i, y_value in enumerate(YP):
        np.save(os.path.join(results_directory, f'yatom_{i}.npy'), y_value)

    return(XP, YP)
import torch
import numpy as np
import warnings
import os
import sys
sys.path.append('../../')
from dictionary_learning.weighted_barycenters import compute_barycenters
warnings.filterwarnings('ignore')

def main():


    features=np.load('./md_clustering/Experiments/Office31/Data/resnet50-all--modern_office31.npy', allow_pickle=True)
    Y1 = np.load('./md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain1.npy', allow_pickle=True)
    Y2 = np.load('./md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain2.npy', allow_pickle=True)
    Y3 = np.load('./md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain3.npy', allow_pickle=True)
    data_list = features.tolist()
    features = []
    for array in data_list:
        tensor = torch.from_numpy(array)
        features.append(tensor)

    # Define hyperparameters
    n_classes = 31
    n_samples = 3000
    batch_size = 128
    ϵ = 0.01
    η_A = 0.0
    lr = 1e-1
    num_iter_max = 20
    num_iter_dil=100

    # Prepare data for the barycenter computation
    Xs=features
    Ys=[torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=31).float(),torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=31).float(),torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=31).float()]

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

if __name__ == "__main__":
    main()
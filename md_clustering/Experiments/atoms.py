import torch
import numpy as np
import json
import warnings
import os
import sys
sys.path.append('../../')
from dictionary_learning.weighted_barycenters import compute_barycenters
warnings.filterwarnings('ignore')

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_pressure_matrices(json_data):
    pressure_matrices = []
    for entry in json_data['pressureData']:
        pressure_matrix = entry["pressureMatrix"]
        pressure_matrices.append({"pressureMatrix": pressure_matrix})
    return pressure_matrices

def extract_features_from_pressure_matrices(pressure_matrices):
    flattened_data = [np.array(item["pressureMatrix"]).flatten() for item in pressure_matrices]
    return np.concatenate(flattened_data, axis=0)

def combine_features(features_posture, features_continuous):
    return [features_posture , features_continuous]

def main():

    # Load continuous sitting data
    continuous_data = load_json_data('Posture_Data/Data/Mayara/SensingMatData_231208_131826.json')
    continuous_1=load_json_data('Posture_Data/Data/Aaron/SensingMatData_231126_230808.json')
    
    pressure_matrices_continuous = extract_pressure_matrices(continuous_data)
    pressure_1=extract_pressure_matrices(continuous_1)
    
    features_continuous = extract_features_from_pressure_matrices(pressure_matrices_continuous)
    features_1=extract_features_from_pressure_matrices(pressure_1)
    
    
    # Combine features if necessary
    Y1 = np.load('Posture_Data\Results\KMeans\Clusters.npy', allow_pickle=True)
    Y2 = np.load('Posture_Data\Results\KMeans\MappedLabels_Continuous.npy', allow_pickle=True)
    
    # Define hyperparameters
    n_classes = 7
    n_samples = 100
    batch_size = 64
    ϵ = 0.01
    η_A = 0.0
    lr = 1e-1
    num_iter_max = 20
    num_iter_dil = 100
    # Prepare data for the barycenter computation
    Ys = [torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=7).float(),
        torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float()]
    
    l=[y.shape for y in Ys]
    features_continuous_tensor = torch.from_numpy(features_continuous).view(l[0][0],-1)
    features_1_tensor = torch.from_numpy(features_1).view(l[1][0],-1)


    Xs = [features_continuous_tensor, features_1_tensor]
    print("Before fit - Xs shapes:", [x.shape for x in Xs])
    print("Before fit - Ys shapes:", [y.shape for y in Ys])
    # Ensure n_samples is not larger than the size of the smallest dataset
    #n_samples = min(n_samples, min(len(Xs), len(Ys[0]), len(Ys[1])))
    atoms = []
    for i in range(0, len(Xs[0]), batch_size):
        X_batch = [X[i:i+batch_size] for X in Xs]
        Y_batch = [Y[i:i+batch_size] for Y in Ys]
        atoms_batch = compute_barycenters(X_batch, Y_batch, n_samples, batch_size, num_iter_dil, n_classes, ϵ, η_A, lr, num_iter_max)
    # Process the batched atoms
        atoms.extend(atoms_batch)
    # Compute the barycenters
    #atoms = compute_barycenters(Xs, Ys, n_samples, batch_size, num_iter_dil,
     #                           n_classes, ϵ, η_A, lr, num_iter_max)
    
    # Getting initialized atoms
    XP = [xatom[0] for xatom in atoms]
    YP = [yatom[1] for yatom in atoms]
    
    print("Before converting types - XP and YP dtypes:", XP[0].dtype, YP[0].dtype)

   
    XP = [x.to(torch.float32) for x in XP]
    YP = [y.to(torch.float32) for y in YP]

    
    print("After converting types - XP and YP dtypes:", XP[0].dtype, YP[0].dtype)

    # Create the Results/Atoms directory if it doesn't exist
    results_directory = "Posture_Data/Results/Atoms"
    os.makedirs(results_directory, exist_ok=True)

    # Save atoms supports as NumPy files
    for i, x_value in enumerate(XP):
        np.save(os.path.join(results_directory, f'xatom_{i}.npy'), x_value)

    for i, y_value in enumerate(YP):
        np.save(os.path.join(results_directory, f'yatom_{i}.npy'), y_value)
if __name__ == "__main__":
    main()
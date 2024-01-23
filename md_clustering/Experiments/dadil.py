import warnings
import sys
import os
sys.path.append('../')
sys.path.append('../../')
warnings.filterwarnings('ignore')
import json
from utils.clustering_utils import clusters
from dictionary_learning.DaDiL_clustering import *
# Now you can use a simple import statement



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
    Y1 = np.load('Posture_Data/Results/KMeans/Clusters.npy', allow_pickle=True)
    Y2 = np.load('Posture_Data/Results/KMeans/MappedLabels_Continuous.npy', allow_pickle=True)
    # Prepare data for the barycenter computation
    Ys = [torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=7).float(),
        torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=7).float()]
    
    l=[y.shape for y in Ys]
    features_continuous_tensor = torch.from_numpy(features_continuous).view(l[0][0],-1)
    features_1_tensor = torch.from_numpy(features_1).view(l[1][0],-1)


    features = [features_continuous_tensor, features_1_tensor]
    Xs = features
    print("Before fit - Xs shapes:", [x.shape for x in Xs])
    print("Before fit - Ys shapes:", [y.shape for y in Ys])
   
    # Load XP and YP NumPy files
    XP = []
    YP = []
    for i in range(len(features)):
        x_file_path = os.path.join(
            'Posture_Data/Results/Atoms', f'xatom_{i}.npy')
        y_file_path = os.path.join(
            'Posture_Data/Results/Atoms', f'yatom_{i}.npy')

        # Load XP
        loaded_x = np.load(x_file_path)
        XP.append(torch.tensor(loaded_x))

        # Load YP
        loaded_y = np.load(y_file_path)
        YP.append(torch.tensor(loaded_y))

    # Define hyperparameters
    n_classes = 7
    n_samples = 100
    batch_size = 64
    n_components = 3
    n_datasets = 3
    reg = 0.0
    reg_labels = 0.0
    num_iter_max = 100


    # Perform the DaDiL clustering
    cluster_labels = dadil_clustering(
        Xs, Ys, XP, YP, n_samples, n_components, reg, reg_labels, batch_size, n_classes, num_iter_max)

    domain_1 = clusters(features[0], cluster_labels[0], n_classes)
    domain_2 = clusters(features[1], cluster_labels[1], n_classes)


    # Cluster data for each domain

    domain_1.cluster_data()
    domain_2.cluster_data()

    results_directory = "Posture_Data/Results/Dadil"
    os.makedirs(results_directory, exist_ok=True)
    np.save(os.path.join(results_directory,'MappedLabels_Posture.npy'),
            cluster_labels[0])

    mapped_labels_domain_2 = domain_2.clusters_mapping(
        domain_1.cluster_tensors)
    np.save(os.path.join(results_directory,'MappedLabels_Continuous.npy'),
            mapped_labels_domain_2)

  


if __name__ == "__main__":
    main()

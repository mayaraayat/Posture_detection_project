
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('../')
sys.path.append('../../')
import json
import os
import numpy as np
from utils.kmeans_utils import perform_kmeans_clustering
from utils.clustering_utils import clusters

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
    return [np.array(item["pressureMatrix"]).flatten() for item in pressure_matrices]

def combine_features(features_posture, features_continuous):
    return features_posture + features_continuous

def main():

    # Load continuous sitting data
    continuous_data = load_json_data('Posture_Data/Data/Mayara/SensingMatData_231208_131826.json')
    continuous_1=load_json_data('Posture_Data/Data/Aaron/SensingMatData_231126_230808.json')
    #print(continuous_data)
    
    # Extract features from the data
    pressure_matrice_continuous = extract_pressure_matrices(continuous_data)
    pressure_matrice_domain1=extract_pressure_matrices(continuous_1)
    
    
    features_continuous = extract_features_from_pressure_matrices(pressure_matrice_continuous)
    features_1=extract_features_from_pressure_matrices(pressure_matrice_domain1)
    print(features_continuous)



    num_clusters = 7  


    # Perform k-means clustering for continuous data of the reference subject
    cluster_labels_continuous, _ = perform_kmeans_clustering(np.array(features_continuous).reshape(-1,1), num_clusters)
    cluster_labels_continuous1,_=perform_kmeans_clustering(np.array(features_1).reshape(-1,1), num_clusters)
   
    # Create cluster objects for each domain
    continuous_domain = clusters(features_continuous, cluster_labels_continuous, num_clusters)
    domain_1= clusters(features_1,cluster_labels_continuous1,num_clusters)
    
    # Cluster data for each domain
    continuous_domain.cluster_data()
    domain_1.cluster_data()
    
    # Save the results
    results_directory = "Posture_Data/Results/KMeans"
    os.makedirs(results_directory, exist_ok=True)
    np.save(os.path.join(results_directory,'Clusters.npy'), cluster_labels_continuous)
    mapped_labels_continuous = domain_1.clusters_mapping(continuous_domain.cluster_tensors)
    np.save(os.path.join(results_directory,'MappedLabels_Continuous.npy'), mapped_labels_continuous)

if __name__ == "__main__":
    main()


import warnings
import numpy as np
import torch
import sys
import os
import json 
import numpy as np 
import random
warnings.filterwarnings('ignore')
sys.path.append('../../../')

import pickle 
from md_clustering.utils.kmeans_utils import perform_kmeans_clustering
from md_clustering.utils.clustering_utils import clusters

import os
import json
import random 

def extract_pressure_matrices(json_data):
    pressure_matrices = []
    for entry in json_data['pressureData']:
        pressure_matrix = entry["pressureMatrix"]
        pressure_matrices.append({"pressureMatrix": pressure_matrix})
    return pressure_matrices

def extract_features_from_pressure_matrices(pressure_matrices):
    
    flattened_data = [np.array(item["pressureMatrix"]).reshape(1,-1) for item in pressure_matrices]
    #print(flattened_data[0].shape)
    return np.concatenate(flattened_data, axis=0)
dic = {}
lab = {}

# Loop through subjects
for i in range(6, 9):
    
    folder_path = f'data/Subject_{i}/Posture_data'

    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    l = []  # List to store feature arrays
     # List to store labels
    labels_list = []
    for f in files:
        x = int(f.split('.')[0][-1])  # Extract label from filename
        file_path = os.path.join(folder_path, f)
        
        # Check if the file is a JSON file
        if file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file) 
                    pressure_matrices = extract_pressure_matrices(data)
                    features = extract_features_from_pressure_matrices(pressure_matrices)
                    l.append(features)
                    
                    # Create label array with 1 at index x
                    label_array = np.zeros((features.shape[0], 7))
                    label_array[:, x-1] = 1
                    labels_list.append(label_array)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
        else:
            print(f"Skipping non-JSON file: {file_path}")
    num = list(range(7))
    random.shuffle(num)
    
    dic[i] = np.concatenate([l[j] for j in num], axis=0)
    print(dic[i].shape)
    lab[i] = np.concatenate(labels_list,axis = 0)


with open ('Results/features_dic.pkl','wb') as file:
    pickle.dump(dic,file)

with open ('Results/labels_dic.pkl','wb') as file:
    pickle.dump(lab,file)  
    
def KMeans_baseline(features,labels):
    

    num_clusters = 7


    # Perform k-means clustering for each domain
    print(np.array(dic[6]).shape)
    cluster_labels_domain_1, _ =perform_kmeans_clustering(np.array(features[0]),num_clusters)
    cluster_labels_domain_2, _ =perform_kmeans_clustering(np.array(features[1]), num_clusters)
    cluster_labels_domain_3, _ =perform_kmeans_clustering(np.array(features[2]), num_clusters)
    # Create cluster objects for each domain

    domain_1 = clusters(features[0], cluster_labels_domain_1, num_clusters)
    domain_2 = clusters(features[1], cluster_labels_domain_2, num_clusters)
    domain_3 = clusters(features[2], cluster_labels_domain_3, num_clusters)

    # Cluster data for each domain

    domain_1.cluster_data()
    domain_2.cluster_data()
    domain_3.cluster_data()
 
 
 
    results_directory = "Results/KMeans"
    os.makedirs(results_directory, exist_ok=True)

    np.save(os.path.join(results_directory,'MappedLabels_Domain1.npy'), cluster_labels_domain_1)

    # Map cluster labels for Domain 2 using Domain 1's cluster tensors

    mapped_labels_domain_2 = domain_2.clusters_mapping(domain_1.cluster_tensors)
    np.save(os.path.join(results_directory,'MappedLabels_Domain2.npy'), mapped_labels_domain_2)

    # Map cluster labels for Domain 3 using Domain 1's cluster tensors

    mapped_labels_domain_3 = domain_3.clusters_mapping(domain_1.cluster_tensors)
    np.save(os.path.join(results_directory,'MappedLabels_Domain3.npy'), mapped_labels_domain_3)
    
    return(cluster_labels_domain_1,mapped_labels_domain_2,mapped_labels_domain_3)
if __name__ == "__main__":
    KMeans_baseline(list(dic.values()),list(lab.values()))


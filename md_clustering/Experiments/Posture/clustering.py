
# Import the necessary functions

import warnings
import sys
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.utils import resample

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import random
import json
sys.path.append('../../../')
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score
from dictionary_learning.DaDiL_clustering import *
from md_clustering.utils.clustering_utils import clusters
with open ('Results/features_dic.pkl','rb') as file:
    dic = pickle.load(file)

with open ('Results/labels_dic.pkl','rb') as file:
    lab = pickle.load(file)  
# Import features , labels and atoms

data_list = list(dic.values())
features = []
for array in data_list:
    tensor = torch.from_numpy(array)
    features.append(tensor)
alldata = features
labels = list(lab.values())
alllabels = [labels[-1], labels[0],labels[1]]

Y1 = np.load(
    'Results/KMeans/MappedLabels_Domain1.npy', allow_pickle=True)
Y2 = np.load(
    'Results/KMeans/MappedLabels_Domain2.npy', allow_pickle=True)
Y3 = np.load(
    'Results/KMeans/MappedLabels_Domain3.npy', allow_pickle=True)

# Load XP and YP NumPy files
XP = []
YP = []
for i in range(len(features)):
    x_file_path = os.path.join(
        'Results/Atoms', f'xatom_{i}.npy')
    y_file_path = os.path.join(
        'Results/Atoms', f'yatom_{i}.npy')

    # Load XP
    loaded_x = np.load(x_file_path)
    XP.append(torch.tensor(loaded_x))

    # Load YP
    loaded_y = np.load(y_file_path)
    YP.append(torch.tensor(loaded_y))
    
    


from scipy.spatial.distance import cdist

def split_clusters(domain, n_classes):
    while len(domain.cluster_tensors) < n_classes:
        # Find the largest cluster
        largest_cluster_index = np.argmax([len(cluster) for cluster in domain.cluster_tensors])
        largest_cluster = domain.cluster_tensors[largest_cluster_index]
        
        # Create a new centroid by perturbing the centroid of the largest cluster
        new_centroid = np.mean(largest_cluster, axis=0) + np.random.normal(scale=0.1, size=largest_cluster.shape[1])
        
        # Split the largest cluster into two around the new centroid
        distances = cdist(largest_cluster, [new_centroid])
        split_point = np.argmax(distances)
        cluster1 = largest_cluster[:split_point]
        cluster2 = largest_cluster[split_point:]
        
        # Replace the largest cluster with the two new clusters
        new_clusters = domain.cluster_tensors[:largest_cluster_index] + [cluster1, cluster2] + domain.cluster_tensors[largest_cluster_index+1:]
        
        # Assign labels based on the closest centroid
        new_centroids = [np.mean(cluster, axis=0) for cluster in new_clusters]
        new_labels = np.argmin(cdist(domain.data, new_centroids), axis=1)
        
        # Update domain 
        domain.cluster_tensors = new_clusters
        domain.labels = new_labels

    return domain

def Dadclustering(features,Y1,Y2,Y3,XP,YP):


    # Define hyperparameters
    n_classes = 7
    n_samples = 3000
    batch_size = 128
    n_components = 3
    n_datasets = 3
    reg = 0.0
    reg_labels = 0.0
    num_iter_max = 80

    Xs = features

    if torch.is_tensor(Y1):
        Ys = [torch.nn.functional.one_hot(Y1.long(), num_classes=n_classes).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=n_classes).float(),
              torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=n_classes).float()]
    else:

        Ys=[torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=n_classes).float(),torch.nn.functional.one_hot(torch.from_numpy(Y2).long(), num_classes=n_classes).float(),torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=n_classes).float()]

    # Perform the DaDiL clustering
    cluster_labels,XAtom,YAtom = dadil_clustering(
        Xs, Ys, XP, YP, n_samples, n_components, reg, reg_labels, batch_size, n_classes, num_iter_max)

    print(cluster_labels[0].shape)
    print(cluster_labels[1].shape)

    domain_1 = clusters(features[0], cluster_labels[0], n_classes)
    domain_2 = clusters(features[1], cluster_labels[1], n_classes)
    domain_3 = clusters(features[2], cluster_labels[2], n_classes)

    # Cluster data for each domain

    domain_1.cluster_data()
    domain_2.cluster_data()
    domain_3.cluster_data()
    print("longeur de domain 1:",len(domain_1.cluster_tensors))
    print("longeur de domain 2:",len(domain_2.cluster_tensors))
    print("longeur de domain 3:",len(domain_3.cluster_tensors))
    print('-------------- resplitting--------------------')
    print('-----------domain1-------------------')
    domain_1 = split_clusters(domain_1, n_classes)
    print("longeur de domain 1:",len(domain_1.cluster_tensors))
    print('-----------domain2-------------------')
    domain_2 = split_clusters(domain_2, n_classes)
    print("longeur de domain 2:",len(domain_2.cluster_tensors))
    print('-----------domain3-------------------')
    domain_3 = split_clusters(domain_3, n_classes)
    
    print("longeur de domain 3:",len(domain_3.cluster_tensors))
    
    print('-------------- resplitting: DONE --------------------')
    

    
    
    results_directory = "Results/DaDil"
    os.makedirs(results_directory, exist_ok=True)
    np.save(os.path.join(results_directory,'MappedLabels_Domain1.npy'),
            domain_1.labels)
    mapped_labels_domain_2 = domain_2.clusters_mapping(
      domain_1.cluster_tensors)
    np.save(os.path.join(results_directory,'MappedLabels_Domain2.npy'),
          mapped_labels_domain_2)

    # Map cluster labels for Domain 3 using Domain 1's cluster tensors

    mapped_labels_domain_3 = domain_3.clusters_mapping(domain_1.cluster_tensors)
    np.save(os.path.join(results_directory,'MappedLabels_Domain3.npy'),mapped_labels_domain_3)
    #print(len(mapped_labels_domain_2))

    #print(len(mapped_labels_domain_3))
    
    return (domain_1.labels,mapped_labels_domain_2,mapped_labels_domain_3,XAtom,YAtom)

if __name__ == "__main__":
    Dadclustering(list(dic.values()),Y1,Y2,Y3,XP,YP)

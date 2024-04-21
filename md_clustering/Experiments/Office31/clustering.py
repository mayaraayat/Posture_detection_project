
import warnings

from dictionary_learning.DaDiL_clusteringV1 import dadil_clustering
from scipy.spatial.distance import cdist
import numpy as np
from md_clustering.utils.clustering_utils import clusters

# Suppress warnings
warnings.filterwarnings('ignore')

def dadclustering(features, Ys, XP, YP, n_samples, reg, reg_labels, batch_size, n_classes, num_iter_max):
    """
    Perform DaDiL clustering.

    Args:
    - features (list of torch.Tensor): List of feature tensors for each domain.
    - Ys (list of torch.Tensor): List of label tensors for each domain.
    - XP (list of np.ndarray): List of XP NumPy arrays.
    - YP (list of np.ndarray): List of YP NumPy arrays.
    - n_samples (int): Number of samples.
    - n_components (int): Number of components.
    - reg (float): Regularization parameter.
    - reg_labels (float): Regularization parameter for labels.
    - batch_size (int): Batch size.
    - n_classes (int): Number of classes.
    - num_iter_max (int): Maximum number of iterations.

    Returns:
    - cluster_labels (list of np.ndarray): Cluster labels for each domain.
    - XAtom (list of torch.Tensor): List of XAtom tensors.
    - YAtom (list of torch.Tensor): List of YAtom tensors.
    """
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
    
    Xs = features
    cluster_labels, XAtom, YAtom = dadil_clustering(
        Xs, Ys, XP, YP, n_samples, reg, reg_labels, batch_size, n_classes, num_iter_max)

    domains=[]
    for i,feature in enumerate(features):
        labels=cluster_labels[i]
        domain = clusters(feature, labels, n_classes)
        domain.cluster_data()
        
        print(f'-----------domain{i}-------------------')
        domain = split_clusters(domain, n_classes)
        domains.append(domain)
    mapped_labels = []
    for i in range(1, len(domains)):
        mapped_labels.append(domains[i].clusters_mapping(domains[0].cluster_tensors))

        # domain_1 = clusters(features[0], cluster_labels[0], n_classes)
        # domain_2 = clusters(features[1], cluster_labels[1], n_classes)
        # domain_3 = clusters(features[2], cluster_labels[2], n_classes)
        #
        # # Cluster data for each domain
        #
        # domain_1.cluster_data()
        # domain_2.cluster_data()
        # domain_3.cluster_data()
        #
        # print("longeur de domain 2", len(domain_2.cluster_tensors))
        #
        #
        # mapped_labels_domain_1 = domain_1.clusters_mapping(domain_2.cluster_tensors)
        #
        # mapped_labels_domain_3 = domain_3.clusters_mapping(domain_2.cluster_tensors)

    # Determine the return values based on the length of mapped_labels
    if len(mapped_labels) == 1:
        return [cluster_labels[0], mapped_labels[0]],XAtom, YAtom
    else:
        return [cluster_labels[0], *mapped_labels],XAtom, YAtom

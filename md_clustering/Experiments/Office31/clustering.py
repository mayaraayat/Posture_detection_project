
import warnings

from dictionary_learning.DaDiL_clusteringV1 import dadil_clustering
from scipy.spatial.distance import cdist
import numpy as np
from md_clustering.utils.clustering_utils import clusters
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
    cluster_labels, XAtom, YAtom, XB = dadil_clustering(
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



    plt.figure(figsize=(10, 8))


    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(Xs[0])
    clus = domains[0].labels
    for label in np.unique(clus):
        indices = np.where(clus == label)
        plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label)
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('Results/dadil_data.png')
    plt.show()
    plt.close()
    
    for i in range(len(domains)):
        plt.figure(figsize=(10, 8))
        tsne = TSNE(n_components=2, random_state=42)
        data=np.concatenate((Xs[i],XB[i]), axis=0)
        embedded_data2 = tsne.fit_transform(data)


        plt.scatter(embedded_data2[:len(Xs[i]), 0], embedded_data2[:len(Xs[i]), 1], label=f'Data{i}')

        # Plot data2
        plt.scatter(embedded_data2[len(Xs[i]):, 0], embedded_data2[len(Xs[i]):, 1], label='Centroids')


        plt.title(f"t-SNE Visualization of the clustering of domain{i} using U-DaDiL")
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig(f'Results/dadil_data{i}.png')
        plt.show()
        plt.close()
    #plt.figure(figsize=(10, 8))
    '''tsne = TSNE(n_components=2, random_state=42)
    data=np.concatenate((Xs[0],XB[0]), axis=0)
    embedded_data2 = tsne.fit_transform(data)


    plt.scatter(embedded_data2[:len(Xs[0]), 0], embedded_data2[:len(Xs[0]), 1], label='Data1')

    # Plot data2
    plt.scatter(embedded_data2[len(Xs[0]):, 0], embedded_data2[len(Xs[0]):, 1], label='Centroids')


    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('Results/dadil_data1.png')
    plt.show()
    plt.close()'''
    
    if len(mapped_labels) == 1:
        return [cluster_labels[0], mapped_labels[0]],XAtom, YAtom
    else:
        return [cluster_labels[0], *mapped_labels],XAtom, YAtom

import warnings
import numpy as np
import torch
import sys
warnings.filterwarnings('ignore')
sys.path.append('.../')
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('.../')

from md_clustering.utils.kmeans_utils import perform_kmeans_clustering
from md_clustering.utils.clustering_utils import clusters

from md_clustering.utils.kmeans_utils import perform_kmeans_clustering
from md_clustering.utils.clustering_utils import clusters

features = np.load('Data/resnet50-all--modern_office31.npy', allow_pickle=True)
labels = np.load('Data/labels-resnet50-all--modern_office31.npy', allow_pickle=True)

def KMeans_baseline(features):





    num_clusters = 31


    # Perform k-means clustering for each domain
    # List to store domain objects
    domains = []

    # List to store cluster labels
    cluster_labels = []

    # Iterate over features and perform clustering for each domain
    for feature in features:
        labels, _ = perform_kmeans_clustering(feature, num_clusters)
        cluster_labels.append(labels)
        domain = clusters(feature, labels, num_clusters)
        domain.cluster_data()
        domains.append(domain)

    # Map cluster labels for each domain using Domain 1's cluster tensors
    mapped_labels = []
    for i in range(1, len(domains)):
        mapped_labels.append(domains[i].clusters_mapping(domains[0].cluster_tensors))

    # Save mapped labels for each domain
    for i, mapped_label in enumerate(mapped_labels, start=2):
        np.save(f'Results/KMeans/MappedLabels_Domain{i}.npy', mapped_label)

    # Determine the return values based on the length of mapped_labels
    if len(mapped_labels) == 1:
        return [cluster_labels[0], mapped_labels[0]]
    else:
        return [cluster_labels[0], *mapped_labels]


if __name__ == "__main__":
    KMeans_baseline()



def KMeans_baselineTest(features,labels):


    #labels = [labels[:len(features[0])], labels[len(features[0]):len(features[1]) + len(features[0])],labels[len(features[1]):len(features[2]) + len(features[1])]]



    num_clusters = 31


    # Perform k-means clustering for each domain

    cluster_labels_domain_1, _ =perform_kmeans_clustering(features[0],num_clusters)
    cluster_labels_domain_2, _ =perform_kmeans_clustering(features[1], num_clusters)
    # Create cluster objects for each domain

    domain_1 = clusters(features[0], cluster_labels_domain_1, num_clusters)
    domain_2 = clusters(features[1], cluster_labels_domain_2, num_clusters)

    # Cluster data for each domain

    domain_1.cluster_data()
    domain_2.cluster_data()


    np.save('Results/KMeans/MappedLabels_Domain1.npy', cluster_labels_domain_1)

    # Map cluster labels for Domain 2 using Domain 1's cluster tensors

    mapped_labels_domain_2 = domain_2.clusters_mapping(domain_1.cluster_tensors)
    np.save('Results/KMeans/MappedLabels_Domain2.npy', mapped_labels_domain_2)

    # Map cluster labels for Domain 3 using Domain 1's cluster tensors

   # np.save('Results/KMeans/MappedLabels_Domain3.npy', mapped_labels_domain_3)

    return(cluster_labels_domain_1,mapped_labels_domain_2)
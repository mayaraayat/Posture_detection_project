import warnings
import numpy as np
import torch
import sys
warnings.filterwarnings('ignore')
sys.path.append('../../../')

import pickle
import sys

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from md_clustering.utils.kmeans_utils import perform_kmeans_clustering
from md_clustering.utils.clustering_utils import clusters
from sklearn.metrics import adjusted_rand_score


features = np.load('Data/resnet50-all--modern_office31.npy', allow_pickle=True)
labels = np.load('Data/labels-resnet50-all--modern_office31.npy', allow_pickle=True)

def KMeans_baseline(features,truelabels,num_clusters):








    # Perform k-means clustering for each domain
    # List to store domain objects
    domains = []

    # List to store cluster labels
    cluster_labels = []
    XB=[]
    # Iterate over features and perform clustering for each domain
    for feature in features:
        labels, centroids = perform_kmeans_clustering(feature, num_clusters)
        cluster_labels.append(labels)
        XB.append(centroids)
        domain = clusters(feature, labels, num_clusters)
        domain.cluster_data()
        domains.append(domain)

    # Map cluster labels for each domain using Domain 1's cluster tensors
    mapped_labels = []
    for i in range(1, len(domains)):
        mapped_labels.append(domains[i].clusters_mapping(domains[0].cluster_tensors))
    np.save(f'Results/KMeans/MappedLabels_Domain1.npy', cluster_labels[0])
    with open('Results/KMeans/features.pkl', 'wb') as file:
        pickle.dump(features, file)

    with open('Results/KMeans/labels.pkl', 'wb') as file:
        pickle.dump(truelabels, file)



    # Save mapped labels for each domain
    for i, mapped_label in enumerate(mapped_labels, start=2):
        np.save(f'Results/KMeans/MappedLabels_Domain{i}.npy', mapped_label)



    
    tsne = TSNE(n_components=2, random_state=42)
    #data=np.concatenate((features[1],XB[1]), axis=0)
    embedded_data = tsne.fit_transform(features[0])

    for label in np.unique(cluster_labels[0]):
        indices = np.where(cluster_labels[0] == label)
        plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label)

    




    plt.title(f't-SNE Visualization of the clustering of domain{i} using K-Means' )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('Results/kmeans_data.png')
    plt.show()
    
    for i in range(len(features)):
        plt.figure(figsize=(10, 8))
        tsne = TSNE(n_components=2, random_state=42)
        data=np.concatenate((features[i],XB[i]), axis=0)
        embedded_data = tsne.fit_transform(data)

        plt.scatter(embedded_data[:len(features[i]), 0], embedded_data[:len(features[i]), 1], label=f'Data{i}')

        # Plot data2
        plt.scatter(embedded_data[len(features[i]):, 0], embedded_data[len(features[i]):, 1], label='Centroids')
        plt.title(f't-SNE Visualization of the clustering of domain{i} using K-Means')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.savefig(f'Results/kmeans_datacentroids_{i}.png')
        plt.show()
        plt.close()
    '''plt.figure(figsize=(10, 8))
    tsne = TSNE(n_components=2, random_state=42)
    data=np.concatenate((features[1],XB[1]), axis=0)
    embedded_data = tsne.fit_transform(data)

    plt.scatter(embedded_data[:len(features[1]), 0], embedded_data[:len(features[1]), 1], label='Data1')

    # Plot data2
    plt.scatter(embedded_data[len(features[1]):, 0], embedded_data[len(features[1]):, 1], label='Centroids')
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('Results/kmeans_datacentroids1.png')
    plt.show()
    plt.close()'''
    # Assuming 'tensor_data' is your input tensor and 'labels' is your corresponding labels
    #tsne = TSNE(n_components=2, random_state=42)
    #embedded_data = tsne.fit_transform(features[0])

    # # Assuming 'labels' is your labels array
    # plt.figure(figsize=(10, 8))
    # for label in np.unique(cluster_labels[0]):
    #     indices = np.where(cluster_labels[0] == label)
    #     plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label)
    #
    # plt.title('t-SNE Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()
    # plt.show()
    # ari = adjusted_rand_score(truelabels[0],  cluster_labels[0])
    # print( cluster_labels[0].shape)
    # print(cluster_labels[0].shape)
    # print(truelabels[0], cluster_labels[0])
    # print(ari)


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
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('.../')

from md_clustering.utils.kmeans_utils import perform_kmeans_clustering
from md_clustering.utils.clustering_utils import clusters

def main():
    features = np.load('Experiments/Office31/Data/resnet50-all--modern_office31.npy', allow_pickle=True)
    labels = np.load('../labels-resnet50-all--modern_office31.npy', allow_pickle=True)

    data_list = features.tolist()
    features = []
    for array in data_list:
        tensor = torch.from_numpy(array)
        features.append(tensor)
    labels = [labels[:len(features[0])], labels[len(features[0]):len(features[1]) + len(features[0])],
                 labels[len(features[1]):len(features[2]) + len(features[1])]]
    ylabels = []
    for array in labels:
        labels = np.argmax(array, axis=1)
        ylabels.append(labels)

    labels = ylabels

    num_clusters = 31


    # Perform k-means clustering for each domain

    cluster_labels_domain_1, _ =perform_kmeans_clustering(features[0],num_clusters)
    cluster_labels_domain_2, _ =perform_kmeans_clustering(features[1], num_clusters)
    cluster_labels_domain_3, _ =perform_kmeans_clustering(features[2], num_clusters)
    # Create cluster objects for each domain

    domain_1 = clusters(features[0], cluster_labels_domain_1, num_clusters)
    domain_2 = clusters(features[1], cluster_labels_domain_2, num_clusters)
    domain_3 = clusters(features[2], cluster_labels_domain_3, num_clusters)

    # Cluster data for each domain

    domain_1.cluster_data()
    domain_2.cluster_data()
    domain_3.cluster_data()


    np.save('../md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain1.npy', cluster_labels_domain_1)

    # Map cluster labels for Domain 2 using Domain 1's cluster tensors

    mapped_labels_domain_2 = domain_2.clusters_mapping(domain_1.cluster_tensors)
    np.save('../md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain2.npy', mapped_labels_domain_2)

    # Map cluster labels for Domain 3 using Domain 1's cluster tensors

    mapped_labels_domain_3 = domain_3.clusters_mapping(domain_1.cluster_tensors)
    np.save('../md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain3.npy', mapped_labels_domain_3)


if __name__ == "__main__":
    main()

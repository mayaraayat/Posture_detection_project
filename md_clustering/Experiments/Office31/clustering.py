
# Import the necessary functions

from dictionary_learning.DaDiL_clustering import *
from md_clustering.utils.clustering_utils import clusters
import warnings
import sys
import os
sys.path.append('../../')
warnings.filterwarnings('ignore')

# Now you can use a simple import statement


def main():
    # Import features , labels and atoms
    features = np.load(
        './md_clustering/Experiments/Office31/Data/resnet50-all--modern_office31.npy', allow_pickle=True)
    data_list = features.tolist()
    features = []
    for array in data_list:
        tensor = torch.from_numpy(array)
        features.append(tensor)
    Y1 = np.load(
        './md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain1.npy', allow_pickle=True)
    Y2 = np.load(
        './md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain2.npy', allow_pickle=True)
    Y3 = np.load(
        './md_clustering/Experiments/Office31/Results/KMeans/MappedLabels_Domain3.npy', allow_pickle=True)

    # Load XP and YP NumPy files
    XP = []
    YP = []
    for i in range(len(features)):
        x_file_path = os.path.join(
            './md_clustering/Experiments/Office31/Results/Atoms', f'xatom_{i}.npy')
        y_file_path = os.path.join(
            './md_clustering/Experiments/Office31/Results/Atoms', f'yatom_{i}.npy')

        # Load XP
        loaded_x = np.load(x_file_path)
        XP.append(torch.tensor(loaded_x))

        # Load YP
        loaded_y = np.load(y_file_path)
        YP.append(torch.tensor(loaded_y))

    # Define hyperparameters
    n_classes = 31
    n_samples = 3000
    batch_size = 128
    n_components = 3
    n_datasets = 3
    reg = 0.0
    reg_labels = 0.0
    num_iter_max = 100

    Xs = features
    Ys = [torch.nn.functional.one_hot(torch.from_numpy(Y1).long(), num_classes=n_classes).float(), torch.nn.functional.one_hot(torch.from_numpy(
        Y2).long(), num_classes=n_classes).float(), torch.nn.functional.one_hot(torch.from_numpy(Y3).long(), num_classes=n_classes).float()]

    # Perform the DaDiL clustering
    cluster_labels = dadil_clustering(
        Xs, Ys, XP, YP, n_samples, n_components, reg, reg_labels, batch_size, n_classes, num_iter_max)

    domain_1 = clusters(features[0], cluster_labels[0], n_classes)
    domain_2 = clusters(features[1], cluster_labels[1], n_classes)
    domain_3 = clusters(features[2], cluster_labels[2], n_classes)

    # Cluster data for each domain

    domain_1.cluster_data()
    domain_2.cluster_data()
    domain_3.cluster_data()

    np.save('./md_clustering/Experiments/Office31/Results/DaDiL/MappedLabels_Domain1.npy',
            cluster_labels[0])

    mapped_labels_domain_2 = domain_2.clusters_mapping(
        domain_1.cluster_tensors)
    np.save('./md_clustering/Experiments/Office31/Results/DaDiL/MappedLabels_Domain2.npy',
            mapped_labels_domain_2)

    # Map cluster labels for Domain 3 using Domain 1's cluster tensors

    mapped_labels_domain_3 = domain_3.clusters_mapping(
        domain_1.cluster_tensors)
    np.save('./md_clustering/Experiments/Office31/Results/DaDiL/MappedLabels_Domain3.npy',
            mapped_labels_domain_3)


if __name__ == "__main__":
    main()

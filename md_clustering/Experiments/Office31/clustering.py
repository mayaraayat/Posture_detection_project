
import warnings

from dictionary_learning.DaDiL_clusteringV1 import dadil_clustering

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
    Xs = features
    cluster_labels, XAtom, YAtom = dadil_clustering(
        Xs, Ys, XP, YP, n_samples, reg, reg_labels, batch_size, n_classes, num_iter_max)

    domains=[]
    for i,feature in enumerate(features):
        labels=cluster_labels[i]
        domain = clusters(feature, labels, n_classes)
        domain.cluster_data()
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

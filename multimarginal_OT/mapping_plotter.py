import matplotlib.pyplot as plt
import numpy as np


def plot_cluster_mapping(clusters1, clusters2, mapping):
    """
    Plots the mapping between clusters from two domain (lists).

    Args:
        clusters1 (list): List of clusters from the first set.
        clusters2 (list): List of clusters from the second set.
        mapping (list): List of tuples representing the mapping between clusters.
                        Each tuple contains the index of the cluster from clusters1
                        and the index of the cluster from clusters2.

    Returns:
        None
    """
    fig, ax = plt.subplots()

    # Assign colors to clusters in domain 1
    colors1 = plt.cm.Paired(np.linspace(0, 1, len(clusters1)))



    # Plot clusters in domain 1
    for i, cluster in enumerate(clusters1):
        cluster = np.array(cluster).T
        ax.scatter(cluster[0], cluster[1], c=colors1[i], label=f'Cluster {i + 1} (Domain 1)')

    # Assign colors to clusters in domain 2
    colors2 = plt.cm.Set3(np.linspace(0, 1, len(clusters2)))

    # Plot clusters in domain 2
    for i, cluster in enumerate(clusters2):
        cluster = np.array(cluster).T
        ax.scatter(cluster[0], cluster[1], c=colors2[i], label=f'Cluster {i + 1} (Domain 2)')

    # Plot arrows for mapping between clusters with unique colors
    arrow_colors = plt.cm.tab10(np.linspace(0, 1, len(mapping)))
    for i, mapping_tuple in enumerate(mapping):
        cluster1_idx, cluster2_idx = mapping_tuple
        cluster1 = np.array(clusters1[cluster1_idx]).T
        cluster2 = np.array(clusters2[cluster2_idx]).T

        # Handle dimension mismatch
        min_dim = min(cluster1.shape[1], cluster2.shape[1])
        cluster1 = cluster1[:, :min_dim]
        cluster2 = cluster2[:, :min_dim]

        for j in range(len(cluster1[0])):
            ax.arrow(cluster1[0][j], cluster1[1][j], cluster2[0][j] - cluster1[0][j], cluster2[1][j] - cluster1[1][j],
                     head_width=0.2, head_length=0.2, fc=arrow_colors[i], ec=arrow_colors[i], alpha=0.5)
    ax.legend(loc='upper right', fontsize=4)

    plt.show()

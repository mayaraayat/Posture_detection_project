# kmeans_utils.py
import numpy as np
from sklearn.cluster import KMeans

def perform_kmeans_clustering(data, num_clusters):
    """
    Perform K-Means clustering on the input data.

r
    Args:
        data (numpy.ndarray): The data to be clustered.
        num_clusters (int): The number of clusters to create.

    Returns:
        numpy.ndarray: Cluster labels assigned to data points.
        numpy.ndarray: Cluster centers in feature space.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

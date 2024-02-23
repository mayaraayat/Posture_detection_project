from multimarginal_OT.solving_MM import solve_multimarginal_optimal_transport
from multimarginal_OT.cost_matrix import wasserstein_cost_matrix
import torch
import warnings
import numpy as np
import sys
sys.path.append('../../../')

warnings.filterwarnings('ignore')


class clusters:
    def __init__(self, data, labels, num_clusters):
        """
        Initialize the ClusteringPseudoLabels class.

        Args:
            data (numpy.ndarray): The data to be clustered.
            labels (numpy.ndarray): The labels corresponding to the data.
            num_clusters (int): The number of clusters to create.
        """
        self.data = data
        self.labels = labels
        self.num_clusters = num_clusters
        self.cluster_tensors = []

    def cluster_data(self):
        """
        Cluster data based on provided cluster labels.

        This method groups data points into clusters based on the cluster labels provided in 'self.labels'.
        """
        unique_values = torch.unique(torch.tensor(self.labels)).tolist()
        #print(unique_values)
        self.cluster_tensors = []
        for cluster_id in unique_values:
            cluster_indices = [i for i in range(
                len(self.data)) if self.labels[i] == cluster_id]
            if cluster_indices:
                #print(cluster_indices)
                cluster_tensor = np.array(self.data)[cluster_indices]
                #cluster_tensor = self.data[cluster_indices]
                self.cluster_tensors.append(cluster_tensor)
            else:
                self.cluster_tensors.append(None)
        #print(len(self.cluster_tensors))
    def clusters_mapping(self, target_data):
        """
        Map labels from source to target domain.

        Args:
            target_data (list): A list of target domain cluster tensors.

        Returns:
            numpy.ndarray: Mapped labels for the target domain.
        """
        # Convert self.cluster_tensors to PyTorch tensors
        self.cluster_tensors = [torch.from_numpy(data) for data in self.cluster_tensors]
        # Convert target_data to PyTorch tensors
        target_data_tensor = [torch.from_numpy(data) for data in target_data]

        # Use the target_data_tensor in the wasserstein_cost_matrix function
        Cost_matrix = wasserstein_cost_matrix(self.cluster_tensors, target_data_tensor)
        cluster_mapping = solve_multimarginal_optimal_transport(Cost_matrix)

        inverse_mapping_dict = {v: k for k, v in cluster_mapping}

        if torch.is_tensor(self.labels):
            self.labels=self.labels.tolist()


        mapped_labels = np.array([inverse_mapping_dict[label]
                                for label in self.labels if label in inverse_mapping_dict])
        return mapped_labels

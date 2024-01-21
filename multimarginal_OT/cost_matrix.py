import torch

import sys
sys.path.append('./')
from dictionary_learning.barycenters import wasserstein_barycenter_with_cost


def wasserstein_cost_matrix(source_distribution, target_distribution):
    """
    Computes the Wasserstein cost matrix between two distributions.

    Args:
        source_distribution (list): A list of source domain cluster tensors.
        target_distribution (list): A list of target domain cluster tensors.

    Returns:
        list: Cost matrix for the multimarginal optimal transport problem .
    """
    cost_matrix = []
    for source_tensor in source_distribution:
        cost_row = []
        for target_tensor in target_distribution:
            # Convert target_tensor to a torch tensor
            target_tensor = torch.tensor(target_tensor)

            # Create a pair of tensors for Wasserstein barycenter calculation
            tensor_pair = [target_tensor, source_tensor]

            # Calculate Wasserstein barycenter cost
            _, cost, _ = wasserstein_barycenter_with_cost(XP=tensor_pair, YP=None,
                                                          n_samples=3,
                                                          ϵ=0.0,
                                                          α=None,
                                                          num_iter_max=10,
                                                          initialization='random',
                                                          propagate_labels=False,
                                                          penalize_labels=False,
                                                          verbose=False,
                                                          τ=1e-9)
            cost_row.append(cost)

        cost_matrix.append(cost_row)

    return cost_matrix

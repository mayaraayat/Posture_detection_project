import torch
import numpy as np
import pytorch_lightning as pl
from dictionary_learning.utils import MultiDomainBalancedBatchSampler
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.lightning_dictionary import LightningDictionary
from dictionary_learning.barycenters import wasserstein_barycenter,wasserstein_barycenter_with_cost
from dictionary_learning.utils import DictionaryDADataset
from sklearn.metrics import pairwise_distances_argmin_min

import ot


def dadil_clustering(Xs, Ys,XP,YP, n_samples,reg,reg_labels, batch_size, n_classes, num_iter_max):
    # Create a MultiDomainBalancedBatchSampler
    train_dataset = DictionaryDADataset(Xs, Ys)
    S = MultiDomainBalancedBatchSampler([Ysk.argmax(dim=1).numpy() for Ysk in Ys],
                                        n_classes=31,
                                        batch_size=batch_size,
                                        n_batches=n_samples // batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=S)
    # Initialize LightningDictionary
    dictionary = LightningDictionary(
        XP=XP,
        YP=YP,
        n_samples=n_samples,
        n_dim=Xs[0].shape[1],
        n_classes=n_classes,
        n_components=len(Xs),
        n_distributions=len(Xs),
        learning_rate_features=1e-2,
        learning_rate_labels=1e-1,
        learning_rate_weights=1e-2,
        reg=reg,
        reg_labels=reg_labels,
        num_iter_barycenter=20,
        loss_fn=JointWassersteinLoss(ϵ=0.0),
        proj_grad=True,
        grad_labels=True,
        pseudo_label=False,
        balanced_sampling=True,
        sampling_with_replacement=True,
        track_atoms=True
    )

    # Create a Lightning Trainer
    trainer = pl.Trainer(max_epochs=num_iter_max, accelerator='cpu', logger=False, enable_checkpointing=False)

    # Train the dictionary
    trainer.fit(dictionary, train_loader)

    # Extract dictionary components
    XP = [XPk.detach() for XPk in dictionary.XP]
    YP = [YPk.detach() for YPk in dictionary.YP]
    A = dictionary.A.detach()
    costs=[]
    # Compute Wasserstein Barycenters
    XB = []
    for αℓ in A:
        XBℓ,_,_ = wasserstein_barycenter_with_cost(XP=XP, YP=YP,
                                     n_samples=n_classes,
                                     ϵ=0.0,
                                     α=αℓ,
                                     num_iter_max=num_iter_max,
                                     initialization='random',
                                     propagate_labels=False,
                                     penalize_labels=False,
                                     verbose=False,
                                     τ=1e-9)
        XB.append(XBℓ)







    #new_clusters = [torch.argmax(torch.transpose(torch.tensor(P), 0, 1), dim=1) for P in transport_plans]



    # mass_sent = torch.transpose(π[0], 0, 1)





    # Traditional Method (with eu distance)
    new_clusters = []
    for ℓ, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
        Cℓ = torch.cdist(XBℓ.cpu(), torch.tensor(XQℓ).float(), p=2) ** 2
        clusters_ℓ = Cℓ.argmin(dim=0)
        centroids=XBℓ

        unique_labels, counts = np.unique(clusters_ℓ, return_counts=True)
        # Determine empty clusters based on centroids
        empty_clusters = []
        for i, centroid in enumerate(XBℓ):
            if i not in unique_labels:
                empty_clusters.append(i)

        for empty_cluster in empty_clusters:
            # Randomly select a non-empty cluster
            non_empty_clusters = np.where(counts > 0)[0]
            non_empty_cluster = np.random.choice(non_empty_clusters)

            new_centroid = centroids[non_empty_cluster] + np.random.normal(scale=1e-6, size=centroids.shape[1])
            centroids[empty_cluster] = new_centroid

            mask = clusters_ℓ == non_empty_cluster
            clusters_ℓ[mask] = empty_cluster
        new_clusters.append(clusters_ℓ)



    # transport_plans=[]
    #
    # for k, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
    #     C_k = torch.cdist(XQℓ,XBℓ, p=2) ** 2
    #     cost_matrix = np.linalg.norm(XQℓ[:, np.newaxis, :] - XBℓ, axis=2)
    #
    #
    #
    #     π_k = ot.emd([], [], np.array(cost_matrix))
    #     print("shape de torch:",torch.tensor(π_k).shape)
    #     transport_plans.append(torch.tensor(π_k))
    # new_clusters_OT = [torch.argmax(torch.tensor(P), dim=1) for P in transport_plans]


    #Final_idea
    centroids=[]
    transport_plans = []
    for k, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
         C_k = torch.cdist(XQℓ,XBℓ, p=2) ** 2
         cost_matrix = np.linalg.norm(XBℓ[:, np.newaxis, :] - XQℓ, axis=2)



         π_k = ot.emd([], [], np.array(cost_matrix))

         n_samples=torch.tensor(π_k).shape[1]
         #centroid = sum([α_k * n_samples * torch.mm(π_k.T, XP_k) for α_k, π_k, XP_k in zip(α, π, XP)])
         centroid=torch.mm(torch.tensor(π_k), XQℓ)
         centroid_sums = centroid.sum(axis=1)

         # Normalize each centroid
         centroid = centroid * centroid_sums[:, np.newaxis]
         centroids.append(centroid)
         transport_plans.append(torch.tensor(π_k))

    new_clusters_OT = [torch.argmax(torch.tensor(P), dim=1) for P in transport_plans]


    new_clusters = []
    for ℓ, (XBℓ, XQℓ) in enumerate(zip(centroids, Xs)):
        Cℓ = torch.cdist(XBℓ.cpu(), torch.tensor(XQℓ).float(), p=2) ** 2
        clusters_ℓ = Cℓ.argmin(dim=0)
        unique_labels, counts = np.unique(clusters_ℓ, return_counts=True)
        # Determine empty clusters based on centroids
        empty_clusters = []
        for i, centroid in enumerate(XBℓ):
            if i not in unique_labels:
                empty_clusters.append(i)

        print("len empty_clusters " , len(empty_clusters))

        new_clusters.append(clusters_ℓ)




    XP1=[x.cpu() for x in XP]
    YP1=[y.cpu() for y in YP]


 #Change new_clusters with new_clusters_OT to use OT method

    return new_clusters,XP1,YP1




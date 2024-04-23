import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from dictionary_learning.utils import MultiDomainBalancedBatchSampler
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.lightning_dictionary import LightningDictionary
from dictionary_learning.barycenters import wasserstein_barycenter,wasserstein_barycenter_with_cost
from dictionary_learning.utils import DictionaryDADataset
from sklearn.metrics import pairwise_distances_argmin_min

import ot


def dadil_clustering(Xs,labels, Ys,XP,YP, n_samples,reg,reg_labels, batch_size, n_classes, num_iter_max):
    # Create a MultiDomainBalancedBatchSampler
    train_dataset = DictionaryDADataset(Xs, Ys)
    S = MultiDomainBalancedBatchSampler([Ysk.argmax(dim=1).numpy() for Ysk in Ys],
                                        n_classes=n_classes,
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
        reg_labels=0.0,
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
    trainer = pl.Trainer(max_epochs=num_iter_max, accelerator='cuda', logger=False, enable_checkpointing=False)

    # Train the dictionary
    trainer.fit(dictionary, train_loader)

    # Extract dictionary components
    XP = [XPk.detach() for XPk in dictionary.XP]
    YP = [YPk.detach() for YPk in dictionary.YP]

    print(XP[0].shape)
    print(XP[1].shape)
    print(XP[2].shape)

    A = dictionary.A.detach()
    costs=[]
    # Compute Wasserstein Barycenters
    XB = []
    for αℓ in A:
        XBℓ,_,_ = wasserstein_barycenter_with_cost(XP=XP, YP=YP,
                                     n_samples=7,
                                     ϵ=0,
                                     α=αℓ,
                                     num_iter_max=num_iter_max,
                                     initialization='random',
                                     propagate_labels=False,
                                     penalize_labels=False,
                                     verbose=False,
                                     τ=1e-9)
        XB.append(XBℓ)

    print("len XB",XB[0].shape)










    #new_clusters = [torch.argmax(torch.transpose(torch.tensor(P), 0, 1), dim=1) for P in transport_plans]



    # mass_sent = torch.transpose(π[0], 0, 1)

    transport_plans=[]
    # Traditional Method (with eu distance)
    new_clusters = []
    for ℓ, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
        Cℓ = torch.cdist(XBℓ.cpu(), torch.tensor(XQℓ).float(), p=2) ** 2
        cost_matrix = np.linalg.norm(XBℓ[:, np.newaxis, :] - XQℓ, axis=2)
        π_k = ot.emd([], [], np.array(cost_matrix))
        transport_plans.append(torch.tensor(π_k))
        clusters_ℓ = Cℓ.argmin(dim=0)
        centroids=XBℓ

        unique_labels, counts = np.unique(clusters_ℓ, return_counts=True)
        new_clusters.append(clusters_ℓ)


    plt.figure(figsize=(10, 8))


    tsne = TSNE(n_components=2, random_state=42)
    data=np.concatenate((Xs[1],XB[1]), axis=0)


    #plt.scatter(embedded_data1[:len(Xs[1]), 0], embedded_data1[:len(Xs[1]), 1], label='Data1')

    # Plot data2
    #plt.scatter(embedded_data1[len(Xs[1]):, 0], embedded_data1[len(Xs[1]):, 1], label='Centroids')


    embedded_data = tsne.fit_transform(Xs[0])

    for label in np.unique(new_clusters[0]):
         indices = np.where(new_clusters[0] == label)
         plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label)

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()





    tsne = TSNE(n_components=2, random_state=42)
    data=np.concatenate((Xs[2],XB[2]), axis=0)
    embedded_data2 = tsne.fit_transform(data)


    plt.scatter(embedded_data2[:len(Xs[2]), 0], embedded_data2[:len(Xs[2]), 1], label='Data1')

    # Plot data2
    plt.scatter(embedded_data2[len(Xs[2]):, 0], embedded_data2[len(Xs[2]):, 1], label='Centroids')


    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()




    tsne = TSNE(n_components=2, random_state=42)
    data=np.concatenate((Xs[0],XB[0]), axis=0)
    embedded_data = tsne.fit_transform(data)


    plt.scatter(embedded_data[:len(Xs[0]), 0], embedded_data[:len(Xs[0]), 1], label='Data1')

    # Plot data2
    plt.scatter(embedded_data[len(Xs[0]):, 0], embedded_data[len(Xs[0]):, 1], label='Centroids')


    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()



    # transport_plans=[]
    #
    # for k, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
    #     C_k = torch.cdist(XBℓ,XQℓ.float() , p=2) ** 2
    #     cost_matrix = np.linalg.norm(XBℓ[:, np.newaxis, :] - XQℓ, axis=2)
    #     π_k = ot.emd([], [], np.array(cost_matrix))
    #     print("shape de torch:",torch.tensor(π_k).shape)
    #     transport_plans.append(torch.tensor(π_k))
    #
    # new_clusters_OT = [torch.argmax(torch.tensor(P), dim=0) for P in transport_plans]


    # new_clusters = []
    # #Final_idea
    # centroids=[]
    # transport_plans = []
    # for k, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
    #      C_k = torch.cdist(XBℓ,XQℓ.float() ,p=2) ** 2
    #      cost_matrix = np.linalg.norm(XBℓ[:, np.newaxis, :] - XQℓ, axis=2)
    #      clusters_ℓ = C_k.argmin(dim=0)
    #      new_clusters.append(clusters_ℓ)
    #
    #      unique_labels, counts = np.unique(clusters_ℓ, return_counts=True)
    #      print(unique_labels)
    #      empty_clusters = []
    #      for i, centroid in enumerate(XBℓ):
    #          if i not in unique_labels:
    #              empty_clusters.append(i)
    #      print("len empty_clusters ", len(empty_clusters))
    #
    #      π_k = ot.emd([], [], np.array(cost_matrix))
    #      transport_plans.append(torch.tensor(π_k))

    #      n_samples=torch.tensor(π_k).shape[1]
    #      #centroid = sum([α_k * n_samples * torch.mm(π_k.T, XP_k) for α_k, π_k, XP_k in zip(α, π, XP)])
    #      centroid=torch.mm(torch.tensor(π_k), XQℓ)
    #      centroid_sums = centroid.sum(axis=1)
    #
    #      # Normalize each centroid
    #      centroid = centroid / centroid_sums[:, np.newaxis]
    #      #centroid=centroid / torch.tensor(n_samples)
    #      centroids.append(centroid)
    #
    # new_clusters_OT = [torch.argmax(torch.tensor(P), dim=1) for P in transport_plans]
    #
    #
    # new_clusters1 = []
    # for ℓ, (XBℓ, XQℓ) in enumerate(zip(centroids, Xs)):
    #     Cℓ = torch.cdist(XBℓ.cpu().float(), torch.tensor(XQℓ).float(), p=2) ** 2
    #     clusters_ℓ = Cℓ.argmin(dim=0)
    #     unique_labels, counts = np.unique(clusters_ℓ, return_counts=True)
    #     print(unique_labels,counts)
    #     new_clusters1.append(clusters_ℓ)



    XP1=[x.cpu() for x in XP]
    YP1=[y.cpu() for y in YP]


 #Change new_clusters with new_clusters_OT to use OT method

    return new_clusters,XP1,YP1




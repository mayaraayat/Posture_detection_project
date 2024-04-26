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
from md_clustering.utils.clustering_utils import clusters
from scipy.spatial.distance import cdist
import ot


def dadil_clustering(Xs, Ys,XP,YP, n_samples,reg,reg_labels, batch_size, n_classes, num_iter_max):
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
    #print(XP[2].shape)

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
    clusters_new = []
    for ℓ, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
        Cℓ = torch.cdist(XBℓ.cpu(), torch.tensor(XQℓ).float(), p=2) ** 2
        cost_matrix = np.linalg.norm(XBℓ[:, np.newaxis, :] - XQℓ, axis=2)
        π_k = ot.emd([], [], np.array(cost_matrix))
        transport_plans.append(torch.tensor(π_k))
        clusters_ℓ = Cℓ.argmin(dim=0)
        centroids=XBℓ

        unique_labels, counts = np.unique(clusters_ℓ, return_counts=True)
        clusters_new.append(clusters_ℓ)

    
    '''def split_clusters(domain, n_classes):
        while len(domain.cluster_tensors) < n_classes:
            # Find the largest cluster
            largest_cluster_index = np.argmax([len(cluster) for cluster in domain.cluster_tensors])
            largest_cluster = domain.cluster_tensors[largest_cluster_index]
            
            # Create a new centroid by perturbing the centroid of the largest cluster
            new_centroid = np.mean(largest_cluster, axis=0) + np.random.normal(scale=0.1, size=largest_cluster.shape[1])
            
            # Split the largest cluster into two around the new centroid
            distances = cdist(largest_cluster, [new_centroid])
            split_point = np.argmax(distances)
            cluster1 = largest_cluster[:split_point]
            cluster2 = largest_cluster[split_point:]
            
            # Replace the largest cluster with the two new clusters
            new_clusters = domain.cluster_tensors[:largest_cluster_index] + [cluster1, cluster2] + domain.cluster_tensors[largest_cluster_index+1:]
            
            # Assign labels based on the closest centroid
            new_centroids = [np.mean(cluster, axis=0) for cluster in new_clusters]
            new_labels = np.argmin(cdist(domain.data, new_centroids), axis=1)
            
            # Update domain 
            domain.cluster_tensors = new_clusters
            domain.labels = new_labels

        return domain
    features = Xs
    domains=[]
    for i,feature in enumerate(features):
        labels=clusters_new[i]
        domain = clusters(feature, labels, n_classes)
        domain.cluster_data()
        
        print(f'-----------domain{i}-------------------')
        print(len(domain.cluster_tensors))
        domain = split_clusters(domain, n_classes)
        print(len(domain.cluster_tensors))
        domains.append(domain)
    
    mapped_labels = []
    for i in range(1, len(domains)):
        mapped_labels.append(domains[i].clusters_mapping(domains[0].cluster_tensors))

    plt.figure(figsize=(10, 8))


    tsne = TSNE(n_components=2, random_state=42)
    data=np.concatenate((Xs[1],XB[1]), axis=0)


    #plt.scatter(embedded_data1[:len(Xs[1]), 0], embedded_data1[:len(Xs[1]), 1], label='Data1')

    # Plot data2
    #plt.scatter(embedded_data1[len(Xs[1]):, 0], embedded_data1[len(Xs[1]):, 1], label='Centroids')


    embedded_data = tsne.fit_transform(Xs[0])
    clus = domains[0].labels
    for label in np.unique(clus):
        indices = np.where(clus == label)
        plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=label)

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('Results/dadil_data.png')
    plt.show()
    plt.close()



    plt.figure(figsize=(10, 8))
    tsne = TSNE(n_components=2, random_state=42)
    data=np.concatenate((Xs[1],XB[1]), axis=0)
    embedded_data2 = tsne.fit_transform(data)


    plt.scatter(embedded_data2[:len(Xs[1]), 0], embedded_data2[:len(Xs[1]), 1], label='Data1')

    # Plot data2
    plt.scatter(embedded_data2[len(Xs[1]):, 0], embedded_data2[len(Xs[1]):, 1], label='Centroids')


    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('Results/dadil_data2.png')
    plt.show()
    plt.close()


    plt.figure(figsize=(10, 8))
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
    plt.savefig('Results/dadil_data1.png')
    plt.show()
    plt.close()

'''

    XP1=[x.cpu() for x in XP]
    YP1=[y.cpu() for y in YP]


 #Change new_clusters with new_clusters_OT to use OT method

    return clusters_new,XP1,YP1,XB




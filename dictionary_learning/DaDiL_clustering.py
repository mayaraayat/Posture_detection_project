import torch
import numpy as np
import pytorch_lightning as pl
from dictionary_learning.utils import MultiDomainBalancedBatchSampler
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.lightning_dictionary import LightningDictionary
from dictionary_learning.barycenters import wasserstein_barycenter
from dictionary_learning.utils import DictionaryDADataset




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

    # Compute Wasserstein Barycenters
    XB = []
    for αℓ in A:
        XBℓ = wasserstein_barycenter(XP=XP, YP=YP,
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

    # Assign data points to clusters based on the computed barycenters
    new_clusters = []
    for ℓ, (XBℓ, XQℓ) in enumerate(zip(XB, Xs)):
        Cℓ = torch.cdist(XBℓ.cpu(), torch.tensor(XQℓ).float(), p=2) ** 2
        clusters_ℓ = Cℓ.argmin(dim=0)
        new_clusters.append(clusters_ℓ)


    print("Data type of yAtom[0]:", YP[0].dtype)

    XP1=[x.cpu() for x in XP]
    YP1=[y.cpu() for y in YP]
    print("Data type of yAtom[0]:", YP1[0].dtype)
    #print(XP1.shape)
    #print(YP1.shape)
    return new_clusters,XP1,YP1




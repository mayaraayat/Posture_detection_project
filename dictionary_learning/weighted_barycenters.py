import torch
import pytorch_lightning as pl
from dictionary_learning.barycentric_regression import WassersteinBarycentricRegression
from dictionary_learning.utils import BalancedBatchSamplerWithTargets
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.utils import DictionaryDADataset
from dictionary_learning.barycenters import wasserstein_barycenter_with_cost


def compute_barycenters(Xs, Ys , n_samples, batch_size,num_iter_dil,
                                  n_classes, ϵ, η_A, lr, num_iter_max):
    # Initialize the maximum shape as None
    max_shape = None
    for tensor in Xs:
        current_shape = tensor.shape[0]
        if max_shape is None or current_shape > max_shape:
            max_shape = current_shape
    # Create a training dataset and dataloader
    train_dataset = DictionaryDADataset(Xs, Ys)
    S = BalancedBatchSamplerWithTargets([Ysk.argmax(dim=1).numpy() for Ysk in Ys],
                                        n_classes=n_classes,
                                        batch_size=batch_size,
                                        n_batches=n_samples // batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=S)

    # Determine the number of atoms based on the number of datasets
    num_atoms = len(Xs)
    names = ['source{}'.format(i) for i in range(num_atoms)]
    # Initialize the loss function and Wasserstein Barycentric Regression model
    loss_fn = JointWassersteinLoss()
    wbr = WassersteinBarycentricRegression(Xs, Ys, n_distributions=num_atoms, loss_fn=loss_fn,
                                           learning_rate=lr, reg=η_A,
                                           domain_names=names,
                                           sampling_with_replacement=True,
                                           weight_initialization='uniform')
    # Initialize the PyTorch Lightning Trainer

    trainer = pl.Trainer(max_epochs=num_iter_dil, accelerator='cpu', logger=False, enable_checkpointing=False)
    trainer.fit(wbr, train_loader)

    # Compute barycenters for each atom
    atom_results = []
    for i in range(num_atoms):
        w_i = wbr.A.detach().squeeze()[i]
        X_atom, Y_atom = wasserstein_barycenter_with_cost(XP=Xs , YP=Ys, n_samples=max_shape, ϵ=ϵ, α=w_i,
                                                β=None, num_iter_max=num_iter_max, verbose=True, propagate_labels=True,
                                                penalize_labels=True)
        atom_results.append((X_atom, Y_atom))

    return atom_results
import ot
import torch
import numpy as np
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score
from dictionary_learning.utils import unif
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.barycenters import wasserstein_barycenter_with_cost


class WassersteinBarycentricRegression(pl.LightningModule):
    def __init__(self,
                 Xs,
                 Ys,
                 A=None,
                 n_distributions=None,
                 weight_initialization='random',
                 loss_fn=None,
                 learning_rate=None,
                 reg=0.0,
                 num_iter_barycenter=10,
                 num_iter_sinkhorn=20,
                 domain_names=None,
                 proj_grad=True,
                 optimizer_name='adam',
                 balanced_sampling=True,
                 sampling_with_replacement=False,
                 pseudo_label=False,
                 barycenter_tol=1e-9,
                 barycenter_beta=None,
                 barycenter_verbose=False,
                 barycenter_label_metric='l2',
                 barycenter_initialization='random',
                 barycenter_covariance_type='diag',
                 dtype='float',
                 batch_size=5,
                 log_gradients=False):
        super(WassersteinBarycentricRegression, self).__init__()

        self.Xs = Xs
        self.Ys = Ys
        self.training_epoch_outputs = []  # Add this list

        self.reg = reg
        self.n_classes = Ys[0].shape[1]
        self.num_iter_barycenter = num_iter_barycenter
        self.num_iter_sinkhorn = num_iter_sinkhorn
        self.optimizer_name = optimizer_name
        self.balanced_sampling = balanced_sampling
        self.sampling_with_replacement = sampling_with_replacement
        self.pseudo_label = pseudo_label
        self.n_components = len(Xs)
        self.barycenter_tol = barycenter_tol
        self.barycenter_beta = barycenter_beta
        self.barycenter_verbose = barycenter_verbose
        self.barycenter_label_metric = barycenter_label_metric
        self.barycenter_initialization = barycenter_initialization
        self.barycenter_covariance_type = barycenter_covariance_type
        self.tensor_dtype = torch.float64 if dtype.lower() == 'double' else torch.float32
        self.log_gradients = log_gradients
        self.track_pseudo_label_acc = False
        self.batch_size = batch_size
        self.training_epoch_outputs = []
        device='gpu'
        self.to('cuda')
        # Initialize weights
        if A is None:
            if n_distributions is None:
                raise ValueError(
                    "If 'A' is not given you should specify 'n_distributions'")
            if weight_initialization == 'random':
                a_data = torch.rand(n_distributions, self.n_components, requires_grad=True).to(
                    self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad():
                        a_data = ot.utils.proj_simplex(a_data.T).T
            else:
                a_data = torch.ones(n_distributions, self.n_components, requires_grad=True).to(
                    self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad():
                        a_data = ot.utils.proj_simplex(a_data.T).T
        else:
            a_data = A
        self.A = torch.nn.parameter.Parameter(data=a_data, requires_grad=True)

        # Defines loss
        self.loss_fn = JointWassersteinLoss() if loss_fn is None else loss_fn

        # Learning rate
        # NOTE: if learning_rate_labels is not given, use learning_rate_features.
        # NOTE: if learning_rate_weights is not given, use learning_rate_features.
        self.learning_rate_weights = learning_rate

        # Whether or not we project on the simplex
        self.proj_grad = proj_grad

        # Sets history
        self.history = {'loss': [], 'label_entropy': [], 'weights': []}

        # Creates batch generator
        self.batch_generator = self.generate_batch_indices_without_replacement(
            batch_size=batch_size * self.n_classes)

        # Set domain names
        if domain_names is None:
            if n_distributions is None:
                raise ValueError(
                    "If 'domain_names' is not given, 'n_distributions' must be provided.")
            self.domain_names = ["Domain {}".format(
                ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

    def optimizer_step(

        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)

        if self.proj_grad:
            with torch.no_grad():
                self.A.data = ot.utils.proj_simplex(self.A.data.T).T

    def get_weights(self):
        if self.proj_grad:
            W = self.A
        else:
            W = self.A.softmax(dim=-1)

        return W.detach()

    def sample_from_atoms(self, n=None, detach=False):
        batch_features, batch_labels = [], []

        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        # Sampling
        for Xsk, Ysk in zip(self.Xs, self.Ys):
            # If balanced sampling, needs to select sampler_per_class from each class
            if self.balanced_sampling:
                # Gets categorical labels
                ysk = Ysk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                sampled_indices = []
                # Loops over each class
                for yu in ysk.unique():
                    # Gets indices from current class
                    ind = torch.where(ysk == yu)[0]
                    # Randomly permutes labels
                    perm = torch.randperm(len(ind))
                    ind = ind[perm]
                    if samples_per_class is None:
                        # If n was not given, samples all samples from the said class
                        sampled_indices.append(ind[:])
                    else:
                        # Samples "samples_per_class" from given class
                        sampled_indices.append(ind[:samples_per_class])
                # Concatenates all indices
                sampled_indices = torch.cat(
                    sampled_indices, dim=0).cpu().numpy()
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(
                    np.arange(self.n_samples), size=n)

            # Creates batch arrays
            features_k, labels_k = Xsk[sampled_indices], Ysk[sampled_indices]

            if detach:
                features_k, labels_k = features_k.detach(), labels_k.detach()

            batch_features.append(features_k)
            batch_labels.append(labels_k)

        return batch_features, batch_labels

    def generate_batch_indices_without_replacement(self, batch_size=None):
        n_batches = self.n_samples // batch_size
        n_classes_per_batch = batch_size // self.n_classes

        for i in range(n_batches + 1):
            batch_indices = []
            for Ysk in self.Ys:
                # Gets categorical labels
                ysk = Ysk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                atom_batch_indices = []
                # Loops over each class
                for yu in ysk.unique():
                    indices = np.where(ysk == yu)[0]
                    atom_batch_indices.append(
                        indices[n_classes_per_batch *
                                i: n_classes_per_batch * (i + 1)]
                    )
                atom_batch_indices = np.concatenate(atom_batch_indices)
                np.random.shuffle(atom_batch_indices)
                batch_indices.append(atom_batch_indices)
            yield batch_indices

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            if self.logger is not None:
                self.logger.experiment.add_histogram(
                    name, params, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.A, 'lr': self.learning_rate_weights}
        ])

    def training_step(self, batch, batch_idx):
        if self.sampling_with_replacement:
            return self.__training_step_with_replacement(batch, batch_idx)
        else:
            return self.__training_step_without_replacement(batch, batch_idx)

    def __training_step_without_replacement(self, batch, batch_idx):
        if len(batch) == 3:
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            gt_target_labels = None
            target_labels = None
        elif len(batch) == 4:
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            gt_target_labels = batch[3]
            target_labels = None
        elif len(batch) == 5:
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            gt_target_labels = batch[3]
            target_labels = batch[4]
        else:
            raise ValueError(
                "Expected batch to have 3, 4 or 5 elements, but got {}".format(len(batch)))

        batch_size = len(target_features)

        XQ = [target_features]
        YQ = [target_labels]

        # Samples from atoms
        try:
            batch_indices = next(self.batch_generator)
            if len(batch_indices[0]) == 0:
                self.batch_generator = self.generate_batch_indices_without_replacement(
                    batch_size=self.n_classes * self.batch_size)
                batch_indices = next(self.batch_generator)
        except StopIteration:
            self.batch_generator = self.generate_batch_indices_without_replacement(
                batch_size=self.n_classes * self.batch_size)
            batch_indices = next(self.batch_generator)

        Xs = [Xsk[ind_k] for Xsk, ind_k in zip(self.Xs, batch_indices)]
        Ys = [Ysk[ind_k] for Ysk, ind_k in zip(self.Ys, batch_indices)]

        L = 0.0
        per_dataset_loss = {}
        for ℓ, (XQℓ, YQℓ, aℓ) in enumerate(zip(XQ, YQ, self.A)):
            # Change of variables
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)

            # XP = [XPk[ind_k] for XPk, ind_k in zip(self.XP, batch_indices)]
            # YP = [YPk[ind_k] for YPk, ind_k in zip(self.YP, batch_indices)]

            # Calculates the Wasserstein Barycenter
            XBℓ, YBℓ = wasserstein_barycenter(XP=Xs, YP=Ys,
                                              n_samples=batch_size,
                                              ϵ=self.reg,
                                              α=αℓ,
                                              β=self.barycenter_beta,
                                              num_iter_max=self.num_iter_barycenter,
                                              num_iter_sinkhorn=self.num_iter_sinkhorn,
                                              initialization=self.barycenter_initialization,
                                              propagate_labels=True,
                                              penalize_labels=True,
                                              verbose=self.barycenter_verbose,
                                              covariance_type=self.barycenter_covariance_type,
                                              label_metric=self.barycenter_label_metric,
                                              τ=self.barycenter_tol)

            # Calculates current loss
            loss_ℓ = self.loss_fn(XQℓ, YQℓ, XBℓ, YBℓ)

            # Accumulates loss
            L += loss_ℓ  # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ

        wc_target = self.loss_fn(
            target_features, gt_target_labels, XBℓ, YBℓ).item()

        if self.pseudo_label:
            return {'loss': L, 'Wc_target': wc_target, **per_dataset_loss}
        return {'loss': L, 'Wc_target': wc_target, **per_dataset_loss}

    def __training_step_with_replacement(self, batch, batch_idx):

        source_features, source_labels = batch[0], batch[1]

        XQ = [xq for xq in source_features]
        YQ = [yq for yq in source_labels]

        batch_size = 128

        # Samples from atoms
        Xs, Ys = self.sample_from_atoms(n=batch_size)

        L = 0.0
        per_dataset_loss = {}
        for ℓ, (XQℓ, YQℓ, aℓ) in enumerate(zip(XQ, YQ, self.A)):
            # Change of variables
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)
            device='cuda'
            # Calculates the Wasserstein Barycenter
            XBℓ, YBℓ = wasserstein_barycenter_with_cost(XP=[XP_k.to(device) for XP_k in Xs],
    YP=[YP_k.to(device) for YP_k in Ys],
                                                        n_samples=batch_size,
                                                        ϵ=self.reg,
                                                        α=αℓ,
                                                        β=self.barycenter_beta,
                                                        num_iter_max=self.num_iter_barycenter,
                                                        num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                        initialization=self.barycenter_initialization,
                                                        propagate_labels=True,
                                                        penalize_labels=True,
                                                        verbose=self.barycenter_verbose,
                                                        covariance_type=self.barycenter_covariance_type,
                                                        label_metric=self.barycenter_label_metric,
                                                        τ=self.barycenter_tol)

            # Calculates current loss
            loss_ℓ = self.loss_fn(XQℓ, YQℓ, XBℓ, YBℓ)

            # Accumulates loss
            L += loss_ℓ  # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ

        #wc_target = self.loss_fn(target_features, gt_target_labels, XBℓ, YBℓ).item()
        self.training_epoch_outputs.append({'loss': L, **per_dataset_loss})
        if self.pseudo_label:
            return {'loss': L, **per_dataset_loss}
        return {'loss': L,  **per_dataset_loss}

    def training_epoch_end(self, outputs):

        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        #avg_wc_target = torch.tensor([x['Wc_target'] for x in outputs]).mean()
        avg_loss_per_dataset = {
            self.domain_names[ℓ]: torch.tensor([x[self.domain_names[ℓ]] for x in outputs]).mean() for ℓ in range(self.A.shape[0])
        }

        _A = ot.utils.proj_simplex(self.A.clone().T).T

        # Adds loss to history
        self.history['loss'].append(avg_loss)
        self.history['weights'].append({
            name: ot.utils.proj_simplex(weights.detach()) for name, weights in zip(self.domain_names, _A)
        })

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar(
                "DiL Loss/Train", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar(
                "Wc/Target/Train", avg_wc_target, self.current_epoch)
            for domain in avg_loss_per_dataset:
                self.logger.experiment.add_scalar(
                    "DiL Loss {}/Train".format(domain), avg_loss_per_dataset[domain], self.current_epoch)

            for ℓ in range(_A.shape[0]):
                if self.proj_grad:
                    αℓ = _A[ℓ].detach().cpu()
                else:
                    αℓ = _A[ℓ].detach().softmax(dim=0).cpu()
                a = np.arange(len(αℓ))
                self.logger.experiment.add_histogram("Weights/" + self.domain_names[ℓ],
                                                     np.random.choice(
                                                         a, p=αℓ.numpy(), size=1000),
                                                     self.current_epoch)
            if self.log_gradients:
                # Logs histograms
                self.custom_histogram_adder()


    '''def on_train_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        # You can calculate other metrics and log information here
        self.log("avg_loss", avg_loss, on_epoch=True)
        avg_wc_target = torch.tensor([x['Wc_target'] for x in outputs]).mean()
        avg_loss_per_dataset = {
            self.domain_names[ℓ]: torch.tensor([x[self.domain_names[ℓ]] for x in outputs]).mean() for ℓ in range(self.A.shape[0])
        }

        _A = ot.utils.proj_simplex(self.A.clone().T).T

        # Adds loss to history
        self.history['loss'].append(avg_loss)
        self.history['weights'].append({
            name: ot.utils.proj_simplex(weights.detach()) for name, weights in zip(self.domain_names, _A)
        })

        if self.logger is not None:
            self.logger.experiment.add_scalar(
                "DiL Loss/Train", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar(
                "Wc/Target/Train", avg_wc_target, self.current_epoch)
            for domain in avg_loss_per_dataset:
                self.logger.experiment.add_scalar(
                    "DiL Loss {}/Train".format(domain), avg_loss_per_dataset[domain], self.current_epoch)

        for ℓ in range(_A.shape[0]):
            if self.proj_grad:
                αℓ = _A[ℓ].detach().cpu()
            else:
                αℓ = _A[ℓ].detach().softmax(dim=0).cpu()
            a = np.arange(len(αℓ))
            self.logger.experiment.add_histogram("Weights/" + self.domain_names[ℓ],
                                                 np.random.choice(
                a, p=αℓ.numpy(), size=1000),
                self.current_epoch)
        if self.log_gradients:
            self.custom_histogram_adder()
'''
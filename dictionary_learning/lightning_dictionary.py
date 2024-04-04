r"""This module implements Algorithm 2 of our main paper, using [Pytorch Lightning](https://www.pytorchlightning.ai/index.html). As such, it should be the preferred method
for learning a dictionary.

## References

[Schmitz et al., 2018] Schmitz, M. A., Heitz, M., Bonneel, N., Ngole, F., Coeurjolly, D., Cuturi, M., ... & Starck, J. L. (2018). Wasserstein dictionary learning: Optimal transport-based unsupervised nonlinear dictionary learning. SIAM Journal on Imaging Sciences, 11(1), 643-678.

[Turrisi et al., 2022] Turrisi, R., Flamary, R., Rakotomamonjy, A., & Pontil, M. (2022, August). Multi-source domain adaptation via weighted joint distributions optimal transport. In Uncertainty in Artificial Intelligence (pp. 1970-1980). PMLR.

[Montesuma, Mboula and Souloumiac, 2023] Multi-Source Domain Adaptation through Dataset
Dictionary Learning in Wasserstein Space, Submitted.

"""

import ot
import torch
import numpy as np
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score
from dictionary_learning.utils import unif
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.barycenters import wasserstein_barycenter








class LightningDictionary(pl.LightningModule):
    r"""Implementation of Dataset Dictionary Learning (DaDiL) using Pytorch Lightning. This class has as parameters
    a set $\mathcal{P} = \set{\hat{P}_{k}}_{k=1}^{K}$ of atoms, and a set $\mathcal{A}=\set{\alpha_{\ell}}_{\ell=1}^{N}$ of weights.

    The atoms are labeled empirical distributions, that is,

    $$\hat{P}_{k}(\mathbf{x},\mathbf{y})=\dfrac{1}{n}\sum_{i=1}^{n}\delta((\mathbf{x},\mathbf{y})-(\mathbf{x}_{i}^{(P_{k})}, \mathbf{y}_{i}^{(P_{k})}))$$

    and the weights are $K-$dimensional vectors whose components are all positive and sum to one, i.e. $\alpha_{\ell} \in \Delta_{K}$.
    """
    def __init__(self,
                 XP=None,
                 YP=None,
                 A=None,
                 n_samples=1024,
                 n_dim=None,
                 n_classes=None,
                 n_components=2,
                 weight_initialization='random',
                 n_distributions=None,
                 loss_fn=None,
                 learning_rate_features=1e-1,
                 learning_rate_labels=None,
                 learning_rate_weights=None,
                 reg=0.0,
                 reg_labels=0.0,
                 num_iter_barycenter=10,
                 num_iter_sinkhorn=20,
                 domain_names=None,
                 proj_grad=True,
                 grad_labels=True,
                 optimizer_name='adam',
                 balanced_sampling=True,
                 sampling_with_replacement=True,
                 pseudo_label=False,
                 barycenter_tol=1e-9,
                 barycenter_beta=None,
                 barycenter_verbose=False,
                 barycenter_label_metric='l2',
                 barycenter_initialization='random',
                 barycenter_covariance_type='diag',
                 dtype='float',
                 batch_size=5,
                 log_gradients=False,
                 track_atoms=False):
        r"""Initializes a LightningDictionary Object. This class should be used for Domain Adaptation.

        Args:
            XP: list of tensors.
                Manual initialization of atom features. List of tensors of shape $(n, d)$ containing
                the support (features) of each atom.
            YP: list of tensors.
                Manual initialization of atom labels. List of tensors of shape $(n, c)$ containing
                the support (labels) of each atom.
            A: Tensor.
                Manual initialization of barycentric coordinates. Tensor of shape $(N, K)$, where each element
                $(ℓ,k)$ indicates the coordinate of $\hat{Q}_{\ell}$ w.r.t. $\hat{P}_{k}$.
            n_samples: Integer.
                Number of samples $n$ in the support of atom distributions.
            n_dim: Integer.
                Number of dimensions $d$ of the support of atom distributions.
                It should be specified if ```XP``` is not given.
            n_classes: Integer.
                Number of classes $c$ in the support of atom distributions.
                It should be specified if ```YP``` is not given.
            n_components: Integer.
                Number of atoms $K$ in the dictionary.
            weight_initialization: String.
                Either 'random' or 'uniform' corresponding to how $\alpha_{\ell,k}$ are initialized.
            n_distributions: Integer.
                Number of distributions in $\mathcal{Q}$. Should be specified if $\mathcal{A}$
                is not given.
            loss_fn: Function.
                Function implementing the loss that is minimized throughout DiL.
                If not specified, uses the c-Wasserstein distance.
            learning_rate_features: Float.
                Learning rate $\eta_{x}$ applied to atom features $\mathbf{X}^{(P_{k})}$.
            learning_rate_labels: Float.
                Learning rate $\eta_{y}$ applied to atom labels $\mathbf{Y}^{(P_{k})}$. If not given, uses $\eta_{x}$ by default.
            learning_rate_weights: Float.
                Learning rate $\eta_{\alpha}$ applied to barycentric coordinates $\alpha_{\ell,k}$. If not given, uses $\eta_{x}$ by default.
            reg: Float.
                Amount of entropic regularization $\epsilon$ used when solving OT.
            reg_labels: Float.
                Penalizes labels with high entropy. Note, following our experiments, this regularization
                term is unecessary, as DiL naturally penalizes labels with high entropy.
            num_iter_barycenter: Integer.
                Number of steps when solving a Wasserstein barycenter
                (Algorithm 1 in [Montesuma, Mboula and Souloumiac, 2023]).
            num_iter_sinkhorn: Integer.
                Number of steps when solving regularized OT. Only used if $\epsilon > 0$.
            domain_names: List of Strings.
                List of names for each domain, for better logging.
            proj_grad: Boolean.
                If True, projects the barycentric coefficients in the simplex as [Turrisi et al., 2022].
                Otherwise, performs a change of variables through a softmax, as in [Schmitz et al., 2018].
            grad_labels: Boolean.
                If True, calculates gradients w.r.t. labels. Setting it to False is equivalent to $\eta_{y} = 0$
            optimizer_name: String.
                Choice of optimizer. Either 'adam' or 'sgd'.
            balanced_sampling: Boolean.
                If True, samples balanced mini-batches from atoms. This is key to the success of DA.
            sampling_with_replacement: Boolean.
                If True, samples from atoms with replacement.
            pseudo_label: Boolean.
                If True, uses pseudo-labels (passed through the dataloader) in the calculation of the loss
                in the target domain.
            barycenter_tol: Float.
                Stopping criteria for barycenter calculation.
            barycenter_beta: Float.
                Label importance in the ground-cost
            barycenter_verbose: Boolean.
                If True, prints info about barycenter calculation.
            barycenter_label_metric: String.
                Either 'l2' or 'delta'. It specifies the metric for which label distances are calculated.
            barycenter_initialization: String.
                Parameter for initializing the barycenter support.
                Either 'random', 'class', 'samples', 'zeros'.
            barycenter_covariance_type: String.
                If barycenter initialization is 'Class', specifies how to calculate the covariance
                (i.e., 'full', 'diag', 'none'). __WARNING__ 'full' may give numerical errors.
            dtype: String.
                Either 'float' or 'double'. Should agree with dtype of XP and A.
            batch_size: Integer.
                Number of samples per class in batches. Effective batch size corresponds to this parameter
                times number of classes.
            log_gradients: Boolean.
                Boolean. If True, logs gradients of variables in Tensorboard.
                __WARNING__ memory intensive.
            track_atoms: Boolean.
                Boolean. If True, saves atoms at each iteration in dictionary.history. __WARNING__ memory intensive.
        """
        super(LightningDictionary, self).__init__()

        self.n_samples = n_samples
        self.reg = reg
        self.reg_labels = reg_labels
        self.num_iter_barycenter = num_iter_barycenter
        self.num_iter_sinkhorn = num_iter_sinkhorn
        self.grad_labels = grad_labels
        self.optimizer_name = optimizer_name
        self.balanced_sampling = balanced_sampling
        self.sampling_with_replacement = sampling_with_replacement
        self.pseudo_label = pseudo_label
        self.n_components = n_components
        self.barycenter_tol = barycenter_tol
        self.barycenter_beta = barycenter_beta
        self.barycenter_verbose = barycenter_verbose
        self.barycenter_label_metric = barycenter_label_metric
        self.barycenter_initialization = barycenter_initialization
        self.barycenter_covariance_type = barycenter_covariance_type
        self.tensor_dtype = torch.float64 if dtype.lower() == 'double' else torch.float32
        self.log_gradients = log_gradients
        self.track_pseudo_label_acc = False
        self.track_atoms = track_atoms
        self.batch_size = batch_size

        self.training_step_outputs = []

        self.save_hyperparameters("n_samples", "reg", "num_iter_barycenter", "num_iter_sinkhorn")

        # Initialize atom features
        if XP is None:
            if n_dim is None:
                raise ValueError("If 'XP' is not given, you should specify 'n_dim'.")
            XP_data = [torch.randn(n_samples, n_dim, requires_grad=True).to(self.tensor_dtype) for _ in range(n_components)]
            self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp, requires_grad=True) for xp in XP_data])
            self.n_dim = n_dim
        else:
            self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp.to(self.tensor_dtype), requires_grad=True) for xp in XP])
            self.n_dim = XP[0].shape[1]

        # Initialize atom labels
        if YP is None:
            if n_classes is None:
                raise ValueError("If 'YP' is not given, you should specify 'n_classes'")
            self.n_classes = n_classes
            samples_per_class = n_samples // n_classes
            if self.n_samples % n_classes != 0:
                self.n_samples = n_classes * samples_per_class
                print("[WARNING] n_samples ({}) is not divisible by n_classes ({}). Using n_samples = {} instead".format(n_samples,
                                                                                                                         n_classes,
                                                                                                                         self.n_samples))
            YP_data = []
            for _ in range(n_components):
                ypk = torch.cat([torch.tensor([c] * samples_per_class) for c in range(self.n_classes)]).long()
                YPk = torch.nn.functional.one_hot(ypk, num_classes=self.n_classes)
                YP_data.append(YPk)
            self.YP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=yp.to(self.tensor_dtype), requires_grad=grad_labels) for yp in YP_data])
        else:
            self.YP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=yp.to(self.tensor_dtype), requires_grad=grad_labels) for yp in YP])
            self.n_classes = YP[0].shape[1]

        # Initialize weights
        if A is None:
            if n_distributions is None:
                raise ValueError("If 'A' is not given you should specify 'n_distributions'")
            if weight_initialization == 'random':
                a_data = torch.rand(n_distributions, n_components, requires_grad=True).to(self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad(): a_data = ot.utils.proj_simplex(a_data.T).T
            else:
                a_data = torch.ones(n_distributions, n_components, requires_grad=True).to(self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad(): a_data = ot.utils.proj_simplex(a_data.T).T
        else:
            a_data = A
        self.A = torch.nn.parameter.Parameter(data=a_data, requires_grad=True)

        # Defines loss
        self.loss_fn = JointWassersteinLoss() if loss_fn is None else loss_fn

        # Learning rate
        self.learning_rate_features = learning_rate_features
        # NOTE: if learning_rate_labels is not given, use learning_rate_features.
        self.learning_rate_labels = learning_rate_labels if learning_rate_labels is not None else learning_rate_features
        # NOTE: if learning_rate_weights is not given, use learning_rate_features.
        self.learning_rate_weights = learning_rate_weights if learning_rate_weights is not None else learning_rate_features

        # Whether or not we project on the simplex
        self.proj_grad = proj_grad

        # Variable Update Tracker
        self.var_tracker = [torch.zeros(n_samples) for _ in range(self.n_components)]

        # Sets history
        self.history = {
            'loss': [],
            'label_entropy': [],
            'weights': [],
            'atoms': {'Atom {}'.format(k): {'Features': [], 'Labels': []} for k in range(self.n_components)}
        }

        # Creates batch generator
        self.batch_generator = self.generate_batch_indices_without_replacement(batch_size=batch_size * n_classes)

        # Set domain names
        if domain_names is None:
            if n_distributions is None:
                raise ValueError("If 'domain_names' is not given, 'n_distributions' must be provided.")
            self.domain_names = ["Domain {}".format(ℓ) for ℓ in range(n_distributions)]
            self.domain_names = ["Domain {}".format(ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, *, on_tpu=None, using_native_amp=None, using_lbfgs=None):
    # implementation of optimizer step
    # implementation of optimizer step
        r"""Updates dictionary variables using gradients."""
        optimizer.step(closure=optimizer_closure) # remove 'closure=optimizer_closure' here
        optimizer.zero_grad()
        self.lr_scheduler.step()


        if self.proj_grad:
            with torch.no_grad():
                self.A.data = ot.utils.proj_simplex(self.A.data.T).T

    def get_weights(self):
        r"""Returns the barycentric coordinates of distributions in $\mathcal{Q}$."""
        if self.proj_grad:
            W = self.A
        else:
            W = self.A.softmax(dim=-1)

        return W.detach()

    def sample_from_atoms(self, n=None, detach=False):
        """Samples (with replacement) $n$ samples from atoms support.

        Args:
            n: Integer.
                Number of samples (with replacement) acquired from the atoms support.
                If $n$ is None, gets all samples from the atoms supports.
            detach: boolean.
                If True, detaches tensors so that gradients are not calculated.
        """
        batch_features, batch_labels = [], []

        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        # Sampling
        for tracker, XPk, YPk in zip(self.var_tracker, self.XP, self.YP):
            # If balanced sampling, needs to select sampler_per_class from each class
            if self.balanced_sampling:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                sampled_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    # Gets indices from current class
                    ind = torch.where(yPk == yu)[0]
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
                sampled_indices = torch.cat(sampled_indices, dim=0).cpu().numpy()
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples), size=n)

            # Adds counter of sampling
            #print(sampled_indices)
            tracker[sampled_indices] += 1

            # Creates batch arrays
            features_k, labels_k = XPk[sampled_indices], YPk[sampled_indices]

            if self.grad_labels:
                labels_k = labels_k.softmax(dim=-1)

            if detach:
                features_k, labels_k = features_k.detach(), labels_k.detach()

            batch_features.append(features_k)
            batch_labels.append(labels_k)

        return batch_features, batch_labels


    def generate_batch_indices_without_replacement(self, batch_size=None):
        r"""Divides the atom indices into mini-batches.

        Args:
            batch_size: Integer.
                Number of samples in each batch.
        """
        n_batches = self.n_samples // batch_size
        n_classes_per_batch = batch_size // self.n_classes

        for i in range(n_batches + 1):
            batch_indices = []
            for YPk in self.YP:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                atom_batch_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    indices = np.where(yPk == yu)[0]
                    atom_batch_indices.append(
                        indices[n_classes_per_batch * i: n_classes_per_batch * (i + 1)]
                    )
                atom_batch_indices = np.concatenate(atom_batch_indices)
                np.random.shuffle(atom_batch_indices)
                batch_indices.append(atom_batch_indices)
            yield batch_indices


    def custom_histogram_adder(self):
        r"""Adds variable histograms to Tensorboard. __WARNING:__ this function generates"""
        for name, params in self.named_parameters():
            if self.logger is not None:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        r"""Configures optimizers for Pytorch Lightning. Adds ```XP``` and ```A``` as optimization
        variables. If ```grad_labels``` is True, then ```YP``` is added as a variable as well."""
        if self.grad_labels:
            return torch.optim.Adam([
                {'params': self.XP, 'lr': self.learning_rate_features},
                {'params': self.YP, 'lr': self.learning_rate_labels},
                {'params': self.A, 'lr': self.learning_rate_weights}
            ])
        else:
            return torch.optim.Adam([
                {'params': self.XP, 'lr': self.learning_rate_features},
                {'params': self.A, 'lr': self.learning_rate_weights}
            ])


    def training_step(self, batch, batch_idx):
        r"""Runs a single optimization step. This function is used internally by Pytorch Lightning.
        The training_step is implemented in ```__training_step_with_replacement``` and ```__training_step_without_replacement```,
        for sampling with and without replacement respectively.

        Args:
            batch: list of $N$ pairs of Tensors of shape $(n_{b}, d)$ and $(n_{b}, c)$.
                Contains a list of pairs of tensors corresponding to a minibatch $(\mathbf{X}^{(Q_{\ell})}, \mathbf{Y}^{(Q_{\ell})})$
                from each dataset $\hat{Q}_{\ell} \in \mathcal{Q}$.
            batch_index: Integer.
                Not used.
        """
        if self.sampling_with_replacement:
            output = self.__training_step_with_replacement(batch, batch_idx)
        else:
            output = self.__training_step_without_replacement(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def __training_step_with_replacement(self, batch, batch_idx):
        # Line 5: Batch Treatment. We need to figure out what was given by the dataloader.
        if len(batch) == 2:
            # If the batch contains 3 elements, then no label information is given.
            source_features, source_labels = batch[0], batch[1]
            gt_target_labels = None
            target_labels = None
        elif len(batch) == 4:
            # If the batch contains 4 elements, the 4th element is the ground-truth labels
            # of the target domain. NOTE, we cannot use this during training, since we are
            # performing Unsupervised Domain Adaptation.
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            target_labels = batch[3]
            #target_labels = None
        elif len(batch) == 5:
            # If the batch contains 5 elements, the 5th element is the pseudo-labels of the
            # target domain, obtained in an unsupervised way. NOTE, we CAN use this information
            # during training, since pseudo-labels are obtained in an unsupervised way. Even if
            # these were provided, we only use if pseudo_label is set to True in the class constructor.
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            gt_target_labels = batch[3]
            target_labels = batch[4]
        else:
            # In any other case, we raise an exception.
            raise ValueError("Expected batch to have 3, 4 or 5 elements, but got {}".format(len(batch)))

        XQ = [xq for xq in source_features]
        YQ = [yq for yq in source_labels]
        batch_size = 128

        L = 0.0
        per_dataset_loss = {}
        for ℓ, (XQℓ, YQℓ, aℓ) in enumerate(zip(XQ, YQ, self.A)):
            # Line 9: Samples from atoms
            XP, YP = self.sample_from_atoms(n=batch_size)

            # NOTE: If we do not project gradients in optimizer_step, we need
            #       an additional step for changing variables.
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)

            # Line 13: Calculates the Wasserstein Barycenter
            XBℓ, YBℓ = wasserstein_barycenter(XP=XP, YP=YP,
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

            # Calculates current loss L(Qℓ, Bℓ).
            # NOTE: If ℓ corresponds to the target domain, YQℓ is None, and we
            #       treat this internally in the loss_fn. This corresponds to the
            #       condition in the definition of L
            loss_ℓ = self.loss_fn(XQℓ, YQℓ, XBℓ, YBℓ, index=ℓ)

            # Line 15: Accumulates loss (summation for defining L = sum(L(Qℓ, Bℓ)))
            L += loss_ℓ # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ

        # Evaluates the entropy of labels
        ΩY = 0.0
        for YPk in self.YP:
            probs = YPk.softmax(dim=1)
            ΩY += -((probs * torch.log(probs)).mean(dim=1)).mean()

        # If reg_labels > 0.0 and optimization is done w.r.t. labels,
        # we penalize labels with high entropy.
        if self.grad_labels and self.reg_labels > 0.0:
            L += self.reg_labels * ΩY

        # For debugging: we calculate the c-Wasserstein distance with ground-truth labels.
        #wc_target = self.loss_fn(target_features, gt_target_labels, XBℓ, YBℓ, index=-1).item()

        return {'loss': L, 'label_entropy': ΩY, **per_dataset_loss}

    def on_train_epoch_end(self):
        r"""Logs information to Tensorboard, if the logger specified in the trainer object
        is not None."""
        avg_loss = torch.tensor([x['loss'] for x in self.training_step_outputs]).mean()
        avg_ΩY = torch.tensor([x['label_entropy'] for x in self.training_step_outputs]).mean()
         #avg_wc_target = torch.tensor([x['Wc_target'] for x in self.training_step_outputs]).mean()

        avg_loss_per_dataset = {
            self.domain_names[ℓ]: torch.tensor([x[self.domain_names[ℓ]] for x in self.training_step_outputs]).mean() for ℓ in range(self.A.shape[0])
        }

        self.training_step_outputs.clear()

        _A = self.A.clone()
        if self.proj_grad:
            _A = ot.utils.proj_simplex(_A.T).T
        else:
            _A = _A.softmax(dim=1)
        _A = _A.detach().cpu()

        # Adds loss to history
        self.history['loss'].append(avg_loss)
        self.history['label_entropy'].append(avg_ΩY)
        self.history['weights'].append({
            name: ot.utils.proj_simplex(weights.detach()) for name, weights in zip(self.domain_names, _A)
        })
        if self.track_atoms:
            with torch.no_grad():
                for k, (_XPk, _YPk) in enumerate(zip(self.XP, self.YP)):
                    self.history['atoms']['Atom {}'.format(k)]['Features'].append(_XPk.data.cpu().clone())
                    self.history['atoms']['Atom {}'.format(k)]['Labels'].append(_YPk.data.softmax(dim=-1).cpu().clone())

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar("DiL Loss/Train", avg_loss, self.current_epoch)
            #self.logger.experiment.add_scalar("Wc/Target/Train", avg_wc_target, self.current_epoch)
            for domain in avg_loss_per_dataset:
                self.logger.experiment.add_scalar("DiL Loss {}/Train".format(domain), avg_loss_per_dataset[domain], self.current_epoch)

            for ℓ in range(_A.shape[0]):
                αℓ = _A[ℓ]
                a = np.arange(len(αℓ))
                self.logger.experiment.add_histogram("Weights/" + self.domain_names[ℓ],
                                                    np.random.choice(a, p=αℓ.numpy(), size=1000),
                                                    self.current_epoch)
            if self.log_gradients:
                # Logs histograms
                self.custom_histogram_adder()









class LightningUnsupervisedDictionary(pl.LightningModule):
    r"""Implementation of Dataset Dictionary Learning (DaDiL) using Pytorch Lightning. This class has as parameters
    a set $\mathcal{P} = \set{\hat{P}_{k}}_{k=1}^{K}$ of atoms, and a set $\mathcal{A} = \set{ \alpha_{\ell}}_{\ell=1}^{N}$ of weights.
    
    The atoms are unlabeled empirical distributions, that is,

    $$\hat{P}_{k}(\mathbf{x})=\dfrac{1}{n}\sum_{i=1}^{n}\delta(\mathbf{x} - \mathbf{x}_{i}^{(P_{k})})$$

    and the weights are $K-$dimensional vectors whose components are all positive and sum to one, i.e. $\alpha_{\ell} \in \Delta_{K}$.

    __NOTE.__ Since this class inherits from ```pl.LightningModule```, some methods on the docs come from the parent class. You should ignore methods
    that do not have text description.
    """
    def __init__(self,
                 XP=None,
                 A=None,
                 n_samples=1024,
                 n_dim=None,
                 n_components=2,
                 weight_initialization='random',
                 n_distributions=None,
                 loss_fn=None,
                 learning_rate_features=1e-1,
                 learning_rate_weights=None,
                 reg=0.0,
                 num_iter_barycenter=10,
                 num_iter_sinkhorn=20,
                 domain_names=None,
                 proj_grad=True,
                 optimizer_name='adam',
                 sampling_with_replacement=True,
                 barycenter_initialization='random',
                 barycenter_covariance_type='full',
                 barycenter_verbose=False,
                 barycenter_tol=1e-9,
                 dtype='float',
                 batch_size=128,
                 log_gradients=False,
                 track_atoms=False):
        r"""Initializes a LightningUnsupervisedDictionary object
        
        Args:
            XP: List of tensors of shape $(n, d)$.
                Manually initializes the atoms support.
            A: Tensor of shape $(N, K)$.
                Manually initializes the barycentric coefficient matrix $\mathcal{A}$.
            n_samples: Integer.
                Number of samples $n$ in the support of atom distributions.
            n_dim: Integer.
                Number of dimensions $d$ of the support of atom distributions.
                It should be specified if ```XP``` is not given.
            n_components: Integer.
                Number of atoms in the dictionary.
            weight_initialization: String.
                Either 'random' or 'uniform' corresponding to how $\alpha_{\ell,k}$ are initialized.
            n_distributions: Integer.
                Number of distributions in $\mathcal{Q}$. Should be specified if $\mathcal{A}$
                is not given.
            loss_fn: Function.
                Function implementing the loss that is minimized throughout DiL.
                If not specified, uses the default 2-Wasserstein distance.
            learning_rate_features: Float.
                Learning rate $\eta$ applied to atom features $\mathbf{X}^{(P_{k})}$.
            learning_rate_weights: Float.
                If given, considers an independent learning rate $\eta_{\alpha}$ for the barycentric
                coordinates. If not given, uses $\eta$ by default.
            reg: Float.
                Amount of entropic regularization $\epsilon$ used when solving OT.
            num_iter_barycenter: Integer.
                Number of steps when solving a Wasserstein barycenter
                (Algorithm 1 in [Montesuma, Mboula and Souloumiac, 2023]).
            num_iter_sinkhorn: Integer.
                Number of steps when solving regularized OT. Only used if $\epsilon > 0$.
            proj_grad: Boolean.
                If True, projects the barycentric coefficients in the simplex as [Turrisi et al., 2022].
                Otherwise, performs a change of variables through a softmax, as in [Schmitz et al., 2018].
            optimizer_name: String.
                Choice of optimizer. Either 'adam' or 'sgd'.
            sampling_with_replacement: Bool.
                If True, samples from atoms with replacement.
            barycenter_initialization: String.
                Parameter for initializing the barycenter support. Either 'random', 'class', 'samples', 'zeros'.
            barycenter_covariance_type: String. If barycenter initialization is 'Class', specifies how to calculate the covariance (i.e., 'full', 'diag', 'none'). WARNING: 'full' may give numerical errors.
            barycenter_verbose: Boolean.
                If True, prints info about barycenter calculation.
            barycenter_tol: Float.
                Stopping criteria for barycenter calculation.
            dtype: string.
                Either 'float' or 'double'. Should agree with dtype of XP and A.
            batch_size: Integer.
                Batch size used during learning. Only used if sampling WITHOUT replacement.
            log_gradients: Boolean. If True, logs gradients of variables in Tensorboard. __WARNING:__ memory intensive.
            track_atoms: Boolean. If True, saves atoms at each iteration in dictionary.history. __WARNING:__ memory intensive.
        """
        super(LightningUnsupervisedDictionary, self).__init__()

        self.n_samples = n_samples
        self.reg = reg
        self.num_iter_barycenter = num_iter_barycenter
        self.num_iter_sinkhorn = num_iter_sinkhorn
        self.optimizer_name = optimizer_name
        self.sampling_with_replacement = sampling_with_replacement
        self.n_components = n_components
        self.barycenter_initialization = barycenter_initialization
        self.barycenter_verbose = barycenter_verbose
        self.barycenter_tol = barycenter_tol
        self.tensor_dtype = torch.float64 if dtype.lower() == 'double' else torch.float32
        self.log_gradients = log_gradients
        self.batch_size = batch_size
        self.barycenter_covariance_type = barycenter_covariance_type
        self.track_atoms = track_atoms

        self.training_step_outputs = []
        self.save_hyperparameters("n_samples", "reg", "num_iter_barycenter", "num_iter_sinkhorn")

        # Initialize atom features
        if XP is None:
            if n_dim is None:
                raise ValueError("If 'XP' is not given, you should specify 'n_dim'.")
            XP_data = [torch.randn(n_samples, n_dim, requires_grad=True).to(self.tensor_dtype) for _ in range(n_components)]
            self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp, requires_grad=True) for xp in XP_data])
            self.n_dim = n_dim
        else:
            self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp.to(self.tensor_dtype), requires_grad=True) for xp in XP])
            self.n_dim = XP[0].shape[1]
        
        # Initialize weights
        if A is None:
            if n_distributions is None:
                raise ValueError("If 'A' is not given you should specify 'n_distributions'")
            if weight_initialization == 'random':
                a_data = torch.rand(n_distributions, n_components, requires_grad=True).to(self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad(): a_data = ot.utils.proj_simplex(a_data.T).T
            else:
                a_data = torch.ones(n_distributions, n_components, requires_grad=True).to(self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad(): a_data = ot.utils.proj_simplex(a_data.T).T
        else:
            a_data = A
        self.A = torch.nn.parameter.Parameter(data=a_data, requires_grad=True)

        # Defines loss
        self.loss_fn = JointWassersteinLoss() if loss_fn is None else loss_fn

        # Learning rate
        # NOTE: if learning_rate_labels is not given, use learning_rate_features.
        # NOTE: if learning_rate_weights is not given, use learning_rate_features.
        self.learning_rate_features = learning_rate_features
        self.learning_rate_weights = learning_rate_weights if learning_rate_weights is not None else learning_rate_features

        # Whether or not we project on the simplex
        self.proj_grad = proj_grad

        # Variable Update Tracker
        self.var_tracker = [torch.zeros(n_samples) for _ in range(self.n_components)]

        # Sets history
        self.history = {'loss': [], 'label_entropy': [], 'weights': [], 'atoms': {'Atom {}'.format(k): {'Features': []} for k in range(self.n_components)}}

        # Creates batch generator
        self.batch_generator = self.generate_batch_indices_without_replacement(batch_size=batch_size)

        # Set domain names
        if domain_names is None:
            if n_distributions is None:
                raise ValueError("If 'domain_names' is not given, 'n_distributions' must be provided.")
            self.domain_names = ["Domain {}".format(ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, *, on_tpu=None, using_native_amp=None, using_lbfgs=None):
    # implementation of optimizer step
        r"""Updates dictionary variables using gradients."""
        optimizer.step(closure=optimizer_closure) # remove 'closure=optimizer_closure' here
        optimizer.zero_grad()
        self.lr_scheduler.step()


    def get_weights(self):
        r"""Returns the barycentric coordinates of distributions in $\mathcal{Q}$."""
        if self.proj_grad:
            W = self.A
        else:
            W = self.A.softmax(dim=-1)

        return W.detach()

    def sample_from_atoms(self, n=None, detach=False):
        """Samples (with replacement) $n$ samples from atoms support.
        
        Args:
            n: Integer.
                Number of samples (with replacement) acquired from the atoms support.
                If $n$ is None, gets all samples from the atoms supports.
            detach: boolean.
                If True, detaches tensors so that gradients are not calculated.
        """
        batch_features = []

        # Sampling
        for tracker, XPk in zip(self.var_tracker, self.XP):
            # Determining the number of samples
            if n is not None:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples), size=n)
            else:
                sampled_indices = np.arange(len(self.n_samples))
            
            # Adds counter of sampling
            tracker[sampled_indices] += 1
            
            # Creates batch arrays
            features_k = XPk[sampled_indices]

            if detach:
                features_k = features_k.detach()

            batch_features.append(features_k)

        return batch_features


    def generate_batch_indices_without_replacement(self, batch_size=None):
        r"""Divides the atom indices into mini-batches.
        
        Args:
            batch_size: Integer.
                Number of samples in each batch.
        """
        n_batches = self.n_samples // batch_size

        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)

        for i in range(n_batches + 1):
            yield indices[i * batch_size:(i + 1) * batch_size]


    def custom_histogram_adder(self):
        r"""Adds variable histograms to Tensorboard. __WARNING:__ this function generates
        heavy Tensorboard logs."""
        for name, params in self.named_parameters():
            if self.logger is not None:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        r"""Configures optimizers for Pytorch Lightning. Adds ```XP``` and ```A``` as optimization
        variables."""
        return torch.optim.Adam([
            {'params': self.XP, 'lr': self.learning_rate_features},
            {'params': self.A, 'lr': self.learning_rate_weights}
        ])

    def training_step(self, batch, batch_idx):
        r"""Runs a single optimization step. This function is used internally by Pytorch Lightning.
        The training_step is implemented in ```__training_step_with_replacement``` and ```__training_step_without_replacement```,
        for sampling with and without replacement respectively.
        
        Args:
            batch: list of $N$ Tensors of shape $(n_{b}, d)$.
                Contains a list of tensors corresponding to a minibatch $\mathbf{X}^{(Q_{\ell})}$ from
                each dataset $\hat{Q}_{\ell} \in \mathcal{Q}$.
            batch_index: Integer.
                Not used.
        """
        if self.sampling_with_replacement:
            output = self.__training_step_with_replacement(batch, batch_idx)
        else:
            output = self.__training_step_without_replacement(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def __training_step_without_replacement(self, batch, batch_idx):
        r"""Training step without replacement. In this case, we get batch indices that were previously divided in
        ```generate_batch_indices_without_replacement```. This gives us an iterator for the batch indices.
        
        Args:
            batch: list of $N$ Tensors of shape $(n_{b}, d)$.
                Contains a list of tensors corresponding to a minibatch $\mathbf{X}^{(Q_{\ell})}$ from
                each dataset $\hat{Q}_{\ell} \in \mathcal{Q}$.
            batch_index: Integer.
                Not used.
        """
        # Line 5: Sample XQℓ from each Qℓ
        XQ = batch

        # Need batch size for generating batch indices
        batch_size = len(XQ[0])

        L = 0.0
        per_dataset_loss = {}
        for ℓ, (XQℓ, aℓ) in enumerate(zip(XQ, self.A)):
            # Line 9: Sample XPk from each Pk
            try:
                batch_indices = next(self.batch_generator)
                if len(batch_indices[0]) == 0:
                    self.batch_generator = self.generate_batch_indices_without_replacement(batch_size=batch_size)
                    batch_indices = next(self.batch_generator)
            except StopIteration:
                self.batch_generator = self.generate_batch_indices_without_replacement(batch_size=batch_size)
                batch_indices = next(self.batch_generator)

            XP = [XPk[ind_k] for XPk, ind_k in zip(self.XP, batch_indices)]

            # NOTE: If we do not project gradients in optimizer_step, we need
            #       an additional step for changing variables.
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)

            # Line 13: Calculates the Wasserstein Barycenter
            XBℓ = wasserstein_barycenter(XP=XP, YP=None,
                                         n_samples=batch_size,
                                         ϵ=self.reg,
                                         α=αℓ,
                                         num_iter_max=self.num_iter_barycenter,
                                         num_iter_sinkhorn=self.num_iter_sinkhorn,
                                         initialization=self.barycenter_initialization,
                                         propagate_labels=False,
                                         penalize_labels=False,
                                         verbose=self.barycenter_verbose,
                                         covariance_type=self.barycenter_covariance_type,
                                         τ=self.barycenter_tol)

            # Calculates current loss L(Qℓ, Bℓ)
            loss_ℓ = self.loss_fn(XQℓ, XBℓ)

            # Line 15: Accumulates loss (summation for defining L = sum(L(Qℓ, Bℓ)))
            L += loss_ℓ # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ
        
            # NOTE: Lines 16, 17 and 18 are implemented internally by Pytorch Lightning.
            #       for more information see the method "optimization_step".
        return {'loss': L, **per_dataset_loss}

    def __training_step_with_replacement(self, batch, batch_idx):
        # Line 5: Sample XQℓ from each Qℓ
        XQ = batch
        
        # NOTE: We need the batch size for specifying the number of
        #       samples in the barycenter support
        batch_size = len(XQ[0])

        L = 0.0
        per_dataset_loss = {}
        for ℓ, (XQℓ, aℓ) in enumerate(zip(XQ, self.A)):
            # Line 9: Sample XPk from each Pk
            XP = self.sample_from_atoms(n=batch_size)

            # NOTE: If we do not project gradients in optimizer_step, we need
            #       an additional step for changing variables.
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)

            # Line 13: Calculates the Wasserstein Barycenter
            XBℓ = wasserstein_barycenter(XP=XP, YP=None,
                                         n_samples=batch_size,
                                         ϵ=self.reg,
                                         α=αℓ,
                                         num_iter_max=self.num_iter_barycenter,
                                         num_iter_sinkhorn=self.num_iter_sinkhorn,
                                         initialization=self.barycenter_initialization,
                                         propagate_labels=False,
                                         penalize_labels=False,
                                         verbose=self.barycenter_verbose,
                                         covariance_type=self.barycenter_covariance_type,
                                         τ=self.barycenter_tol)

            # Calculates current loss L(Qℓ, Bℓ)
            loss_ℓ = self.loss_fn(XQℓ, XBℓ)

            # Line 15: Accumulates loss (summation for defining L = sum(L(Qℓ, Bℓ)))
            L += loss_ℓ # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ
        
            # NOTE: Lines 16, 17 and 18 are implemented internally by Pytorch Lightning.
            #       for more information see the method "optimization_step".
        return {'loss': L, **per_dataset_loss}

    def on_train_epoch_end(self):
        r"""Logs information to Tensorboard, if the logger specified in the trainer object
        is not None."""
        avg_loss = torch.tensor([x['loss'] for x in self.training_step_outputs]).mean()
        avg_loss_per_dataset = {
            self.domain_names[ℓ]: torch.tensor([x[self.domain_names[ℓ]] for x in self.training_step_outputs]).mean() for ℓ in range(self.A.shape[0])
        }

        _A = self.A.clone()
        if self.proj_grad:
            _A = ot.utils.proj_simplex(_A.T).T
        else:
            _A = _A.softmax(dim=1)
        _A = _A.detach().cpu()

        # Adds loss to history
        self.history['loss'].append(avg_loss)
        self.history['weights'].append({
            name: weights.detach() for name, weights in zip(self.domain_names, _A)
        })
        if self.track_atoms:
            with torch.no_grad():
                for k, XPk in enumerate(self.XP):
                    self.history['atoms']['Atom {}'.format(k)]['Features'].append(XPk.data.cpu().clone())

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar("DiL Loss/Train", avg_loss, self.current_epoch)
            for domain in avg_loss_per_dataset:
                self.logger.experiment.add_scalar("DiL Loss {}/Train".format(domain), avg_loss_per_dataset[domain], self.current_epoch)
            
            for ℓ in range(_A.shape[0]):
                αℓ = _A[ℓ]
                a = np.arange(len(αℓ))
                self.logger.experiment.add_histogram("Weights/" + self.domain_names[ℓ],
                                                    np.random.choice(a, p=αℓ.numpy(), size=1000),
                                                    self.current_epoch)
            if self.log_gradients:
                # Logs histograms
                self.custom_histogram_adder()

    def reconstruct(self, α=None, n_samples_atoms=None, n_samples_barycenter=None):
        r"""Obtains a given reconstruction using the barycentric coordinates $\alpha$, i.e.,
        calculates $\mathcal{B}(\alpha;\mathcal{P})$.
        
        Args:
            α: Tensor of shape (K,).
                Must correspond to a barycentric coordinate vector, i.e., its components must be
                positive and it must sum to one.
            n_samples_atoms: Integer.
                Number of samples to be acquired from atom distributions.
            n_samples_barycenter: Integer.
                Number of samples generated in the support of the Barycenter distribution.
        """
        XP = self.sample_from_atoms(n=n_samples_atoms)
        n_samples_barycenter = len(XP[0]) if n_samples_barycenter is None else n_samples_barycenter
        with torch.no_grad():
            if α is None:
                Q_rec = []
                for aℓ in self.A:
                    if self.proj_grad:
                        αℓ = aℓ
                    else:
                        αℓ = aℓ.softmax(dim=0)
                    XB = wasserstein_barycenter(XP=XP, YP=None,
                                                n_samples=n_samples_barycenter,
                                                ϵ=self.reg,
                                                α=αℓ,
                                                num_iter_max=self.num_iter_barycenter,
                                                num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                initialization=self.barycenter_initialization,
                                                propagate_labels=False,
                                                penalize_labels=False,
                                                verbose=self.barycenter_verbose,
                                                covariance_type=self.barycenter_covariance_type,
                                                τ=self.barycenter_tol)
                    Q_rec.append(XB)
            else:
                XB = wasserstein_barycenter(XP=XP, YP=None,
                                            n_samples=n_samples_barycenter,
                                            ϵ=self.reg,
                                            α=αℓ,
                                            num_iter_max=self.num_iter_barycenter,
                                            num_iter_sinkhorn=self.num_iter_sinkhorn,
                                            initialization=self.barycenter_initialization,
                                            propagate_labels=False,
                                            penalize_labels=False,
                                            verbose=self.barycenter_verbose,
                                            covariance_type=self.barycenter_covariance_type,
                                            τ=self.barycenter_tol)
                Q_rec = [XB]
        return Q_rec


class LightningDictionary(pl.LightningModule):
    r"""Implementation of Dataset Dictionary Learning (DaDiL) using Pytorch Lightning. This class has as parameters
    a set $\mathcal{P} = \set{\hat{P}_{k}}_{k=1}^{K}$ of atoms, and a set $\mathcal{A}=\set{\alpha_{\ell}}_{\ell=1}^{N}$ of weights.
    
    The atoms are labeled empirical distributions, that is,

    $$\hat{P}_{k}(\mathbf{x},\mathbf{y})=\dfrac{1}{n}\sum_{i=1}^{n}\delta((\mathbf{x},\mathbf{y})-(\mathbf{x}_{i}^{(P_{k})}, \mathbf{y}_{i}^{(P_{k})}))$$

    and the weights are $K-$dimensional vectors whose components are all positive and sum to one, i.e. $\alpha_{\ell} \in \Delta_{K}$.
    """
    def __init__(self,
                 XP=None,
                 YP=None,
                 A=None,
                 n_samples=1024,
                 n_dim=None,
                 n_classes=None,
                 n_components=2,
                 weight_initialization='random',
                 n_distributions=None,
                 loss_fn=None,
                 learning_rate_features=1e-1,
                 learning_rate_labels=None,
                 learning_rate_weights=None,
                 reg=0.0,
                 reg_labels=0.0,
                 num_iter_barycenter=10,
                 num_iter_sinkhorn=20,
                 domain_names=None,
                 proj_grad=True,
                 grad_labels=True,
                 optimizer_name='adam',
                 balanced_sampling=True,
                 sampling_with_replacement=True,
                 pseudo_label=False,
                 barycenter_tol=1e-9,
                 barycenter_beta=None,
                 barycenter_verbose=False,
                 barycenter_label_metric='l2',
                 barycenter_initialization='random',
                 barycenter_covariance_type='diag',
                 dtype='float',
                 batch_size=5,
                 log_gradients=False,
                 track_atoms=False):
        r"""Initializes a LightningDictionary Object. This class should be used for Domain Adaptation.
        
        Args:
            XP: list of tensors.
                Manual initialization of atom features. List of tensors of shape $(n, d)$ containing
                the support (features) of each atom.
            YP: list of tensors.
                Manual initialization of atom labels. List of tensors of shape $(n, c)$ containing
                the support (labels) of each atom.
            A: Tensor.
                Manual initialization of barycentric coordinates. Tensor of shape $(N, K)$, where each element
                $(ℓ,k)$ indicates the coordinate of $\hat{Q}_{\ell}$ w.r.t. $\hat{P}_{k}$.
            n_samples: Integer.
                Number of samples $n$ in the support of atom distributions.
            n_dim: Integer.
                Number of dimensions $d$ of the support of atom distributions.
                It should be specified if ```XP``` is not given.
            n_classes: Integer.
                Number of classes $c$ in the support of atom distributions.
                It should be specified if ```YP``` is not given.
            n_components: Integer.
                Number of atoms $K$ in the dictionary.
            weight_initialization: String.
                Either 'random' or 'uniform' corresponding to how $\alpha_{\ell,k}$ are initialized.
            n_distributions: Integer.
                Number of distributions in $\mathcal{Q}$. Should be specified if $\mathcal{A}$
                is not given.
            loss_fn: Function.
                Function implementing the loss that is minimized throughout DiL.
                If not specified, uses the c-Wasserstein distance.
            learning_rate_features: Float.
                Learning rate $\eta_{x}$ applied to atom features $\mathbf{X}^{(P_{k})}$.
            learning_rate_labels: Float.
                Learning rate $\eta_{y}$ applied to atom labels $\mathbf{Y}^{(P_{k})}$. If not given, uses $\eta_{x}$ by default.
            learning_rate_weights: Float.
                Learning rate $\eta_{\alpha}$ applied to barycentric coordinates $\alpha_{\ell,k}$. If not given, uses $\eta_{x}$ by default.
            reg: Float.
                Amount of entropic regularization $\epsilon$ used when solving OT.
            reg_labels: Float.
                Penalizes labels with high entropy. Note, following our experiments, this regularization
                term is unecessary, as DiL naturally penalizes labels with high entropy.
            num_iter_barycenter: Integer.
                Number of steps when solving a Wasserstein barycenter
                (Algorithm 1 in [Montesuma, Mboula and Souloumiac, 2023]).
            num_iter_sinkhorn: Integer.
                Number of steps when solving regularized OT. Only used if $\epsilon > 0$.
            domain_names: List of Strings.
                List of names for each domain, for better logging.
            proj_grad: Boolean.
                If True, projects the barycentric coefficients in the simplex as [Turrisi et al., 2022].
                Otherwise, performs a change of variables through a softmax, as in [Schmitz et al., 2018].
            grad_labels: Boolean.
                If True, calculates gradients w.r.t. labels. Setting it to False is equivalent to $\eta_{y} = 0$
            optimizer_name: String.
                Choice of optimizer. Either 'adam' or 'sgd'.
            balanced_sampling: Boolean.
                If True, samples balanced mini-batches from atoms. This is key to the success of DA.
            sampling_with_replacement: Boolean.
                If True, samples from atoms with replacement.
            pseudo_label: Boolean.
                If True, uses pseudo-labels (passed through the dataloader) in the calculation of the loss
                in the target domain.
            barycenter_tol: Float.
                Stopping criteria for barycenter calculation.
            barycenter_beta: Float.
                Label importance in the ground-cost
            barycenter_verbose: Boolean.
                If True, prints info about barycenter calculation.
            barycenter_label_metric: String.
                Either 'l2' or 'delta'. It specifies the metric for which label distances are calculated.
            barycenter_initialization: String.
                Parameter for initializing the barycenter support.
                Either 'random', 'class', 'samples', 'zeros'.
            barycenter_covariance_type: String.
                If barycenter initialization is 'Class', specifies how to calculate the covariance
                (i.e., 'full', 'diag', 'none'). __WARNING__ 'full' may give numerical errors.
            dtype: String.
                Either 'float' or 'double'. Should agree with dtype of XP and A.
            batch_size: Integer.
                Number of samples per class in batches. Effective batch size corresponds to this parameter
                times number of classes.
            log_gradients: Boolean.
                Boolean. If True, logs gradients of variables in Tensorboard.
                __WARNING__ memory intensive.
            track_atoms: Boolean.
                Boolean. If True, saves atoms at each iteration in dictionary.history. __WARNING__ memory intensive.
        """
        super(LightningDictionary, self).__init__()

        self.n_samples = n_samples
        self.reg = reg
        self.reg_labels = reg_labels
        self.num_iter_barycenter = num_iter_barycenter
        self.num_iter_sinkhorn = num_iter_sinkhorn
        self.grad_labels = grad_labels
        self.optimizer_name = optimizer_name
        self.balanced_sampling = balanced_sampling
        self.sampling_with_replacement = sampling_with_replacement
        self.pseudo_label = pseudo_label
        self.n_components = n_components
        self.barycenter_tol = barycenter_tol
        self.barycenter_beta = barycenter_beta
        self.barycenter_verbose = barycenter_verbose
        self.barycenter_label_metric = barycenter_label_metric
        self.barycenter_initialization = barycenter_initialization
        self.barycenter_covariance_type = barycenter_covariance_type
        self.tensor_dtype = torch.float64 if dtype.lower() == 'double' else torch.float32
        self.log_gradients = log_gradients
        self.track_pseudo_label_acc = False
        self.track_atoms = track_atoms
        self.batch_size = batch_size

        self.training_step_outputs = []

        self.save_hyperparameters("n_samples", "reg", "num_iter_barycenter", "num_iter_sinkhorn")

        # Initialize atom features
        if XP is None:
            if n_dim is None:
                raise ValueError("If 'XP' is not given, you should specify 'n_dim'.")
            XP_data = [torch.randn(n_samples, n_dim, requires_grad=True).to(self.tensor_dtype) for _ in range(n_components)]
            self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp, requires_grad=True) for xp in XP_data])
            self.n_dim = n_dim
        else:
            self.XP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=xp.to(self.tensor_dtype), requires_grad=True) for xp in XP])
            self.n_dim = XP[0].shape[1]
        
        # Initialize atom labels
        if YP is None:
            if n_classes is None:
                raise ValueError("If 'YP' is not given, you should specify 'n_classes'")
            self.n_classes = n_classes
            samples_per_class = n_samples // n_classes
            if self.n_samples % n_classes != 0:
                self.n_samples = n_classes * samples_per_class
                print("[WARNING] n_samples ({}) is not divisible by n_classes ({}). Using n_samples = {} instead".format(n_samples,
                                                                                                                         n_classes,
                                                                                                                         self.n_samples))
            YP_data = []
            for _ in range(n_components):
                ypk = torch.cat([torch.tensor([c] * samples_per_class) for c in range(self.n_classes)]).long()
                YPk = torch.nn.functional.one_hot(ypk, num_classes=self.n_classes)
                YP_data.append(YPk)
            self.YP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=yp.to(self.tensor_dtype), requires_grad=grad_labels) for yp in YP_data])
        else:
            self.YP = torch.nn.ParameterList([torch.nn.parameter.Parameter(data=yp.to(self.tensor_dtype), requires_grad=grad_labels) for yp in YP])
            self.n_classes = YP[0].shape[1]

        # Initialize weights
        if A is None:
            if n_distributions is None:
                raise ValueError("If 'A' is not given you should specify 'n_distributions'")
            if weight_initialization == 'random':
                a_data = torch.rand(n_distributions, n_components, requires_grad=True).to(self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad(): a_data = ot.utils.proj_simplex(a_data.T).T
            else:
                a_data = torch.ones(n_distributions, n_components, requires_grad=True).to(self.tensor_dtype)
                if proj_grad:
                    with torch.no_grad(): a_data = ot.utils.proj_simplex(a_data.T).T
        else:
            a_data = A
        self.A = torch.nn.parameter.Parameter(data=a_data, requires_grad=True)

        # Defines loss
        self.loss_fn = JointWassersteinLoss() if loss_fn is None else loss_fn

        # Learning rate
        self.learning_rate_features = learning_rate_features
        # NOTE: if learning_rate_labels is not given, use learning_rate_features.
        self.learning_rate_labels = learning_rate_labels if learning_rate_labels is not None else learning_rate_features
        # NOTE: if learning_rate_weights is not given, use learning_rate_features.
        self.learning_rate_weights = learning_rate_weights if learning_rate_weights is not None else learning_rate_features

        # Whether or not we project on the simplex
        self.proj_grad = proj_grad

        # Variable Update Tracker
        self.var_tracker = [torch.zeros(n_samples) for _ in range(self.n_components)]

        # Sets history
        self.history = {
            'loss': [],
            'label_entropy': [],
            'weights': [],
            'atoms': {'Atom {}'.format(k): {'Features': [], 'Labels': []} for k in range(self.n_components)}
        }

        # Creates batch generator
        self.batch_generator = self.generate_batch_indices_without_replacement(batch_size=batch_size * n_classes)

        # Set domain names
        if domain_names is None:
            if n_distributions is None:
                raise ValueError("If 'domain_names' is not given, 'n_distributions' must be provided.")
            self.domain_names = ["Domain {}".format(ℓ) for ℓ in range(n_distributions)]
        else:
            self.domain_names = domain_names

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, *, on_tpu=None, using_native_amp=None, using_lbfgs=None):
    # implementation of optimizer step
        
    # implementation of optimizer step
        r"""Updates dictionary variables using gradients."""
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        #self.lr_scheduler.step()
        if self.proj_grad:
            with torch.no_grad():
                self.A.data = ot.utils.proj_simplex(self.A.data.T).T

    def get_weights(self):
        r"""Returns the barycentric coordinates of distributions in $\mathcal{Q}$."""
        if self.proj_grad:
            W = self.A
        else:
            W = self.A.softmax(dim=-1)

        return W.detach()

    def sample_from_atoms(self, n=None, detach=False):
        """Samples (with replacement) $n$ samples from atoms support.
        
        Args:
            n: Integer.
                Number of samples (with replacement) acquired from the atoms support.
                If $n$ is None, gets all samples from the atoms supports.
            detach: boolean.
                If True, detaches tensors so that gradients are not calculated.
        """
        batch_features, batch_labels = [], []

        # Determining the number of samples
        if n is not None:
            samples_per_class = n // self.n_classes
        else:
            samples_per_class = None

        # Sampling
        for tracker, XPk, YPk in zip(self.var_tracker, self.XP, self.YP):
            # If balanced sampling, needs to select sampler_per_class from each class
            if self.balanced_sampling:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                sampled_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    # Gets indices from current class
                    ind = torch.where(yPk == yu)[0]
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
                sampled_indices = torch.cat(sampled_indices, dim=0).cpu().numpy()
            else:
                # In this case, we randomly select samples
                sampled_indices = np.random.choice(np.arange(self.n_samples), size=n)
            
            # Adds counter of sampling
            #print(sampled_indices)
            tracker[sampled_indices] += 1

            # Creates batch arrays
            features_k, labels_k = XPk[sampled_indices], YPk[sampled_indices]

            if self.grad_labels:
                labels_k = labels_k.softmax(dim=-1)

            if detach:
                features_k, labels_k = features_k.detach(), labels_k.detach()

            batch_features.append(features_k)
            batch_labels.append(labels_k)

        return batch_features, batch_labels


    def generate_batch_indices_without_replacement(self, batch_size=None):
        r"""Divides the atom indices into mini-batches.

        Args:
            batch_size: Integer.
                Number of samples in each batch.
        """
        n_batches = self.n_samples // batch_size
        n_classes_per_batch = batch_size // self.n_classes

        for i in range(n_batches + 1):
            batch_indices = []
            for YPk in self.YP:
                # Gets categorical labels
                yPk = YPk.detach().argmax(dim=1)
                # Initializes list of sampled indices
                atom_batch_indices = []
                # Loops over each class
                for yu in yPk.unique():
                    indices = np.where(yPk == yu)[0]
                    atom_batch_indices.append(
                        indices[n_classes_per_batch * i: n_classes_per_batch * (i + 1)]
                    )
                atom_batch_indices = np.concatenate(atom_batch_indices)
                np.random.shuffle(atom_batch_indices)
                batch_indices.append(atom_batch_indices)
            yield batch_indices


    def custom_histogram_adder(self):
        r"""Adds variable histograms to Tensorboard. __WARNING:__ this function generates"""
        for name, params in self.named_parameters():
            if self.logger is not None:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        r"""Configures optimizers for Pytorch Lightning. Adds ```XP``` and ```A``` as optimization
        variables. If ```grad_labels``` is True, then ```YP``` is added as a variable as well."""
        if self.grad_labels:
            return torch.optim.Adam([
                {'params': self.XP, 'lr': self.learning_rate_features},
                {'params': self.YP, 'lr': self.learning_rate_labels},
                {'params': self.A, 'lr': self.learning_rate_weights}
            ])
        else:
            return torch.optim.Adam([
                {'params': self.XP, 'lr': self.learning_rate_features},
                {'params': self.A, 'lr': self.learning_rate_weights}
            ])


    def training_step(self, batch, batch_idx):
        r"""Runs a single optimization step. This function is used internally by Pytorch Lightning.
        The training_step is implemented in ```__training_step_with_replacement``` and ```__training_step_without_replacement```,
        for sampling with and without replacement respectively.

        Args:
            batch: list of $N$ pairs of Tensors of shape $(n_{b}, d)$ and $(n_{b}, c)$.
                Contains a list of pairs of tensors corresponding to a minibatch $(\mathbf{X}^{(Q_{\ell})}, \mathbf{Y}^{(Q_{\ell})})$
                from each dataset $\hat{Q}_{\ell} \in \mathcal{Q}$.
            batch_index: Integer.
                Not used.
        """
        if self.sampling_with_replacement:
            output = self.__training_step_with_replacement(batch, batch_idx)
        else:
            output = self.__training_step_without_replacement(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def __training_step_without_replacement(self, batch, batch_idx):
        # Line 5: Batch Treatment. We need to figure out what was given by the dataloader.
        if len(batch) == 2:
            # If the batch contains 3 elements, then no label information is given.
            source_features, source_labels = batch[0], batch[1]
            gt_target_labels = None
            target_labels = None
        elif len(batch) == 4:
            # If the batch contains 4 elements, the 4th element is the ground-truth labels
            # of the target domain. NOTE, we cannot use this during training, since we are
            # performing Unsupervised Domain Adaptation.
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            gt_target_labels = batch[3]
            target_labels = None
        elif len(batch) == 5:
            # If the batch contains 5 elements, the 5th element is the pseudo-labels of the
            # target domain, obtained in an unsupervised way. NOTE, we CAN use this information
            # during training, since pseudo-labels are obtained in an unsupervised way. Even if
            # these were provided, we only use if pseudo_label is set to True in the class constructor.
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            gt_target_labels = batch[3]
            target_labels = batch[4]
        else:
            # In any other case, we raise an exception.
            raise ValueError("Expected batch to have 3, 4 or 5 elements, but got {}".format(len(batch)))

        XQ = [xq for xq in source_features]
        YQ = [yq for yq in source_labels]

        batch_size = 128

        L = 0.0
        per_dataset_loss = {}
        for ℓ, (XQℓ, YQℓ, aℓ) in enumerate(zip(XQ, YQ, self.A)):
            # Line 9: Samples XPk and YPk from atoms for optimization.
            try:
                batch_indices = next(self.batch_generator)
                if len(batch_indices[0]) == 0:
                    self.batch_generator = self.generate_batch_indices_without_replacement(batch_size=self.n_classes * self.batch_size)
                    batch_indices = next(self.batch_generator)
            except StopIteration:
                self.batch_generator = self.generate_batch_indices_without_replacement(batch_size=self.n_classes * self.batch_size)
                batch_indices = next(self.batch_generator)

            XP = [XPk[ind_k] for XPk, ind_k in zip(self.XP, batch_indices)]
            YP = [YPk[ind_k] for YPk, ind_k in zip(self.YP, batch_indices)]

            # NOTE: If we do not project gradients in optimizer_step, we need
            #       an additional step for changing variables.
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)

            # Line 13: Calculates the Wasserstein Barycenter
            XBℓ, YBℓ = wasserstein_barycenter(XP=XP, YP=YP,
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

            # Calculates current loss L(Qℓ, Bℓ).
            # NOTE: If ℓ corresponds to the target domain, YQℓ is None, and we
            #       treat this internally in the loss_fn. This corresponds to the
            #       condition in the definition of L
            loss_ℓ = self.loss_fn(XQℓ, YQℓ, XBℓ, YBℓ, index=ℓ)

            # Line 15: Accumulates loss (summation for defining L = sum(L(Qℓ, Bℓ)))
            L += loss_ℓ # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ

        # Evaluates the entropy of labels
        ΩY = 0.0
        for YPk in self.YP:
            probs = YPk.softmax(dim=1)
            ΩY += -((probs * torch.log(probs)).mean(dim=1)).mean()

        # If reg_labels > 0.0 and optimization is done w.r.t. labels,
        # we penalize labels with high entropy.
        if self.grad_labels and self.reg_labels > 0.0:
            L += self.reg_labels * ΩY


        # For debugging: we calculate the c-Wasserstein distance with ground-truth labels.
        #wc_target = self.loss_fn(target_features, gt_target_labels, XBℓ, YBℓ, index=-1).item()

        return {'loss': L, 'label_entropy': ΩY,  **per_dataset_loss}

    def __training_step_with_replacement(self, batch, batch_idx):
        # Line 5: Batch Treatment. We need to figure out what was given by the dataloader.
        if len(batch) == 2:
            # If the batch contains 3 elements, then no label information is given.
            source_features, source_labels = batch[0], batch[1]
            gt_target_labels = None
            target_labels = None
        elif len(batch) == 4:
            # If the batch contains 4 elements, the 4th element is the ground-truth labels
            # of the target domain. NOTE, we cannot use this during training, since we are
            # performing Unsupervised Domain Adaptation.
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            target_labels = batch[3]
            #target_labels = None
        elif len(batch) == 5:
            # If the batch contains 5 elements, the 5th element is the pseudo-labels of the
            # target domain, obtained in an unsupervised way. NOTE, we CAN use this information
            # during training, since pseudo-labels are obtained in an unsupervised way. Even if
            # these were provided, we only use if pseudo_label is set to True in the class constructor.
            source_features, source_labels, target_features = batch[0], batch[1], batch[2]
            gt_target_labels = batch[3]
            target_labels = batch[4]
        else:
            # In any other case, we raise an exception.
            raise ValueError("Expected batch to have 3, 4 or 5 elements, but got {}".format(len(batch)))

        XQ = [xq for xq in source_features]
        YQ = [yq for yq in source_labels]

        batch_size = 128

        L = 0.0
        per_dataset_loss = {}
        for ℓ, (XQℓ, YQℓ, aℓ) in enumerate(zip(XQ, YQ, self.A)):
            # Line 9: Samples from atoms
            XP, YP = self.sample_from_atoms(n=batch_size)

            # NOTE: If we do not project gradients in optimizer_step, we need
            #       an additional step for changing variables.
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)

            # Line 13: Calculates the Wasserstein Barycenter
            XBℓ, YBℓ = wasserstein_barycenter(XP=XP, YP=YP,
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

            # Calculates current loss L(Qℓ, Bℓ).
            # NOTE: If ℓ corresponds to the target domain, YQℓ is None, and we
            #       treat this internally in the loss_fn. This corresponds to the
            #       condition in the definition of L
            loss_ℓ = self.loss_fn(XQℓ, YQℓ, XBℓ, YBℓ, index=ℓ)

            # Line 15: Accumulates loss (summation for defining L = sum(L(Qℓ, Bℓ)))
            L += loss_ℓ # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ

        # Evaluates the entropy of labels
        ΩY = 0.0
        for YPk in self.YP:
            probs = YPk.softmax(dim=1)
            ΩY += -((probs * torch.log(probs)).mean(dim=1)).mean()

        # If reg_labels > 0.0 and optimization is done w.r.t. labels,
        # we penalize labels with high entropy.
        if self.grad_labels and self.reg_labels > 0.0:
            L += self.reg_labels * ΩY

        # For debugging: we calculate the c-Wasserstein distance with ground-truth labels.
        #wc_target = self.loss_fn(target_features, gt_target_labels, XBℓ, YBℓ, index=-1).item()

        return {'loss': L, 'label_entropy': ΩY, **per_dataset_loss}

    def on_train_epoch_end(self):
        r"""Logs information to Tensorboard, if the logger specified in the trainer object
        is not None."""
        avg_loss = torch.tensor([x['loss'] for x in self.training_step_outputs]).mean()
        avg_ΩY = torch.tensor([x['label_entropy'] for x in self.training_step_outputs]).mean()
         #avg_wc_target = torch.tensor([x['Wc_target'] for x in self.training_step_outputs]).mean()

        avg_loss_per_dataset = {
            self.domain_names[ℓ]: torch.tensor([x[self.domain_names[ℓ]] for x in self.training_step_outputs]).mean() for ℓ in range(self.A.shape[0])
        }

        self.training_step_outputs.clear()

        _A = self.A.clone()
        if self.proj_grad:
            _A = ot.utils.proj_simplex(_A.T).T
        else:
            _A = _A.softmax(dim=1)
        _A = _A.detach().cpu()

        # Adds loss to history
        self.history['loss'].append(avg_loss)
        self.history['label_entropy'].append(avg_ΩY)
        self.history['weights'].append({
            name: ot.utils.proj_simplex(weights.detach()) for name, weights in zip(self.domain_names, _A)
        })
        if self.track_atoms:
            with torch.no_grad():
                for k, (_XPk, _YPk) in enumerate(zip(self.XP, self.YP)):
                    self.history['atoms']['Atom {}'.format(k)]['Features'].append(_XPk.data.cpu().clone())
                    self.history['atoms']['Atom {}'.format(k)]['Labels'].append(_YPk.data.softmax(dim=-1).cpu().clone())

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar("DiL Loss/Train", avg_loss, self.current_epoch)
            #self.logger.experiment.add_scalar("Wc/Target/Train", avg_wc_target, self.current_epoch)
            for domain in avg_loss_per_dataset:
                self.logger.experiment.add_scalar("DiL Loss {}/Train".format(domain), avg_loss_per_dataset[domain], self.current_epoch)
            
            for ℓ in range(_A.shape[0]):
                αℓ = _A[ℓ]
                a = np.arange(len(αℓ))
                self.logger.experiment.add_histogram("Weights/" + self.domain_names[ℓ],
                                                    np.random.choice(a, p=αℓ.numpy(), size=1000),
                                                    self.current_epoch)
            if self.log_gradients:
                # Logs histograms
                self.custom_histogram_adder()

    def reconstruct(self, α=None, n_samples_atoms=None, n_samples_barycenter=None):
        r"""Obtains a given reconstruction using the barycentric coordinates $\alpha$, i.e.,
        calculates $\mathcal{B}(\alpha;\mathcal{P})$.
        
        Args:
            α: Tensor of shape (K,).
                Must correspond to a barycentric coordinate vector, i.e., its components must be
                positive and it must sum to one.
            n_samples_atoms: Integer.
                Number of samples to be acquired from atom distributions.
            n_samples_barycenter: Integer.
                Number of samples generated in the support of the Barycenter distribution.
        """
        XP, YP = self.sample_from_atoms(n=n_samples_atoms)
        n_samples_barycenter = len(XP[0]) if n_samples_barycenter is None else n_samples_barycenter
        with torch.no_grad():
            if α is None:
                Q_rec = []
                for aℓ in self.A:
                    if self.proj_grad:
                        αℓ = aℓ
                    else:
                        αℓ = aℓ.softmax(dim=0)
                    XB, YB = wasserstein_barycenter(XP=XP, YP=YP,
                                                    n_samples=n_samples_barycenter,
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
                    Q_rec.append([XB, YB])
            else:
                XB, YB = wasserstein_barycenter(XP=XP, YP=YP,
                                                n_samples=n_samples_barycenter,
                                                ϵ=self.reg,
                                                α=α,
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
                Q_rec = [XB, YB]
        return Q_rec

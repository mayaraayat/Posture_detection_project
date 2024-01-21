import ot
import torch
import numpy as np
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score
from dictionary_learning.utils import unif
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.barycenters import wasserstein_barycenter


class FullDictionary(pl.LightningModule):
    """Implementation of Dataset Dictionary Learning (DaDiL) using Pytorch Lightning. This class has as parameters
    a set $\mathcal{P} = \set{\hat{P}_{k}}_{k=1}^{K}$ of atoms, and a set $\mathcal{A}=\set{\alpha_{\ell}}_{\ell=1}^{N}$ of weights.
    
    The atoms are labeled empirical distributions, that is,

    $$\hat{P}_{k}(\mathbf{x},\mathbf{y})=\dfrac{1}{n}\sum_{i=1}^{n}\delta((\mathbf{x},\mathbf{y})-(\mathbf{x}_{i}^{(P_{k})}, \mathbf{y}_{i}^{(P_{k})}))$$

    and the weights are $K-$dimensional vectors whose components are all positive and sum to one, i.e. $\alpha_{\ell} \in \Delta_{K}$.
    """
    def __init__(self, n_samples=1024, n_dim=None, n_classes=None, XP=None,
                 YP=None, A=None, n_components=2, weight_initialization='random',
                 barycenter_initialization='random', n_distributions=None, loss_fn=None,
                 learning_rate=1e-1, learning_rate_weights=None, reg=0.0, num_iter_barycenter=10, num_iter_sinkhorn=20,
                 domain_names=None, proj_grad=True, grad_labels=True, optimizer_name='adam',
                 balanced_sampling=True, pseudo_label=False, barycenter_verbose=False,
                 barycenter_tol=1e-9, dtype='double', log_gradients=False):
        super(FullDictionary, self).__init__()

        self.n_samples = n_samples
        self.reg = reg
        self.num_iter_barycenter = num_iter_barycenter
        self.num_iter_sinkhorn = num_iter_sinkhorn
        self.grad_labels = grad_labels
        self.optimizer_name = optimizer_name
        self.balanced_sampling = balanced_sampling
        self.pseudo_label = pseudo_label
        self.n_components = n_components
        self.barycenter_initialization = barycenter_initialization
        self.barycenter_verbose = barycenter_verbose
        self.barycenter_tol = barycenter_tol
        self.tensor_dtype = torch.float64 if dtype.lower() == 'double' else torch.float32
        self.log_gradients = log_gradients
        self.track_pseudo_label_acc = False

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
                """
                YP_data.append(
                    torch.nn.functional.one_hot(
                        torch.from_numpy(np.random.randint(low=0, high=n_classes, size=n_samples)).long(),
                        num_classes=n_classes
                    ).to(self.tensor_dtype)
                )
                """
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
                a_data = torch.randn(n_distributions, n_components, requires_grad=True).to(self.tensor_dtype)
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
        self.learning_rate = learning_rate
        # NOTE: if learning_rate_weights is not given, use learning_rate of atoms.
        self.learning_rate_weights = learning_rate_weights if learning_rate_weights is not None else learning_rate

        # Whether or not we project on the simplex
        self.proj_grad = proj_grad

        # Sets history
        self.history = {'loss': []}

        # Set domain names
        if domain_names is None:
            if n_distributions is None:
                raise ValueError("If 'domain_names' is not given, 'n_distributions' must be provided.")
            self.domain_names = ["Domain {}".format(ℓ) for ℓ in range(n_distributions)]
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
            for name, params in self.named_parameters():
                if name == 'A':
                    with torch.no_grad(): params.data = ot.utils.proj_simplex(params.T).T

    def get_weights(self):
        if self.proj_grad:
            W = self.A
        else:
            W = self.A.softmax(dim=-1)

        return W.detach()

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            if self.logger is not None:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        print(self.learning_rate, self.learning_rate_weights)
        if self.grad_labels:
            return torch.optim.Adam([
                {'params': self.XP, 'lr': self.learning_rate},
                {'params': self.YP, 'lr': self.learning_rate},
                {'params': self.A, 'lr': self.learning_rate_weights}
            ])
        else:
            return torch.optim.Adam([
                {'params': self.XP, 'lr': self.learning_rate},
                {'params': self.A, 'lr': self.learning_rate_weights}
            ])

    def training_step(self, batch, batch_idx):
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
            raise ValueError("Expected batch to have 3, 4 or 5 elements, but got {}".format(len(batch)))

        source_features, source_labels = batch[0], batch[1]
        target_features, gt_target_labels = batch[2], batch[3]

        XQ = [xq.squeeze() for xq in source_features] + [target_features.squeeze()]
        YQ = [yq.squeeze() for yq in source_labels] + [target_labels.squeeze()]

        L = 0.0
        per_dataset_loss = {}

        # Calculates the Wasserstein Barycenter
        _XP, _YP = [XPk for XPk in self.XP], [YPk.softmax(dim=1) for YPk in self.YP]

        for ℓ, (XQℓ, YQℓ, aℓ) in enumerate(zip(XQ, YQ, self.A)):
            # Change of variables
            if self.proj_grad:
                αℓ = aℓ
            else:
                αℓ = aℓ.softmax(dim=0)

            XBℓ, YBℓ = wasserstein_barycenter(XP=_XP, YP=_YP,
                                              n_samples=len(self.XP[0]),
                                              ϵ=self.reg,
                                              α=αℓ,
                                              num_iter_max=self.num_iter_barycenter,
                                              num_iter_sinkhorn=self.num_iter_sinkhorn,
                                              initialization=self.barycenter_initialization,
                                              propagate_labels=True,
                                              penalize_labels=True,
                                              verbose=self.barycenter_verbose,
                                              τ=self.barycenter_tol)

            # Calculates current loss
            loss_ℓ = self.loss_fn(XQℓ, YQℓ, XBℓ, YBℓ)

            # Accumulates loss
            L += loss_ℓ # / len(XQ)

            # Adds loss to per dataset loss
            per_dataset_loss[self.domain_names[ℓ]] = loss_ℓ
        wc_target = self.loss_fn(target_features.squeeze(), gt_target_labels.squeeze(), XBℓ, YBℓ).item()
        
        return {'loss': L, 'Wc_target': wc_target, **per_dataset_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        avg_wc_target = torch.tensor([x['Wc_target'] for x in outputs]).mean()
        avg_loss_per_dataset = {
            self.domain_names[ℓ]: torch.tensor([x[self.domain_names[ℓ]] for x in outputs]).mean() for ℓ in range(self.A.shape[0])
        }

        # Adds loss to history
        self.history['loss'].append(avg_loss)

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar("DiL Loss/Train", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar("Wc/Target/Train", avg_wc_target, self.current_epoch)
            for domain in avg_loss_per_dataset:
                self.logger.experiment.add_scalar("DiL Loss {}/Train".format(domain), avg_loss_per_dataset[domain], self.current_epoch)
            
            for ℓ in range(self.A.shape[0]):
                if self.proj_grad:
                    αℓ = self.A[ℓ].detach().cpu()
                else:
                    αℓ = self.A[ℓ].detach().softmax(dim=0).cpu()
                a = np.arange(len(αℓ))
                self.logger.experiment.add_histogram("Weights/" + self.domain_names[ℓ],
                                                    np.random.choice(a, p=αℓ.numpy(), size=1000),
                                                    self.current_epoch)
            if self.log_gradients:
                # Logs histograms
                self.custom_histogram_adder()

    def reconstruct(self, α=None):
        XP, YP = [], []

        for XPk, YPk in zip(self.XP, self.YP):
            XP.append(XPk.data)
            YP.append(YPk.data.softmax(dim=1))

        with torch.no_grad():
            if α is None:
                Q_rec = []
                for aℓ in self.A:
                    if self.proj_grad:
                        αℓ = aℓ
                    else:
                        αℓ = aℓ.softmax(dim=0)
                    XB, YB = wasserstein_barycenter(XP=XP, YP=YP, α=αℓ,
                                                    n_samples=len(XP[0]), ϵ=self.reg,
                                                    num_iter_max=self.num_iter_barycenter,
                                                    num_iter_sinkhorn=self.num_iter_sinkhorn,
                                                    initialization=self.barycenter_initialization,
                                                    propagate_labels=True,
                                                    penalize_labels=True,
                                                    verbose=self.barycenter_verbose,
                                                    τ=self.barycenter_tol)
                    Q_rec.append([XB, YB])
            else:
                XB, YB = wasserstein_barycenter(XP=XP, YP=YP, α=α,
                                                n_samples=len(XP[0]), ϵ=self.reg,
                                                num_iter_max=self.num_iter_barycenter,
                                                num_iter_sinkhorn=self.num_iter_barycenter,
                                                initialization=self.barycenter_initialization,
                                                propagate_labels=True,
                                                penalize_labels=True,
                                                verbose=self.barycenter_verbose,
                                                τ=self.barycenter_tol)
                Q_rec = [XB, YB]
        return Q_rec
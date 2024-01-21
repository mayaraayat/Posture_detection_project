<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/lightning_dictionary.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `lightning_dictionary`
This module implements Algorithm 2 of our main paper, using [Pytorch Lightning](https://www.pytorchlightning.ai/index.html). As such, it should be the preferred method for learning a dictionary. 

## References 

[Schmitz et al., 2018] Schmitz, M. A., Heitz, M., Bonneel, N., Ngole, F., Coeurjolly, D., Cuturi, M., ... & Starck, J. L. (2018). Wasserstein dictionary learning: Optimal transport-based unsupervised nonlinear dictionary learning. SIAM Journal on Imaging Sciences, 11(1), 643-678. 

[Turrisi et al., 2022] Turrisi, R., Flamary, R., Rakotomamonjy, A., & Pontil, M. (2022, August). Multi-source domain adaptation via weighted joint distributions optimal transport. In Uncertainty in Artificial Intelligence (pp. 1970-1980). PMLR. 

[Montesuma, Mboula and Souloumiac, 2023] Multi-Source Domain Adaptation through Dataset Dictionary Learning in Wasserstein Space, Submitted. 



---

<a href="../dictionary_learning/lightning_dictionary.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LightningUnsupervisedDictionary`
Implementation of Dataset Dictionary Learning (DaDiL) using Pytorch Lightning. This class has as parameters a set $\mathcal{P} = \set{\hat{P}_{k}}_{k=1}^{K}$ of atoms, and a set $\mathcal{A} = \set{ \alpha_{\ell}}_{\ell=1}^{N}$ of weights. 

The atoms are unlabeled empirical distributions, that is, 

$$\hat{P}_{k}(\mathbf{x})=\dfrac{1}{n}\sum_{i=1}^{n}\delta(\mathbf{x} - \mathbf{x}_{i}^{(P_{k})})$$ 

and the weights are $K-$dimensional vectors whose components are all positive and sum to one, i.e. $\alpha_{\ell} \in \Delta_{K}$. 

__NOTE.__ Since this class inherits from ```pl.LightningModule```, some methods on the docs come from the parent class. You should ignore methods that do not have text description. 

<a href="../dictionary_learning/lightning_dictionary.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.__init__`

```python
__init__(
    XP=None,
    A=None,
    n_samples=1024,
    n_dim=None,
    n_components=2,
    weight_initialization='random',
    n_distributions=None,
    loss_fn=None,
    learning_rate_features=0.1,
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
    barycenter_tol=1e-09,
    dtype='float',
    batch_size=128,
    log_gradients=False,
    track_atoms=False
)
```

Initializes a LightningUnsupervisedDictionary object 



**Args:**
 
 - <b>`XP`</b>:  List of tensors of shape $(n, d)$.  Manually initializes the atoms support. 
 - <b>`A`</b>:  Tensor of shape $(N, K)$.  Manually initializes the barycentric coefficient matrix $\mathcal{A}$. 
 - <b>`n_samples`</b>:  Integer.  Number of samples $n$ in the support of atom distributions. 
 - <b>`n_dim`</b>:  Integer.  Number of dimensions $d$ of the support of atom distributions.  It should be specified if ```XP``` is not given. 
 - <b>`n_components`</b>:  Integer.  Number of atoms in the dictionary. 
 - <b>`weight_initialization`</b>:  String.  Either 'random' or 'uniform' corresponding to how $\alpha_{\ell,k}$ are initialized. 
 - <b>`n_distributions`</b>:  Integer.  Number of distributions in $\mathcal{Q}$. Should be specified if $\mathcal{A}$  is not given. 
 - <b>`loss_fn`</b>:  Function.  Function implementing the loss that is minimized throughout DiL.  If not specified, uses the default 2-Wasserstein distance. 
 - <b>`learning_rate_features`</b>:  Float.  Learning rate $\eta$ applied to atom features $\mathbf{X}^{(P_{k})}$. 
 - <b>`learning_rate_weights`</b>:  Float.  If given, considers an independent learning rate $\eta_{\alpha}$ for the barycentric  coordinates. If not given, uses $\eta$ by default. 
 - <b>`reg`</b>:  Float.  Amount of entropic regularization $\epsilon$ used when solving OT. 
 - <b>`num_iter_barycenter`</b>:  Integer.  Number of steps when solving a Wasserstein barycenter  (Algorithm 1 in [Montesuma, Mboula and Souloumiac, 2023]). 
 - <b>`num_iter_sinkhorn`</b>:  Integer.  Number of steps when solving regularized OT. Only used if $\epsilon > 0$. 
 - <b>`proj_grad`</b>:  Boolean.  If True, projects the barycentric coefficients in the simplex as [Turrisi et al., 2022].  Otherwise, performs a change of variables through a softmax, as in [Schmitz et al., 2018]. 
 - <b>`optimizer_name`</b>:  String.  Choice of optimizer. Either 'adam' or 'sgd'. 
 - <b>`sampling_with_replacement`</b>:  Bool.  If True, samples from atoms with replacement. 
 - <b>`barycenter_initialization`</b>:  String.  Parameter for initializing the barycenter support. Either 'random', 'class', 'samples', 'zeros'. 
 - <b>`barycenter_covariance_type`</b>:  String. If barycenter initialization is 'Class', specifies how to calculate the covariance (i.e., 'full', 'diag', 'none'). WARNING: 'full' may give numerical errors. 
 - <b>`barycenter_verbose`</b>:  Boolean.  If True, prints info about barycenter calculation. 
 - <b>`barycenter_tol`</b>:  Float.  Stopping criteria for barycenter calculation. 
 - <b>`dtype`</b>:  string.  Either 'float' or 'double'. Should agree with dtype of XP and A. 
 - <b>`batch_size`</b>:  Integer.  Batch size used during learning. Only used if sampling WITHOUT replacement. 
 - <b>`log_gradients`</b>:  Boolean. If True, logs gradients of variables in Tensorboard. __WARNING:__ memory intensive. 
 - <b>`track_atoms`</b>:  Boolean. If True, saves atoms at each iteration in dictionary.history. __WARNING:__ memory intensive. 


---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.device





---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.dtype





---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.fabric





---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> LightningUnsupervisedDictionary.trainer







---

<a href="../dictionary_learning/lightning_dictionary.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.configure_optimizers`

```python
configure_optimizers()
```

Configures optimizers for Pytorch Lightning. Adds ```XP``` and ```A``` as optimization variables. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.custom_histogram_adder`

```python
custom_histogram_adder()
```

Adds variable histograms to Tensorboard. __WARNING:__ this function generates heavy Tensorboard logs. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.generate_batch_indices_without_replacement`

```python
generate_batch_indices_without_replacement(batch_size=None)
```

Divides the atom indices into mini-batches. 



**Args:**
 
 - <b>`batch_size`</b>:  Integer.  Number of samples in each batch. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.get_weights`

```python
get_weights()
```

Returns the barycentric coordinates of distributions in $\mathcal{Q}$. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.on_train_epoch_end`

```python
on_train_epoch_end()
```

Logs information to Tensorboard, if the logger specified in the trainer object is not None. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.optimizer_step`

```python
optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
```

Updates dictionary variables using gradients. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L457"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.reconstruct`

```python
reconstruct(α=None, n_samples_atoms=None, n_samples_barycenter=None)
```

Obtains a given reconstruction using the barycentric coordinates $\alpha$, i.e., calculates $\mathcal{B}(\alpha;\mathcal{P})$. 



**Args:**
 
 - <b>`α`</b>:  Tensor of shape (K,).  Must correspond to a barycentric coordinate vector, i.e., its components must be  positive and it must sum to one. 
 - <b>`n_samples_atoms`</b>:  Integer.  Number of samples to be acquired from atom distributions. 
 - <b>`n_samples_barycenter`</b>:  Integer.  Number of samples generated in the support of the Barycenter distribution. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.sample_from_atoms`

```python
sample_from_atoms(n=None, detach=False)
```

Samples (with replacement) $n$ samples from atoms support. 



**Args:**
 
 - <b>`n`</b>:  Integer.  Number of samples (with replacement) acquired from the atoms support.  If $n$ is None, gets all samples from the atoms supports. 
 - <b>`detach`</b>:  boolean.  If True, detaches tensors so that gradients are not calculated. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningUnsupervisedDictionary.training_step`

```python
training_step(batch, batch_idx)
```

Runs a single optimization step. This function is used internally by Pytorch Lightning. The training_step is implemented in ```__training_step_with_replacement``` and ```__training_step_without_replacement```, for sampling with and without replacement respectively. 



**Args:**
 
 - <b>`batch`</b>:  list of $N$ Tensors of shape $(n_{b}, d)$.  Contains a list of tensors corresponding to a minibatch $\mathbf{X}^{(Q_{\ell})}$ from  each dataset $\hat{Q}_{\ell} \in \mathcal{Q}$. 
 - <b>`batch_index`</b>:  Integer.  Not used. 


---

<a href="../dictionary_learning/lightning_dictionary.py#L510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LightningDictionary`
Implementation of Dataset Dictionary Learning (DaDiL) using Pytorch Lightning. This class has as parameters a set $\mathcal{P} = \set{\hat{P}_{k}}_{k=1}^{K}$ of atoms, and a set $\mathcal{A}=\set{\alpha_{\ell}}_{\ell=1}^{N}$ of weights. 

The atoms are labeled empirical distributions, that is, 

$$\hat{P}_{k}(\mathbf{x},\mathbf{y})=\dfrac{1}{n}\sum_{i=1}^{n}\delta((\mathbf{x},\mathbf{y})-(\mathbf{x}_{i}^{(P_{k})}, \mathbf{y}_{i}^{(P_{k})}))$$ 

and the weights are $K-$dimensional vectors whose components are all positive and sum to one, i.e. $\alpha_{\ell} \in \Delta_{K}$. 

<a href="../dictionary_learning/lightning_dictionary.py#L520"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.__init__`

```python
__init__(
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
    learning_rate_features=0.1,
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
    barycenter_tol=1e-09,
    barycenter_beta=None,
    barycenter_verbose=False,
    barycenter_label_metric='l2',
    barycenter_initialization='random',
    barycenter_covariance_type='diag',
    dtype='float',
    batch_size=5,
    log_gradients=False,
    track_atoms=False
)
```

Initializes a LightningDictionary Object. This class should be used for Domain Adaptation. 



**Args:**
 
 - <b>`XP`</b>:  list of tensors.  Manual initialization of atom features. List of tensors of shape $(n, d)$ containing  the support (features) of each atom. 
 - <b>`YP`</b>:  list of tensors.  Manual initialization of atom labels. List of tensors of shape $(n, c)$ containing  the support (labels) of each atom. 
 - <b>`A`</b>:  Tensor.  Manual initialization of barycentric coordinates. Tensor of shape $(N, K)$, where each element  $(ℓ,k)$ indicates the coordinate of $\hat{Q}_{\ell}$ w.r.t. $\hat{P}_{k}$. 
 - <b>`n_samples`</b>:  Integer.  Number of samples $n$ in the support of atom distributions. 
 - <b>`n_dim`</b>:  Integer.  Number of dimensions $d$ of the support of atom distributions.  It should be specified if ```XP``` is not given. 
 - <b>`n_classes`</b>:  Integer.  Number of classes $c$ in the support of atom distributions.  It should be specified if ```YP``` is not given. 
 - <b>`n_components`</b>:  Integer.  Number of atoms $K$ in the dictionary. 
 - <b>`weight_initialization`</b>:  String.  Either 'random' or 'uniform' corresponding to how $\alpha_{\ell,k}$ are initialized. 
 - <b>`n_distributions`</b>:  Integer.  Number of distributions in $\mathcal{Q}$. Should be specified if $\mathcal{A}$  is not given. 
 - <b>`loss_fn`</b>:  Function.  Function implementing the loss that is minimized throughout DiL.  If not specified, uses the c-Wasserstein distance. 
 - <b>`learning_rate_features`</b>:  Float.  Learning rate $\eta_{x}$ applied to atom features $\mathbf{X}^{(P_{k})}$. 
 - <b>`learning_rate_labels`</b>:  Float.  Learning rate $\eta_{y}$ applied to atom labels $\mathbf{Y}^{(P_{k})}$. If not given, uses $\eta_{x}$ by default. 
 - <b>`learning_rate_weights`</b>:  Float.  Learning rate $\eta_{\alpha}$ applied to barycentric coordinates $\alpha_{\ell,k}$. If not given, uses $\eta_{x}$ by default. 
 - <b>`reg`</b>:  Float.  Amount of entropic regularization $\epsilon$ used when solving OT. 
 - <b>`reg_labels`</b>:  Float.  Penalizes labels with high entropy. Note, following our experiments, this regularization  term is unecessary, as DiL naturally penalizes labels with high entropy. 
 - <b>`num_iter_barycenter`</b>:  Integer.  Number of steps when solving a Wasserstein barycenter  (Algorithm 1 in [Montesuma, Mboula and Souloumiac, 2023]). 
 - <b>`num_iter_sinkhorn`</b>:  Integer.  Number of steps when solving regularized OT. Only used if $\epsilon > 0$. 
 - <b>`domain_names`</b>:  List of Strings.  List of names for each domain, for better logging. 
 - <b>`proj_grad`</b>:  Boolean.  If True, projects the barycentric coefficients in the simplex as [Turrisi et al., 2022].  Otherwise, performs a change of variables through a softmax, as in [Schmitz et al., 2018]. 
 - <b>`grad_labels`</b>:  Boolean.  If True, calculates gradients w.r.t. labels. Setting it to False is equivalent to $\eta_{y} = 0$ 
 - <b>`optimizer_name`</b>:  String.  Choice of optimizer. Either 'adam' or 'sgd'. 
 - <b>`balanced_sampling`</b>:  Boolean.  If True, samples balanced mini-batches from atoms. This is key to the success of DA. 
 - <b>`sampling_with_replacement`</b>:  Boolean.  If True, samples from atoms with replacement. 
 - <b>`pseudo_label`</b>:  Boolean.  If True, uses pseudo-labels (passed through the dataloader) in the calculation of the loss  in the target domain. 
 - <b>`barycenter_tol`</b>:  Float.  Stopping criteria for barycenter calculation. 
 - <b>`barycenter_beta`</b>:  Float.  Label importance in the ground-cost 
 - <b>`barycenter_verbose`</b>:  Boolean.  If True, prints info about barycenter calculation. 
 - <b>`barycenter_label_metric`</b>:  String.  Either 'l2' or 'delta'. It specifies the metric for which label distances are calculated. 
 - <b>`barycenter_initialization`</b>:  String.  Parameter for initializing the barycenter support.  Either 'random', 'class', 'samples', 'zeros'. 
 - <b>`barycenter_covariance_type`</b>:  String.  If barycenter initialization is 'Class', specifies how to calculate the covariance  (i.e., 'full', 'diag', 'none'). __WARNING__ 'full' may give numerical errors. 
 - <b>`dtype`</b>:  String.  Either 'float' or 'double'. Should agree with dtype of XP and A. 
 - <b>`batch_size`</b>:  Integer.  Number of samples per class in batches. Effective batch size corresponds to this parameter  times number of classes. 
 - <b>`log_gradients`</b>:  Boolean.  Boolean. If True, logs gradients of variables in Tensorboard.  __WARNING__ memory intensive. 
 - <b>`track_atoms`</b>:  Boolean.  Boolean. If True, saves atoms at each iteration in dictionary.history. __WARNING__ memory intensive. 


---

#### <kbd>property</kbd> LightningDictionary.automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> LightningDictionary.current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> LightningDictionary.device





---

#### <kbd>property</kbd> LightningDictionary.dtype





---

#### <kbd>property</kbd> LightningDictionary.example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> LightningDictionary.fabric





---

#### <kbd>property</kbd> LightningDictionary.global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> LightningDictionary.global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> LightningDictionary.hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> LightningDictionary.hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> LightningDictionary.local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> LightningDictionary.logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> LightningDictionary.loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> LightningDictionary.on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> LightningDictionary.trainer







---

<a href="../dictionary_learning/lightning_dictionary.py#L869"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.configure_optimizers`

```python
configure_optimizers()
```

Configures optimizers for Pytorch Lightning. Adds ```XP``` and ```A``` as optimization variables. If ```grad_labels``` is True, then ```YP``` is added as a variable as well. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L863"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.custom_histogram_adder`

```python
custom_histogram_adder()
```

Adds variable histograms to Tensorboard. __WARNING:__ this function generates 

---

<a href="../dictionary_learning/lightning_dictionary.py#L834"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.generate_batch_indices_without_replacement`

```python
generate_batch_indices_without_replacement(batch_size=None)
```

Divides the atom indices into mini-batches. 



**Args:**
 
 - <b>`batch_size`</b>:  Integer.  Number of samples in each batch. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L762"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.get_weights`

```python
get_weights()
```

Returns the barycentric coordinates of distributions in $\mathcal{Q}$. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L1092"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.on_train_epoch_end`

```python
on_train_epoch_end()
```

Logs information to Tensorboard, if the logger specified in the trainer object is not None. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.optimizer_step`

```python
optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
```

Updates dictionary variables using gradients. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L1140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.reconstruct`

```python
reconstruct(α=None, n_samples_atoms=None, n_samples_barycenter=None)
```

Obtains a given reconstruction using the barycentric coordinates $\alpha$, i.e., calculates $\mathcal{B}(\alpha;\mathcal{P})$. 



**Args:**
 
 - <b>`α`</b>:  Tensor of shape (K,).  Must correspond to a barycentric coordinate vector, i.e., its components must be  positive and it must sum to one. 
 - <b>`n_samples_atoms`</b>:  Integer.  Number of samples to be acquired from atom distributions. 
 - <b>`n_samples_barycenter`</b>:  Integer.  Number of samples generated in the support of the Barycenter distribution. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L771"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.sample_from_atoms`

```python
sample_from_atoms(n=None, detach=False)
```

Samples (with replacement) $n$ samples from atoms support. 



**Args:**
 
 - <b>`n`</b>:  Integer.  Number of samples (with replacement) acquired from the atoms support.  If $n$ is None, gets all samples from the atoms supports. 
 - <b>`detach`</b>:  boolean.  If True, detaches tensors so that gradients are not calculated. 

---

<a href="../dictionary_learning/lightning_dictionary.py#L885"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LightningDictionary.training_step`

```python
training_step(batch, batch_idx)
```

Runs a single optimization step. This function is used internally by Pytorch Lightning. The training_step is implemented in ```__training_step_with_replacement``` and ```__training_step_without_replacement```, for sampling with and without replacement respectively. 



**Args:**
 
 - <b>`batch`</b>:  list of $N$ pairs of Tensors of shape $(n_{b}, d)$ and $(n_{b}, c)$.  Contains a list of pairs of tensors corresponding to a minibatch $(\mathbf{X}^{(Q_{\ell})}, \mathbf{Y}^{(Q_{\ell})})$  from each dataset $\hat{Q}_{\ell} \in \mathcal{Q}$. 
 - <b>`batch_index`</b>:  Integer.  Not used. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

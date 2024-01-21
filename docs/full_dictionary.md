<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/full_dictionary.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `full_dictionary`






---

<a href="../dictionary_learning/full_dictionary.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FullDictionary`
Implementation of Dataset Dictionary Learning (DaDiL) using Pytorch Lightning. This class has as parameters a set $\mathcal{P} = \set{\hat{P}_{k}}_{k=1}^{K}$ of atoms, and a set $\mathcal{A}=\set{lpha_{\ell}}_{\ell=1}^{N}$ of weights. 

The atoms are labeled empirical distributions, that is, 

$$\hat{P}_{k}(\mathbf{x},\mathbf{y})=\dfrac{1}{n}\sum_{i=1}^{n}\delta((\mathbf{x},\mathbf{y})-(\mathbf{x}_{i}^{(P_{k})}, \mathbf{y}_{i}^{(P_{k})}))$$ 

and the weights are $K-$dimensional vectors whose components are all positive and sum to one, i.e. $lpha_{\ell} \in \Delta_{K}$. 

<a href="../dictionary_learning/full_dictionary.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.__init__`

```python
__init__(
    n_samples=1024,
    n_dim=None,
    n_classes=None,
    XP=None,
    YP=None,
    A=None,
    n_components=2,
    weight_initialization='random',
    barycenter_initialization='random',
    n_distributions=None,
    loss_fn=None,
    learning_rate=0.1,
    learning_rate_weights=None,
    reg=0.0,
    num_iter_barycenter=10,
    num_iter_sinkhorn=20,
    domain_names=None,
    proj_grad=True,
    grad_labels=True,
    optimizer_name='adam',
    balanced_sampling=True,
    pseudo_label=False,
    barycenter_verbose=False,
    barycenter_tol=1e-09,
    dtype='double',
    log_gradients=False
)
```






---

#### <kbd>property</kbd> FullDictionary.automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> FullDictionary.current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> FullDictionary.device





---

#### <kbd>property</kbd> FullDictionary.dtype





---

#### <kbd>property</kbd> FullDictionary.example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> FullDictionary.fabric





---

#### <kbd>property</kbd> FullDictionary.global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> FullDictionary.global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> FullDictionary.hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> FullDictionary.hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> FullDictionary.local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> FullDictionary.logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> FullDictionary.loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> FullDictionary.on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> FullDictionary.trainer







---

<a href="../dictionary_learning/full_dictionary.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.configure_optimizers`

```python
configure_optimizers()
```





---

<a href="../dictionary_learning/full_dictionary.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.custom_histogram_adder`

```python
custom_histogram_adder()
```





---

<a href="../dictionary_learning/full_dictionary.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.get_weights`

```python
get_weights()
```





---

<a href="../dictionary_learning/full_dictionary.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.optimizer_step`

```python
optimizer_step(
    epoch,
    batch_idx,
    optimizer,
    optimizer_idx,
    optimizer_closure,
    on_tpu=False,
    using_lbfgs=False
)
```





---

<a href="../dictionary_learning/full_dictionary.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.reconstruct`

```python
reconstruct(α=None)
```





---

<a href="../dictionary_learning/full_dictionary.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.training_epoch_end`

```python
training_epoch_end(outputs)
```





---

<a href="../dictionary_learning/full_dictionary.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullDictionary.training_step`

```python
training_step(batch, batch_idx)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

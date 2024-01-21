<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dictionary_fixed_atoms`




**Global Variables**
---------------
- **ICML**


---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DictionaryFixedAtoms`




<a href="../dictionary_learning/dictionary_fixed_atoms.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.__init__`

```python
__init__(
    XP,
    YP,
    n_samples=1024,
    A=None,
    weight_initialization='random',
    barycenter_initialization='random',
    n_distributions=None,
    loss_fn=None,
    learning_rate=0.1,
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

#### <kbd>property</kbd> DictionaryFixedAtoms.automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.device





---

#### <kbd>property</kbd> DictionaryFixedAtoms.dtype





---

#### <kbd>property</kbd> DictionaryFixedAtoms.example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.fabric





---

#### <kbd>property</kbd> DictionaryFixedAtoms.global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> DictionaryFixedAtoms.trainer







---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.configure_optimizers`

```python
configure_optimizers()
```





---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.custom_histogram_adder`

```python
custom_histogram_adder()
```





---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.get_weights`

```python
get_weights()
```





---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.optimizer_step`

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

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.reconstruct`

```python
reconstruct(α=None, n_samples_atoms=None, n_samples_barycenter=None)
```





---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.sample_from_atoms`

```python
sample_from_atoms(n=None, detach=False)
```





---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.training_epoch_end`

```python
training_epoch_end(outputs)
```





---

<a href="../dictionary_learning/dictionary_fixed_atoms.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DictionaryFixedAtoms.training_step`

```python
training_step(batch, batch_idx)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

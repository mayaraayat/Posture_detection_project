<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/dadil_jdot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dadil_jdot`





---

<a href="../dictionary_learning/dadil_jdot.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `torch_accuracy_score`

```python
torch_accuracy_score(y_true, y_pred)
```






---

<a href="../dictionary_learning/dadil_jdot.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DaDiLJDOT`




<a href="../dictionary_learning/dadil_jdot.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.__init__`

```python
__init__(
    task,
    n_samples=1024,
    n_dim=None,
    n_classes=None,
    XP=None,
    YP=None,
    A=None,
    n_components=2,
    weight_initialization='random',
    n_distributions=None,
    loss_fn=None,
    learning_rate=0.1,
    learning_rate_task=1e-05,
    reg=0.0,
    num_iter_barycenter=10,
    num_iter_sinkhorn=20,
    domain_names=None,
    proj_grad=True,
    grad_labels=True,
    optimizer_name='adam',
    balanced_sampling=True
)
```






---

#### <kbd>property</kbd> DaDiLJDOT.automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> DaDiLJDOT.current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> DaDiLJDOT.device





---

#### <kbd>property</kbd> DaDiLJDOT.dtype





---

#### <kbd>property</kbd> DaDiLJDOT.example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> DaDiLJDOT.fabric





---

#### <kbd>property</kbd> DaDiLJDOT.global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> DaDiLJDOT.global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> DaDiLJDOT.hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> DaDiLJDOT.hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> DaDiLJDOT.local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> DaDiLJDOT.logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> DaDiLJDOT.loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> DaDiLJDOT.on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> DaDiLJDOT.trainer







---

<a href="../dictionary_learning/dadil_jdot.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.configure_optimizers`

```python
configure_optimizers()
```





---

<a href="../dictionary_learning/dadil_jdot.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.custom_histogram_adder`

```python
custom_histogram_adder()
```





---

<a href="../dictionary_learning/dadil_jdot.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.optimizer_step`

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

<a href="../dictionary_learning/dadil_jdot.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.reconstruct`

```python
reconstruct(α=None, n=None)
```





---

<a href="../dictionary_learning/dadil_jdot.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.sample_from_atoms`

```python
sample_from_atoms(n=None)
```





---

<a href="../dictionary_learning/dadil_jdot.py#L273"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.training_epoch_end`

```python
training_epoch_end(outputs)
```





---

<a href="../dictionary_learning/dadil_jdot.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.training_step`

```python
training_step(batch, batch_idx)
```





---

<a href="../dictionary_learning/dadil_jdot.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.validation_epoch_end`

```python
validation_epoch_end(outputs)
```





---

<a href="../dictionary_learning/dadil_jdot.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DaDiLJDOT.validation_step`

```python
validation_step(batch, batch_index)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/barycentric_regression.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `barycentric_regression`






---

<a href="../dictionary_learning/barycentric_regression.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WassersteinBarycentricRegression`




<a href="../dictionary_learning/barycentric_regression.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.__init__`

```python
__init__(
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
    barycenter_tol=1e-09,
    barycenter_beta=None,
    barycenter_verbose=False,
    barycenter_label_metric='l2',
    barycenter_initialization='random',
    barycenter_covariance_type='diag',
    dtype='float',
    batch_size=5,
    log_gradients=False
)
```






---

#### <kbd>property</kbd> WassersteinBarycentricRegression.automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.device





---

#### <kbd>property</kbd> WassersteinBarycentricRegression.dtype





---

#### <kbd>property</kbd> WassersteinBarycentricRegression.example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.fabric





---

#### <kbd>property</kbd> WassersteinBarycentricRegression.global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> WassersteinBarycentricRegression.trainer







---

<a href="../dictionary_learning/barycentric_regression.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.configure_optimizers`

```python
configure_optimizers()
```





---

<a href="../dictionary_learning/barycentric_regression.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.custom_histogram_adder`

```python
custom_histogram_adder()
```





---

<a href="../dictionary_learning/barycentric_regression.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.generate_batch_indices_without_replacement`

```python
generate_batch_indices_without_replacement(batch_size=None)
```





---

<a href="../dictionary_learning/barycentric_regression.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.get_weights`

```python
get_weights()
```





---

<a href="../dictionary_learning/barycentric_regression.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.optimizer_step`

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

<a href="../dictionary_learning/barycentric_regression.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.sample_from_atoms`

```python
sample_from_atoms(n=None, detach=False)
```





---

<a href="../dictionary_learning/barycentric_regression.py#L362"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.training_epoch_end`

```python
training_epoch_end(outputs)
```





---

<a href="../dictionary_learning/barycentric_regression.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `WassersteinBarycentricRegression.training_step`

```python
training_step(batch, batch_idx)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

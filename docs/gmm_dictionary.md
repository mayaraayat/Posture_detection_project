<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/gmm_dictionary.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `gmm_dictionary`






---

<a href="../dictionary_learning/gmm_dictionary.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GaussianMixtureDictionary`




<a href="../dictionary_learning/gmm_dictionary.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GaussianMixtureDictionary.__init__`

```python
__init__()
```






---

#### <kbd>property</kbd> GaussianMixtureDictionary.automatic_optimization

If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.current_epoch

The current epoch in the ``Trainer``, or 0 if not attached. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.device





---

#### <kbd>property</kbd> GaussianMixtureDictionary.dtype





---

#### <kbd>property</kbd> GaussianMixtureDictionary.example_input_array

The example input array is a specification of what the module can consume in the :meth:`forward` method. The return type is interpreted as follows: 


-   Single tensor: It is assumed the model takes a single argument, i.e.,  ``model.forward(model.example_input_array)`` 
-   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,  ``model.forward(*model.example_input_array)`` 
-   Dict: The input array represents named keyword arguments, i.e.,  ``model.forward(**model.example_input_array)`` 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.fabric





---

#### <kbd>property</kbd> GaussianMixtureDictionary.global_rank

The index of the current process across all nodes and devices. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.global_step

Total training batches seen across all epochs. 

If no Trainer is attached, this propery is 0. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.hparams

The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`. 



**Returns:**
  Mutable hyperparameters dictionary 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.hparams_initial

The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only. Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`. 



**Returns:**
 
 - <b>`AttributeDict`</b>:  immutable initial hyperparameters 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.local_rank

The index of the current process within a single node. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.logger

Reference to the logger object in the Trainer. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.loggers

Reference to the list of loggers in the Trainer. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.on_gpu

Returns ``True`` if this model is currently located on a GPU. 

Useful to set flags around the LightningModule for different CPU vs GPU behavior. 

---

#### <kbd>property</kbd> GaussianMixtureDictionary.trainer





---

#### <kbd>property</kbd> GaussianMixtureDictionary.truncated_bptt_steps

Enables `Truncated Backpropagation Through Time` in the Trainer when set to a positive integer. 

It represents the number of times :meth:`training_step` gets called before backpropagation. If this is > 0, the :meth:`training_step` receives an additional argument ``hiddens`` and is expected to return a hidden state. 






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

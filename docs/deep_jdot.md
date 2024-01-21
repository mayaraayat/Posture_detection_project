<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../msda/deep_jdot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `deep_jdot`






---

<a href="../msda/deep_jdot.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DeepJDOT`




<a href="../msda/deep_jdot.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DeepJDOT.__init__`

```python
__init__(
    encoder,
    task,
    λ,
    feature_importance,
    label_importance,
    reg,
    encoder_optimizer,
    task_optimizer,
    loss_fn
)
```






---

#### <kbd>property</kbd> DeepJDOT.activity_regularizer

Optional regularizer function for the output of this layer. 

---

#### <kbd>property</kbd> DeepJDOT.compute_dtype

The dtype of the layer's computations. 

This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless mixed precision is used, this is the same as `Layer.dtype`, the dtype of the weights. 

Layers automatically cast their inputs to the compute dtype, which causes computations and the output to be in the compute dtype as well. This is done by the base Layer class in `Layer.__call__`, so you do not have to insert these casts if implementing your own layer. 

Layers often perform certain internal computations in higher precision when `compute_dtype` is float16 or bfloat16 for numeric stability. The output will still typically be float16 or bfloat16 in such cases. 



**Returns:**
  The layer's compute dtype. 

---

#### <kbd>property</kbd> DeepJDOT.distribute_strategy

The `tf.distribute.Strategy` this model was created under. 

---

#### <kbd>property</kbd> DeepJDOT.dtype

The dtype of the layer weights. 

This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless mixed precision is used, this is the same as `Layer.compute_dtype`, the dtype of the layer's computations. 

---

#### <kbd>property</kbd> DeepJDOT.dtype_policy

The dtype policy associated with this layer. 

This is an instance of a `tf.keras.mixed_precision.Policy`. 

---

#### <kbd>property</kbd> DeepJDOT.dynamic

Whether the layer is dynamic (eager-only); set in the constructor. 

---

#### <kbd>property</kbd> DeepJDOT.inbound_nodes

Return Functional API nodes upstream of this layer. 

---

#### <kbd>property</kbd> DeepJDOT.input

Retrieves the input tensor(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input tensor or list of input tensors. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If called in Eager mode. 
 - <b>`AttributeError`</b>:  If no inbound nodes are found. 

---

#### <kbd>property</kbd> DeepJDOT.input_mask

Retrieves the input mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Input mask tensor (potentially None) or list of input  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> DeepJDOT.input_shape

Retrieves the input shape(s) of a layer. 

Only applicable if the layer has exactly one input, i.e. if it is connected to one incoming layer, or if all inputs have the same shape. 



**Returns:**
  Input shape, as an integer shape tuple  (or list of shape tuples, one tuple per input tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined input_shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> DeepJDOT.input_spec

`InputSpec` instance(s) describing the input format for this layer. 

When you create a layer subclass, you can set `self.input_spec` to enable the layer to run input compatibility checks when it is called. Consider a `Conv2D` layer: it can only be called on a single input tensor of rank 4. As such, you can set, in `__init__()`: 

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
``` 

Now, if you try to call the layer on an input that isn't rank 4 (for instance, an input of shape `(2,)`, it will raise a nicely-formatted error: 

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
``` 

Input checks that can be specified via `input_spec` include: 
- Structure (e.g. a single input, a list of 2 inputs, etc) 
- Shape 
- Rank (ndim) 
- Dtype 

For more information, see `tf.keras.layers.InputSpec`. 



**Returns:**
  A `tf.keras.layers.InputSpec` instance, or nested structure thereof. 

---

#### <kbd>property</kbd> DeepJDOT.layers





---

#### <kbd>property</kbd> DeepJDOT.losses

List of losses added using the `add_loss()` API. 

Variable regularization tensors are created when this property is accessed, so it is eager safe: accessing `losses` under a `tf.GradientTape` will propagate gradients back to the corresponding variables. 



**Examples:**
 

``` class MyLayer(tf.keras.layers.Layer):```
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
``` l = MyLayer()``` ``` l(np.ones((10, 1)))```
``` l.losses``` [1.0] 

``` inputs = tf.keras.Input(shape=(10,))```
``` x = tf.keras.layers.Dense(10)(inputs)``` ``` outputs = tf.keras.layers.Dense(1)(x)```
``` model = tf.keras.Model(inputs, outputs)``` ``` # Activity regularization.```
``` len(model.losses)``` 0 ``` model.add_loss(tf.abs(tf.reduce_mean(x)))```
``` len(model.losses)``` 1 

``` inputs = tf.keras.Input(shape=(10,))```
``` d = tf.keras.layers.Dense(10, kernel_initializer='ones')``` ``` x = d(inputs)```
``` outputs = tf.keras.layers.Dense(1)(x)``` ``` model = tf.keras.Model(inputs, outputs)```
``` # Weight regularization.``` ``` model.add_loss(lambda: tf.reduce_mean(d.kernel))```
``` model.losses``` [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>] 



**Returns:**
  A list of tensors. 

---

#### <kbd>property</kbd> DeepJDOT.metrics

Returns the model's metrics added using `compile()`, `add_metric()` APIs. 

Note: Metrics passed to `compile()` are available only after a `keras.Model` has been trained/evaluated on actual data. 



**Examples:**
 

``` inputs = tf.keras.layers.Input(shape=(3,))```
``` outputs = tf.keras.layers.Dense(2)(inputs)``` ``` model = tf.keras.models.Model(inputs=inputs, outputs=outputs)```
``` model.compile(optimizer="Adam", loss="mse", metrics=["mae"])``` ``` [m.name for m in model.metrics]```
[]

``` x = np.random.random((2, 3))``` ``` y = np.random.randint(0, 2, (2, 2))```
``` model.fit(x, y)``` ``` [m.name for m in model.metrics]```
['loss', 'mae']

``` inputs = tf.keras.layers.Input(shape=(3,))``` ``` d = tf.keras.layers.Dense(2, name='out')```
``` output_1 = d(inputs)``` ``` output_2 = d(inputs)```
``` model = tf.keras.models.Model(``` ...    inputs=inputs, outputs=[output_1, output_2]) ``` model.add_metric(```
...    tf.reduce_sum(output_2), name='mean', aggregation='mean')
``` model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])``` ``` model.fit(x, (y, y))```
``` [m.name for m in model.metrics]``` ['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae', 'out_1_acc', 'mean'] 

---

#### <kbd>property</kbd> DeepJDOT.metrics_names

Returns the model's display labels for all outputs. 

Note: `metrics_names` are available only after a `keras.Model` has been trained/evaluated on actual data. 



**Examples:**
 

``` inputs = tf.keras.layers.Input(shape=(3,))```
``` outputs = tf.keras.layers.Dense(2)(inputs)``` ``` model = tf.keras.models.Model(inputs=inputs, outputs=outputs)```
``` model.compile(optimizer="Adam", loss="mse", metrics=["mae"])``` ``` model.metrics_names```
[]

``` x = np.random.random((2, 3))``` ``` y = np.random.randint(0, 2, (2, 2))```
``` model.fit(x, y)``` ``` model.metrics_names```
['loss', 'mae']

``` inputs = tf.keras.layers.Input(shape=(3,))``` ``` d = tf.keras.layers.Dense(2, name='out')```
``` output_1 = d(inputs)``` ``` output_2 = d(inputs)```
``` model = tf.keras.models.Model(``` ...    inputs=inputs, outputs=[output_1, output_2]) ``` model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])```
``` model.fit(x, (y, y))``` ``` model.metrics_names```
['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
'out_1_acc']


---

#### <kbd>property</kbd> DeepJDOT.name

Name of the layer (string), set in the constructor. 

---

#### <kbd>property</kbd> DeepJDOT.name_scope

Returns a `tf.name_scope` instance for this class. 

---

#### <kbd>property</kbd> DeepJDOT.non_trainable_variables





---

#### <kbd>property</kbd> DeepJDOT.non_trainable_weights





---

#### <kbd>property</kbd> DeepJDOT.outbound_nodes

Return Functional API nodes downstream of this layer. 

---

#### <kbd>property</kbd> DeepJDOT.output

Retrieves the output tensor(s) of a layer. 

Only applicable if the layer has exactly one output, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output tensor or list of output tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming  layers. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> DeepJDOT.output_mask

Retrieves the output mask tensor(s) of a layer. 

Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer. 



**Returns:**
  Output mask tensor (potentially None) or list of output  mask tensors. 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer is connected to more than one incoming layers. 

---

#### <kbd>property</kbd> DeepJDOT.output_shape

Retrieves the output shape(s) of a layer. 

Only applicable if the layer has one output, or if all outputs have the same shape. 



**Returns:**
  Output shape, as an integer shape tuple  (or list of shape tuples, one tuple per output tensor). 



**Raises:**
 
 - <b>`AttributeError`</b>:  if the layer has no defined output shape. 
 - <b>`RuntimeError`</b>:  if called in Eager mode. 

---

#### <kbd>property</kbd> DeepJDOT.run_eagerly

Settable attribute indicating whether the model should run eagerly. 

Running eagerly means that your model will be run step by step, like Python code. Your model might run slower, but it should become easier for you to debug it by stepping into individual layer calls. 

By default, we will attempt to compile your model to a static graph to deliver the best execution performance. 



**Returns:**
  Boolean, whether the model should run eagerly. 

---

#### <kbd>property</kbd> DeepJDOT.state_updates

Deprecated, do NOT use! 

Returns the `updates` from all layers that are stateful. 

This is useful for separating training updates and state updates, e.g. when we need to update a layer's internal state during prediction. 



**Returns:**
  A list of update ops. 

---

#### <kbd>property</kbd> DeepJDOT.stateful





---

#### <kbd>property</kbd> DeepJDOT.submodules

Sequence of all sub-modules. 

Submodules are modules which are properties of this module, or found as properties of modules which are properties of this module (and so on). 

``` a = tf.Module()```
``` b = tf.Module()``` ``` c = tf.Module()```
``` a.b = b``` ``` b.c = c```
``` list(a.submodules) == [b, c]``` True ``` list(b.submodules) == [c]```
True
``` list(c.submodules) == []``` True 



**Returns:**
  A sequence of all submodules. 

---

#### <kbd>property</kbd> DeepJDOT.supports_masking

Whether this layer supports computing a mask using `compute_mask`. 

---

#### <kbd>property</kbd> DeepJDOT.trainable





---

#### <kbd>property</kbd> DeepJDOT.trainable_variables





---

#### <kbd>property</kbd> DeepJDOT.trainable_weights





---

#### <kbd>property</kbd> DeepJDOT.updates





---

#### <kbd>property</kbd> DeepJDOT.variable_dtype

Alias of `Layer.dtype`, the dtype of the weights. 

---

#### <kbd>property</kbd> DeepJDOT.variables

Returns the list of all layer variables/weights. 

Alias of `self.weights`. 

Note: This will not track the weights of nested `tf.Modules` that are not themselves Keras layers. 



**Returns:**
  A list of variables. 

---

#### <kbd>property</kbd> DeepJDOT.weights

Returns the list of all layer variables/weights. 

Note: This will not track the weights of nested `tf.Modules` that are not themselves Keras layers. 



**Returns:**
  A list of variables. 



---

<a href="../msda/deep_jdot.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DeepJDOT.compile`

```python
compile(metrics=None)
```





---

<a href="../msda/deep_jdot.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DeepJDOT.compute_ot`

```python
compute_ot(hs, ht, ys, yt_pred)
```





---

<a href="../msda/deep_jdot.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DeepJDOT.compute_pairwise_distances`

```python
compute_pairwise_distances(xs, xt)
```





---

<a href="../msda/deep_jdot.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DeepJDOT.test_step`

```python
test_step(data)
```





---

<a href="../msda/deep_jdot.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DeepJDOT.train_step`

```python
train_step(data)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

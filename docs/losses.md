<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/losses.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `losses`
Module for losses between probability distributions: $\mathcal{L}(P,Q)$. 


---

<a href="../dictionary_learning/losses.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_pairwise_wasserstein`

```python
compute_pairwise_wasserstein(Xs, Ys)
```






---

<a href="../dictionary_learning/losses.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `label_loss`

```python
label_loss(YA, YB)
```

Computes the 0-1 label loss between one-hot encoded label vectors, 

$$d_{\mathcal{Y}}(\mathbf{Y}^{(P)},\mathbf{Y}^{(Q)}) = \delta(\mathbf{Y}^{(P)} - \mathbf{Y}^{(Q)})$$ 

__NOTE:__ this function is not differentiable w.r.t. YA nor YB. 



**Args:**
 
 - <b>`YA`</b>:  labels for samples in P. 
 - <b>`YB`</b>:  labels for samples in Q. 
 - <b>`device`</b>:  device to allocate the matrix. Either 'cpu' or 'cuda'. 


---

<a href="../dictionary_learning/losses.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `parametric_bures_wasserstein`

```python
parametric_bures_wasserstein(mP, mQ, sP, sQ)
```






---

<a href="../dictionary_learning/losses.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DifferentiableDeltaLabelLoss`




<a href="../dictionary_learning/losses.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DifferentiableDeltaLabelLoss.__init__`

```python
__init__(precomputed_M=None, n_classes_P=None, n_classes_Q=None)
```








---

<a href="../dictionary_learning/losses.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DifferentiableDeltaLabelLoss.forward`

```python
forward(YP, YQ)
```

Computes the 0-1 label loss between one-hot encoded label vectors, 

$$d_{\mathcal{Y}}(\mathbf{Y}^{(P)},\mathbf{Y}^{(Q)}) = \delta(\mathbf{Y}^{(P)} - \mathbf{Y}^{(Q)})$$ 

__NOTE:__ this function is not differentiable w.r.t. YA nor YB. 



**Args:**
 
 - <b>`YA`</b>:  labels for samples in P. 
 - <b>`YB`</b>:  labels for samples in Q. 
 - <b>`device`</b>:  device to allocate the matrix. Either 'cpu' or 'cuda'. 


---

<a href="../dictionary_learning/losses.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EnvelopeWassersteinLoss`
Wasserstein loss using the Primal Kantorovich formulation. Gradients are computed using the Envelope Theorem. 

<a href="../dictionary_learning/losses.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EnvelopeWassersteinLoss.__init__`

```python
__init__(ε=0.0, num_iter_sinkhorn=20, debias=False)
```

Creates the loss object. 



**Args:**
 
 - <b>`ϵ`</b>:  entropic regularization penalty. 
 - <b>`num_iter_sinkhorn`</b>:  maximum number of sinkhorn iterations. Only used for ϵ > 0. 
 - <b>`debias`</b>:  whether or not compute the debiased sinkhorn loss. Only used when ϵ > 0. 




---

<a href="../dictionary_learning/losses.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EnvelopeWassersteinLoss.forward`

```python
forward(XP, XQ)
```

Computes the Wasserstien loss between samples XP ~ P and XQ ~ Q, 

$$\mathcal{L}(\mathbf{X}^{(P)},\mathbf{X}^{(Q)}) = W_{2}(P,Q) = \underset{\pi\in U(\mathbf{u}_{n},\mathbf{u}_{m})}{\text{argmin}}\sum_{i=1}^{n}\sum_{j=1}^{m}\pi_{i,j}\lVert \mathbf{x}_{i}^{(P)} - \mathbf{x}_{j}^{(Q)} \rVert_{2}^{2}$$ 



**Args:**
 
 - <b>`XP`</b>:  Tensor of shape (n, d) containing i.i.d samples from distribution P 
 - <b>`XQ`</b>:  Tensor of shape (m, d) containing i.i.d samples from distribution Q 


---

<a href="../dictionary_learning/losses.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `JointWassersteinLoss`
Wasserstein loss between joint distributions of labels and features, using the Primal Kantorovich formulation. Gradients are computed using the Envelope Theorem. 

<a href="../dictionary_learning/losses.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `JointWassersteinLoss.__init__`

```python
__init__(
    ε=0.0,
    τ=0.0,
    β=None,
    label_metric=None,
    num_iter_sinkhorn=20,
    max_val=None,
    p=2,
    q=2
)
```

Creates the loss object. 



**Args:**
 
 - <b>`ϵ`</b>:  entropic regularization penalty. 
 - <b>`τ`</b>:  marginal OT plan relaxation. __remark:__ not used in the paper. Should be set to 0. 
 - <b>`num_iter_sinkhorn`</b>:  maximum number of sinkhorn iterations. Only used for ϵ > 0. 




---

<a href="../dictionary_learning/losses.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `JointWassersteinLoss.forward`

```python
forward(XQ, YQ, XP, YP, index=None)
```

Computes the Wasserstien loss between samples XP ~ P and XQ ~ Q, 

$$\mathcal{L}(\mathbf{X}^{(P)}, \mathbf{Y}^{(P)},\mathbf{X}^{(Q)}, \mathbf{Y}^{(Q)}) = W_{2}(P,Q) = \underset{\pi\in U(\mathbf{u}_{n},\mathbf{u}_{m})}{\text{argmin}}\sum_{i=1}^{n}\sum_{j=1}^{m}\pi_{i,j}(\lVert \mathbf{x}_{i}^{(P)} - \mathbf{x}_{j}^{(Q)} \rVert_{2}^{2}+\beta\lVert \mathbf{Y}_{i}^{(P)} - \mathbf{Y}_{j}^{(Q)} \rVert_{2}^{2})$$ 

__Remark:__ as in the paper, we set $\beta = \text{max}_{i,j}\lVert \mathbf{x}_{i}^{(P)} - \mathbf{x}_{j}^{(Q)} \rVert_{2}^{2}$ 



**Args:**
 
 - <b>`XP`</b>:  Tensor of shape (n, d) containing i.i.d features from distribution P 
 - <b>`YP`</b>:  Tensor of shape (n, nc) containing i.i.d labels from distribution P 
 - <b>`XQ`</b>:  Tensor of shape (m, d) containing i.i.d samples from distribution Q 
 - <b>`YQ`</b>:  Tensor of shape (n, nc) containing i.i.d labels from distribution Q 


---

<a href="../dictionary_learning/losses.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SupervisedPartialWassersteinDistance`




<a href="../dictionary_learning/losses.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SupervisedPartialWassersteinDistance.__init__`

```python
__init__(n_dummies=1, m=0.9, β=None, label_metric='l2')
```








---

<a href="../dictionary_learning/losses.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SupervisedPartialWassersteinDistance.forward`

```python
forward(XQ, YQ, XP, YP, index=None)
```






---

<a href="../dictionary_learning/losses.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RenyiEntropy`
Rényi Entropy regularization 

<a href="../dictionary_learning/losses.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RenyiEntropy.__init__`

```python
__init__(β=1)
```

For a random variable $X$ assuming discrete values $1,\cdots,n$ with probabilities $\alpha_{i}, i=1,\cdots,n$, $\sum_{i}\alpha_{i}=1$, the Renyi entropy is, 

$$H_{\beta}(\alpha)=\dfrac{\beta}{1-\beta}\log \lVert \alpha \rVert_{\alpha}$$ 



**args:**
 
 - <b>`β`</b>:  parameter for the Renyi entropy. 




---

<a href="../dictionary_learning/losses.py#L259"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RenyiEntropy.forward`

```python
forward(x)
```

computes the renyi entropy. __remark:__ x must be non-negative and sum to 1. 


---

<a href="../dictionary_learning/losses.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `JointDeltaWassersteinLoss`




<a href="../dictionary_learning/losses.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `JointDeltaWassersteinLoss.__init__`

```python
__init__(M, β=None, p=2, q=2)
```








---

<a href="../dictionary_learning/losses.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `JointDeltaWassersteinLoss.forward`

```python
forward(XQ, YQ, XP, YP, index=None)
```






---

<a href="../dictionary_learning/losses.py#L308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SlicedWassersteinLoss`




<a href="../dictionary_learning/losses.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SlicedWassersteinLoss.__init__`

```python
__init__(n_projections=50, use_max=False, p=2)
```








---

<a href="../dictionary_learning/losses.py#L316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SlicedWassersteinLoss.forward`

```python
forward(XP, XQ)
```






---

<a href="../dictionary_learning/losses.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MaximumMeanDiscrepancy`




<a href="../dictionary_learning/losses.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MaximumMeanDiscrepancy.__init__`

```python
__init__(kernel='linear', bandwidth=None)
```








---

<a href="../dictionary_learning/losses.py#L330"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MaximumMeanDiscrepancy.forward`

```python
forward(XP, XQ)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/dictionary.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dictionary`
Module for Dictionary Learning 



---

<a href="../dictionary_learning/dictionary.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AbstractDictionary`
Abstract Dictionary class 

<a href="../dictionary_learning/dictionary.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractDictionary.__init__`

```python
__init__(
    n_samples=64,
    n_components=2,
    n_dim=2,
    lr=0.01,
    device='cpu',
    loss_fn=None,
    regularizer_fn=None,
    weight_initialization='uniform',
    num_iter_barycenter=5,
    num_iter_sinkhorn=20
)
```

Initializes an abstract dictionary. 



**Args:**
 
 - <b>`n_samples`</b>:  number of samples in the barycenters support. 
 - <b>`n_components`</b>:  number of atoms. 
 - <b>`n_dim`</b>:  number of dimensions in the data. 
 - <b>`lr`</b>:  learning_rate in gradient descent. 
 - <b>`loss_fn`</b>:  loss between distributions. 
 - <b>`regularizer`</b>:  regularizer for dictionary weights 
 - <b>`weight_initialization`</b>:  either 'uniform' or 'random'. 
 - <b>`num_iter_barycenter`</b>:  number of iterations used in the fixed-point algorithm. 
 - <b>`num_iter_sinkhorn`</b>:  entropic regularization penalty. 




---

<a href="../dictionary_learning/dictionary.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractDictionary.fit`

```python
fit()
```





---

<a href="../dictionary_learning/dictionary.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractDictionary.get_atoms`

```python
get_atoms()
```





---

<a href="../dictionary_learning/dictionary.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractDictionary.get_weights`

```python
get_weights()
```





---

<a href="../dictionary_learning/dictionary.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractDictionary.initialize`

```python
initialize()
```





---

<a href="../dictionary_learning/dictionary.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractDictionary.reconstruct`

```python
reconstruct(**kwargs)
```





---

<a href="../dictionary_learning/dictionary.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractDictionary.transform`

```python
transform()
```






---

<a href="../dictionary_learning/dictionary.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EmpiricalDictionary`
Empirical Dictionary class. This represents a dictionary with empirical distributions as atoms, namely $\mathcal{P} = \{\hat{P}\}_{k=1}^{K}$, where each $\hat{P}_{k} = \frac{1}{n}\sum_{i=1}^{n}\delta_{\mathbf{x}_{i}^{(P_{k})}}$. 

__Remark:__ this class is abstract, as we do not implement the fit method. 

<a href="../dictionary_learning/dictionary.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalDictionary.__init__`

```python
__init__(
    n_samples=64,
    n_components=2,
    n_dim=2,
    lr=0.01,
    η_A=0.0,
    ε=0.01,
    device='cpu',
    loss_fn=None,
    regularizer_fn=None,
    weight_initialization='uniform',
    num_iter_barycenter=5,
    num_iter_sinkhorn=20
)
```

Initializes an empirical dictionary. 



**Args:**
 
 - <b>`n_samples`</b>:  number of samples in the barycenters support. 
 - <b>`n_components`</b>:  number of atoms. 
 - <b>`n_dim`</b>:  number of dimensions in the data. 
 - <b>`lr`</b>:  learning_rate in gradient descent. 
 - <b>`ϵ`</b>:  entropic regularization penalty. 
 - <b>`η_A`</b>:  sparse regularization for the atoms weights. __remark:__ not used in our paper. 
 - <b>`loss_fn`</b>:  loss between distributions. 
 - <b>`regularizer`</b>:  regularizer for dictionary weights 
 - <b>`weight_initialization`</b>:  either 'uniform' or 'random'. 
 - <b>`num_iter_barycenter`</b>:  number of iterations used in the fixed-point algorithm. 
 - <b>`num_iter_sinkhorn`</b>:  entropic regularization penalty. 




---

<a href="../dictionary_learning/dictionary.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalDictionary.fit`

```python
fit()
```





---

<a href="../dictionary_learning/dictionary.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalDictionary.get_atoms`

```python
get_atoms()
```

Gets the entire support for each atom. 

---

<a href="../dictionary_learning/dictionary.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalDictionary.get_weights`

```python
get_weights()
```

Gets the learned weights. 

---

<a href="../dictionary_learning/dictionary.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalDictionary.initialize`

```python
initialize(n_datasets)
```





---

<a href="../dictionary_learning/dictionary.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalDictionary.reconstruct`

```python
reconstruct()
```





---

<a href="../dictionary_learning/dictionary.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalDictionary.transform`

```python
transform()
```






---

<a href="../dictionary_learning/dictionary.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MinibatchDictionary`
Minibatch Dictionary Class. This class represents a dictionary where we can sample from its atoms for learning. This is exactly the same as EmpiricalDictionary, except that we can call the fit method. 

<a href="../dictionary_learning/dictionary.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.__init__`

```python
__init__(
    n_samples=1000,
    batch_size=128,
    n_components=2,
    n_dim=2,
    lr_features=0.01,
    lr_labels=None,
    lr_weights=None,
    grad_labels=True,
    ε=0.0,
    device='cpu',
    loss_fn=None,
    regularizer_fn=None,
    weight_initialization='uniform',
    num_iter_barycenter=5,
    num_iter_sinkhorn=20
)
```

Initializes a minibatch dictionary. 



**Args:**
 
 - <b>`n_samples`</b>:  number of samples in the barycenters support. 
 - <b>`batch_size`</b>:  number of samples on each mini-batch. __remark:__ should be at most n_samples. 
 - <b>`n_components`</b>:  number of atoms. 
 - <b>`n_dim`</b>:  number of dimensions in the data. 
 - <b>`lr`</b>:  learning_rate in gradient descent. 
 - <b>`ϵ`</b>:  entropic regularization penalty. 
 - <b>`η_A`</b>:  sparse regularization for the atoms weights. __remark:__ not used in our paper. 
 - <b>`loss_fn`</b>:  loss between distributions. 
 - <b>`regularizer`</b>:  regularizer for dictionary weights 
 - <b>`weight_initialization`</b>:  either 'uniform' or 'random'. 
 - <b>`num_iter_barycenter`</b>:  number of iterations used in the fixed-point algorithm. 
 - <b>`num_iter_sinkhorn`</b>:  entropic regularization penalty. 




---

<a href="../dictionary_learning/dictionary.py#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.fit`

```python
fit(datasets, num_iter_max=100, batches_per_it=100)
```

Dictionary Learning method. In this method we minimize, 

$$L(\mathcal{P},\mathcal{A}) = \dfrac{1}{N}\sum_{\ell=1}^{N}\mathcal{L}(\hat{Q}_{\ell},\hat{B}_{\ell}),$$ 

where $\hat{B}_{\ell} = \mathcal{B}(\alpha;\mathcal{P})$. Optimization is carried w.r.t. $\mathbf{x}_{j}^{(P_{k})}$ and $\alpha_{\ell} \in \Delta_{K}$. 

__Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf), we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator, 

$$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$ 



**Args:**
 
 - <b>`datasets`</b>:  List of arrays $\mathbf{X}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}$. 
 - <b>`num_iter_max`</b>:  maximum number of iterations for dictionary learning. 
 - <b>`batches_per_it`</b>:  number of batches sampled per iteration. 

---

<a href="../dictionary_learning/dictionary.py#L287"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.generate_batch_indices_without_replacement`

```python
generate_batch_indices_without_replacement(batch_size=None)
```





---

<a href="../dictionary_learning/dictionary.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.get_atoms`

```python
get_atoms()
```

Gets the entire support for each atom. 

---

<a href="../dictionary_learning/dictionary.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.get_weights`

```python
get_weights()
```

Gets the learned weights. 

---

<a href="../dictionary_learning/dictionary.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.initialize`

```python
initialize(n_datasets)
```





---

<a href="../dictionary_learning/dictionary.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.reconstruct`

```python
reconstruct()
```





---

<a href="../dictionary_learning/dictionary.py#L234"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.sample_from_atoms`

```python
sample_from_atoms(n=None, detach=False)
```





---

<a href="../dictionary_learning/dictionary.py#L368"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MinibatchDictionary.transform`

```python
transform(datasets, num_iter_max=10, batches_per_it=10)
```

Embeds distributions $\mathcal{Q} = \{\hat{Q}_{\ell}\}_{\ell=1}^{N}$ into the simplex $\Delta_{K}$. It is equivalent to the Barycentric Coordinate Regression method of [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf), 

$$\alpha^{\star}_{\ell} = \varphi(\hat{Q}) = \underset{\alpha \in \Delta_{K}}{\text{argmin}}\sum_{k=1}^{K}\alpha_{k}W_{2}(\hat{P}_{k},\hat{Q}_{\ell}).$$ 

__Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf), we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator, 

$$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$ 



**Args:**
 
 - <b>`datasets`</b>:  List of arrays $\mathbf{X}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}$. 
 - <b>`num_iter_max`</b>:  maximum number of iterations for optimizing the cost function. 
 - <b>`batches_per_it`</b>:  number of mini-batches sampled per iteration. 


---

<a href="../dictionary_learning/dictionary.py#L414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LabeledMinibatchDictionary`
Class for dictionary learning when the support of distributions is labeled 

<a href="../dictionary_learning/dictionary.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.__init__`

```python
__init__(
    n_samples=1000,
    batch_size=128,
    n_components=2,
    n_dim=2,
    n_classes=None,
    lr_features=0.01,
    lr_labels=None,
    lr_weights=None,
    ε=0.0,
    device='cpu',
    loss_fn=None,
    regularizer_fn=None,
    proj_grad=True,
    grad_labels=True,
    balanced_sampling=True,
    sample_with_repetition=True,
    weight_initialization='uniform',
    barycenter_initialization='samples',
    num_iter_barycenter=5,
    num_iter_sinkhorn=20,
    penalize_labels=True,
    names=None,
    track_atoms=False
)
```

Initializes a labeled minibatch dictionary. 



**Args:**
 
 - <b>`n_samples`</b>:  number of samples in the barycenters support. 
 - <b>`batch_size`</b>:  number of samples on each mini-batch. __remark:__ should be at most n_samples. 
 - <b>`n_components`</b>:  number of atoms. 
 - <b>`n_dim`</b>:  number of dimensions in the data. 
 - <b>`lr`</b>:  learning_rate in gradient descent. 
 - <b>`ϵ`</b>:  entropic regularization penalty. 
 - <b>`η_A`</b>:  sparse regularization for the atoms weights. __remark:__ not used in our paper. 
 - <b>`loss_fn`</b>:  loss between distributions. __remark:__ this function must accept 4 arguments in its \_\_call\_\_,  namely XP, YP, XQ, YQ 
 - <b>`regularizer`</b>:  regularizer for dictionary weights 
 - <b>`weight_initialization`</b>:  either 'uniform' or 'random'. 
 - <b>`num_iter_barycenter`</b>:  number of iterations used in the fixed-point algorithm. 
 - <b>`num_iter_sinkhorn`</b>:  entropic regularization penalty. 
 - <b>`penalize_labels`</b>:  If True, includes a class-based penalty in the OT plan estimation. 




---

<a href="../dictionary_learning/dictionary.py#L685"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.fit`

```python
fit(datasets, num_iter_max=100, batches_per_it=100)
```

Dictionary Learning method. In this method we minimize, 

$$L(\mathcal{P},\mathcal{A}) = \dfrac{1}{N}\sum_{\ell=1}^{N}\mathcal{L}(\hat{Q}_{\ell},\hat{B}_{\ell}),$$ 

where $\hat{B}_{\ell} = \mathcal{B}(\alpha;\mathcal{P})$. Optimization is carried w.r.t. $\mathbf{x}_{j}^{(P_{k})}$ and $\alpha_{\ell} \in \Delta_{K}$. 

__Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf), we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator, 

$$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$ 



**Args:**
 
 - <b>`datasets`</b>:  List of tuples containing 2 arrays, $\mathbf{X}^{(Q_{\ell})}$ and $\mathbf{Y}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}(X,Y)$. 
 - <b>`num_iter_max`</b>:  maximum number of iterations for dictionary learning. 
 - <b>`batches_per_it`</b>:  number of batches sampled per iteration. 

---

<a href="../dictionary_learning/dictionary.py#L595"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.fit_without_replacement`

```python
fit_without_replacement(datasets, num_iter_max=100, batches_per_it=100)
```

Dictionary Learning method. In this method we minimize, 

$$L(\mathcal{P},\mathcal{A}) = \dfrac{1}{N}\sum_{\ell=1}^{N}\mathcal{L}(\hat{Q}_{\ell},\hat{B}_{\ell}),$$ 

where $\hat{B}_{\ell} = \mathcal{B}(\alpha;\mathcal{P})$. Optimization is carried w.r.t. $\mathbf{x}_{j}^{(P_{k})}$ and $\alpha_{\ell} \in \Delta_{K}$. 

__Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf), we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator, 

$$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$ 



**Args:**
 
 - <b>`datasets`</b>:  List of tuples containing 2 arrays, $\mathbf{X}^{(Q_{\ell})}$ and $\mathbf{Y}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}(X,Y)$. 
 - <b>`num_iter_max`</b>:  maximum number of iterations for dictionary learning. 
 - <b>`batches_per_it`</b>:  number of batches sampled per iteration. 

---

<a href="../dictionary_learning/dictionary.py#L567"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.generate_batch_indices_without_replacement`

```python
generate_batch_indices_without_replacement(batch_size=None)
```





---

<a href="../dictionary_learning/dictionary.py#L589"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.get_atoms`

```python
get_atoms()
```

Gets the features and labels from learned atoms. 

---

<a href="../dictionary_learning/dictionary.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.get_weights`

```python
get_weights()
```

Gets the learned weights. 

---

<a href="../dictionary_learning/dictionary.py#L483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.initialize`

```python
initialize(n_datasets, n_classes)
```

Initialization of atoms and weights. For $k=1,\cdots,K$, $j=1,\cdots,n$, $c=1,\cdots,n_{c}$, and $\ell=1,\cdots,N$, 

$$\mathbf{x}_{j}^{(P_{k})} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d}),$$ $$\mathbf{y}_{j}^{(P_{k})} = \dfrac{e^{p_{c}}}{\sum_{c'}e^{p_{c'}}}, p_{c} \sim \mathcal{N}(0, 1).$$ $$a_{k,\ell} \sim \mathcal{N}(0, 1).$$ 



**Args:**
 
 - <b>`n_datasets`</b>:  equivalent to $\ell$. Number of datasets in Dictionary Learning. 
 - <b>`n_classes`</b>:  number of classes in the datasets. 

---

<a href="../dictionary_learning/dictionary.py#L832"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.reconstruct`

```python
reconstruct(weights=None, n_samples_atoms=None, n_samples_barycenter=None)
```

Uses Wasserstein barycenters for reconstructing samples given $\alpha$, 

$$\hat{B} = \mathcal{B}(\alpha;\mathcal{P}) = \underset{B}{\text{argmin}}\sum_{k=1}^{K}\alpha_{k}W_{2}(B, \hat{P}_{k}) $$ 



**Args:**
 
 - <b>`α`</b>:  List of weights $\{\alpha_{\ell}\}_{\ell=1}^{N}$, $\alpha_{\ell} \in \Delta_{K}$ which will be used  to reconstruct samples from a given distribution. 
 - <b>`n_samples_atoms`</b>:  number of samples sampled from each atom for the reconstruction. 
 - <b>`n_samples_barycenter`</b>:  number of samples in the barycenter's support. 

---

<a href="../dictionary_learning/dictionary.py#L514"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.sample_from_atoms`

```python
sample_from_atoms(n=None, detach=False)
```





---

<a href="../dictionary_learning/dictionary.py#L783"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledMinibatchDictionary.transform`

```python
transform(Q, num_iter_max=100, batches_per_it=10)
```

Embeds distributions $\mathcal{Q} = \{\hat{Q}_{\ell}\}_{\ell=1}^{N}$ into the simplex $\Delta_{K}$. It is equivalent to the Barycentric Coordinate Regression method of [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf), 

$$\alpha^{\star}_{\ell} = \varphi(\hat{Q}) = \underset{\alpha \in \Delta_{K}}{\text{argmin}}\sum_{k=1}^{K}\alpha_{k}W_{2}(\hat{P}_{k},\hat{Q}_{\ell}).$$ 

__Remark:__ As [[Bonneel et al., 2016]](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinBarycentricCoordinates/WBC_lowres.pdf), we optimize the loss function w.r.t. $a_{k} \in \mathbb{R}^{K}$. We retrieve $\alpha \in \Delta_{K}$ using the softmax operator, 

$$\alpha_{k} = \dfrac{e^{a_{k}}}{\sum_{k'}e^{a_{k'}}}$$ 



**Args:**
 
 - <b>`datasets`</b>:  List of tuples containing 2 arrays, $\mathbf{X}^{(Q_{\ell})}$ and $\mathbf{Y}^{(Q_{\ell})}$ representing empirical distributions $\hat{Q}_{\ell}(X,Y)$. 
 - <b>`num_iter_max`</b>:  maximum number of iterations for optimizing the cost function. 
 - <b>`batches_per_it`</b>:  number of mini-batches sampled per iteration. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

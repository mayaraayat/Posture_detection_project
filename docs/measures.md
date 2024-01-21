<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/measures.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `measures`
Module for sampling data from datasets. 



---

<a href="../dictionary_learning/measures.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AbstractMeasure`
Abstract Measure Class 

<a href="../dictionary_learning/measures.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractMeasure.__init__`

```python
__init__()
```








---

<a href="../dictionary_learning/measures.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AbstractMeasure.sample`

```python
sample(n)
```






---

<a href="../dictionary_learning/measures.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EmpiricalMeasure`
Empirical Measure. 

Given a support $\mathbf{X}^{(P)} = [\mathbf{x}_{i}^{(P)}]_{i=1}^{n}$ with $\mathbf{x}_{i}^{(P)} \sim P$ with probability $0 \leq a_{i}$ \leq 1, where $a_{i}$ reflects the sample weight, samples $\mathbf{x}_{i}^{(P)} according to $\mathbf{a} \in \Delta_{n}$. 

<a href="../dictionary_learning/measures.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalMeasure.__init__`

```python
__init__(support, weights=None, device='cpu')
```

Initializes an empirical measure, 

$$\hat{P} = \sum_{i=1}^{n}a_{i}\delta_{\mathbf{x}_{i}^{(P)}}$$ 



**Args:**
 
 - <b>`support`</b>:  tensor of shape (n, d) containing samples $\mathbf{x}_{i}^{(P)} \in \mathbb{R}^{d}$ 
 - <b>`weights`</b>:  tensor of shape (n,) of non-negative entries that sum to 1. Correspond to sample weights. If not  given, assumes uniform weights (i.i.d. hypothesis). 
 - <b>`device`</b>:  either 'cpu' or 'gpu'. 




---

<a href="../dictionary_learning/measures.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalMeasure.sample`

```python
sample(n)
```

Gets $n$ samples from $\mathbf{X}^{(P)}$ according to $\mathbf{a} \in \Delta_{n}$. 

---

<a href="../dictionary_learning/measures.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EmpiricalMeasure.to`

```python
to(device)
```

Moves weights and support to device. 


---

<a href="../dictionary_learning/measures.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DatasetMeasure`
Dataset Measure class. This corresponds to an EmpiricalMeasure with i.i.d. hypothesis, namely, 

$$\hat{P} = \dfrac{1}{n}\sum_{i=1}^{n}\delta_{\mathbf{x}_{i}^{(P)}}$$ 

<a href="../dictionary_learning/measures.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DatasetMeasure.__init__`

```python
__init__(features, transforms=None, batch_size=64, device='cpu')
```

Initializes a DatasetMeasure object. 



**Args:**
 
 - <b>`features`</b>:  numpy array containing raw data. 
 - <b>`transforms`</b>:  pre-processing steps for data samples. 
 - <b>`batch_size`</b>:  size of batches to be sampled from the support $\mathbf{X}^{(P)}$ 
 - <b>`device`</b>:  either 'cpu' or 'gpu'. 




---

<a href="../dictionary_learning/measures.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DatasetMeasure.sample`

```python
sample(n=None)
```

Samples $n$ points from the measure support. 



**Args:**
 
 - <b>`n`</b>:  if given, samples $n$ samples from the support. If $n$ is None, then samples self.batch_size samples. 


---

<a href="../dictionary_learning/measures.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LabeledDatasetMeasure`
Labeled Dataset Measure class. This corresponds to an EmpiricalMeasure with i.i.d. hypothesis, namely, 

$$\hat{P} = \dfrac{1}{n}\sum_{i=1}^{n}\delta_{(\mathbf{x}_{i}^{(P)},y_{i}^{(P)})}$$ 

<a href="../dictionary_learning/measures.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledDatasetMeasure.__init__`

```python
__init__(
    features,
    labels,
    transforms=None,
    batch_size=64,
    n_classes=None,
    stratify=False,
    device='cpu'
)
```

Initializes a LabeledDatasetMeasure object. 



**Args:**
 
 - <b>`features`</b>:  numpy array containing raw data. 
 - <b>`labels`</b>:  numpy array containing the labels of each sample. 
 - <b>`transforms`</b>:  pre-processing steps for data samples. 
 - <b>`batch_size`</b>:  size of batches to be sampled from the support $\mathbf{X}^{(P)}$ 
 - <b>`n_classes`</b>:  number of classes in the labels array. If not given, infers automatically by searching the unique  values in labels. 
 - <b>`stratify`</b>:  whether or not stratify mini-batches. If mini-batches are stratified, classes are balanced on  each mini-batch. 
 - <b>`device`</b>:  either 'cpu' or 'gpu'. 




---

<a href="../dictionary_learning/measures.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LabeledDatasetMeasure.sample`

```python
sample(n=None)
```

Samples $n$ points from the measure support. 



**Args:**
 
 - <b>`n`</b>:  if given, samples $n$ samples from the support. If $n$ is None, then samples self.batch_size samples. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

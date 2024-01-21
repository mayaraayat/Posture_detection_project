<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/mapping.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `mapping`






---

<a href="../dictionary_learning/mapping.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BarycentricMapping`




<a href="../dictionary_learning/mapping.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BarycentricMapping.__init__`

```python
__init__(reg=0.0, num_iter_sinkhorn=50)
```








---

<a href="../dictionary_learning/mapping.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BarycentricMapping.fit`

```python
fit(XP, XQ, p=None, q=None)
```





---

<a href="../dictionary_learning/mapping.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BarycentricMapping.forward`

```python
forward(XP, XQ, p=None, q=None)
```






---

<a href="../dictionary_learning/mapping.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SupervisedBarycentricMapping`




<a href="../dictionary_learning/mapping.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SupervisedBarycentricMapping.__init__`

```python
__init__(reg=0.0, num_iter_sinkhorn=50, label_importance=None)
```








---

<a href="../dictionary_learning/mapping.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SupervisedBarycentricMapping.fit`

```python
fit(XP, XQ, YP, YQ, p=None, q=None)
```





---

<a href="../dictionary_learning/mapping.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SupervisedBarycentricMapping.forward`

```python
forward(XP, XQ, YP, YQ, p=None, q=None)
```






---

<a href="../dictionary_learning/mapping.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearOptimalTransportMapping`




<a href="../dictionary_learning/mapping.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearOptimalTransportMapping.__init__`

```python
__init__(reg=1e-06)
```








---

<a href="../dictionary_learning/mapping.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearOptimalTransportMapping.dist`

```python
dist()
```





---

<a href="../dictionary_learning/mapping.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearOptimalTransportMapping.fit`

```python
fit(XP, XQ)
```





---

<a href="../dictionary_learning/mapping.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearOptimalTransportMapping.forward`

```python
forward(XP, XQ, p=None, q=None)
```






---

<a href="../dictionary_learning/mapping.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `OracleMapping`




<a href="../dictionary_learning/mapping.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `OracleMapping.__init__`

```python
__init__(mP, mQ, sP, sQ, reg=1e-06)
```








---

<a href="../dictionary_learning/mapping.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `OracleMapping.fit`

```python
fit()
```





---

<a href="../dictionary_learning/mapping.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `OracleMapping.forward`

```python
forward(XP, XQ)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

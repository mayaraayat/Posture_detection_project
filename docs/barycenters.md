<!-- markdownlint-disable -->

<p align="center">
  <img src="../assets/dadil.png" width="300"/>
</p>

<a href="../dictionary_learning/barycenters.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `barycenters`
Module for computing Wasserstein barycenters 

## References 

[Cuturi and Doucet, 2014] Cuturi, M., & Doucet, A. (2014, June). Fast computation of Wasserstein barycenters.  In International conference on machine learning (pp. 685-693). PMLR. 

[Álvarez-Esteban et al., 2016] Álvarez-Esteban, P. C., Del Barrio, E., Cuesta-Albertos, J. A., & Matrán, C. (2016).  A fixed-point approach to barycenters in Wasserstein space. Journal of Mathematical  Analysis and Applications, 441(2), 744-762. 

[Cuturi, 2013] Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.  In Advances in neural information processing systems, 26. 


---

<a href="../dictionary_learning/barycenters.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `legacy_wasserstein_barycenter`

```python
legacy_wasserstein_barycenter(
    XP,
    YP,
    ε=0.0,
    α=None,
    β=None,
    num_iter_max=100,
    num_iter_sinkhorn=1000,
    τ=0.0001,
    verbose=False,
    inner_verbose=False,
    initialization='random',
    penalize_labels=False,
    device='cpu',
    log=False,
    covariance_type='full',
    label_metric='l2'
)
```






---

<a href="../dictionary_learning/barycenters.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `wasserstein_barycenter`

```python
wasserstein_barycenter(
    XP,
    YP=None,
    n_samples=None,
    ε=0.0,
    α=None,
    β=None,
    num_iter_max=100,
    num_iter_sinkhorn=1000,
    τ=0.0001,
    verbose=False,
    inner_verbose=False,
    initialization='random',
    propagate_labels=False,
    penalize_labels=False,
    device='cpu',
    log=False,
    covariance_type='full',
    label_metric='l2'
)
```

Computes the Wasserstein Barycenter 

$$ \hat{B}( \alpha, \mathcal{P} ) = \underset{P}{ \text{argmin} }\sum_{k=1}^{K} \alpha_{k} W_{c}(P, P_{k} ),$$ 

for a list of distributions $\mathcal{P}$, containing $\hat{P}_{1}, \cdots ,\hat{P}_{K}$ and weights $\alpha \in \Delta_{K}$. Each distribution is parametrized through their support $\mathbf{X}^{( P_{k} )}, k=1, \cdots ,K$. This consists on a implementation of the Free-Support Wasserstien Barycenter of [[Cuturi and Doucet, 2014]](https://proceedings.mlr.press/v32/cuturi14.html). Our implementation relies on the fixed-point iteration of [[Alvarez-Esteban et al., 2016]](https://arxiv.org/abs/1511.05355), 

$$ \hat{B}^{(it+1)} = \psi( \hat{B}^{(it)} ), $$ 

where $\psi(\hat{P}) = T_{it,\sharp}\hat{P}$, $T_{it} = \sum_{k}\alpha_{k}T_{k,it}$, for $T_{k,it}$, the  barycentric mapping between $\hat{P}_{k}$ and $\hat{B}^{(it)}$. 



**Args:**
 
 - <b>`XP`</b>:  List of numpy arrays/torch tensors containing the support of each measure 
 - <b>`YP`</b>:  Optional. List of numpy arrays/torch tensors containing the labels of each support point. 
 - <b>`n_samples`</b>:  Optional. Number of points in the Barycenter's support. If not given, assumes that barycenter is  supported on sum(nk) points, for nk being the number of points the support of the k-th measure. 
 - <b>`ϵ`</b>:  Optional. 0 if not given. Entropic Regularization penalty. For ϵ, OT is solved using the Sinkhorn algorithm of [[Cuturi, 2013]](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html). 
 - <b>`num_iter_max`</b>:  Optional. Maximum number of iterations for computing the barycenter. 
 - <b>`num_iter_sinkhorn`</b>:  Optional. When ϵ > 0, maximum number of iterations for the Sinkhorn algorithm. Ignored if ϵ = 0.0. 
 - <b>`τ`</b>:  Optional. Tolerance between barycentric iterations. If $\lVert \mathbf{X}^{(B^{(it+1)})} - \mathbf{X}^{(B^{(it)})} \rVert_{2}^{2} < \tau$, the algorithm stops. 
 - <b>`verbose`</b>:  Optional. If True, displays information about barycenter iterations 
 - <b>`inner_verbose`</b>:  Optional. If True, displays information about Sinkhorn's iterations 
 - <b>`initialization`</b>:  Optional. Either 'random' or 'samples'. If 'random', initializes the Barycenter support  randomly, i.e., $\mathbf{x}_{i}^{(B^{(0)})} \sim \mathcal{N}(\mathbf{0},\mathbf{I}_{d})$.  If 'samples', sub-samples the support of each measure for creating the initial barycenter  support 
 - <b>`propagate_labels`</b>:  Optional. If True, uses Wasserstein propagation for generating barycenter labels.  Ignored if labels are not provided. 
 - <b>`penalize_labels`</b>:  Optional.  If True, penalizes OT plans that mix classes 
 - <b>`device`</b>:  Optional. Either 'cpu' or 'cuda'. 'cuda' should only be provided if GPUs are available. 
 - <b>`log`</b>:  Optional. If True, saves additional information in a dictionary. 


---

<a href="../dictionary_learning/barycenters.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `linear_wasserstein_barycenter`

```python
linear_wasserstein_barycenter(
    XP,
    YP=None,
    n_samples=None,
    ε=0.0,
    α=None,
    num_iter_max=100,
    τ=0.0001,
    verbose=False,
    initialization='random',
    propagate_labels=False,
    penalize_labels=False,
    device='cpu',
    log=False
)
```






---

<a href="../dictionary_learning/barycenters.py#L325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bures_wasserstein_barycenter`

```python
bures_wasserstein_barycenter(
    means,
    covs,
    weights,
    num_iter_max=10,
    tol=1e-06,
    verbose=False,
    extra_ret=False,
    return_cost=False
)
```






---

<a href="../dictionary_learning/barycenters.py#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `barycenter_gmms`

```python
barycenter_gmms(gmms, weights, tol_solution=1e-09, **kwargs)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

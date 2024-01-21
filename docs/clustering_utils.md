<!-- markdownlint-disable -->

# <kbd>module</kbd> `clustering_utils`






---

## <kbd>class</kbd> `clusters`




### <kbd>method</kbd> `__init__`

```python
__init__(data, labels, num_clusters)
```

Initialize the ClusteringPseudoLabels class. 



**Args:**
 
 - <b>`data`</b> (numpy.ndarray):  The data to be clustered. 
 - <b>`labels`</b> (numpy.ndarray):  The labels corresponding to the data. 
 - <b>`num_clusters`</b> (int):  The number of clusters to create. 




---

### <kbd>method</kbd> `cluster_data`

```python
cluster_data()
```

Cluster data based on provided cluster labels. 

This method groups data points into clusters based on the cluster labels provided in 'self.labels'. 

---

### <kbd>method</kbd> `clusters_mapping`

```python
clusters_mapping(target_data)
```

Map labels from source to target domain. 



**Args:**
 
 - <b>`target_data`</b> (list):  A list of target domain cluster tensors. 



**Returns:**
 
 - <b>`numpy.ndarray`</b>:  Mapped labels for the target domain. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

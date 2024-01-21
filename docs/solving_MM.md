<!-- markdownlint-disable -->

# <kbd>module</kbd> `solving_MM`





---

## <kbd>function</kbd> `solve_multimarginal_optimal_transport`

```python
solve_multimarginal_optimal_transport(cost_matrix)
```

Solves the multimarginal optimal transport problem based on the given cost matrix. 



**Args:**
 
 - <b>`cost_matrix`</b> (ndarray):  The cost matrix representing the transportation costs between clusters. 



**Returns:**
 
 - <b>`list`</b>:  A list of tuples representing the optimal mapping between source and target clusters.  Each tuple contains the index of the source cluster and the index of the target cluster. 


---

## <kbd>function</kbd> `plot_objective_history`

```python
plot_objective_history(objective_history)
```

Plots the objective function value history. 



**Args:**
 
 - <b>`objective_history`</b> (list):  List of objective function values at each iteration. 



**Returns:**
 None 


---

## <kbd>function</kbd> `visualize_optimal_mapping`

```python
visualize_optimal_mapping(solution, cost_matrix)
```

Visualizes the optimal mapping between source and target clusters. 



**Args:**
 
 - <b>`solution`</b> (list):  List of tuples representing the optimal mapping. 
 - <b>`cost_matrix`</b> (ndarray):  The cost matrix. 



**Returns:**
 None 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

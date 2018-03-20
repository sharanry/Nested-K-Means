#  Nested Mini-Batch K-Means

Implementation of Nested Mini-Batch K-Means in numpy - Newling et al.

[arxiv 1602.02934](https://arxiv.org/abs/1602.02934)

## Usage
1) Import the class
```python
from NestedKMeans import NestedKMeans
```
2) Initialize the model
```python
nkmeans = NestedKMeans(<datapoints>, <No. of Clusters>, <No. of starting Points>>, rho=<>, earlyStop=<To stop early assign True>)
```

3) Train
```python
nkmeans.train(<No. of max iters>)
```

4) Show the trained result
```python
# works only for 2 dimensions
nkmeans.show()
```


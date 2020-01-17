![](https://github.com/gmodena/tensorfm/workflows/build/badge.svg)

# tensor-fm

A sklearn compatible order 2 Factorization Machine, implemented atop TensorFlow 2.
The algorithm is described in http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf. For an higher level
overview of the method see http://nowave.it/factorization-machines-with-tensorflow.html.

This package is a port of the code presented in the blog post to Tensorflow 2. The goal of this project is
to be and experimentation framework for different optimization strategies on classical ML models.

## Install

```
pip install git+https://github.com/gmodena/tensor-fm
```

## Use

`tensorfm.sklearn` exposes two sklearn compatible estimators: `FactorizationMachineRegressor`
and `FactorizationMachineClassifier`.

Example
```
from tensorfm.sklearn import FactorizationMachineRegressor
...
fm = FactorizationMachineRegressor()
fm.fit(X, y)
fm.predict(X)
```

See also `examples/movielens.py`

## Limitations and known issues

Operations on sparse matrices are currently not supported.
Training continues till `max_iter` is reached, we should stop if performance does not improve for a certain number
of iterations.

## Performance

All parameters and settings being equal, I noticed a considerable performance degradation (MSE on train/test)
on movielens compared to the tensorflow 1 implementation from http://nowave.it/factorization-machines-with-tensorflow.html

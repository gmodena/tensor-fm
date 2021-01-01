![](https://github.com/gmodena/tensor-fm/workflows/build/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tensor-fm/badge/?version=latest)](https://tensor-fm.readthedocs.io/en/latest/?badge=latest)

# tensor-fm

A scikit-learn compatible order 2 Factorization Machine, implemented atop TensorFlow 2.
The algorithm is described in http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf. For an higher level
overview of the method see http://nowave.it/factorization-machines-with-tensorflow.html.

This package is a port to Tensorflow 2 of the code presented in that blog post. The goal of this project is
to experiment with different optimization strategies for classical ML models, and scalability of
TF2 backends.

## Install

The latest development version of `tensorfm` can be installed from its
[github repo](git+https://github.com/gmodena/tensor-f) with:
```
pip install git+https://github.com/gmodena/tensor-fm
```

## Usage examples

Tensorlow and scikit-learn APIs are provided.

### Tensorflow 

The tensorflow implementation of Factorization Machines lives under `tensor-fm/tensorfm/base.py`.
An example of how to work with this API can be found in `tensor-fm/tests/test_base.py`. 

### Scikit-learn estimator
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

## Performance

All parameters and settings being equal, I noticed a considerable performance degradation of
`FactorizationMachineRegressor` (MSE on train/test) on movielens compared to the tensorflow 1 implementation
from http://nowave.it/factorization-machines-with-tensorflow.html.
Possibly related, a test in the `check_regressors_train` suite (`sklearn`) fails due to a low `R^2`. As a workaround
`FactorizationMachineRegressor` sets the `poor_score` tag to `True`.

## Limitations and known issues

Operations on sparse matrices are currently not supported.
Training continues till `max_iter` is reached, we should stop if performance does not improve for a certain number
of iterations.


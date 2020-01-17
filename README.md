# tensor-fm

A sklearn compatible order 2 Factorization Machine, implemented atop TensorFlow 2.
The algorithm is described in http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf. For an higher level
overview of the method see http://nowave.it/factorization-machines-with-tensorflow.html.


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

See also `examples/movielense.py`

## Limitations
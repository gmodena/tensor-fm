import pytest
import tensorflow as tf
from tensorfm.base import FactorizationMachine
from tensorfm.base import check_X_y, check_X
from tensorfm.sklearn import FactorizationMachineClassifier

import numpy as np


# Example dummy data from Rendle 2010
x_data = np.array([
    #     Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
    #    A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
])
y_data = np.array([5, 3, 1, 4, 5, 1, 5])
y_data.shape += (1, )


def test_base_fit():
    train_dataset = check_X_y(x_data, y_data)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=tf.constant(0.01))

    model = FactorizationMachine(train_dataset, optimizer=optimizer)
    clf = model.fit()
    assert clf is not None

    # predict multiple instances
    r = model.predict(check_X(x_data))
    assert r.shape == [x_data.shape[0], 1]

    # predict single instance
    x = check_X(x_data[0])
    pred = model.predict(x)
    assert pred.shape == [1, ]


def test_sklearn_classifier():
    clf = FactorizationMachineClassifier()

    assert clf is not None

    clf.fit(x_data, y_data)
    pred = clf.predict(x_data)


import pytest
import tensorflow as tf
from tensorfm.base import FactorizationMachine

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
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((tf.reshape(x_data, [-1, x_data.shape[1]]), y_data))
            .batch(200)
            .shuffle(1000)
    )

    model = FactorizationMachine(train_dataset)
    model.fit()
    print(model.predict(tf.cast(np.array([
        #     Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
        #    A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
        [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ]]), tf.float32)))

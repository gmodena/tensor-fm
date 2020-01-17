import numpy as np
import tensorflow as tf
import pytest

seed = 12345
dtype = np.float32
eta = 0.001


@pytest.fixture(scope="module")
def optimizer(request):
    return tf.keras.optimizers.Adam(learning_rate=tf.constant(eta))


@pytest.fixture(scope="module")
def rendle_dataset(request):
    # Example dummy data from Rendle 2010
    X = np.array([
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
    y = np.array([5, 3, 1, 4, 5, 1, 5])
    y.shape += (1, )
    return X, y

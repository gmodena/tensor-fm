import pytest
from tensorfm.base import train, fm
from tensorfm.base import to_tf_shuffled_dataset, to_tf_dataset_X
from tensorflow.keras.losses import MSE, binary_crossentropy


def test_base_regr(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)

    w0, W, V = train(train_dataset, optimizer=optimizer, loss=MSE)
    assert w0 is not None
    assert W is not None
    assert V is not None

    # predict multiple instances
    pred = fm(to_tf_dataset_X(x_data), w0, W, V)
    assert pred.shape == [x_data.shape[0], 1]

    # predict single instance
    x = to_tf_dataset_X(x_data[0])
    pred = fm(x, w0, W, V)
    assert pred.shape == [1, ]


def test_base_clf(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)

    w0, W, V = train(train_dataset, optimizer=optimizer, loss=binary_crossentropy)
    assert w0 is not None
    assert W is not None
    assert V is not None

    # predict multiple instances
    r = fm(to_tf_dataset_X(x_data), w0, W, V)
    assert r.shape == [x_data.shape[0], 1]

    # predict single instance
    x = to_tf_dataset_X(x_data[0])
    pred = fm(x, w0, W, V)
    assert pred.shape == [1, ]


def test_base_regr_invalid_param(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)
    with pytest.raises(ValueError):
        train(train_dataset, num_factors=0, optimizer=optimizer, loss=MSE)
        train(train_dataset, num_factors=-1, optimizer=optimizer, loss=MSE)
        train(train_dataset, max_iter=0, optimizer=optimizer, loss=MSE)
        train(train_dataset, max_iter=-1, optimizer=optimizer, loss=MSE)
        train(train_dataset, optimizer=optimizer, loss=MSE, C=0)
        train(train_dataset, optimizer=optimizer, loss=MSE, C=-1)
        train(train_dataset, optimizer=optimizer, loss=None)

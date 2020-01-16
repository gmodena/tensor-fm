import pytest
from tensorfm.base import fit_2d_fm, model
from tensorfm.base import to_tf_shuffled_dataset, to_tf_dataset_X
from tensorfm.base import l1_norm, l2_norm
from tensorflow.keras.losses import MSE, binary_crossentropy


def test_base_regr(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)

    w0, W, V = fit_2d_fm(train_dataset, optimizer=optimizer, loss=MSE)
    assert w0 is not None
    assert W is not None
    assert V is not None

    # predict multiple instances
    pred = model(to_tf_dataset_X(x_data), w0, W, V)
    assert pred.shape == [x_data.shape[0], 1]

    # predict single instance
    x = to_tf_dataset_X(x_data[0])
    pred = model(x, w0, W, V)
    assert pred.shape == [1, ]


def test_base_clf(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)

    w0, W, V = fit_2d_fm(train_dataset, optimizer=optimizer, loss=binary_crossentropy)
    assert w0 is not None
    assert W is not None
    assert V is not None

    # predict multiple instances
    r = model(to_tf_dataset_X(x_data), w0, W, V)
    assert r.shape == [x_data.shape[0], 1]

    # predict single instance
    x = to_tf_dataset_X(x_data[0])
    pred = model(x, w0, W, V)
    assert pred.shape == [1, ]


def test_base_regr_invalid_param(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)
    with pytest.raises(ValueError):
        fit_2d_fm(train_dataset, num_factors=0, optimizer=optimizer, loss=MSE)
        fit_2d_fm(train_dataset, num_factors=-1, optimizer=optimizer, loss=MSE)
        fit_2d_fm(train_dataset, max_iter=0, optimizer=optimizer, loss=MSE)
        fit_2d_fm(train_dataset, max_iter=-1, optimizer=optimizer, loss=MSE)
        fit_2d_fm(train_dataset, optimizer=optimizer, loss=MSE, C=0)
        fit_2d_fm(train_dataset, optimizer=optimizer, loss=MSE, C=-1)
        fit_2d_fm(train_dataset, optimizer=optimizer, loss=None)

import pytest
from tensorfm.base import FactorizationMachine
from tensorfm.base import to_tf_shuffled_dataset, to_tf_dataset_X
from tensorfm.base import l1_norm, l2_norm
from tensorflow.keras.losses import MSE, binary_crossentropy


def test_base_regr(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)

    model = FactorizationMachine(optimizer=optimizer, loss=MSE)
    regr = model.fit(train_dataset)
    assert regr is not None

    # predict multiple instances
    pred = model.predict(to_tf_dataset_X(x_data))
    assert pred.shape == [x_data.shape[0], 1]

    # predict single instance
    x = to_tf_dataset_X(x_data[0])
    pred = model.predict(x)
    assert pred.shape == [1, ]


def test_base_clf(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)

    model = FactorizationMachine(optimizer=optimizer, loss=binary_crossentropy)
    clf = model.fit(train_dataset)
    assert clf is not None

    # predict multiple instances
    r = model.predict(to_tf_dataset_X(x_data))
    assert r.shape == [x_data.shape[0], 1]

    # predict single instance
    x = to_tf_dataset_X(x_data[0])
    pred = model.predict(x)
    assert pred.shape == [1, ]


def test_base_regr_invalid_param(rendle_dataset, optimizer):
    x_data, y_data = rendle_dataset
    train_dataset = to_tf_shuffled_dataset(x_data, y_data)
    with pytest.raises(ValueError):
        FactorizationMachine(num_factors=0, optimizer=optimizer, loss=MSE).fit(train_dataset)
        FactorizationMachine(num_factors=-1, optimizer=optimizer, loss=MSE).fit(train_dataset)
        FactorizationMachine(max_iter=0, optimizer=optimizer, loss=MSE).fit(train_dataset)
        FactorizationMachine(max_iter=-1, optimizer=optimizer, loss=MSE).fit(train_dataset)
        FactorizationMachine(optimizer=optimizer, loss=MSE, C=0).fit(train_dataset)
        FactorizationMachine(optimizer=optimizer, loss=MSE, C=-1).fit(train_dataset)
        FactorizationMachine(optimizer=optimizer, loss=None).fit(train_dataset)

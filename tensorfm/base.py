import tensorflow as tf
import numpy as np

TF_DATASET_BATCH_SIZE = 30000


def to_tf_dataset(
    X, y, dtype=tf.float32, batch_size=TF_DATASET_BATCH_SIZE, shuffle_buffer_size=None
):
    if isinstance(X, (list, tuple)):
        X = np.array(X)
    if isinstance(y, (list, tuple)):
        y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            " The number of training "
            "examples is not the same as the number of "
            "labels"
        )
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X, dtype=dtype), tf.cast(y, dtype=dtype))
    ).batch(batch_size)

    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    return dataset


def to_tf_dataset_X(X, dtype=tf.float32):
    X = np.array(X)
    return tf.cast(X, dtype=dtype)


def check_loss(loss, penalty):
    if not loss:
        raise ValueError(
            f"must specify a loss function. Either tensorflow.keras.losses.MSE"
            f"or tensorflow.keras.losses.binary_crossentropy are supported"
        )
    if loss.__name__ not in ("mean_squared_error", "binary_crossentropy"):
        raise ValueError(f"{loss.__name__} function not supported")
    if penalty and penalty.__name__ not in ("l1_norm", "l2_norm"):
        raise ValueError(f"{penalty.__name__} not supported")


def l1_norm(V, W, lambda_=0.001):
    l1_norm = tf.reduce_sum(
        tf.add(tf.multiply(lambda_, tf.abs(W, 2)), tf.multiply(lambda_, tf.abs(V, 2)))
    )
    return l1_norm


def l2_norm(V, W, lambda_=0.001):
    l2_norm = tf.reduce_sum(
        tf.add(tf.multiply(lambda_, tf.pow(W, 2)), tf.multiply(lambda_, tf.pow(V, 2)))
    )
    return l2_norm


def fm(X, w0, W, V):
    linear_terms = X * W
    interactions = tf.subtract(
        tf.pow(tf.tensordot(X, tf.transpose(V), 1), 2),
        tf.tensordot(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)), 1),
    )

    if X.ndim > 1:
        linear_terms = tf.reduce_sum(linear_terms, 1, keepdims=True)
        interactions = tf.reduce_sum(interactions, 1, keepdims=True)

    else:
        # One dimensional data: e.g. passed when we call fm() for inference
        linear_terms = tf.reduce_sum(linear_terms)
        interactions = tf.reduce_sum(interactions)

    return w0 + linear_terms + 0.5 * interactions


def train(
    train_dataset,
    num_factors=2,
    max_iter=10,
    penalty=None,
    C=1.0,
    loss=None,
    optimizer=None,
    activation=None,
    loss_kwargs={},
    random_state=None,
    dtype=tf.float32,
):
    """Fit a degree 2 polynomial factorization machine, implemented atop tensorflow 2.
       This class contains the generic code for training a FM. Regressors and classifiers can be learnt
       by minimizing appropriate loss functions (e.g. MSE or cross entropy)."""
    tf.random.set_seed(random_state)
    check_loss(loss, penalty)
    if C < 0:
        raise ValueError(f"Inverse regularization term must be positive; got (C={C})")
    if max_iter < 1:
        raise ValueError(f"max_iter must be > zero. Got {max_iter}")
    if num_factors < 1:
        raise ValueError(f"num_factors must be >= 1. Got {num_factors}")

    # Get the number of feature columns
    p = train_dataset.element_spec[0].shape[1]
    # bias and weights
    w0 = tf.Variable(tf.zeros([1], dtype=dtype))
    W = tf.Variable(tf.zeros([p], dtype=dtype))
    # interaction factors, randomly initialized
    V = tf.Variable(
        tf.random.normal([num_factors, p], stddev=0.01, dtype=dtype, seed=random_state)
    )

    for epoch_count in range(max_iter):
        for (x, y) in train_dataset:
            with tf.GradientTape() as tape:
                pred = fm(x, w0, W, V)
                if activation:
                    pred = activation(pred)
                loss_ = loss(y, pred, **loss_kwargs)
                if penalty:
                    loss_ += penalty(V, W, lambda_=1.0 / C)

            # Update gradients
            grads = tape.gradient(loss_, [w0, W, V])
            optimizer.apply_gradients(zip(grads, [w0, W, V]))
    return w0, W, V

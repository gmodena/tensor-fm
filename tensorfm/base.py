from . import logger
import tensorflow as tf


def l1_norm(V, W, lambda_=0.001):
    l1_norm = tf.reduce_sum(
        tf.add(tf.multiply(lambda_, tf.abs(W)), tf.multiply(lambda_, tf.abs(V)))
    )
    return l1_norm


def l2_norm(V, W, lambda_=0.001):
    l2_norm = tf.reduce_sum(
        tf.add(tf.multiply(lambda_, tf.pow(W, 2)), tf.multiply(lambda_, tf.pow(V, 2)))
    )
    return l2_norm


def noop_norm(V, W, lambda_=None):
    return 0


def mse(y, y_hat):
    return tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))


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
    random_state=None,
    dtype=tf.float32,
):
    """Fit a degree 2 polynomial factorization machine, implemented atop tensorflow 2.
       This class contains the generic code for training a FM. Regressors and classifiers can be learnt
       by minimizing appropriate loss functions (e.g. MSE or cross entropy)."""
    tf.random.set_seed(random_state)
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
        tf.random.normal(
            [num_factors, p], mean=0.0, stddev=0.01, dtype=dtype, seed=random_state
        )
    )

    for epoch_count in range(max_iter):
        for batch, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                pred = fm(x, w0, W, V)
                loss_ = loss(y, pred) + penalty(V, W, lambda_=1.0 / C)
                # Update gradients
            grads = tape.gradient(loss_, [w0, W, V])
            optimizer.apply_gradients(zip(grads, [w0, W, V]))
            logger.debug(f"Epoch: {epoch_count}, batch: {batch} loss:, {loss_.numpy()}")
    return w0, W, V

import tensorflow as tf
import numpy as np

TF_DATASET_BATCH_SIZE = 200
TF_DATASET_SHUFFLE = 1000


def to_tf_shuffled_dataset(X, y, dtype=tf.float32):
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError(" The number of training "
                     "examples is not the same as the number of "
                     "labels")
    dataset = (
        tf.data.Dataset.from_tensor_slices((tf.reshape(tf.cast(X, dtype=dtype), [-1, X.shape[1]]),
                                            tf.cast(y, dtype=dtype)))
            .batch(TF_DATASET_BATCH_SIZE)
            .shuffle(TF_DATASET_SHUFFLE))

    return dataset


def to_tf_dataset_X(X, dtype=tf.float32):
    X = np.array(X)
    return tf.cast(X, dtype=dtype)


def check_loss(loss, penalty):
    if not loss:
        raise ValueError(f"must specify a loss function. Either tensorflow.keras.losses.MSE"
                         f"or tensorflow.keras.losses.binary_crossentropy are supported")
    if loss.__name__ not in ('mean_squared_error', 'binary_crossentropy'):
        raise ValueError(f"{loss.__name__} function not supported")
    if penalty and penalty.__name__ not in ('l1_norm', 'l2_norm'):
        raise ValueError(f"{penalty.__name__} not supported")


def mse(y, y_hat):
    return tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))


def l1_norm(V, W, lambda_=0.001):
    l1_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_, tf.cast(tf.abs(W, 2), tf.float32)),
            tf.multiply(lambda_, tf.cast(tf.abs(V, 2), tf.float32)))))


    return l1_norm


def l2_norm(V, W, lambda_=0.001):
    l2_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_, tf.cast(tf.pow(W, 2), tf.float32)),
            tf.multiply(lambda_, tf.cast(tf.pow(V, 2), tf.float32)))))

    return l2_norm


def model(X, w0, W, V):
    linear_terms = W * X
    interactions = tf.subtract(
        tf.pow(tf.tensordot(X, tf.transpose(V), 1), 2),
        tf.tensordot(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)), 1))

    if X.ndim > 1:
        linear_terms = tf.reduce_sum(linear_terms, 1, keepdims=True)
        interactions = tf.reduce_sum(interactions, 1, keepdims=True)

    else:
        linear_terms = tf.reduce_sum(linear_terms) #tf.tensordot(X, tf.transpose(self.W_), 1)
        interactions = tf.reduce_sum(interactions)

    return w0 + linear_terms + 0.5 * interactions

def fit_2d_fm(train_dataset, num_factors=2, max_iter=100, penalty=None, C=1.0, loss=None,
        optimizer=None, activation=None, loss_kwargs={}, random_state=None):
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
    w0 = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.zeros([p]))
    # interaction factors, randomly initialized
    V = tf.Variable(
        tf.random.normal([num_factors, p], stddev=0.01, dtype=tf.dtypes.float32, seed=random_state))

    for epoch_count in range(max_iter):
        for (x, y) in train_dataset:
            with tf.GradientTape() as tape:
                pred = model(x, w0, W, V)
                if activation:
                    pred = activation(pred)
                loss_ = loss(y, pred, **loss_kwargs)
                if penalty:
                    loss_ += penalty(V, W, lambda_=1.0 / C)

            # Update gradients
            grads = tape.gradient(loss_, [W, w0])
            optimizer.apply_gradients(zip(grads, [W, w0]))
    return w0, W, V

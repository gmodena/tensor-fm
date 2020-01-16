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
    if loss.__name__ not in ('mean_squared_error', 'binary_crossentropy', 'sigmoid_cross_entropy_with_logits_v2'):
        raise ValueError(f"{loss.__name__} function not supported")
    if penalty and penalty.__name__ not in ('l1_norm', 'l2_norm'):
        raise ValueError(f"{penalty.__name__} not supported")


def mse(y, y_hat):
    return tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))


def l1_norm(y, y_hat, V, W, lambda_=0.001):
    l1_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_, tf.cast(tf.abs(W, 2), tf.float32)),
            tf.multiply(lambda_, tf.cast(tf.abs(V, 2), tf.float32)))))


    return l1_norm


def l2_norm(y, y_hat, V, W, lambda_=0.001):
    l2_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_, tf.cast(tf.pow(W, 2), tf.float32)),
            tf.multiply(lambda_, tf.cast(tf.pow(V, 2), tf.float32)))))

    return l2_norm


class FactorizationMachine:
    """A 2 polynomial factorization machine, implemented atop tensorflow 2.
    This class contains the generic code for training a FM. Regressors and classifiers can be learnt
    by minimizing appropriate loss functions (e.g. MSE or cross entropy).
    """
    def __init__(self, num_factors=2, max_iter=100, penalty=None, C=1.0, loss=None,
                 optimizer=None, activation=None, loss_kwargs={}, random_state=None):
        self.num_factors = num_factors
        self.max_iter = max_iter
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.optimizer = optimizer
        self.activation = activation
        self.loss_kwargs = loss_kwargs

        # bias, weight and latent factors
        self.w0_ = None
        self.W_ = None
        self.V_ = None

        tf.random.set_seed(random_state)
        self.random_state = random_state

    def model(self, X, activation=None):
        linear_terms = self.W_ * X
        interactions = tf.subtract(
            tf.pow(tf.tensordot(X, tf.transpose(self.V_), 1), 2),
            tf.tensordot(tf.pow(X, 2), tf.transpose(tf.pow(self.V_, 2)), 1))

        if X.ndim > 1:
            linear_terms = tf.reduce_sum(linear_terms, 1, keepdims=True)
            interactions = tf.reduce_sum(interactions, 1, keepdims=True)

        else:
            linear_terms = tf.reduce_sum(linear_terms) #tf.tensordot(X, tf.transpose(self.W_), 1)
            interactions = tf.reduce_sum(interactions)

        y_hat = self.w0_ + linear_terms + 0.5 * interactions
        if activation:
            y_hat = activation(y_hat)
        return y_hat

    def fit(self, train_dataset):
        check_loss(self.loss, self.penalty)
        if self.C < 0:
            raise ValueError(f"Inverse regularization term must be positive; got (C={self.C})")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be > zero. Got {self.max_iter}")
        if self.num_factors < 1:
            raise ValueError(f"num_factors must be >= 1. Got {self.num_factors}")

        # Get the number of feature columns
        p = train_dataset.element_spec[0].shape[1]
        # bias and weights
        self.w0_ = tf.Variable(tf.zeros([1]))
        self.W_ = tf.Variable(tf.zeros([p]))
        # interaction factors, randomly initialized
        self.V_ = tf.Variable(
            tf.random.normal([self.num_factors, p], stddev=0.01, dtype=tf.dtypes.float32, seed=self.random_state))

        for epoch_count in range(self.max_iter):
            for (x, y) in train_dataset:
                with tf.GradientTape() as tape:
                    pred = self.model(x)
                    loss = self.loss(y, pred, **self.loss_kwargs)
                    if self.penalty:
                        loss += self.penalty(y, pred, self.V_, self.W_, lambda_=1.0 / self.C)

                # Update gradients
                grads = tape.gradient(loss, [self.W_, self.w0_])
                self.optimizer.apply_gradients(zip(grads, [self.W_, self.w0_]))
        return self

    def predict(self, X):
        return self.model(X)
import tensorflow as tf

TF_DATASET_BATCH_SIZE = 200
TF_DATASET_SHUFFLE = 1000


def check_X_y(X, y, dtype=tf.float32):
    dataset = (
        tf.data.Dataset.from_tensor_slices((tf.reshape(tf.cast(X, dtype=dtype), [-1, X.shape[1]]),
                                            tf.cast(y, dtype=dtype)))
            .batch(TF_DATASET_BATCH_SIZE)
            .shuffle(TF_DATASET_SHUFFLE))

    return dataset


def check_X(X, dtype=tf.float32):
    return tf.cast(X, dtype=dtype)


def mse(y, y_hat):
    return tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))


def _l1_loss(y, y_hat, V, W):
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l1_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.cast(tf.abs(W, 2), tf.float32)),
            tf.multiply(lambda_v, tf.cast(tf.abs(V, 2), tf.float32)))))


    return l1_norm


def _l2_loss(y, y_hat, V, W):
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.cast(tf.pow(W, 2), tf.float32)),
            tf.multiply(lambda_v, tf.cast(tf.pow(V, 2), tf.float32)))))

    return l2_norm

class FactorizationMachine:
    """Base class for factorization machines.
    """
    def __init__(self, train_dataset, k=5, epochs=100, optimizer=None, loss=None):
        self.k = k
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer

        # Get the number of feature columns
        p = train_dataset.element_spec[0].shape[1]
        self.train_dataset = train_dataset

        # bias and weights
        self.w0_ = tf.Variable(tf.zeros([1]))
        self.W_ = tf.Variable(tf.zeros([p]))

        # interaction factors, randomly initialized
        self.V_ = tf.Variable(tf.random.normal([self.k, p], stddev=0.01, dtype=tf.dtypes.float32))


    def _validate_params(self):
        if self.epochs < 1:
            raise ValueError("epoch must be > 0")
        if self.k < 1:
            raise ValueError("k must be > 0")

    def model(self, X):

        if X.ndim > 1:
            linear_terms = tf.add(self.w0_,
                                  tf.reduce_sum(
                                      tf.multiply(self.W_, X), 1, keepdims=True))

            interactions = (tf.multiply(0.5,
                                        tf.reduce_sum(
                                            tf.subtract(
                                                tf.pow(tf.matmul(X, tf.transpose(self.V_)), 2),
                                                tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(self.V_, 2)))),
                                            1, keepdims=True)))
        else:
            linear_terms = tf.add(self.w0_, tf.tensordot(X, tf.transpose(self.W_), 1))


            interactions = (tf.multiply(0.5,
                                        tf.reduce_sum(
                                        tf.subtract(
                                            tf.pow(tf.tensordot(X, tf.transpose(self.V_), 1), 2),
                                            tf.tensordot(tf.pow(X, 2), tf.transpose(tf.pow(self.V_, 2)), 1 )))))





        return tf.add(linear_terms, tf.cast(interactions, tf.float32))

    def fit(self):
        self._validate_params()
        for epoch_count in range(self.epochs):
            for (x_, y_) in self.train_dataset:
                with tf.GradientTape() as tape:
                    pred = self.model(tf.cast(x_, tf.float32))
                    loss = mse(y_, pred) + _l2_loss(tf.cast(y_, tf.float32), pred, self.V_, self.W_)
                # Update gradients
                grads = tape.gradient(loss, [self.W_, self.w0_])
                self.optimizer.apply_gradients(zip(grads, [self.W_, self.w0_]))
        return self

    def predict(self, X):
        return self.model(X)
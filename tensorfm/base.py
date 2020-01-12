import tensorflow as tf


def _l1_loss(y, y_hat, V, W):
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l1_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.cast(tf.abs(W, 2), tf.float32)),
            tf.multiply(lambda_v, tf.cast(tf.abs(V, 2), tf.float32)))))

    error = tf.reduce_mean(tf.abs(tf.subtract(y, y_hat)))
    return error + l1_norm

def _l2_loss(y, y_hat, V, W):
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.cast(tf.pow(W, 2), tf.float32)),
            tf.multiply(lambda_v, tf.cast(tf.pow(V, 2), tf.float32)))))

    error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
    return error + l2_norm

class BaseFactorizationMachine:
    """
    Base class for factorization machines.

    Parameters
    ----------
    k : number of latent factors

    max_iter : number of training epochs (TODO (gmodena 2019-12-22) naming is confusing)

    penalty : string, 'l1' or 'l2' (default='l2')
        Specifies the norm used in the penalization

    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

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
        self.w0 = tf.Variable(tf.zeros([1]))
        self.W = tf.Variable(tf.zeros([p]))

        # interaction factors, randomly initialized
        self.V = tf.Variable(tf.random.normal([self.k, p], stddev=0.01, dtype=tf.dtypes.float32))


    def _validate_params(self):
        if self.epochs < 1:
            raise ValueError("epoch must be > 0")
        if self.k < 1:
            raise ValueError("k must be > 0")

    def model(self, X):
        linear_terms = tf.add(self.w0,
                              tf.reduce_sum(
                                  tf.multiply(self.W, X), 1, keepdims=True))

        interactions = (tf.multiply(0.5,
                                    tf.reduce_sum(
                                        tf.subtract(
                                            tf.pow( tf.matmul(X, tf.transpose(self.V)), 2),
                                            tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(self.V, 2)))),
                                        1, keepdims=True)))

        return tf.add(linear_terms, tf.cast(interactions, tf.float32))

    def fit(self):
        self._validate_params()
        for epoch_count in range(self.epochs):
            for i, (x_, y_) in enumerate(self.train_dataset):
                with tf.GradientTape() as tape:
                    pred = self.model(tf.cast(x_, tf.float32))
                    print(epoch_count, pred)
                    loss = _l2_loss(tf.cast(y_, tf.float32), pred, self.V, self.W)
                # Update gradients
                grads = tape.gradient(loss, [self.W, self.w0])
                self.optimizer.apply_gradients(zip(grads, [self.W, self.w0]))
        return self

    def predict(self, X):
        return self.model(X)
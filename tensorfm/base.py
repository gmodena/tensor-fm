import tensorflow as tf



def _model(X, V, W, w0):
    linear_terms = tf.add(w0,
                          tf.reduce_sum(
                              tf.multiply(W, X), 1, keepdims=True))

    interactions = (tf.multiply(0.5,
                                tf.reduce_sum(
                                    tf.subtract(
                                        tf.pow( tf.matmul(X, tf.transpose(V)), 2),
                                        tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                                    1, keepdims=True)))

    return linear_terms + tf.cast(interactions, tf.float32)

def _compute_loss(y, y_hat, V, W):
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    l2_norm = (tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.cast(tf.pow(W, 2), tf.float32)),
            tf.multiply(lambda_v, tf.cast(tf.pow(V, 2), tf.float32)))))

    error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
    return error + l2_norm

class Base:
    pass

class FactorizationMachine(Base):
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
    def __init__(self, train_dataset, k=5, epochs=100, eta=0.1, penalty='l2', C=10e-5, loss=None):
        self.k = k
        self.epochs = epochs
        self.penalty = penalty
        self.C = C
        self.loss = loss

        # Get the number of feature columns
        p = train_dataset.element_spec[0].shape[1]
        self.train_dataset = train_dataset
        # bias and weights
        self.w0 = tf.Variable(tf.zeros([1]))
        self.W = tf.Variable(tf.zeros([p]))

        # interaction factors, randomly initialized
        self.V = tf.Variable(tf.random.normal([self.k, p], stddev=0.01, dtype=tf.dtypes.float32))
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=tf.constant(eta))

    def _session_run(self,  X, y):
        pass


    def fit(self):
        y_hat = None
        for epoch_count in range(self.epochs):
            for (x_, y_) in self.train_dataset:
                with tf.GradientTape() as tape:
                    y_hat = _model(tf.cast(x_, tf.float32), self.V, self.W, self.w0)
                    loss = _compute_loss(tf.cast(y_, tf.float32), y_hat, self.V, self.W)
                # Update gradients
                grads = tape.gradient(loss, [self.W, self.w0])
                self.optimizer.apply_gradients(zip(grads, [self.W, self.w0]))
        return self

    def predict(self, X):
        return _model(X, self.V, self.W, self.w0)
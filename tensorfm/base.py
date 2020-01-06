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

    # his is the model
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
    def __init__(self, k=5, epochs=100, penalty='l2', C=10e-5, loss=None):
        self.k = k
        self.epochs = epochs
        self.penalty = penalty
        self.C = C
        self.loss = loss


    def _session_run(self,  X, y):
        pass


    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        self : object
        """
        k = self.k

        n, p = X.shape
        # bias and weights
        w0 = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.zeros([p]))

        # interaction factors, randomly initialized
        V = tf.Variable(tf.random.normal([k, p], stddev=0.01, dtype=tf.dtypes.float32))


        # estimate of y, initialized to 0.
        y_hat = tf.Variable(tf.zeros([n, 1]))

        eta = tf.constant(0.1)
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=eta)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((tf.reshape(X, [-1, X.shape[1]]), y))
                .batch(200)
                .shuffle(1000)
        )

        for epoch_count in range(self.epochs):
            for i, (x_, y_) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    y_hat  = _model(tf.cast(x_, tf.float32), V, W, w0)
                    loss = _compute_loss(tf.cast(y_, tf.float32), y_hat, V, W)

            grads = tape.gradient(loss, [W, w0])
            optimizer.apply_gradients(zip(grads, [W, w0]))

    def predict(self, X):
        pass
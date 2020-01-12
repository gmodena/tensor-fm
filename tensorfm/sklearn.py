"""
Custom scikit-learn estimators for supervised learning with Factorization Machines

"""
from .base import FactorizationMachine
from .base import _l2_loss, _l1_loss, check_X_y, check_X
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf

class FactorizationMachineClassifier(BaseEstimator, RegressorMixin):
    def __init__(self, n_factors=5, max_iter=100, eta=0.01, penalty='l2'):
        """Factorization machine for regularized regression

        :param n_factors: number of latent factor vectors
        :param max_iter: iterations to convergence
        :param eta: learning rate for adaptive optimizer
        :param penalty: regularization (l1, l2)
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        if penalty not in ('l1', 'l2'):
            raise ValueError(f"penalty must be l1 or l2")
        self.penalty = penalty
        self.loss = _l2_loss if penalty == 'l2' else _l1_loss
        if eta <= 0:
            raise ValueError(f"learning rate eta must be > 0")
        self.eta = eta
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=tf.constant(eta))
        self.fm = None

    def fit(self, X, y):
        """Fit a factorization machine model

        Internally, X and y are converted to to ShuffleDataset with types (float32, float32)

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        :return: an instance of self.
        """
        train_dataset = check_X_y(X, y)
        self.fm = FactorizationMachine(train_dataset,
                                       k=self.n_factors,
                                       epochs=self.max_iter,
                                       optimizer=self.optimizer,
                                       loss=self.loss)
        self.fm.fit()
        return self

    def predict(self, X):
        """Predict using a factorization machine model

        :param X: array-like or sparse matrix, shape (n_samples, n_features).
                    The input samples. Internally, it will be converted to a float32 Tensor
                    with shape (n_samples, n_features)
        :return y: array, shape (n_samples,)
        """
        # TODO(gmodena, 2020-01-12): we should look at something like sklearn.utils.validation.check_is_fitted
        if not self.fm:
            raise ValueError('Estimator not fitted')
        X = check_X(X)
        pred = self.fm.predict(X)
        # TODO(gmodena, 2020-01-12): convert back to ndarray
        return pred.numpy



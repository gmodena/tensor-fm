"""
Custom scikit-learn estimators for supervised learning with Factorization Machines

"""
from .base import train, fm
from .base import l2_norm, l1_norm, to_tf_shuffled_dataset, to_tf_dataset_X
from sklearn import utils
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils.validation import check_is_fitted
from tensorflow.keras.losses import MSE, binary_crossentropy


from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    FLOAT_DTYPES,
    column_or_1d,
)

import tensorflow as tf


class BaseFactorizationMachine(BaseEstimator):
    def __init__(self, n_factors=2, max_iter=100, eta=0.001, penalty='l2', C=1.0, random_state=None):
        """Factorization machine for regularized regression

        :param n_factors: number of latent factor vectors
        :param max_iter: iterations to convergence
        :param eta: learning rate for adaptive optimizer.
        :param penalty: regularization (l1, l2 or None). Default l2.
        :param C: inverse of regularization strength
        :param random_state: int, random state
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.C = C
        if penalty and penalty not in ('l1', 'l2'):
            raise ValueError(f"penalty must be l1, l2 or None")
        self.penalty = penalty
        self.penalty_function = None
        if penalty:
            self.penalty_function = l2_norm if penalty == 'l2' else l1_norm
        self.eta = eta
        self.C = C
        self.random_state = random_state


class FactorizationMachineRegressor(BaseFactorizationMachine, RegressorMixin):
    def __init__(self, n_factors=2, max_iter=100, eta=0.001, penalty='l2', C=1.0, random_state=None):
        super().__init__(
            n_factors=n_factors,
            max_iter=max_iter,
            eta=eta,
            penalty=penalty,
            C=C,
            random_state=random_state)
        self.loss = MSE



    def fit(self, X, y):
        """Fit a factorization machine model

        Internally, X and y are converted to to ShuffleDataset with types (float32, float32)

        :param X: {array-like} of shape (n_samples, n_features)
            Training data.
        :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        :return: an instance of self.
        """
        X, y = utils.check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        column_or_1d(y)

        train_dataset = to_tf_shuffled_dataset(X, y)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta)
        self.w0_, self.W_, self.V_ = train(train_dataset, num_factors=self.n_factors,
                                           max_iter=self.max_iter,
                                           optimizer=self.optimizer,
                                           loss=self.loss,
                                           C=self.C,
                                           penalty=self.penalty_function,
                                           random_state=self.random_state)

        return self

    def predict(self, X):
        """Predict using a factorization machine model

        :param X: array-like , shape (n_samples, n_features).
                    The input samples. Internally, it will be converted to a float32 Tensor
                    with shape (n_samples, n_features)
        :return y: array, shape (n_samples,)
        """
        check_is_fitted(self)
        X = to_tf_dataset_X(X)
        pred = fm(X, self.w0_, self.W_, self.V_).numpy()
        pred = column_or_1d(pred, warn=True)

        return pred

    def _more_tags(self):
        tags = super()._more_tags()
        tags['poor_score'] = True

        return tags


class FactorizationMachineClassifier(BaseFactorizationMachine, ClassifierMixin):
    def __init__(self, n_factors=2, max_iter=100, eta=0.001, penalty='l2', C=1.0, random_state=None):
        super().__init__(
            n_factors=n_factors,
            max_iter=max_iter,
            eta=eta,
            penalty = penalty,
            C=C,
            random_state=random_state)
        self.loss = binary_crossentropy

    def fit(self, X, y):
            """Fit a factorization machine binary classifier

            Internally, X and y are converted to to ShuffleDataset with types (float32, float32)

            :param X: {array-like} of shape (n_samples, n_features)
                Training data.
            :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values.
            :return: an instance of self.
            """
            X, y = utils.check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
            column_or_1d(y)
            self.label_binarizer = LabelBinarizer().fit(y)
            y = self.label_binarizer.transform(y)
            train_dataset = to_tf_shuffled_dataset(X, y)

            self.classes_ = self.label_binarizer.classes_
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta)
            self.w0_, self.W_, self.V_ = train(train_dataset,
                                               num_factors=self.n_factors,
                                               max_iter=self.max_iter,
                                               optimizer=self.optimizer,
                                               loss=self.loss,
                                               penalty=self.penalty_function,
                                               activation=tf.nn.sigmoid,
                                               loss_kwargs={'from_logits': True},
                                               random_state=self.random_state)
            return self

    def _predict(self, X):
        check_is_fitted(self)
        X = to_tf_dataset_X(X)
        try:
            pred = fm(X, self.w0_, self.W_, self.V_)
        except:
            raise ValueError()
        return pred

    def predict(self, X):
        """Predict using a factorization machine model

        :param X: array-like, shape (n_samples, n_features).
                    The input samples. Internally, it will be converted to a float32 Tensor
                    with shape (n_samples, n_features)
        :return y: array, shape (n_samples,)
        """
        pred = self._predict(X).numpy() > 0.5

        return self.label_binarizer.inverse_transform(pred)

    def _more_tags(self):
        tags = super()._more_tags()
        tags['binary_only'] = True
        return tags


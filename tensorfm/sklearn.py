"""
Custom scikit-learn estimators for supervised learning with Factorization Machines

"""
from .base import fit_2d_fm, model
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
import numpy as np

class BaseFactorizationMachine(BaseEstimator):
    """Base class for factorization machine regressor and classifier"""
    def __init__(self, n_factors=2, max_iter=500, eta=0.01, penalty='l2', C=1.0, random_state=None):
        """Factorization machine for regularized regression

        :param n_factors: number of latent factor vectors
        :param max_iter: iterations to convergence
        :param eta: learning rate for adaptive optimizer
        :param penalty: regularization (l1, l2). Default l2.
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.C = C
        if penalty not in ('l1', 'l2'):
            raise ValueError(f"penalty must be l1 or l2")
        self.penalty = penalty
        self.penalty_function = l2_norm if penalty == 'l2' else l1_norm
        self.loss = MSE
        self.eta = eta
        self.C = C
        self.random_state = random_state

class FactorizationMachineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_factors=2, max_iter=500, eta=0.01, penalty='l2', C=1.0, random_state=None):
        """Factorization machine for regularized regression

        :param n_factors: number of latent factor vectors
        :param max_iter: iterations to convergence
        :param eta: learning rate for adaptive optimizer
        :param penalty: regularization (l1, l2). Default l2.
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.C = C
        if penalty not in ('l1', 'l2'):
            raise ValueError(f"penalty must be l1 or l2")
        self.penalty = penalty
        self.penalty_function = l2_norm if penalty == 'l2' else l1_norm
        self.loss = MSE
        self.eta = eta
        self.C = C
        self.random_state = random_state


    def _more_tags(self):
        tags = super()._more_tags()
        tags['poor_score'] = True

        return tags

    def fit(self, X, y):
        """Fit a factorization machine model

        Internally, X and y are converted to to ShuffleDataset with types (float32, float32)

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        :return: an instance of self.
        """
        X, y = utils.check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        column_or_1d(y)

        train_dataset = to_tf_shuffled_dataset(X, y)
        #tf.random.set_seed(self.random_state)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta)
        self.w0_, self.W_, self.V_ = fit_2d_fm(train_dataset, num_factors=self.n_factors,
                                       max_iter=self.max_iter,
                                       optimizer=self.optimizer,
                                       loss=self.loss,
                                       C=self.C,
                                       penalty=self.penalty_function,
                                       random_state=self.random_state)

        return self

    def predict(self, X):
        """Predict using a factorization machine model

        :param X: array-like or sparse matrix, shape (n_samples, n_features).
                    The input samples. Internally, it will be converted to a float32 Tensor
                    with shape (n_samples, n_features)
        :return y: array, shape (n_samples,)
        """
        check_is_fitted(self)
        X = to_tf_dataset_X(X)
        pred = model(X, self.w0_, self.W_, self.V_).numpy()
        pred = column_or_1d(pred, warn=True)

        return pred


class FactorizationMachineClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_factors=5, max_iter=100, eta=0.01, penalty='l2', C=1.0, random_state=None):
        """Factorization machine for regularized regression

        :param n_factors: number of latent factor vectors
        :param max_iter: iterations to convergence
        :param eta: learning rate for adaptive optimizer
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        if penalty not in ('l1', 'l2'):
            raise ValueError(f"penalty must be l1 or l2")
        self.penalty = penalty
        self.penalty_function = l2_norm if penalty == 'l2' else l1_norm
        self.C = C
        self.loss = binary_crossentropy
        self.eta = eta
        self.random_state = random_state

    def fit(self, X, y):
            """Fit a factorization machine model

            Internally, X and y are converted to to ShuffleDataset with types (float32, float32)

            :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
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

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=tf.constant(self.eta))
            self.w0_, self.W_, self.V_ = fit_2d_fm(train_dataset,
                                                   num_factors=self.n_factors,
                                                   max_iter=self.max_iter,
                                                   optimizer=self.optimizer,
                                                   loss=self.loss,
                                                   penalty=self.penalty_function,
                                                   activation=tf.nn.sigmoid, # logit binary classification
                                                   loss_kwargs={'from_logits': True},
                                                   random_state=self.random_state)
            return self

    def _get_proba(self, X):
        """

        :param X:
        :return:
        """
        check_is_fitted(self)
        X = to_tf_dataset_X(X)
        try:
            pred = model(X, self.w0_, self.W_, self.V_)
        except:
            raise ValueError("fixme")
        return pred

    def get_proba(self, X):
        return self._get_proba(X).numpy()

    def predict(self, X):
        """Predict using a factorization machine model

        :param X: array-like or sparse matrix, shape (n_samples, n_features).
                    The input samples. Internally, it will be converted to a float32 Tensor
                    with shape (n_samples, n_features)
        :return y: array, shape (n_samples,)
        """
        pred = self._get_proba(X).numpy() > 0.5

        return self.label_binarizer.inverse_transform(pred)


    def _more_tags(self):
        tags = super()._more_tags()
        tags['binary_only'] = True
        return tags
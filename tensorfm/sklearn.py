"""
Custom scikit-learn estimators for supervised learning with Factorization Machines

"""
from .base import train, fm
from .base import l2_norm, l1_norm, noop_norm

from .util import to_tf_dataset, to_tf_tensor, TF_DATASET_BATCH_SIZE

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.losses import MSE, binary_crossentropy

from sklearn import utils
from sklearn.utils.validation import (
    check_is_fitted,
    FLOAT_DTYPES,
    column_or_1d,
)
from functools import partial

from tensorflow.python.framework.errors_impl import (
    InvalidArgumentError as TensoFlowInvalidArgumentError,
)
import tensorflow as tf


class BaseFactorizationMachine(BaseEstimator):
    def __init__(
        self,
        n_factors=2,
        max_iter=10,
        eta=0.001,
        penalty="l2",
        C=1.0,
        batch_size=TF_DATASET_BATCH_SIZE,
        random_state=None,
    ):
        """A base class for factorization machines

        :param n_factors: number of latent factor vectors
        :param max_iter: iterations to convergence
        :param eta: learning rate for adaptive optimizer.
        :param penalty: regularization (l1, l2 or None). Default l2.
        :param C: inverse of regularization strength
        :param batch_size: training batch size
        :param random_state: int, random state
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        if penalty and penalty not in ("l1", "l2"):
            raise ValueError(f"penalty must be l1, l2 or None")
        self.penalty = penalty
        self.penalty_function = noop_norm
        if penalty:
            self.penalty_function = l2_norm if penalty == "l2" else l1_norm

        self.eta = eta
        self.C = C
        self.batch_size = batch_size
        self.random_state = random_state


class FactorizationMachineRegressor(BaseFactorizationMachine, RegressorMixin):
    def __init__(
        self,
        n_factors=2,
        max_iter=100,
        eta=0.001,
        penalty="l2",
        C=1.0,
        batch_size=TF_DATASET_BATCH_SIZE,
        random_state=None,
    ):
        super().__init__(
            n_factors=n_factors,
            max_iter=max_iter,
            eta=eta,
            penalty=penalty,
            C=C,
            batch_size=batch_size,
            random_state=random_state,
        )
        self.loss = MSE

    def fit(self, X, y):
        """Fit a factorization machine regressor

        Internally, X and y are converted to a Tensorflow Dataset with types (float32, float32)

        :param X: {array-like} of shape (n_samples, n_features)
            Training data.
        :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        :return: an instance of self.
        """
        X, y = utils.check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        column_or_1d(y)

        train_dataset = to_tf_dataset(X, y, batch_size=self.batch_size,)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta)
        self.w0_, self.W_, self.V_ = train(
            train_dataset,
            num_factors=self.n_factors,
            max_iter=self.max_iter,
            optimizer=self.optimizer,
            loss=self.loss,
            C=self.C,
            penalty=self.penalty_function,
            random_state=self.random_state,
        )
        return self

    def predict(self, X):
        """Predict using a factorization machine model

        :param X: array-like , shape (n_samples, n_features).
                    The input samples. Internally, it will be converted to a float32 Tensor
                    with shape (n_samples, n_features)
        :return y: array, shape (n_samples,)
        """
        check_is_fitted(self)
        X = utils.check_array(X)
        X = to_tf_tensor(X)
        pred = fm(X, self.w0_, self.W_, self.V_).numpy()
        pred = column_or_1d(pred, warn=True)

        return pred

    def _more_tags(self):
        tags = super()._more_tags()
        # TODO(gmodena): this needs investigation; also wrt performance degradation on movielens
        tags["poor_score"] = True

        return tags


class FactorizationMachineClassifier(BaseFactorizationMachine, ClassifierMixin):
    def __init__(
        self,
        n_factors=2,
        max_iter=10,
        eta=0.001,
        penalty="l2",
        C=1.0,
        batch_size=TF_DATASET_BATCH_SIZE,
        random_state=None,
    ):
        super().__init__(
            n_factors=n_factors,
            max_iter=max_iter,
            eta=eta,
            penalty=penalty,
            C=C,
            batch_size=batch_size,
            random_state=random_state,
        )
        self.loss = partial(binary_crossentropy, from_logits=True)

    def fit(self, X, y):
        """Fit a factorization machine binary classifier

            Internally, X and y are converted to to Dataset with types (float32, float32)

            :param X: {array-like} of shape (n_samples, n_features)
                Training data.
            :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
                Target values.
            :return: an instance of self
            """
        X, y = utils.check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        column_or_1d(y)
        self.label_binarizer = LabelBinarizer().fit(y)
        y = self.label_binarizer.transform(y)
        train_dataset = to_tf_dataset(X, y, batch_size=self.batch_size)

        self.classes_ = self.label_binarizer.classes_
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta)
        self.w0_, self.W_, self.V_ = train(
            train_dataset,
            num_factors=self.n_factors,
            max_iter=self.max_iter,
            optimizer=self.optimizer,
            loss=self.loss,
            penalty=self.penalty_function,
            random_state=self.random_state,
        )
        return self

    def _predict(self, X):
        check_is_fitted(self)
        X = utils.check_array(X)
        X = to_tf_tensor(X)
        try:
            pred = tf.nn.sigmoid(fm(X, self.w0_, self.W_, self.V_))
        except TensoFlowInvalidArgumentError as e:
            # Re-raise a sklearn friendly exception
            raise ValueError(str(e))
        return pred

    def predict(self, X, threshold=0.5):
        """Predict using a factorization machine model

        :param X: array-like, shape (n_samples, n_features).
                    The input samples. Internally, it will be converted to a float32 Tensor
                    with shape (n_samples, n_features)
        :param threshold: decision boundary between classes
        :return y: array, shape (n_samples,)
        """
        pred = self._predict(X).numpy() > threshold

        return self.label_binarizer.inverse_transform(pred)

    def _more_tags(self):
        tags = super()._more_tags()
        tags["binary_only"] = True
        return tags

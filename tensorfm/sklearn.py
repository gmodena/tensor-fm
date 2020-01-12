"""
Custom scikit-learn estimators for supervised learning with Factorization Machines

"""
from base import BaseFactorizationMachine
from base import _l2_loss, _l1_loss
from sklearn.base import BaseEstimator, RegressionMixin
import tensortflow as tf

class FactorizationMachineClassifier(BaseEstimator, RegressionMixin):
    TF_DATASET_BATCH_SIZE = 200
    TF_DATASET_SHUFFLE = 1000

    def _validate_params(self):
        pass

    def __init__(self, n_factors=5, max_iter=100, eta=0.01, penalty='l2'):
        self._validate_params()

        loss = _l2_loss if penalty == 'l2' else _l1_loss
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=tf.constant(eta))
        self.fm = BaseFactorizationMachine(k=n_factors, epochs=max_iter, optimizer = optimizer, loss = loss)

    def fit(self, X, y):
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((tf.reshape(X, [-1, X.shape[1]]), y))
                .batch(self.TF_DATASET_BATCH_SIZE)
                .shuffle(self.TF_DATASET_SHUFFLE)
        )
        self.fm.fit(train_dataset)
        return self

    def predict(self, X):
        return self.fm.predict(X)




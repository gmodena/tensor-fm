import pytest
from tensorfm.sklearn import FactorizationMachineRegressor, FactorizationMachineClassifier

from sklearn.utils import estimator_checks
regressor_checks = (
    estimator_checks.check_regressors_train,
    estimator_checks.check_regressor_data_not_an_array,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_regressors_no_decision_function,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_regressors_int,
    estimator_checks.check_estimators_unfitted,
)

classifier_checks = (
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_classifiers_one_label,
    estimator_checks.check_classifiers_classes, # TODO(gmodena) the generated dataset is a degenerate case for this FM implementation
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_classifiers_train,
    estimator_checks.check_classifiers_regression_target,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_estimators_unfitted,
    estimator_checks.check_decision_proba_consistency
)


@pytest.mark.parametrize("test_fn", regressor_checks)
def test_estimator_checks_regression(test_fn):
    estimator = FactorizationMachineRegressor()
    name = type(estimator).__name__
    test_fn(name, estimator)


@pytest.mark.parametrize("test_fn", classifier_checks)
def test_estimator_checks_classifier(test_fn):
    estimator = FactorizationMachineClassifier()
    name = type(estimator).__name__
    test_fn(name, estimator)
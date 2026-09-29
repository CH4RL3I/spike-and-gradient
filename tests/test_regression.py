import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge

from mlfs.regression import Standardizer, fit_closed_form, fit_gd, mse_loss


@pytest.fixture(scope="module")
def data():
    X, y = load_diabetes(return_X_y=True)
    return X, y


def test_ols_matches_sklearn(data):
    X, y = data
    ref = LinearRegression().fit(X, y)
    m = fit_closed_form(X, y)
    np.testing.assert_allclose(m.coef_, ref.coef_, rtol=1e-9, atol=1e-9)
    assert m.intercept_ == pytest.approx(ref.intercept_, abs=1e-8)


@pytest.mark.parametrize("alpha", [0.01, 1.0, 50.0])
def test_ridge_matches_sklearn(data, alpha):
    X, y = data
    ref = Ridge(alpha=alpha).fit(X, y)
    m = fit_closed_form(X, y, alpha=alpha)
    np.testing.assert_allclose(m.coef_, ref.coef_, rtol=1e-9, atol=1e-9)
    assert m.intercept_ == pytest.approx(ref.intercept_, abs=1e-8)


def test_ols_on_unscaled_raw_features(data):
    X, y = data
    X = X * np.linspace(1, 1e3, X.shape[1])  # badly scaled columns
    ref = LinearRegression().fit(X, y)
    np.testing.assert_allclose(fit_closed_form(X, y).coef_, ref.coef_, rtol=1e-7)


def test_standardizer(data):
    Xs = Standardizer().fit_transform(data[0])
    np.testing.assert_allclose(Xs.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(0), 1, atol=1e-12)


def test_batch_gd_converges_to_closed_form(data):
    X, y = data
    Xs = Standardizer().fit_transform(X)
    ref = fit_closed_form(Xs, y)
    m, hist = fit_gd(Xs, y, lr=0.45, epochs=20000)
    np.testing.assert_allclose(m.coef_, ref.coef_, atol=1e-6)
    assert hist[-1] <= hist[0]


def test_ridge_gd_converges_to_closed_form(data):
    X, y = data
    Xs = Standardizer().fit_transform(X)
    ref = fit_closed_form(Xs, y, alpha=20.0)
    m, _ = fit_gd(Xs, y, lr=0.45, epochs=20000, alpha=20.0)
    np.testing.assert_allclose(m.coef_, ref.coef_, atol=1e-5)


def test_minibatch_gd_reaches_neighbourhood_of_optimum(data):
    X, y = data
    Xs = Standardizer().fit_transform(X)
    ref = fit_closed_form(Xs, y)
    best = mse_loss(Xs, y, ref.coef_, ref.intercept_)
    m, hist = fit_gd(Xs, y, lr=0.02, epochs=300, batch_size=32)
    assert hist[-1] < best * 1.01  # within 1 percent of the optimal loss


def test_divergent_learning_rate_is_detected(data):
    Xs = Standardizer().fit_transform(data[0])
    _, hist = fit_gd(Xs, data[1], lr=5.0, epochs=200)
    assert hist[-1] > hist[0] and len(hist) < 202

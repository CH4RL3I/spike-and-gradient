import numpy as np
import pytest

from mlfs.nn import MLP, SGD, Adam, load_mnist, mnist_available, softmax_cross_entropy, train


def numerical_grad(model, X, y, param, eps=1e-6):
    g = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = param[i]
        param[i] = old + eps
        lp = model.loss_and_grads(X, y)[0]
        param[i] = old - eps
        lm = model.loss_and_grads(X, y)[0]
        param[i] = old
        g[i] = (lp - lm) / (2 * eps)
    return g


def rel_error(a, b):
    return np.abs(a - b).max() / max(np.abs(a).max() + np.abs(b).max(), 1e-12)


@pytest.fixture
def tiny():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 6))
    y = rng.integers(0, 4, size=12)
    model = MLP([6, 8, 7, 4], seed=2)
    # non-zero biases so that the bias gradients are exercised properly
    for b in model.b:
        b += rng.normal(scale=0.3, size=b.shape)
    return model, X, y


def test_gradient_check_every_layer(tiny):
    model, X, y = tiny
    _, grads = model.loss_and_grads(X, y)
    names = [f"{k}{l + 1}" for l in range(len(model.W)) for k in ("W", "b")]
    assert len(grads) == len(model.params) == 6
    for name, p, g in zip(names, model.params, grads, strict=True):
        err = rel_error(g, numerical_grad(model, X, y, p))
        assert err < 1e-7, f"{name}: relative error {err:.2e}"


def test_softmax_cross_entropy_is_stable_and_correct():
    logits = np.array([[1000.0, 0.0, -1000.0], [0.0, 0.0, 0.0]])
    loss, grad = softmax_cross_entropy(logits, np.array([0, 2]))
    assert np.isfinite(loss) and np.isfinite(grad).all()
    assert loss == pytest.approx((0.0 + np.log(3)) / 2)
    np.testing.assert_allclose(grad.sum(axis=1), 0, atol=1e-12)


def test_he_init_variance():
    m = MLP([400, 300, 10], seed=0)
    assert m.W[0].std() == pytest.approx(np.sqrt(2 / 400), rel=0.03)
    assert not m.b[0].any()


@pytest.mark.parametrize(
    "make_opt",
    [lambda p: SGD(p, lr=0.05, momentum=0.9), lambda p: Adam(p, lr=0.01)],
    ids=["sgd_momentum", "adam"],
)
def test_overfits_tiny_batch(make_opt):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(16, 20)), rng.integers(0, 5, size=16)  # random labels: memorise only
    model = MLP([20, 64, 5], seed=0)
    hist = train(
        model, make_opt(model.params), X, y, X, y, epochs=300, batch_size=16, verbose=False
    )
    assert hist["loss"][-1] < 0.01
    assert hist["train_acc"][-1] == 1.0


def test_minibatch_training_learns_separable_problem():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=600)
    X = rng.normal(size=(600, 5)) + 3 * np.eye(3, 5)[y]
    model = MLP([5, 16, 3], seed=0)
    hist = train(
        model, Adam(model.params, lr=0.01), X, y, X, y, epochs=20, batch_size=32, verbose=False
    )
    assert hist["val_acc"][-1] > 0.95


@pytest.mark.skipif(not mnist_available(), reason="MNIST not cached in data/")
def test_mnist_loads_and_one_epoch_learns():
    x_tr, y_tr, x_te, y_te = load_mnist(download=False)
    assert x_tr.shape == (60000, 784) and x_te.shape == (10000, 784)
    assert 0.0 <= x_tr.min() and x_tr.max() == 1.0
    assert np.bincount(y_tr).min() > 5000
    model = MLP([784, 128, 10], seed=0, dtype=np.float32)
    train(
        model,
        Adam(model.params),
        x_tr[:20000],
        y_tr[:20000],
        x_te[:10],
        y_te[:10],
        epochs=2,
        verbose=False,
    )
    assert (model.predict(x_te) == y_te).mean() > 0.93

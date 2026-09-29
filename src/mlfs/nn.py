"""A multi-layer perceptron with hand-written forward and backward passes.

Forward, for layers l = 1..L:  z_l = a_{l-1} W_l + b_l,  a_l = relu(z_l)  (last layer: logits).
Loss: softmax cross-entropy averaged over the batch. Backward (delta = dL/dz):

    delta_L = (softmax(z_L) - onehot(y)) / n
    dW_l = a_{l-1}^T delta_l,   db_l = sum_rows(delta_l)
    delta_{l-1} = (delta_l W_l^T) * 1[z_{l-1} > 0]

Run ``python -m mlfs.nn`` to download MNIST (cached in ``data/``), train and write ``docs/nn.png``.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---- data ----------------------------------------------------------------------------------


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    _, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if dtype_code != 0x08:
        raise ValueError(f"{path.name}: expected unsigned-byte IDX, got type {dtype_code:#x}")
    shape = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    arr = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path.name}: truncated file")
    return arr.reshape(shape)


def mnist_available(data_dir: Path = DATA_DIR) -> bool:
    return all((Path(data_dir) / name).exists() for name in MNIST_FILES.values())


def load_mnist(data_dir: Path = DATA_DIR, download: bool = True):
    """Return (x_train, y_train, x_test, y_test); images are float32 in [0, 1], shape (n, 784)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES.values():
        path = data_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            urllib.request.urlretrieve(MNIST_URL + name, path)
    x_tr, y_tr, x_te, y_te = (_read_idx(data_dir / MNIST_FILES[k]) for k in MNIST_FILES)
    assert x_tr.shape == (60000, 28, 28) and x_te.shape == (10000, 28, 28)
    assert y_tr.shape == (60000,) and y_te.shape == (10000,)
    flat = lambda x: x.reshape(len(x), -1).astype(np.float32) / 255.0  # noqa: E731
    return flat(x_tr), y_tr.astype(np.int64), flat(x_te), y_te.astype(np.int64)


# ---- model ---------------------------------------------------------------------------------


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy of softmax(logits) and its gradient w.r.t. the logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)  # stability: softmax is shift-invariant
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_p = shifted - log_norm
    n = len(y)
    loss = -log_p[np.arange(n), y].mean()
    grad = np.exp(log_p)
    grad[np.arange(n), y] -= 1.0
    return float(loss), grad / n


class MLP:
    """Fully connected ReLU network; ``sizes = [in, hidden..., classes]``, He-initialised."""

    def __init__(self, sizes: list[int], seed: int = 0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        self.W = [
            (rng.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(dtype)  # He: Var = 2 / fan_in
            for i, o in zip(sizes[:-1], sizes[1:], strict=True)
        ]
        self.b = [np.zeros(o, dtype=dtype) for o in sizes[1:]]

    @property
    def params(self) -> list[np.ndarray]:
        return [p for wb in zip(self.W, self.b, strict=True) for p in wb]

    def forward(self, X: np.ndarray, cache: list | None = None) -> np.ndarray:
        a = X
        for l, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            z = a @ W + b
            if cache is not None:
                cache.append((a, z))
            a = z if l == len(self.W) - 1 else np.maximum(z, 0.0)
        return a

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Loss and gradients ordered like ``params`` (W1, b1, W2, b2, ...)."""
        cache: list = []
        loss, delta = softmax_cross_entropy(self.forward(X, cache), y)
        grads: list[np.ndarray] = []
        for l in range(len(self.W) - 1, -1, -1):
            a_prev, _ = cache[l]
            dW, db = a_prev.T @ delta, delta.sum(axis=0)
            grads = [dW, db, *grads]
            if l > 0:
                delta = (delta @ self.W[l].T) * (cache[l - 1][1] > 0)
        return loss, grads

    def predict(self, X: np.ndarray, batch: int = 10000) -> np.ndarray:
        return np.concatenate(
            [self.forward(X[i : i + batch]).argmax(1) for i in range(0, len(X), batch)]
        )


# ---- optimisers ----------------------------------------------------------------------------


class SGD:
    """v <- mu v - lr g;  p <- p + v   (momentum mu = 0 gives plain SGD)."""

    def __init__(self, params: list[np.ndarray], lr: float = 0.1, momentum: float = 0.9):
        self.params, self.lr, self.mu = params, lr, momentum
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        for p, g, v in zip(self.params, grads, self.v, strict=True):
            v *= self.mu
            v -= self.lr * g
            p += v


class Adam:
    """Adam (Kingma & Ba): bias-corrected first and second moment estimates."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


# ---- training ------------------------------------------------------------------------------


def train(model, opt, X, y, X_val, y_val, epochs=10, batch_size=128, seed=0, verbose=True):
    """Mini-batch training; returns per-epoch mean train loss, train and validation accuracy."""
    rng = np.random.default_rng(seed)
    hist = {"loss": [], "train_acc": [], "val_acc": []}
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        losses = []
        for s in range(0, len(X), batch_size):
            j = idx[s : s + batch_size]
            loss, grads = model.loss_and_grads(X[j], y[j])
            opt.step(grads)
            losses.append(loss)
        hist["loss"].append(float(np.mean(losses)))
        hist["train_acc"].append(float((model.predict(X) == y).mean()))
        hist["val_acc"].append(float((model.predict(X_val) == y_val).mean()))
        if verbose:
            print(
                f"epoch {ep + 1:2d}  loss {hist['loss'][-1]:.4f}  "
                f"train {hist['train_acc'][-1]:.4f}  val {hist['val_acc'][-1]:.4f}"
            )
    return hist


def main(out: str = "docs/nn.png", epochs: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_tr, y_tr, x_te, y_te = load_mnist()
    x_fit, y_fit, x_val, y_val = x_tr[:55000], y_tr[:55000], x_tr[55000:], y_tr[55000:]
    model = MLP([784, 256, 128, 10], seed=0, dtype=np.float32)
    hist = train(model, Adam(model.params, lr=1e-3), x_fit, y_fit, x_val, y_val, epochs=epochs)
    pred = model.predict(x_te)
    acc = float((pred == y_te).mean())
    print(f"test accuracy: {acc:.4f} ({int((pred != y_te).sum())} errors of {len(y_te)})")

    wrong = np.flatnonzero(pred != y_te)
    fig = plt.figure(figsize=(11, 4.2), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    ax = fig.add_subplot(gs[0])
    ax.plot(range(1, epochs + 1), hist["loss"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (cross-entropy)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1])
    ax.plot(range(1, epochs + 1), hist["train_acc"], label="train", color="#1f77b4")
    ax.plot(range(1, epochs + 1), hist["val_acc"], label="validation", color="#d62728")
    ax.axhline(acc, color="gray", ls="--", lw=0.8, label=f"test {acc:.4f}")
    ax.set_ylim(0.95, 1.001)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    sub = gs[2].subgridspec(4, 6, wspace=0.1, hspace=0.5)
    for k, i in enumerate(wrong[:24]):
        a = fig.add_subplot(sub[k // 6, k % 6])
        a.imshow(x_te[i].reshape(28, 28), cmap="gray_r")
        a.set_title(f"{y_te[i]}>{pred[i]}", fontsize=7, pad=2)
        a.axis("off")
    fig.text(0.79, 0.96, "misclassified (true > predicted)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

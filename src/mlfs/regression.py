"""Linear regression from first principles: closed form, gradient descent, ridge.

Model: y_hat = X w + b. Loss (ridge, intercept unpenalised):

    L(w, b) = 1/(2n) * (||y - Xw - b||^2 + alpha * ||w||^2)

Run ``python -m mlfs.regression`` to reproduce ``docs/regression.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Standardizer:
    """Zero-mean, unit-variance feature scaling (statistics from the fit data only)."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)  # constant columns stay at zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class LinearModel:
    coef_: np.ndarray
    intercept_: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


def fit_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> LinearModel:
    """Least squares (alpha=0) or ridge via QR, never forming (X^T X)^-1.

    Centering X and y removes the intercept from the penalised problem. Ridge is then
    ordinary least squares on the augmented system [Xc; sqrt(alpha) I] w = [yc; 0],
    solved by a thin QR factorisation and a triangular solve R w = Q^T y.
    """
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    if alpha > 0:
        p = X.shape[1]
        Xc = np.vstack([Xc, np.sqrt(alpha) * np.eye(p)])
        yc = np.concatenate([yc, np.zeros(p)])
    Q, R = np.linalg.qr(Xc)
    w = np.linalg.solve(R, Q.T @ yc)  # R is upper triangular and well conditioned relative to X^T X
    return LinearModel(w, float(y_mean - x_mean @ w))


def mse_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, alpha: float = 0.0) -> float:
    r = X @ w + b - y
    return float((r @ r + alpha * (w @ w)) / (2 * len(y)))


def fit_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int = 200,
    batch_size: int | None = None,
    alpha: float = 0.0,
    seed: int = 0,
) -> tuple[LinearModel, list[float]]:
    """Gradient descent on the loss above. ``batch_size=None`` is full-batch; otherwise
    each epoch shuffles the data and takes one step per mini-batch.

    Returns the model and the full-data loss recorded at the start of every epoch (plus the
    final one). If the loss becomes non-finite or explodes, training stops early.
    """
    n, p = X.shape
    w, b = np.zeros(p), 0.0
    rng = np.random.default_rng(seed)
    bs = n if batch_size is None else batch_size
    history = [mse_loss(X, y, w, b, alpha)]
    for _ in range(epochs):
        idx = rng.permutation(n) if bs < n else np.arange(n)
        for s in range(0, n, bs):
            j = idx[s : s + bs]
            r = X[j] @ w + b - y[j]
            m = len(j)
            w = w - lr * (X[j].T @ r / m + alpha * w / n)  # the penalty is spread over the epoch
            b = b - lr * r.mean()
        history.append(mse_loss(X, y, w, b, alpha))
        if not np.isfinite(history[-1]) or history[-1] > 1e12:
            break
    return LinearModel(w, b), history


def make_demo_data(n: int = 400, p: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Correlated features on different scales, so standardisation and conditioning matter."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, p))
    X = rng.normal(size=(n, p)) @ A * rng.uniform(1, 50, size=p) + rng.uniform(-30, 30, size=p)
    y = X @ (rng.normal(size=p) / rng.uniform(1, 50, size=p)) + 5 + rng.normal(size=n)
    return X, y


def main(out: str = "docs/regression.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y = make_demo_data()
    Xs = Standardizer().fit_transform(X)
    lam_max = np.linalg.eigvalsh(Xs.T @ Xs / len(y))[-1]
    lr_crit = 2 / lam_max  # full-batch GD on a quadratic diverges above 2 / lambda_max(Hessian)
    star = fit_closed_form(Xs, y)
    best = mse_loss(Xs, y, star.coef_, star.intercept_)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for frac, color in [(0.02, "#9ecae1"), (0.1, "#4292c6"), (0.5, "#08519c"), (1.05, "#d62728")]:
        lr = frac * lr_crit
        _, h = fit_gd(Xs, y, lr=lr, epochs=150)
        label = f"lr = {lr:.3f}" + ("  (diverges)" if frac > 1 else "")
        ax.plot(np.maximum(np.array(h) - best, 1e-12), color=color, label=label, lw=1.8)
    _, h = fit_gd(Xs, y, lr=0.1 * lr_crit, epochs=150, batch_size=32)
    ax.plot(
        np.maximum(np.array(h) - best, 1e-12),
        color="#2ca02c",
        ls="--",
        lw=1.4,
        label=f"lr = {0.1 * lr_crit:.3f}, mini-batch 32",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss above closed-form optimum")
    ax.set_title(f"Gradient descent on standardised features (critical lr = {lr_crit:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

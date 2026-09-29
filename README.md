# spike-and-gradient: machine learning from first principles

[![CI](https://github.com/CH4RL3I/spike-and-gradient/actions/workflows/ci.yml/badge.svg)](https://github.com/CH4RL3I/spike-and-gradient/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Linear regression, a biological neuron and a neural network, built from first principles in NumPy and checked against trusted references.

Everything numerical is written by hand (QR solve, gradient descent, RK4 integration of the Hodgkin-Huxley equations, backpropagation, Adam). scikit-learn and SciPy appear only in the tests, as the reference to compare with. Runtime dependencies are NumPy and Matplotlib.

```bash
uv sync --group dev
uv run python -m mlfs.regression   # docs/regression.png
uv run python -m mlfs.neuron       # docs/neuron.png   (about 10 s)
uv run python -m mlfs.nn           # docs/nn.png       (downloads MNIST into data/, about 10 s)
uv run pytest
```

## 1. Linear regression

![Gradient descent loss curves](docs/regression.png)

Model and loss (ridge, intercept not penalised):

$$\hat y = Xw + b, \qquad L(w,b) = \frac{1}{2n}\left(\lVert y - Xw - b\rVert^2 + \alpha \lVert w\rVert^2\right)$$

The closed form centres $X$ and $y$, stacks $\sqrt{\alpha}\,I$ under the design matrix and solves the resulting least-squares problem with a thin QR factorisation, $Rw = Q^\top y$. No inverse and no normal equations, so the condition number is that of $X$ and not of $X^\top X$. Gradient descent uses $\nabla_w L = \frac{1}{m} X_B^\top (X_B w + b - y_B) + \frac{\alpha}{n} w$ on the full batch or on shuffled mini-batches.

The figure runs full-batch gradient descent on standardised, correlated synthetic features. For a quadratic the iteration diverges once the learning rate exceeds $2/\lambda_{\max}$ of the Hessian, and the red curve (about 5 percent above that threshold) does exactly that. Below the threshold, larger steps converge faster; the mini-batch run trades a noisy curve for more updates per epoch.

## 2. Biological neuron

![Hodgkin-Huxley action potential and f-I curves](docs/neuron.png)

Hodgkin-Huxley with the standard squid-axon parameters ($C=1\,\mu\text{F/cm}^2$, $g_{Na}=120$, $g_K=36$, $g_L=0.3$ mS/cm$^2$, $E_{Na}=50$, $E_K=-77$, $E_L=-54.387$ mV):

$$C\dot V = I - g_{Na}m^3h\,(V-E_{Na}) - g_K n^4 (V-E_K) - g_L (V-E_L), \qquad \dot x = \alpha_x(V)(1-x) - \beta_x(V)\,x$$

for $x \in \{m,h,n\}$, integrated with classical RK4. The left panel shows a spike train after a current step: $m$ (sodium activation) rises first and drives the upstroke, then $h$ (sodium inactivation) falls and $n$ (potassium activation) rises and repolarise the membrane. The middle panel is the firing rate against injected current. HH switches from silence to about 60 Hz abruptly, at roughly 6.25 $\mu$A/cm$^2$ in my sweep. The right panel shows the leaky integrate-and-fire model for contrast,

$$\tau\dot V = -(V - V_{rest}) + RI, \qquad f = \left[t_{ref} + \tau \ln\frac{RI + V_{rest} - V_{reset}}{RI + V_{rest} - V_{th}}\right]^{-1},$$

whose rate starts continuously from zero at its threshold current. The simulated LIF rates lie on the analytic curve.

## 3. Neural network

![MNIST training curves and misclassified digits](docs/nn.png)

A fully connected ReLU network with He initialisation, softmax cross-entropy loss, and hand-written backpropagation. With $\delta_L = (\mathrm{softmax}(z_L) - \text{onehot}(y))/n$ the backward recursion is

$$\frac{\partial L}{\partial W_l} = a_{l-1}^\top \delta_l, \qquad \frac{\partial L}{\partial b_l} = \sum_i \delta_{l,i}, \qquad \delta_{l-1} = (\delta_l W_l^\top) \odot \mathbf 1[z_{l-1} > 0].$$

Optimisers are SGD with momentum and Adam, both on shuffled mini-batches. The network below has two hidden layers (784-256-128-10), was trained with Adam (learning rate 1e-3, batch 128, 15 epochs) on 55,000 training images with 5,000 held out for validation, and evaluated once on the 10,000 test images. The misclassified digits are mostly ones a person would also hesitate over.

## Results

| Check | Result |
| --- | --- |
| OLS coefficients vs `sklearn.LinearRegression` (diabetes, 10 features) | max abs difference 5.7e-13 |
| Ridge (alpha 1) vs `sklearn.Ridge` | max abs difference 6.0e-13 |
| Batch GD (standardised features) vs closed form | max abs difference 8e-13 after 20,000 steps |
| HH resting potential | -65.0 mV (integration settles at -64.996) |
| HH onset of sustained spiking | 6.25 $\mu$A/cm$^2$ (asserted to lie in 5 to 8) |
| RK4 order from step-halving | 3.78 and 3.89 (asserted 3.6 to 4.6) |
| LIF simulated vs analytic rate | within 0.5 percent |
| MLP gradient check, every layer | relative error below 1e-7 (asserted) |
| MNIST test accuracy (784-256-128-10, Adam, 15 epochs) | **97.81 percent** (219 errors in 10,000) |

The MNIST figure is a single run with seed 0 and no tuning on the test set; run-to-run variation is a few tenths of a percent.

## How it is verified

- **Regression:** coefficients and intercept compared with scikit-learn on the diabetes dataset (OLS, three ridge strengths, and badly scaled features); gradient descent must converge to the closed form, and a too-large learning rate must be flagged as divergent.
- **Neuron:** no spike for small current, resting potential near -65 mV, gating variables stay in [0, 1], repetitive spiking above the rheobase found by the code's own sweep, RK4 order measured by halving the step, LIF rate against the analytic formula.
- **Network:** central-difference numerical gradients for every weight and bias of every layer. To confirm that this test can fail, I once flipped a sign in the hidden-layer backward step: the check reported a relative error of 1.0 on the affected layer, and I restored the code afterwards. The network must also memorise a small random-label batch with both optimisers, and MNIST tests are skipped if `data/` has no cached files.
- **CI:** GitHub Actions runs `ruff check`, `ruff format --check` and `pytest`. Tests run offline (about 15 s); nothing is downloaded except by `python -m mlfs.nn`.

## Layout

```
src/mlfs/{regression,neuron,nn}.py   the three modules
tests/                               reference and property tests
docs/                                figures
data/                                MNIST cache (gitignored)
```

BLAS thread pools are capped at 4 by default (`OMP_NUM_THREADS` and friends can override this).

MIT license, Emilio Gappa, 2026.

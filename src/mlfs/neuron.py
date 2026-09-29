"""Hodgkin-Huxley and leaky integrate-and-fire neurons.

Hodgkin-Huxley (squid giant axon, modern sign convention, rest near -65 mV):

    C dV/dt = I - gNa m^3 h (V - ENa) - gK n^4 (V - EK) - gL (V - EL)
    dx/dt   = alpha_x(V) (1 - x) - beta_x(V) x,   x in {m, h, n}

integrated with classical fourth-order Runge-Kutta. Units: mV, ms, uA/cm^2, mS/cm^2, uF/cm^2.

Run ``python -m mlfs.neuron`` to reproduce ``docs/neuron.png``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HHParams:
    C: float = 1.0
    gNa: float = 120.0
    gK: float = 36.0
    gL: float = 0.3
    ENa: float = 50.0
    EK: float = -77.0
    EL: float = -54.387


def _vtrap(x: np.ndarray, y: float) -> np.ndarray:
    """x / (1 - exp(-x / y)), with its removable singularity at x = 0 handled (limit = y)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x / y) < 1e-6
    safe = np.where(small, 1.0, x)
    return np.where(small, y * (1 - x / (2 * y)), safe / -np.expm1(-safe / y))


def rates(V):
    """Voltage-dependent opening (alpha) and closing (beta) rates for m, h, n, in 1/ms."""
    am = 0.1 * _vtrap(V + 40.0, 10.0)
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = 0.01 * _vtrap(V + 55.0, 10.0)
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def steady_state(V):
    am, bm, ah, bh, an, bn = rates(V)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def hh_rhs(state: np.ndarray, I, p: HHParams = HHParams()) -> np.ndarray:
    """Right-hand side; ``state`` has shape (4, ...) holding V, m, h, n (broadcasts over I)."""
    V, m, h, n = state
    am, bm, ah, bh, an, bn = rates(V)
    i_ion = p.gNa * m**3 * h * (V - p.ENa) + p.gK * n**4 * (V - p.EK) + p.gL * (V - p.EL)
    return np.array(
        [
            (I - i_ion) / p.C,
            am * (1 - m) - bm * m,
            ah * (1 - h) - bh * h,
            an * (1 - n) - bn * n,
        ]
    )


def rk4_step(f, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_hh(
    I,
    T: float = 50.0,
    dt: float = 0.01,
    V0: float = -65.0,
    p: HHParams = HHParams(),
    current_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate HH from the resting steady state at V0.

    ``I`` is a constant current (scalar or array: one independent neuron per entry), unless
    ``current_fn(t)`` is given. Returns times (steps + 1,) and states (steps + 1, 4, *I.shape).
    """
    I = np.asarray(I, dtype=float)
    y = np.array([np.full(I.shape, V0), *[np.full(I.shape, x) for x in steady_state(V0)]])
    steps = int(round(T / dt))
    out = np.empty((steps + 1, *y.shape))
    out[0] = y
    for k in range(steps):
        if current_fn is None:
            y = rk4_step(lambda s: hh_rhs(s, I, p), y, dt)
        else:
            t = k * dt  # current sampled at step start; use constant-in-step drive
            y = rk4_step(lambda s, Ik=current_fn(t): hh_rhs(s, Ik, p), y, dt)
        out[k + 1] = y
    return np.linspace(0.0, steps * dt, steps + 1), out


def count_spikes(V: np.ndarray, t: np.ndarray, threshold: float = 0.0, t_min: float = 0.0):
    """Upward threshold crossings of V (time on axis 0) after t_min; works on arrays of neurons."""
    up = (
        (V[:-1] < threshold)
        & (V[1:] >= threshold)
        & (t[1:] > t_min).reshape((-1,) + (1,) * (V.ndim - 1))
    )
    return up.sum(axis=0)


def hh_firing_rate(I, T: float = 500.0, dt: float = 0.02, t_settle: float = 100.0) -> np.ndarray:
    """Steady firing rate in Hz for constant currents I: spikes after t_settle over the rest."""
    t, s = simulate_hh(I, T=T, dt=dt)
    return count_spikes(s[:, 0], t, t_min=t_settle) / ((T - t_settle) / 1000.0)


def hh_rheobase(lo: float = 3.0, hi: float = 10.0, **kw) -> float:
    """Smallest constant current with sustained repetitive spiking, found by a two-pass sweep.

    A coarse grid over [lo, hi] locates the first spiking current; a fine grid just below it
    refines the answer to 0.05 µA/cm². All currents in a pass are integrated at once.
    HH onset is a subcritical Hopf bifurcation, so this is measured from a step at rest.
    """
    kw = {"T": 300.0, "dt": 0.025, "t_settle": 100.0, **kw}
    coarse = np.arange(lo, hi + 1e-9, 0.5)
    first = coarse[np.argmax(hh_firing_rate(coarse, **kw) > 0)]
    fine = np.arange(first - 0.5, first + 1e-9, 0.05)
    return float(fine[np.argmax(hh_firing_rate(fine, **kw) > 0)])


# ---- leaky integrate-and-fire --------------------------------------------------------------


@dataclass(frozen=True)
class LIFParams:
    tau: float = 10.0  # membrane time constant, ms
    R: float = 10.0  # membrane resistance, MOhm  (R * I in mV for I in nA)
    V_rest: float = -65.0
    V_th: float = -50.0
    V_reset: float = -65.0
    t_ref: float = 2.0  # absolute refractory period, ms


def lif_rate_analytic(I, p: LIFParams = LIFParams()) -> np.ndarray:
    """f = 1 / (t_ref + tau ln[(RI + V_rest - V_reset) / (RI + V_rest - V_th)]) above threshold."""
    I = np.asarray(I, dtype=float)
    drive = p.R * I + p.V_rest
    active = drive > p.V_th
    with np.errstate(divide="ignore", invalid="ignore"):
        isi = p.t_ref + p.tau * np.log((drive - p.V_reset) / (drive - p.V_th))
    return np.where(active, 1000.0 / isi, 0.0)


def simulate_lif(I: float, T: float = 1000.0, dt: float = 0.01, p: LIFParams = LIFParams()):
    """Simulate tau dV/dt = -(V - V_rest) + R I with reset and refractory period.

    Input is constant, so each step uses the exact exponential solution of the linear ODE.
    Returns times, voltages and spike times (spikes are drawn at V_th, then reset).
    """
    steps = int(round(T / dt))
    decay = np.exp(-dt / p.tau)
    V_inf = p.V_rest + p.R * I
    V = np.empty(steps + 1)
    V[0] = p.V_rest
    spikes, ref_until = [], -1.0
    for k in range(steps):
        t = (k + 1) * dt
        if t <= ref_until:
            V[k + 1] = p.V_reset
            continue
        v = V_inf + (V[k] - V_inf) * decay
        if v >= p.V_th:
            spikes.append(t)
            v, ref_until = p.V_reset, t + p.t_ref
        V[k + 1] = v
    return np.linspace(0.0, steps * dt, steps + 1), V, np.array(spikes)


def main(out: str = "docs/neuron.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, s = simulate_hh(10.0, T=40.0, dt=0.01, current_fn=lambda t: 10.0 if t >= 5.0 else 0.0)
    V, m, h, n = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

    currents = np.arange(0.0, 41.0, 1.0)
    f_hh = hh_firing_rate(currents)
    rheo = hh_rheobase()
    lif_I = np.linspace(0, 4.0, 200)  # nA, same R I scale as the LIF parameters
    p = LIFParams()

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
    ax[0].plot(t, V, color="k", lw=1.5)
    ax[0].axhline(-65, color="gray", lw=0.6, ls=":")
    ax[0].set_xlabel("t (ms)")
    ax[0].set_ylabel("V (mV)")
    ax[0].set_title("Action potential (10 µA/cm² step at 5 ms)")
    tw = ax[0].twinx()
    for x, c, name in [(m, "#d62728", "m"), (h, "#1f77b4", "h"), (n, "#2ca02c", "n")]:
        tw.plot(t, x, color=c, lw=1.2, label=name)
    tw.set_ylabel("gating variable")
    tw.legend(loc="upper right", ncol=3, fontsize=8)

    ax[1].plot(currents, f_hh, "o-", ms=3, color="k", label="Hodgkin-Huxley")
    ax[1].axvline(rheo, color="gray", ls="--", lw=0.8)
    ax[1].text(rheo + 0.5, 5, f"onset ~ {rheo:.2f}", fontsize=8, color="gray")
    ax[1].set_xlabel("I (µA/cm²)")
    ax[1].set_ylabel("firing rate (Hz)")
    ax[1].set_title("f-I curve: HH (abrupt onset)")
    ax[1].grid(alpha=0.3)

    ax[2].plot(lif_I, lif_rate_analytic(lif_I, p), color="#1f77b4", label="LIF analytic")
    sim = [len(simulate_lif(i, T=1000.0, dt=0.01, p=p)[2]) for i in lif_I[::20]]
    ax[2].plot(lif_I[::20], sim, "o", color="#d62728", ms=4, label="LIF simulated")
    ax[2].axvline((p.V_th - p.V_rest) / p.R, color="gray", ls="--", lw=0.8)
    ax[2].set_xlabel("I (nA)")
    ax[2].set_ylabel("firing rate (Hz)")
    ax[2].set_title("f-I curve: LIF (continuous onset)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"HH sustained-spiking onset: {rheo:.2f} µA/cm²; wrote {out}")


if __name__ == "__main__":
    main()

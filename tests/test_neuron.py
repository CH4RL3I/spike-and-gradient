import numpy as np
import pytest

from mlfs.neuron import (
    LIFParams,
    count_spikes,
    hh_firing_rate,
    hh_rheobase,
    lif_rate_analytic,
    rates,
    simulate_hh,
    simulate_lif,
    steady_state,
)


def test_rate_functions_finite_at_singularities():
    for v in (-55.0, -40.0):
        assert all(np.isfinite(r) for r in rates(np.array(v)))
    a, b = rates(np.array(-40.0))[0], rates(np.array(-40.0 + 1e-4))[0]
    assert a == pytest.approx(b, rel=1e-4)


def test_resting_potential_near_minus_65():
    t, s = simulate_hh(0.0, T=100.0, dt=0.05)
    assert s[-1, 0] == pytest.approx(-65.0, abs=1.0)
    assert count_spikes(s[:, 0], t) == 0
    m, h, n = steady_state(-65.0)
    assert (m, h, n) == pytest.approx((0.053, 0.596, 0.318), abs=1e-3)


def test_no_spike_for_small_current():
    t, s = simulate_hh(np.array([1.0, 2.0]), T=200.0, dt=0.05)
    assert count_spikes(s[:, 0], t).tolist() == [0, 0]
    assert s[:, 0].max() < -40.0


def test_action_potential_shape():
    t, s = simulate_hh(10.0, T=30.0, dt=0.01)
    V = s[:, 0]
    assert 20.0 < V.max() < 50.0  # overshoot toward ENa = 50 mV
    assert V.min() < -70.0  # afterhyperpolarisation toward EK = -77 mV
    assert np.all((s[:, 1:] >= 0) & (s[:, 1:] <= 1))  # gating variables are probabilities


def test_repetitive_spiking_above_rheobase_and_rheobase_range():
    rheo = hh_rheobase()
    assert 5.0 < rheo < 8.0  # literature: about 6-7 uA/cm2 for the standard parameters
    assert hh_firing_rate(np.array([rheo - 1.0]), T=300, dt=0.025)[0] == 0
    rates_hz = hh_firing_rate(np.array([rheo + 1.0, 15.0]), T=300, dt=0.025)
    assert np.all(rates_hz > 40)  # HH onset is abrupt, at roughly 50-60 Hz


def test_rk4_is_fourth_order():
    def v_end(dt):
        _, s = simulate_hh(10.0, T=8.0, dt=dt)
        return s[-1]

    dts = [0.05, 0.025, 0.0125, 0.00625]
    ends = [v_end(dt) for dt in dts]
    errs = [np.abs(ends[i] - ends[i + 1]).max() for i in range(3)]  # step-halving differences
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(3.6 < o < 4.6 for o in orders), orders


@pytest.mark.parametrize("I", [1.6, 2.0, 3.0])
def test_lif_rate_matches_analytic(I):
    p = LIFParams()
    _, _, spikes = simulate_lif(I, T=2000.0, dt=0.01, p=p)
    rate = len(spikes) / 2.0
    assert rate == pytest.approx(float(lif_rate_analytic(I, p)), rel=0.02)


def test_lif_silent_below_threshold():
    p = LIFParams()
    I = 0.99 * (p.V_th - p.V_rest) / p.R
    assert lif_rate_analytic(I, p) == 0
    assert len(simulate_lif(I, T=500.0, p=p)[2]) == 0

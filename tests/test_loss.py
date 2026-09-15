"""T5 catastrophe loss: vulnerability, ground-up loss, AAL, OEP/AEP curves."""
from __future__ import annotations

import numpy as np
import pytest

from vhagar.models import loss


def test_vulnerability_monotone_bounded():
    dr = loss.vulnerability_damage_ratio(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.all(np.diff(dr) > 0)
    assert dr.min() >= 0.0 and dr.max() <= 1.0
    assert abs(dr[2] - 0.5) < 1e-9                 # midpoint -> 50% damage


def test_ground_up_and_structure_loss():
    assert loss.ground_up_loss([0.0, 0.5, 1.0], [100.0, 200.0, 300.0]) == 400.0
    assert loss.structure_event_loss([True, False, True], [100.0, 200.0, 300.0]) == 400.0
    with pytest.raises(ValueError):
        loss.ground_up_loss([0.5], [1.0, 2.0])


def test_average_annual_loss():
    assert loss.average_annual_loss([100.0, 400.0], [0.1, 0.01]) == 14.0
    assert loss.average_annual_loss([100.0, 200.0], 0.5) == 150.0    # scalar rate


def test_oep_curve_closed_form():
    c = loss.oep_curve([100.0, 400.0], [0.1, 0.02])
    assert np.allclose(c["threshold"], [0.0, 100.0, 400.0])
    assert abs(c["oep"][0] - (1 - np.exp(-0.12))) < 1e-9   # both events qualify
    assert abs(c["oep"][1] - (1 - np.exp(-0.02))) < 1e-9   # only the 400 event
    assert c["oep"][-1] == 0.0                              # above max loss
    assert abs(c["return_period"][1] - 1.0 / (1 - np.exp(-0.02))) < 1e-6
    assert np.isinf(c["return_period"][-1])
    assert c["aal"] == loss.average_annual_loss([100.0, 400.0], [0.1, 0.02])


def test_oep_monotone_non_increasing():
    c = loss.oep_curve([50.0, 100.0, 400.0, 1000.0], [0.2, 0.1, 0.02, 0.005])
    assert np.all(np.diff(c["oep"]) <= 1e-12)              # exceedance falls with threshold


def test_aep_matches_analytic_aal_and_monotone():
    losses = [100.0, 400.0, 1000.0]
    rates = [0.1, 0.02, 0.005]
    c = loss.aep_curve(losses, rates, n_years=200_000, seed=1)
    # analytic AAL = 10 + 8 + 5 = 23; MC AAL should be within a few percent
    assert abs(c["aal_analytic"] - 23.0) < 1e-9
    assert abs(c["aal"] - c["aal_analytic"]) / c["aal_analytic"] < 0.05
    assert np.all(np.diff(c["aep"]) <= 1e-12)             # non-increasing in threshold
    assert c["aep"].min() >= 0.0 and c["aep"].max() <= 1.0

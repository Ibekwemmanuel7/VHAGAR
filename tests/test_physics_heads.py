"""Physics-constrained head tests. Skipped cleanly without the torch extra."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vhagar.models.physics_heads import (  # noqa: E402
    CensoredMSELoss,
    ConstrainedFRPHead,
    PhysicsConsistencyLoss,
    PlanckMixtureDecoder,
    planck_radiance_torch,
)
from vhagar.physics.planck import planck_radiance  # noqa: E402


def test_torch_planck_matches_numpy_planck():
    for lam in (3.9, 11.0):
        for t in (250.0, 300.0, 900.0, 1300.0):
            got = float(planck_radiance_torch(lam, torch.tensor([t])))
            want = float(planck_radiance(lam, t))
            assert got == pytest.approx(want, rel=1e-6)


def test_torch_planck_is_differentiable_and_increasing_in_temperature():
    t = torch.tensor([600.0], requires_grad=True)
    planck_radiance_torch(3.9, t).backward()
    assert float(t.grad) > 0


def test_constrained_head_starts_as_pure_physics():
    """Zero-initialised correction: the model begins exactly at Wooster."""
    head = ConstrainedFRPHead(in_features=6)
    z = torch.randn(8, 6)
    lm, lb = torch.full((8,), 0.35), torch.full((8,), 0.02)
    area, tau = torch.full((8,), 140625.0), torch.full((8,), 0.69)
    assert torch.allclose(head(z, lm, lb, area, tau), head.physics_only(lm, lb, area, tau))


def test_constrained_head_cannot_produce_negative_frp():
    head = ConstrainedFRPHead(in_features=4)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 5.0)
    out = head(
        torch.randn(32, 4),
        torch.rand(32) * 0.5,
        torch.rand(32) * 0.5,     # background may exceed the pixel
        torch.full((32,), 1e5),
        torch.full((32,), 0.7),
    )
    assert bool((out >= 0).all())


def test_constrained_head_correction_is_bounded():
    """The learned term may adjust FRP by at most a factor of 3 either way."""
    head = ConstrainedFRPHead(in_features=4, max_log_correction=1.0986)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 20.0)
    z = torch.randn(64, 4)
    lm, lb = torch.full((64,), 0.35), torch.full((64,), 0.02)
    area, tau = torch.full((64,), 1e5), torch.full((64,), 0.7)
    ratio = head(z, lm, lb, area, tau) / head.physics_only(lm, lb, area, tau)
    assert bool(((ratio > 1 / 3.01) & (ratio < 3.01)).all())


def test_head_scales_correctly_with_area_and_transmittance():
    head = ConstrainedFRPHead(in_features=3)
    lm, lb = torch.tensor([0.35]), torch.tensor([0.02])
    base = head.physics_only(lm, lb, torch.tensor([1e5]), torch.tensor([1.0]))
    assert float(head.physics_only(lm, lb, torch.tensor([2e5]), torch.tensor([1.0]))) == pytest.approx(
        2 * float(base)
    )
    assert float(head.physics_only(lm, lb, torch.tensor([1e5]), torch.tensor([0.5]))) == pytest.approx(
        2 * float(base)
    )


def test_decoder_round_trips_a_known_state():
    dec = PlanckMixtureDecoder()
    p, tf, tb = torch.tensor([0.004]), torch.tensor([900.0]), torch.tensor([300.0])
    lm, lt = dec(p, tf, tb)
    pv = float(p)
    want = pv * float(planck_radiance(3.9, 900.0)) + (1 - pv) * float(planck_radiance(3.9, 300.0))
    assert float(lm) == pytest.approx(want, rel=1e-6)
    assert float(lt) > 0


def test_decoder_gradients_flow_to_all_three_state_variables():
    dec = PlanckMixtureDecoder()
    p = torch.tensor([0.004], requires_grad=True)
    tf = torch.tensor([900.0], requires_grad=True)
    tb = torch.tensor([300.0], requires_grad=True)
    lm, lt = dec(p, tf, tb)
    (lm + lt).sum().backward()
    for g in (p.grad, tf.grad, tb.grad):
        assert g is not None and torch.isfinite(g).all() and float(g.abs()) > 0


def test_latent_mapping_stays_inside_the_physical_box():
    dec = PlanckMixtureDecoder()
    p, tf, tb = dec.from_latent(torch.randn(500, 3) * 50.0)
    assert bool(((p > 0) & (p <= 1)).all())
    assert bool(((tf >= 600.0) & (tf <= 1400.0)).all())
    assert bool(((tb >= 250.0) & (tb <= 350.0)).all())


def test_learned_inversion_can_actually_fit_a_synthetic_pixel():
    """End-to-end sanity: optimise the latent through the fixed physics decoder."""
    torch.manual_seed(0)
    dec = PlanckMixtureDecoder()
    p_true, tf_true, tb_true = 0.004, 950.0, 300.0
    lm_obs, lt_obs = dec(
        torch.tensor([p_true]), torch.tensor([tf_true]), torch.tensor([tb_true])
    )
    z = torch.zeros(1, 3, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.15)
    for _ in range(1500):
        opt.zero_grad()
        p, tf, tb = dec.from_latent(z)
        lm, lt = dec(p, tf, tb)
        # Log-space loss: radiances span three orders of magnitude between the
        # two channels, so a plain MSE would optimise TIR and ignore MIR.
        loss = (torch.log(lm) - torch.log(lm_obs)) ** 2 + (torch.log(lt) - torch.log(lt_obs)) ** 2
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        p, tf, _ = dec.from_latent(z)
    assert float(p) == pytest.approx(p_true, rel=0.25)
    assert float(tf) == pytest.approx(tf_true, rel=0.15)


def test_frp_from_state_matches_stefan_boltzmann():
    got = float(PlanckMixtureDecoder.frp_from_state(
        torch.tensor([0.004]), torch.tensor([900.0]), torch.tensor([1e5])
    ))
    want = 5.670374419e-8 * 0.004 * 1e5 * 900.0**4 / 1e6
    assert got == pytest.approx(want, rel=1e-6)


def test_censored_loss_forgives_over_prediction_on_saturated_pixels():
    loss = CensoredMSELoss()
    pred = torch.tensor([400.0, 400.0])
    obs = torch.tensor([311.0, 311.0])
    censored = torch.tensor([True, False])
    assert float(loss(pred[:1], obs[:1], censored[:1])) == 0.0
    assert float(loss(pred[1:], obs[1:], censored[1:])) > 0.0


def test_censored_loss_still_penalises_under_prediction_when_saturated():
    loss = CensoredMSELoss()
    v = float(loss(torch.tensor([280.0]), torch.tensor([311.0]), torch.tensor([True])))
    assert v == pytest.approx(31.0**2)


def test_physics_consistency_loss_is_zero_when_the_two_routes_agree():
    loss = PhysicsConsistencyLoss()
    a = torch.tensor([10.0, 50.0, 100.0])
    assert float(loss(a, a)) == pytest.approx(0.0)
    assert float(loss(a, a * 2.0)) > 0.0

"""Physics-constrained model heads and a differentiable forward model.

Three ideas, in increasing ambition and decreasing certainty of payoff.

1. :class:`ConstrainedFRPHead`. **the one I would actually ship.** Rather than
   regressing FRP from scratch, keep the Wooster equation and learn a *bounded
   multiplicative correction*:

       FRP_hat = (A_pix * sigma) / (a * tau) * (L_MIR - L_bg) * g(z),   g > 0

   Physical structure comes free (monotonicity in radiance contrast and in
   1/tau, positivity, correct area and transmittance scaling); the network only
   has to learn what the physics leaves out. It cannot produce a negative FRP or
   invert the response to radiance, whatever the training data does.

2. :class:`PlanckMixtureDecoder`, a **non-trainable, differentiable** forward
   model. Encoder predicts ``(p, T_f, T_b)``, this renders the radiances back,
   and the loss is in radiance space. This is the encoder + fixed-physics-decoder
   autoencoder that has worked well elsewhere in EO retrieval (the PROSAIL
   inversion literature), and it fits fire *better* than it fits vegetation
   because the fire forward model is analytically simple -- a two-component
   Planck mixture. You do not need a neural emulator to make it differentiable;
   you need about thirty lines of PyTorch, which is what this is.

3. :class:`CensoredMSELoss`, saturation is censoring, not noise. Above the
   sensor's MIR saturation temperature the truth is known only to be
   ``>= observed``, so a prediction that exceeds it should not be penalised.
   Without this, a model systematically under-predicts the largest fires.

**Honest expectation.** For *detection and false alarms*, a well-featured
gradient-boosted tree using :mod:`vhagar.features.physics_features` will be
close to anything here; the physics-in-the-loop payoff is in FRP accuracy,
sub-pixel characterisation, and calibrated uncertainty. There is no published
physics-informed neural network for wildfire detection or FRP retrieval, so
this is genuine research risk -- structure the programme so stages 1-3 of
``docs/03_PHYSICS.md`` deliver value on their own.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    nn = object  # type: ignore[assignment]

from vhagar.physics.planck import C1_L, C2_UM_K, STEFAN_BOLTZMANN

__all__ = [
    "CensoredMSELoss",
    "ConstrainedFRPHead",
    "PlanckMixtureDecoder",
    "planck_radiance_torch",
]


if _TORCH:

    def planck_radiance_torch(wavelength_um: float, temperature_k: torch.Tensor) -> torch.Tensor:
        """Planck function as a differentiable torch op.

        ``expm1`` is used rather than ``exp(x) - 1`` because at 11 um / 300 K,
        ``x ~ 4.4`` and at 3.9 um / 300 K, ``x ~ 12.3`` -- the naive form loses
        precision exactly where the fire signal lives.
        """
        t = torch.clamp(temperature_k, min=1.0)
        x = C2_UM_K / (wavelength_um * t)
        return C1_L / (wavelength_um**5 * torch.expm1(x))

    class ConstrainedFRPHead(nn.Module):
        """Wooster FRP with a learned, bounded multiplicative correction.

        The network predicts ``log g`` and the head returns
        ``FRP_wooster * exp(clamp(log g))``. ``max_log_correction`` caps how far
        the model may depart from physics -- default ``ln(3)``, i.e. the learned
        term can adjust FRP by at most a factor of 3 either way. If your model
        wants more than that, the problem is your inputs, not your capacity.

        >>> import torch
        >>> head = ConstrainedFRPHead(in_features=8)
        >>> z = torch.zeros(4, 8)
        >>> frp = head(z, l_mir=torch.full((4,), 0.35), l_bg=torch.full((4,), 0.02),
        ...            pixel_area_m2=torch.full((4,), 140625.0),
        ...            transmittance=torch.full((4,), 0.69))
        >>> bool((frp > 0).all())
        True
        """

        def __init__(
            self,
            in_features: int,
            hidden: int = 64,
            a_constant: float = 3.0e-9,
            max_log_correction: float = 1.0986,  # ln(3)
        ) -> None:
            super().__init__()
            self.a_constant = a_constant
            self.max_log_correction = max_log_correction
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            # Start at g = 1 exactly: the model begins as pure physics and has
            # to earn every departure from it.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, features, l_mir, l_bg, pixel_area_m2, transmittance):
            contrast = torch.clamp(l_mir - l_bg, min=0.0)
            frp_physics = (
                pixel_area_m2 * STEFAN_BOLTZMANN / (self.a_constant * torch.clamp(transmittance, min=1e-3))
            ) * contrast / 1e6
            log_g = torch.tanh(self.net(features).squeeze(-1)) * self.max_log_correction
            return frp_physics * torch.exp(log_g)

        def physics_only(self, l_mir, l_bg, pixel_area_m2, transmittance):
            """The uncorrected Wooster FRP, your permanent baseline."""
            contrast = torch.clamp(l_mir - l_bg, min=0.0)
            return (
                pixel_area_m2 * STEFAN_BOLTZMANN / (self.a_constant * torch.clamp(transmittance, min=1e-3))
            ) * contrast / 1e6

    class PlanckMixtureDecoder(nn.Module):
        """Non-trainable differentiable forward model: state -> radiances.

        Given ``(p, T_f, T_b)`` and a transmittance, render top-of-atmosphere
        MIR and TIR radiance. Put this after an encoder and train on the
        radiance reconstruction loss and you have a learned inversion whose
        outputs are physically consistent by construction.

        Parameterisation matters: the encoder should emit *unconstrained* reals
        which :meth:`from_latent` maps into the physical box via sigmoid/softplus.
        Predicting ``p`` directly in [0, 1] with a clamp gives dead gradients at
        the boundary, and ``p`` spans several decades, so it lives in log space.

        >>> import torch
        >>> dec = PlanckMixtureDecoder()
        >>> lm, lt = dec(torch.tensor([0.004]), torch.tensor([900.0]), torch.tensor([300.0]))
        >>> bool(lm > 0 and lt > 0)
        True
        """

        def __init__(
            self,
            lam_mir_um: float = 3.9,
            lam_tir_um: float = 11.0,
            t_fire_range: tuple[float, float] = (600.0, 1400.0),
            log_p_range: tuple[float, float] = (-9.0, 0.0),
        ) -> None:
            super().__init__()
            self.lam_mir = lam_mir_um
            self.lam_tir = lam_tir_um
            self.t_fire_range = t_fire_range
            self.log_p_range = log_p_range

        def from_latent(self, z):
            """Map 3 unconstrained reals -> (p, T_f, T_b) inside the physical box."""
            lo_t, hi_t = self.t_fire_range
            lo_p, hi_p = self.log_p_range
            log_p = lo_p + (hi_p - lo_p) * torch.sigmoid(z[..., 0])
            t_f = lo_t + (hi_t - lo_t) * torch.sigmoid(z[..., 1])
            t_b = 250.0 + 100.0 * torch.sigmoid(z[..., 2])
            return torch.exp(log_p), t_f, t_b

        def forward(self, fire_fraction, t_fire_k, t_background_k, transmittance=1.0):
            p = fire_fraction
            l_mir = p * planck_radiance_torch(self.lam_mir, t_fire_k) + (1 - p) * planck_radiance_torch(
                self.lam_mir, t_background_k
            )
            l_tir = p * planck_radiance_torch(self.lam_tir, t_fire_k) + (1 - p) * planck_radiance_torch(
                self.lam_tir, t_background_k
            )
            return l_mir * transmittance, l_tir * transmittance

        @staticmethod
        def frp_from_state(fire_fraction, t_fire_k, pixel_area_m2, emissivity: float = 1.0):
            """Stefan-Boltzmann FRP implied by the retrieved state, in MW.

            Use this as a *consistency* term against the Wooster FRP: the two
            routes to FRP must agree, and penalising their disagreement is a
            physically meaningful regulariser that costs nothing to compute.
            """
            return (
                emissivity
                * STEFAN_BOLTZMANN
                * fire_fraction
                * pixel_area_m2
                * t_fire_k**4
            ) / 1e6

    class CensoredMSELoss(nn.Module):
        """MSE that does not penalise over-prediction on saturated pixels.

        Above the sensor's MIR saturation temperature the observation is a lower
        bound, not a measurement. Treating the clipped value as truth teaches
        the model to under-predict exactly the largest, most consequential fires.

        >>> import torch
        >>> loss = CensoredMSELoss()
        >>> pred = torch.tensor([400.0]); obs = torch.tensor([311.0])
        >>> cens = torch.tensor([True])
        >>> float(loss(pred, obs, cens))
        0.0
        """

        def __init__(self, reduction: str = "mean") -> None:
            super().__init__()
            self.reduction = reduction

        def forward(self, prediction, observation, censored):
            residual = prediction - observation
            # Censored and predicting above the bound -> consistent -> no penalty.
            residual = torch.where(censored & (residual > 0), torch.zeros_like(residual), residual)
            sq = residual**2
            if self.reduction == "mean":
                return sq.mean()
            if self.reduction == "sum":
                return sq.sum()
            return sq

    class PhysicsConsistencyLoss(nn.Module):
        """Penalise disagreement between the two independent routes to FRP.

        Route A: Wooster, from observed MIR radiance contrast.
        Route B: Stefan-Boltzmann, from the retrieved ``(p, T_f)``.

        They are derived from the same physics under different approximations,
        so their ratio should sit near 1 wherever the retrieval is trustworthy.
        Computed in log space because FRP errors are multiplicative.
        """

        def forward(self, frp_wooster, frp_from_state, weight=None):
            eps = 1e-6
            r = torch.log(torch.clamp(frp_from_state, min=eps)) - torch.log(
                torch.clamp(frp_wooster, min=eps)
            )
            sq = r**2
            return (sq * weight).mean() if weight is not None else sq.mean()

else:  # pragma: no cover

    def _missing(*_a, **_k):
        raise ImportError("vhagar.models.physics_heads requires torch: pip install 'vhagar[torch]'")

    planck_radiance_torch = _missing  # type: ignore[assignment]
    ConstrainedFRPHead = _missing  # type: ignore[assignment]
    PlanckMixtureDecoder = _missing  # type: ignore[assignment]
    CensoredMSELoss = _missing  # type: ignore[assignment]
    PhysicsConsistencyLoss = _missing  # type: ignore[assignment]

import astropy.constants as const
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from optax import adam
from spectracles import (
    FourierGP,
    Kernel,
    Parameter,
    PhaseConfig,
    SpatialDataLVM,
    SpatialModel,
    SpectralSpatialModel,
    l_bounded,
)

A_LOWER = 1e-5

C_KMS = const.c.to_value("km/s")

N_MODES = (101, 101)
rng = np.random.default_rng(0)


def neg_ln_posterior(model, λ, xy_data, data, u_data, mask):
    # Model predictions
    pred = jax.vmap(model, in_axes=(0, None))(λ, xy_data)
    # Likelihood
    ln_like = jnp.sum(
        jnp.where(
            mask,
            jax.scipy.stats.norm.logpdf(x=pred, loc=data, scale=u_data),
            0.0,
        )
    )
    ln_prior_line1 = model.line1.A_raw.prior_logpdf() + model.line1.v.prior_logpdf()
    ln_prior_line2 = model.line2.A_raw.prior_logpdf() + model.line2.v.prior_logpdf()
    return -1 * (ln_like + ln_prior_line1 + ln_prior_line2)


class EmissionLine(SpectralSpatialModel):
    # Line centre in Angstroms
    μ: Parameter
    # Model components / line quantities
    A_raw: SpatialModel  # Unconstrained line flux
    v: SpatialModel  # Radial velocity in rest frame in km/s
    σ_lsf: Parameter  # Global LSF width in Angstroms

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data)
        peak = self.A(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        return peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def v_obs(self, s) -> Array:
        return self.v(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s) -> Array:
        return self.σ_lsf.val**2


class TwoLineMixture(SpectralSpatialModel):
    # Model components
    line1: EmissionLine  # Emission line model
    line2: EmissionLine  # Emission line model

    def __init__(
        self,
        line_centre1: Parameter,
        line_centre2: Parameter,
        n_modes: tuple[int, int],
        A_kernel1: Kernel,
        A_kernel2: Kernel,
        v_kernel1: Kernel,
        v_kernel2: Kernel,
        σ_lsf1: Parameter,
        σ_lsf2: Parameter,
    ):
        self.line1 = EmissionLine(
            μ=line_centre1,
            A_raw=FourierGP(n_modes=n_modes, kernel=A_kernel1),
            v=FourierGP(n_modes=n_modes, kernel=v_kernel1),
            σ_lsf=σ_lsf1,
        )
        self.line2 = EmissionLine(
            μ=line_centre2,
            A_raw=FourierGP(n_modes=n_modes, kernel=A_kernel2),
            v=FourierGP(n_modes=n_modes, kernel=v_kernel2),
            σ_lsf=σ_lsf2,
        )

    def __call__(self, λ, spatial_data):
        return self.line1(λ, spatial_data) + self.line2(λ, spatial_data)


Δloss = 1e-2


# TODO: Tune optimisation phases to match the new model

N_STEPS = 500


A_coeffs_init = PhaseConfig(
    n_steps=N_STEPS,
    optimiser=adam(1e-2),
    Δloss_criterion=Δloss,
    fix_status_updates={
        # Allowed to vary:
        "line1.A_raw.coefficients": False,
        "line2.A_raw.coefficients": False,
        # Fixed:
        "line1.σ_lsf": True,
        "line2.σ_lsf": True,
        "line1.v.coefficients": True,
        "line2.v.coefficients": True,
    },
    param_val_updates={
        "line1.A_raw.coefficients": jnp.array(rng.standard_normal(N_MODES)),
        "line2.A_raw.coefficients": jnp.array(rng.standard_normal(N_MODES)),
    },
)
v_coeffs_init = PhaseConfig(
    n_steps=N_STEPS,
    optimiser=adam(1e-2),
    Δloss_criterion=Δloss,
    fix_status_updates={
        # Allowed to vary:
        "line1.v.coefficients": False,
        "line2.v.coefficients": False,
        # Fixed:
        "line1.σ_lsf": True,
        "line2.σ_lsf": True,
        "line1.A_raw.coefficients": True,
        "line2.A_raw.coefficients": True,
    },
    param_val_updates={
        "line1.v.coefficients": jnp.array(rng.standard_normal(N_MODES)),
        "line2.v.coefficients": jnp.array(rng.standard_normal(N_MODES)),
    },
)
both_coeffs = PhaseConfig(
    n_steps=N_STEPS,
    optimiser=adam(1e-2),
    Δloss_criterion=Δloss,
    fix_status_updates={
        # Allowed to vary:
        "line1.A_raw.coefficients": False,
        "line2.A_raw.coefficients": False,
        "line1.v.coefficients": False,
        "line2.v.coefficients": False,
        # Fixed:
        "line1.σ_lsf": True,
        "line2.σ_lsf": True,
    },
)
A_coeffs = PhaseConfig(
    n_steps=N_STEPS,
    optimiser=adam(1e-3),
    Δloss_criterion=Δloss,
    fix_status_updates={
        # Allowed to vary:
        "line1.A_raw.coefficients": False,
        "line2.A_raw.coefficients": False,
        # Fixed:
        "line1.σ_lsf": True,
        "line2.σ_lsf": True,
        "line1.v.coefficients": True,
        "line2.v.coefficients": True,
    },
)
v_coeffs = PhaseConfig(
    n_steps=N_STEPS,
    optimiser=adam(1e-3),
    Δloss_criterion=Δloss,
    fix_status_updates={
        # Allowed to vary:
        "line1.v.coefficients": False,
        "line2.v.coefficients": False,
        # Fixed:
        "line1.σ_lsf": True,
        "line2.σ_lsf": True,
        "line1.A_raw.coefficients": True,
        "line2.A_raw.coefficients": True,
    },
)

all_phases = [
    A_coeffs_init,
    v_coeffs_init,
    both_coeffs,
    A_coeffs,
    v_coeffs,
]

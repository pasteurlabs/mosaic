"""External state-VJP intervention; production solver and benchmark are untouched."""
from __future__ import annotations

import hashlib
import threading
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.fft import dctn, idctn

BANDS = {"large": (0.0, 4.0), "medium": (4.0, 16.0), "small": (16.0, float("inf"))}


def mask(q, band):
    low, high = BANDS[band]
    return (q > low) & (q <= high)


@lru_cache(maxsize=4)
def bases(n):
    k = np.arange(1, n, dtype=np.float64)
    sine = np.sqrt(2/n)*np.sin(np.pi*np.arange(1, n)[:, None]*k[None, :]/n)
    cosine = np.sqrt(2/n)*np.cos(np.pi*(np.arange(n)[:, None]+0.5)*k[None, :]/n)
    lam = 2*np.sin(np.pi*k/(2*n))
    denominator = np.sqrt(lam[:, None]**2 + lam[None, :]**2)
    q = np.sqrt(k[:, None]**2 + k[None, :]**2)
    scalar_k = np.arange(n)
    scalar_q = np.sqrt(scalar_k[:, None]**2 + scalar_k[None, :]**2)
    return sine, cosine, lam, denominator, q, scalar_q


def coefficients(n, seed, update, step):
    # Band is deliberately absent: each paired arm filters the same draws.
    rng = np.random.default_rng(np.random.SeedSequence([seed, update, step]))
    return rng.standard_normal((n-1, n-1)), rng.standard_normal((n, n))


def fields(n, seed, update, step, band):
    sine, _, _, denominator, q, scalar_q = bases(n)
    velocity_coeff, scalar_coeff = coefficients(n, seed, update, step)
    velocity_coeff *= mask(q, band)
    scalar_coeff *= mask(scalar_q, band)
    # Divide by each mode's discrete curl norm BEFORE mixing Gaussian draws.
    # These are compatible solenoidal MAC bands, not no-slip Stokes eigenmodes.
    psi = np.pad(sine @ (velocity_coeff/denominator) @ sine.T, ((1, 1), (1, 1)))
    velocity = np.stack((np.diff(psi, axis=1)[1:], -np.diff(psi, axis=0)[:, 1:]), axis=-1)
    scalar = idctn(scalar_coeff, norm="ortho")
    return velocity, scalar, velocity_coeff, scalar_coeff


def projected_velocity_norm(velocity):
    n = velocity.shape[0]
    sine, cosine, lam, denominator, _, _ = bases(n)
    u = sine.T @ velocity[:-1, :, 0] @ cosine
    v = cosine.T @ velocity[:, :-1, 1] @ sine
    coefficients = (u*lam[None, :] - v*lam[:, None])/denominator
    return float(np.linalg.norm(coefficients))


def centered_velocity(velocity):
    ux = np.pad(velocity[..., 0], ((1, 0), (0, 0)))
    uy = np.pad(velocity[..., 1], ((0, 0), (1, 0)))
    return np.stack((0.5*(ux[1:]+ux[:-1]), 0.5*(uy[:, 1:]+uy[:, :-1])), axis=-1)


def spectrum(field):
    q = bases(field.shape[0])[-1]
    coeff = dctn(field, axes=(0, 1), norm="ortho")
    energy = np.square(coeff)
    if energy.ndim == 3:
        energy = energy.sum(axis=-1)
    total = float(energy.sum())
    return {**{band: float(energy[mask(q, band)].sum()/total) if total else 0.0 for band in BANDS},
            "dc": float(energy[0, 0]/total) if total else 0.0}


def perturb(clean, band, sigma, seed, update, step):
    """Independent velocity/scalar budgets use FULL clean field L2 norms."""
    clean = np.asarray(clean)
    if clean.ndim != 3 or clean.shape[0] != clean.shape[1] or clean.shape[-1] != 3:
        raise ValueError("expected canonical square MAC state cotangent")
    original = clean.astype(np.float64)
    velocity_norm = float(np.linalg.norm(original[..., :2]))
    scalar_norm = float(np.linalg.norm(original[..., 2]))
    record = {"update": update, "step": step, "band": band, "sigma": sigma,
              "clean_velocity_l2": velocity_norm, "clean_scalar_l2": scalar_norm,
              "clean_projected_velocity_l2": projected_velocity_norm(original[..., :2])}
    if sigma == 0:
        record.update(velocity_relative_l2=0.0, scalar_relative_l2=0.0)
        return clean, record
    velocity, scalar, vc, sc = fields(clean.shape[0], seed, update, step, band)
    velocity *= sigma*velocity_norm/max(float(np.linalg.norm(velocity)), 1e-300)
    scalar *= sigma*scalar_norm/max(float(np.linalg.norm(scalar)), 1e-300)
    intended = np.concatenate((velocity, scalar[..., None]), axis=-1)
    corrupted = (original+intended).astype(clean.dtype)
    realized = corrupted.astype(np.float64)-original
    if not np.isfinite(corrupted).all():
        raise FloatingPointError("nonfinite corrupted state VJP")
    n = clean.shape[0]
    ux = np.pad(realized[..., 0], ((1, 0), (0, 0)))
    uy = np.pad(realized[..., 1], ((0, 0), (1, 0)))
    div = np.diff(ux, axis=0)+np.diff(uy, axis=1)
    q, sq = bases(n)[-2:]
    record.update(
        velocity_relative_l2=float(np.linalg.norm(realized[..., :2]))/velocity_norm if velocity_norm else 0.0,
        scalar_relative_l2=float(np.linalg.norm(realized[..., 2]))/scalar_norm if scalar_norm else 0.0,
        velocity_mode_outside_energy=float(np.square(vc[~mask(q, band)]).sum()),
        scalar_mode_outside_energy=float(np.square(sc[~mask(sq, band)]).sum()),
        scalar_dc_coefficient=float(sc[0, 0]),
        noise_divergence_l2_relative=float(np.linalg.norm(div))/max(float(np.linalg.norm(realized[..., :2])), 1e-300),
        noise_normal_wall_max=float(max(np.abs(realized[-1, :, 0]).max(), np.abs(realized[:, -1, 1]).max())),
        velocity_centered_dct_energy_fractions=spectrum(centered_velocity(realized[..., :2])),
        scalar_dct_energy_fractions=spectrum(realized[..., 2]),
    )
    return corrupted, record


def state_digest(state):
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()


class Controller:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.arm = "clean"
        self.sigma = 0.0
        self.noise_seed = 17000
        self.update = 0
        self.forward_index = 0
        self.pending = defaultdict(list)
        self.records = []
        self.capture = False
        self.trace = []

    def configure(self, arm, model_seed):
        self.disable()
        self.arm = arm
        self.sigma = 0.0 if arm == "clean" else 0.1
        self.noise_seed = 17000 + model_seed
        self.records = []

    def begin(self, update):
        with self.lock:
            self.active = True
            self.update = int(update)
            self.forward_index = 0
            self.pending.clear()

    def disable(self):
        with self.lock:
            self.active = False
            self.pending.clear()

    def forward(self, state, result):
        with self.lock:
            if self.capture:
                self.trace.append((state_digest(state), state_digest(result)))
            if self.active:
                self.pending[state_digest(state)].append(self.forward_index)
                self.forward_index += 1

    def backward(self, state, gradient):
        with self.lock:
            if not self.active:
                return gradient
            key = state_digest(state)
            if not self.pending[key]:
                raise RuntimeError("state VJP had no paired forward call; refusing unkeyed noise")
            step = self.pending[key].pop()
            corrupted, record = perturb(gradient, self.arm, self.sigma, self.noise_seed, self.update, step)
            self.records.append(record)
            return corrupted


controller = Controller()

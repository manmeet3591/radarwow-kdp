"""Shared PHIDP preparation for every KDP method.

Measured PHIDP differs by source: 0-360 or +-180 storage, system offsets from 0 to ~200 deg,
folding, noise gates and non-rain echo. Before differentiation we
1. keep gates that are signal and look like rain (rhoHV above the band threshold),
2. unfold each ray (np.unwrap on the valid sequence, after a 5-gate median),
3. flip the sign if the source stores PHIDP decreasing with range (core.moments.phidp_direction),
4. remove the system offset (median of the first valid gates of the ray),
5. linearly bridge gaps between valid gates (outside the first/last valid gate: hold).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from radarwow.core.moments import phidp_direction
from radarwow.core.noise import signal_mask


def rain_mask(ds, rhohv_min: float) -> np.ndarray:
    m = signal_mask(ds) & np.isfinite(ds["PHIDP"].values)
    if "RHOHV" in ds:
        m &= ds["RHOHV"].values > rhohv_min
    return m


def prepare_ray(phi: np.ndarray, valid: np.ndarray, n_offset: int = 10):
    """Return (psi_filled, valid, offset) for one ray; psi is unfolded and offset-free."""
    psi = np.full(phi.shape, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size < 5:
        return psi, np.zeros_like(valid), np.nan
    seq = phi[idx].astype(float)
    smooth = median_filter(seq, size=5, mode="nearest")
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(smooth), discont=np.pi))
    seq = seq + (unwrapped - smooth)  # apply the same 360-deg jumps to the raw values
    offset = float(np.median(seq[:n_offset]))
    psi[idx] = seq - offset
    psi = np.interp(np.arange(phi.size), idx, psi[idx])  # bridge gaps, hold at ends
    return psi, valid, offset


def prepare(ds, rhohv_min: float):
    sign, _ = phidp_direction(ds)
    phi = sign * ds["PHIDP"].values.astype(float)
    valid = rain_mask(ds, rhohv_min)
    psi = np.full(phi.shape, np.nan)
    offsets = np.full(phi.shape[0], np.nan)
    for i in range(phi.shape[0]):
        psi[i], valid[i], offsets[i] = prepare_ray(phi[i], valid[i])
    return psi, valid, offsets


def gate_km(ds) -> float:
    return float(np.median(np.diff(ds["range"].values))) / 1e3

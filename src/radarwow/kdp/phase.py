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
from radarwow.core.noise import signal_mask, snr_estimate


def keep_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only runs of at least ``min_run`` consecutive True gates along each ray."""
    if min_run <= 1:
        return mask
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        m = np.concatenate(([False], mask[i], [False]))
        edges = np.flatnonzero(np.diff(m.astype(np.int8)))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_run:
                out[i, s:e] = True
    return out


def rain_mask(ds, rhohv_min: float, min_run_km: float = 2.0, snr_min_db: float = 10.0) -> np.ndarray:
    """Signal, SNR >= ``snr_min_db``, finite PHIDP, rhoHV above threshold, in runs of at least ``min_run_km``.

    The SNR test matters because rhoHV itself is unreliable at low SNR (NEXRAD KTLX: 15 % of signal
    gates have rhoHV > 1.02): weak echo behind a storm can pass the rhoHV test with random PHIDP.

    The run length matters: noise gates far from the storm pass a rhoHV test by chance, and
    bridging PHIDP out to them creates fake PHIDP rises (seen on ARM CSAPR2 and NEXRAD KTLX).
    """
    m = signal_mask(ds) & np.isfinite(ds["PHIDP"].values) & (snr_estimate(ds) >= snr_min_db)
    if "RHOHV" in ds:
        m &= ds["RHOHV"].values > rhohv_min
    return keep_runs(m, max(1, int(round(min_run_km / gate_km(ds)))))


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

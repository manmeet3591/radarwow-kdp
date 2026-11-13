"""Shared PHIDP preparation for every KDP method.

Measured PHIDP differs by source: 0-360 or +-180 storage, system offsets from 0 to ~200 deg,
folding, noise gates and non-rain echo. Before differentiation we
0. rotate the sweep by the circular mean of its rain-gate PHIDP and wrap to +-180 deg, so a system offset near the
   0/360 boundary does not fold every ray (FMI Petajavesi stores ~356 deg; without this the median-based unfolding
   below left 356 deg jumps that ZPHI and R(A) read as heavy rain),
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


def rain_mask(ds, rhohv_min: float, min_run_km: float = 2.0, snr_min_db: float = 10.0, max_clutter_db: float = 10.0) -> np.ndarray:
    """Signal, SNR >= ``snr_min_db``, finite PHIDP, rhoHV above threshold, in runs of at least ``min_run_km``.

    The SNR test matters because rhoHV itself is unreliable at low SNR (NEXRAD KTLX: 15 % of signal
    gates have rhoHV > 1.02): weak echo behind a storm can pass the rhoHV test with random PHIDP.

    The run length matters: noise gates far from the storm pass a rhoHV test by chance, and
    bridging PHIDP out to them creates fake PHIDP rises (seen on ARM CSAPR2 and NEXRAD KTLX).

    Clutter: when the file has both total (DBTH) and clutter-filtered (DBZH) reflectivity, gates where the signal
    processor removed more than ``max_clutter_db`` are excluded, because the processor filters reflectivity but not
    the phase. A precaution: on the Tablazo (Colombia) mountain-top radar it changed little (radar/gauge bias 2.16 ->
    2.09); the spurious PHIDP there was caught by the ray consistency test in radarwow-attenuation instead.
    """
    m = signal_mask(ds) & np.isfinite(ds["PHIDP"].values) & (snr_estimate(ds) >= snr_min_db)
    if "DBTH" in ds:  # phase is not clutter-filtered: where the processor removed > max_clutter_db, PHIDP is unreliable
        removed = ds["DBTH"].values - ds["DBZH"].values
        m &= ~(np.isfinite(removed) & (removed > max_clutter_db))
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


def center_phidp(phi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """PHIDP rotated by the circular mean of the valid gates and wrapped to [-180, 180)."""
    v = phi[valid & np.isfinite(phi)]
    if v.size == 0:
        return phi
    c = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(v)))))
    return (phi - c + 180.0) % 360.0 - 180.0


def prepare(ds, rhohv_min: float):
    valid = rain_mask(ds, rhohv_min)
    ds = ds.assign(PHIDP=ds["PHIDP"].copy(data=center_phidp(ds["PHIDP"].values.astype(float), valid)))
    sign, _ = phidp_direction(ds)
    phi = sign * ds["PHIDP"].values.astype(float)
    psi = np.full(phi.shape, np.nan)
    offsets = np.full(phi.shape[0], np.nan)
    for i in range(phi.shape[0]):
        psi[i], valid[i], offsets[i] = prepare_ray(phi[i], valid[i])
    return psi, valid, offsets


def gate_km(ds) -> float:
    return float(np.median(np.diff(ds["range"].values))) / 1e3

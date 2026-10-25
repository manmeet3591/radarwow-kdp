"""KDP by iterative window differentiation (Vulpiani et al. 2012, Sect. 3).

KDP'(r) = [Psi(r + L/2) - Psi(r - L/2)] / (2 L). Values outside the physical range
[kdp_min, kdp_max] are set to 0, PHIDP is rebuilt as 2*integral(KDP), and KDP is re-derived;
repeating I times lowers the noise (sigma_KDP ~ sigma_Psi / (L sqrt(2 N I))).
Folding is handled during PHIDP preparation (unwrap) rather than by the paper's -180/L test.
"""

from __future__ import annotations

import numpy as np

from radarwow.core import Algorithm, Reference, register

from .phase import gate_km, prepare


def window_derivative(psi: np.ndarray, h: int, dr_km: float) -> np.ndarray:
    n = psi.shape[-1]
    i = np.arange(n)
    hi = np.clip(i + h, 0, n - 1)
    lo = np.clip(i - h, 0, n - 1)
    span_km = (hi - lo) * dr_km
    with np.errstate(invalid="ignore", divide="ignore"):
        return (psi[..., hi] - psi[..., lo]) / (2.0 * span_km)


@register
class Vulpiani2012KDP(Algorithm):
    name = "vulpiani2012_kdp"
    stage = "kdp"
    reference = Reference("G. Vulpiani, M. Montopoli, L. Delli Passeri, A. G. Gioia, P. Giordano, F. S. Marzano", 2012,
                          "On the Use of Dual-Polarized C-Band Radar for Operational Rainfall Retrieval in Mountainous Areas",
                          "J. Appl. Meteor. Climatol. 51, 405-425", "10.1175/JAMC-D-10-05024.1")
    requires = ("PHIDP", "DBZH")
    optional = ("RHOHV", "SNRH", "SQIH")
    provides = ("KDP", "PHIDP_REC")
    notes = "Window L from the band config (kdp_window_km); physical range kdp_min..kdp_max; 10 iterations."
    params = dict(iterations=10)

    def run_sweep(self, ds, cfg):
        dr = gate_km(ds)
        h = max(1, int(round(cfg["kdp_window_km"] / dr / 2)))
        psi, valid, offsets = prepare(ds, cfg["rhohv_rain_min"])
        phi = psi.copy()
        kdp = np.zeros_like(psi)
        for _ in range(max(1, self.params["iterations"])):
            kdp = window_derivative(phi, h, dr)
            kdp = np.where((kdp < cfg["kdp_min"]) | (kdp > cfg["kdp_max"]) | ~np.isfinite(kdp), 0.0, kdp)
            phi = 2.0 * np.cumsum(kdp, axis=1) * dr
        has = np.isfinite(psi[:, :1])
        out = ds.copy()
        out["KDP"] = (("azimuth", "range"), np.where(valid & has, kdp, np.nan).astype("float32"),
                      {"units": "degrees/km", "long_name": "KDP (Vulpiani 2012)"})
        out["PHIDP_REC"] = (("azimuth", "range"), np.where(has, phi, np.nan).astype("float32"),
                            {"units": "degrees", "long_name": "reconstructed PHIDP, offset removed"})
        out["PHIDP_OFFSET"] = (("azimuth",), offsets.astype("float32"))
        return out

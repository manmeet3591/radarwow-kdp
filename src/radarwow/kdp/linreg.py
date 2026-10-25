"""KDP from a median-filtered PHIDP and a moving linear regression, as used to build the
RainForest database (Wolfensberger et al. 2021, Appendix A: same window, best at 6 km)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from radarwow.core import Algorithm, Reference, register

from .phase import gate_km, prepare


@register
class MovingRegressionKDP(Algorithm):
    name = "wolfensberger2021_linreg_kdp"
    stage = "kdp"
    reference = Reference("D. Wolfensberger, M. Gabella, M. Boscacci, U. Germann, A. Berne", 2021,
                          "RainForest: a random forest algorithm for quantitative precipitation estimation over Switzerland",
                          "Atmos. Meas. Tech. 14, 3169-3193", "10.5194/amt-14-3169-2021")
    requires = ("PHIDP", "DBZH")
    optional = ("RHOHV",)
    provides = ("KDP", "PHIDP_REC")
    params = dict(window_km=6.0)

    def run_sweep(self, ds, cfg):
        dr = gate_km(ds)
        n = max(3, int(round(self.params["window_km"] / dr)) | 1)
        psi, valid, offsets = prepare(ds, cfg["rhohv_rain_min"])
        phi = median_filter(np.nan_to_num(psi), size=(1, n), mode="nearest")
        r = np.arange(psi.shape[1]) * dr
        # slope of a moving least-squares line: cov(r, phi) / var(r)
        mr = uniform_filter1d(r, n, mode="nearest")
        mp = uniform_filter1d(phi, n, axis=1, mode="nearest")
        mrp = uniform_filter1d(phi * r[None, :], n, axis=1, mode="nearest")
        mrr = uniform_filter1d(r * r, n, mode="nearest")
        slope = (mrp - mr[None, :] * mp) / (mrr - mr * mr)[None, :]
        out = ds.copy()
        out["KDP"] = (("azimuth", "range"), np.where(valid, slope / 2.0, np.nan).astype("float32"),
                      {"units": "degrees/km", "long_name": "KDP (median + moving regression)"})
        out["PHIDP_REC"] = (("azimuth", "range"), np.where(np.isfinite(psi), phi, np.nan).astype("float32"))
        out["PHIDP_OFFSET"] = (("azimuth",), offsets.astype("float32"))
        return out

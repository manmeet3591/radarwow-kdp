"""PHIDP/KDP by linear programming (Giangrande, McGraw & Lei 2013).

Per ray: minimise sum_i w_i |x_i - Psi_i| (L1, robust to backscatter-phase spikes) subject to
a monotonic PHIDP. The paper constrains a Savitzky-Golay derivative (D x >= 0), which leaves
"weakly monotonic" wiggles and unconstrained ray ends; we constrain every first difference
(x_{i+1} >= x_i), which makes PHIDP strictly monotonic so the SG-derivative KDP is >= 0 at
every gate including the ends.
Gap gates get w = 0. The solution is smoothed with the paper's 5-point filter
{0.1, 0.25, 0.3, 0.25, 0.1} and KDP is the SG derivative / 2. Solved with SciPy's HiGHS
(Py-ART's phase_proc_lp needs cvxopt/cylp instead).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from radarwow.core import Algorithm, Reference, register

from .phase import gate_km, prepare

SMOOTH = np.array([0.1, 0.25, 0.3, 0.25, 0.1])


def sg_derivative_coeffs(m: int) -> np.ndarray:
    """First-derivative Savitzky-Golay weights (linear/quadratic fit) for a window of 2m+1 gates."""
    j = np.arange(-m, m + 1, dtype=float)
    return j / np.sum(j * j)


def solve_ray(b: np.ndarray, w: np.ndarray, m: int) -> np.ndarray | None:
    n = b.size
    rows = n - 1
    if rows <= 0:
        return None
    # D: first differences, D x >= 0 <=> monotonic non-decreasing PHIDP
    D = sparse.diags([-np.ones(rows), np.ones(rows)], [0, 1], shape=(rows, n), format="csr")
    I = sparse.identity(n, format="csr")
    # variables z = [x, t]; minimise w.t
    A = sparse.vstack([sparse.hstack([I, -I]), sparse.hstack([-I, -I]), sparse.hstack([-D, sparse.csr_matrix((rows, n))])]).tocsr()
    ub = np.concatenate([b, -b, np.zeros(rows)])
    cost = np.concatenate([np.zeros(n), w])
    bounds = [(None, None)] * n + [(0, None)] * n
    res = linprog(cost, A_ub=A, b_ub=ub, bounds=bounds, method="highs")
    return res.x[:n] if res.status == 0 else None


@register
class Giangrande2013LP(Algorithm):
    name = "giangrande2013_lp"
    stage = "kdp"
    reference = Reference("S. E. Giangrande, R. McGraw, L. Lei", 2013,
                          "An Application of Linear Programming to Polarimetric Radar Differential Phase Processing",
                          "J. Atmos. Oceanic Technol. 30, 1716-1729", "10.1175/JTECH-D-12-00147.1")
    requires = ("PHIDP", "DBZH")
    optional = ("RHOHV", "SNRH", "SQIH")
    provides = ("KDP", "PHIDP_REC")
    notes = "Monotonic PHIDP => KDP >= 0 by construction. sg_window_km sets the derivative filter length."
    params = dict(sg_window_km=3.0, min_valid=20, min_delta_phi=3.0)

    def run_sweep(self, ds, cfg):
        dr = gate_km(ds)
        m = max(2, int(round(self.params["sg_window_km"] / dr / 2)))
        psi, valid, offsets = prepare(ds, cfg["rhohv_rain_min"])
        x_all = np.full(psi.shape, np.nan)
        kdp = np.full(psi.shape, np.nan)
        c = sg_derivative_coeffs(m)
        for i in range(psi.shape[0]):
            idx = np.flatnonzero(valid[i])
            if idx.size < self.params["min_valid"]:
                continue
            s, e = idx[0], idx[-1] + 1
            b = psi[i, s:e]
            w = valid[i, s:e].astype(float)
            if np.nanpercentile(b[w > 0], 95) - np.nanpercentile(b[w > 0], 5) < self.params["min_delta_phi"]:
                x = np.full(b.shape, float(np.median(b[w > 0])))  # no measurable phase shift: KDP = 0
            else:
                x = solve_ray(b, w, m)
                if x is None:
                    continue
            xs = np.convolve(np.pad(x, 2, mode="edge"), SMOOTH, mode="valid")
            k = np.convolve(np.pad(xs, m, mode="edge"), c[::-1], mode="valid") / (2.0 * dr)
            x_all[i, s:e] = xs
            kdp[i, s:e] = k
        out = ds.copy()
        out["KDP"] = (("azimuth", "range"), np.where(valid, kdp, np.nan).astype("float32"),
                      {"units": "degrees/km", "long_name": "KDP (Giangrande 2013 LP)"})
        out["PHIDP_REC"] = (("azimuth", "range"), x_all.astype("float32"),
                            {"units": "degrees", "long_name": "LP-processed PHIDP, offset removed"})
        out["PHIDP_OFFSET"] = (("azimuth",), offsets.astype("float32"))
        return out

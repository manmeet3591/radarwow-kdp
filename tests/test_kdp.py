import numpy as np
import pytest
import xarray as xr

from radarwow.core.testing import synthetic_ray
from radarwow.kdp import Giangrande2013LP, MovingRegressionKDP, Vulpiani2012KDP

METHODS = [Vulpiani2012KDP, Giangrande2013LP, MovingRegressionKDP]


def ray_sweep(n_rays=6, **kw):
    rays = [synthetic_ray(seed=s, **kw) for s in range(n_rays)]
    r_km = rays[0][0]
    psi = np.stack([x[1] for x in rays])
    kdp = np.stack([x[3] for x in rays])
    ds = xr.Dataset({"PHIDP": (("azimuth", "range"), psi),
                     "DBZH": (("azimuth", "range"), np.full(psi.shape, 30.0)),
                     "RHOHV": (("azimuth", "range"), np.full(psi.shape, 0.99)),
                     "SNRH": (("azimuth", "range"), np.full(psi.shape, 30.0))},
                    coords={"azimuth": np.arange(n_rays, dtype=float), "range": r_km * 1e3 + 125.0})
    return ds, kdp


@pytest.mark.parametrize("cls", METHODS)
@pytest.mark.parametrize("band", ["S", "C"])
def test_kdp_recovers_cell(cls, band):
    ds, truth = ray_sweep(centers=(40.0,), widths=(6.0,), peaks=(3.0,), noise=3.0)
    out = cls()(ds, band=band)
    k = out["KDP"].values
    rmse = np.sqrt(np.nanmean((k - truth) ** 2))
    assert rmse < 0.6, rmse
    total = np.nansum(k, axis=1) * 0.25 * 2
    assert np.allclose(total, np.sum(truth, axis=1) * 0.25 * 2, rtol=0.2)


@pytest.mark.parametrize("cls", METHODS)
def test_kdp_handles_wrapping_and_offset(cls):
    ds, truth = ray_sweep(offset=150.0, wrap=True, centers=(40.0,), widths=(6.0,), peaks=(4.0,), noise=2.0)
    out = cls()(ds, band="C")
    assert np.nanmax(out["KDP"].values) < 8 and np.nanmin(out["KDP"].values) > -2.5


def test_lp_never_negative():
    ds, truth = ray_sweep(centers=(40.0,), widths=(6.0,), peaks=(3.0,), noise=4.0, delta_peak=15.0)
    out = Giangrande2013LP()(ds, band="C")
    assert np.nanmin(out["KDP"].values) > -0.05


@pytest.mark.parametrize("cls", METHODS)
def test_offset_near_360_does_not_create_phase_rise(cls):
    # FMI Petajavesi: system offset ~356 deg stored 0-360, so noise folds every ray around 0/360
    ds, truth = ray_sweep(n_rays=12, offset=356.0, centers=(40.0,), widths=(6.0,), peaks=(2.0,), noise=3.0)
    ds["PHIDP"] = ds["PHIDP"] % 360.0
    out = cls()(ds, band="C")
    span_true = np.sum(truth, axis=1) * 0.25 * 2
    span = np.nansum(out["KDP"].values, axis=1) * 0.25 * 2
    assert np.allclose(span, span_true, rtol=0.2), (span, span_true)

#!/usr/bin/env python
"""Fit the SDSS reference SF (TotalDat.fits) with the SAME SF_linmix used on the
ZTF pickles, so gamma / A_365 are directly comparable.

Two bridges are needed:
  1. SF_linmix indexes .left/.right/.mid/.length -> the bare lag grid must be
     rebuilt as a CategoricalIndex of Intervals (geometric bin edges).
  2. SF_linmix rejects the WHOLE object if any yerr <= 0; the reference has
     ~5% zero-width error bins, so those bins are dropped first.

FRAME: DT_REST_* is rest frame. SF_wnoise lags are observed. --frame decides
which one both sides are expressed in; A_365 is meaningless across frames.
"""
import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from newSF import SF_linmix

BANDS = ("g", "r", "i")
RA_DEC = re.compile(r"^(\d+\.\d+)_([+-]?\d+\.\d+)$")


def interval_index(lag):
    """Bare lag centres -> CategoricalIndex of Intervals, geometric edges.

    SF_linmix uses .mid (arithmetic) rather than the lag itself; on a 0.15 dex
    grid that is a ~0.7% offset, negligible for gamma but not exactly the lag.
    """
    lag = np.asarray(lag, float)
    inner = np.sqrt(lag[:-1] * lag[1:])                  # geometric midpoints
    first = lag[0] ** 2 / inner[0]
    last = lag[-1] ** 2 / inner[-1]
    edges = np.concatenate([[first], inner, [last]])
    iv = pd.IntervalIndex.from_breaks(edges, closed="right")
    return pd.CategoricalIndex(iv, categories=iv)


def ref_dict(t, i, band, z, frame):
    """Build an SF_linmix-compatible dict for row i, or None if unusable."""
    def col(name):
        v = np.atleast_1d(np.asarray(t[name][i], float)).ravel()
        v[np.isin(v, (-99.0, -999.0))] = np.nan
        return v

    lag = col(f"DT_REST_{band}")
    sf = col(f"SF_{band}")
    lo = col(f"SF_{band}_ERR_L")
    hi = col(f"SF_{band}_ERR_U")

    # SF_linmix bails on the whole object if ANY error is <= 0
    ok = (np.isfinite(lag) & np.isfinite(sf) & np.isfinite(lo) & np.isfinite(hi)
          & (sf > 0) & (lo > 0) & (hi > 0) & (lag > 0))
    if ok.sum() < 4:
        return None, int(ok.sum())

    lag, sf, lo, hi = lag[ok], sf[ok], lo[ok], hi[ok]
    if frame == "obs":
        lag = lag * (1 + z)          # rest -> observed

    order = np.argsort(lag)
    lag, sf, lo, hi = lag[order], sf[order], lo[order], hi[order]
    idx = interval_index(lag)
    return {
        "SF": pd.Series(sf, index=idx),
        "SFmaxerr": pd.Series(hi, index=idx),
        "SFminerr": pd.Series(lo, index=idx),
        "RA": float(t["RA"][i]),
        "band": band,
        "z": z,
        "frame": frame,
    }, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="~/SDSS_S82_QSO/TotalDat.fits")
    ap.add_argument("--root", default="~/SDSS_S82_QSO",
                    help="only fit rows matching an RA_DEC dir here (omit with --all)")
    ap.add_argument("--all", action="store_true", help="fit every FITS row")
    ap.add_argument("--band", default="g", choices=BANDS)
    ap.add_argument("--frame", default="rest", choices=("rest", "obs"),
                    help="'obs' multiplies the reference lags by (1+z)")
    ap.add_argument("--z-col", default="Z")
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--outdir", default="~/results/sf_vs_sdss/ref_linmix")
    a = ap.parse_args()

    with fits.open(os.path.expanduser(a.fits)) as h:
        t = Table(h[1].data)
    cat = SkyCoord(np.asarray(t["RA"], float) * u.deg,
                   np.asarray(t["DEC"], float) * u.deg)

    if a.all:
        rows = list(range(len(t)))
    else:
        rows = []
        for d in sorted(Path(os.path.expanduser(a.root)).iterdir()):
            m = RA_DEC.match(d.name) if d.is_dir() else None
            if not m:
                continue
            src = SkyCoord(float(m.group(1)) * u.deg, float(m.group(2)) * u.deg)
            j, sep, _ = src.match_to_catalog_sky(cat)
            if sep.arcsec.item() <= a.tol:
                rows.append(int(j))
        rows = sorted(set(rows))

    outdir = Path(os.path.expanduser(a.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} reference rows, band {a.band}, {a.frame} frame")

    import signal

    class T(Exception):
        pass

    def _alarm(s, f):
        raise T

    n_ok = 0
    for i in rows:
        z = float(t[a.z_col][i])
        d, n_used = ref_dict(t, i, a.band, z, a.frame)
        tag = f"{float(t['RA'][i]):.5f}_{float(t['DEC'][i]):+.5f}"
        if d is None:
            print(f"SKIP {tag}: only {n_used} usable bins")
            continue
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(a.timeout)
        try:
            fitted = SF_linmix(d, amp_at=365, verbose=False,
                               plot=False, save_plot=False, save_pkl=False)
        except T:
            print(f"TIMEOUT {tag}")
            continue
        except Exception as e:
            print(f"ERROR {tag}: {type(e).__name__}: {e}")
            continue
        finally:
            signal.alarm(0)

        if fitted is None:
            print(f"SKIP {tag}: SF_linmix returned None ({n_used} bins in grid)")
            continue
        out = outdir / f"{tag}_ref_{a.band}_{a.frame}_linmixfit.pkl"
        with open(out, "wb") as f:
            pickle.dump(fitted, f)
        n_ok += 1
        print(f"{tag}: gamma={fitted['gamma_spl']:.3f} "
              f"A_365={fitted['A_365_spl']:.4f} z={z:.3f} -> {out.name}")

    print(f"\n{n_ok}/{len(rows)} fitted")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Overplot ZTF SF (from <RUN>/sf_out/<RA_DEC>.pkl) against the SDSS reference FITS.

--inspect first: prints column shapes, dtypes and sentinel candidates, so the
plotting path is not built on a guess about whether SF_g is scalar or vector.

FRAME WARNING: the FITS lags are DT_REST_* (rest frame); SF_wnoise lags are
observed frame. Pass --z (or --z-col) or the two curves are offset by (1+z) in
lag and the comparison is meaningless.
"""
import argparse
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

BANDS = ("g", "r", "i")


def load_fits(path):
    with fits.open(os.path.expanduser(path)) as hdul:
        return Table(hdul[1].data)


def inspect(t):
    keys = ["RA", "DEC"] + [f"{p}_{b}" for b in BANDS
                            for p in ("SF", "DT_REST", "SF_ERR_L_x", "SF_ERR_U_x")]
    print(f"{len(t)} rows\n")
    for c in t.colnames:
        col = t[c]
        shape = col.shape[1:] if col.ndim > 1 else "scalar"
        line = f"{c:16s} dtype={str(col.dtype):10s} shape={shape}"
        try:
            v = np.asarray(col, dtype=float)
            finite = np.isfinite(v)
            line += (f" finite={finite.sum()}/{v.size}"
                     f" min={np.nanmin(v[finite]) if finite.any() else np.nan:.4g}"
                     f" max={np.nanmax(v[finite]) if finite.any() else np.nan:.4g}")
            for sent in (-99, -999, 0):
                n = int((v == sent).sum())
                if n:
                    line += f" [=={sent}: {n}]"
        except Exception:
            pass
        print(line)

    # is ERR_L an absolute SF value or an error magnitude?
    for b in BANDS:
        sf, lo, up = (f"SF_{b}", f"SF_{b}_ERR_L", f"SF_{b}_ERR_U")
        if all(k in t.colnames for k in (sf, lo, up)):
            s = np.asarray(t[sf], float).ravel()
            l = np.asarray(t[lo], float).ravel()
            m = np.isfinite(s) & np.isfinite(l)
            if m.any():
                frac = (l[m] < s[m]).mean()
                print(f"\n{lo} < {sf} in {frac:.0%} of finite entries -> "
                      f"{'error magnitudes (use as-is in yerr)' if frac > 0.9 else 'ABSOLUTE bounds (subtract from SF)'}")


def match_row(t, ra, dec, tol_arcsec):
    cat = SkyCoord(np.asarray(t["RA"], float) * u.deg,
                   np.asarray(t["DEC"], float) * u.deg)
    src = SkyCoord(ra * u.deg, dec * u.deg)
    i, sep, _ = src.match_to_catalog_sky(cat)
    i, sep = int(i), sep.arcsec.item()
    if sep > tol_arcsec:
        raise ValueError(f"no FITS match within {tol_arcsec}\" (nearest {sep:.2f}\")")
    return i, sep


def own_sf(pkl):
    d = pickle.load(open(pkl, "rb"))
    sf = d["SF_dict"][0]["SF"]
    lo = d["SF_dict"][0]["SFminerr"].reindex(sf.index)
    hi = d["SF_dict"][0]["SFmaxerr"].reindex(sf.index)
    good = np.isfinite(np.asarray(sf, float))
    iv = sf.index.categories[sf.index.codes]
    return (np.asarray(iv.mid, float)[good],
            np.asarray(iv.length, float)[good] / 2,
            np.asarray(sf, float)[good],
            np.abs(np.asarray(lo, float)[good]),
            np.abs(np.asarray(hi, float)[good]), d)


def ref_sf(t, i, band, err_absolute, lag_min=0.0):
    """Return (lag_rest, sf, err_lo, err_hi, n_zero_err); NaN bins dropped.

    DT_REST_* is a fixed rest-frame grid shared by every row; SF_* is NaN in
    bins the object's baseline does not reach, so the mask must come from SF.
    """
    def col(name):
        v = np.atleast_1d(np.asarray(t[name][i], float)).ravel()
        v[np.isin(v, (-99.0, -999.0))] = np.nan
        return v

    lag, sf = col(f"DT_REST_{band}"), col(f"SF_{band}")
    lo, hi = col(f"SF_{band}_ERR_L"), col(f"SF_{band}_ERR_U")
    if err_absolute:                       # columns hold bounds, not widths
        lo, hi = sf - lo, hi - sf
    m = np.isfinite(lag) & np.isfinite(sf) & (lag > lag_min)
    n_zero = int(((lo[m] == 0) | (hi[m] == 0)).sum())
    return lag[m], sf[m], np.abs(lo[m]), np.abs(hi[m]), n_zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="~/SDSS_S82_QSO/TotalDat.fits")
    ap.add_argument("--root", default="~/SDSS_S82_QSO")
    ap.add_argument("--object", help="RA_DEC dir name; omit to do all pickles")
    ap.add_argument("--band", default="g", choices=BANDS)
    ap.add_argument("--z", type=float, help="redshift, to put own lags in rest frame")
    ap.add_argument("--z-col", default="Z",
                    help="FITS column holding redshift (overrides --z); '' to disable")
    ap.add_argument("--lag-min", type=float, default=0.1,
                    help="drop reference bins below this rest lag [d]; the grid "
                         "reaches 2e-4 d, which squashes the ZTF range")
    ap.add_argument("--tol", type=float, default=1.0, help="match radius, arcsec")
    ap.add_argument("--err-absolute", action="store_true",
                    help="SF_*_ERR_L/U are absolute bounds, not widths (see --inspect)")
    ap.add_argument("--outdir", default="~/results/sf_vs_sdss")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    t = load_fits(a.fits)
    if a.inspect:
        inspect(t)
        return

    root = Path(os.path.expanduser(a.root))
    outdir = Path(os.path.expanduser(a.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    pkls = (sorted(root.glob(f"{a.object}/ztfphot_stars_*/sf_out/*.pkl")) if a.object
            else sorted(root.glob("*/ztfphot_stars_*/sf_out/*.pkl")))
    if not pkls:
        raise SystemExit(f"no pickles under {root}")

    for pkl in pkls:
        name = pkl.stem
        try:
            ra, dec = (float(x) for x in name.rsplit("_", 1))
        except ValueError:
            print(f"SKIP {name}: cannot parse RA_DEC")
            continue
        try:
            i, sep = match_row(t, ra, dec, a.tol)
        except ValueError as e:
            print(f"SKIP {name}: {e}")
            continue

        lag, half, sf, elo, ehi, d = own_sf(pkl)
        z = float(t[a.z_col][i]) if a.z_col else a.z
        if z is not None and not np.isfinite(z):
            print(f"SKIP {name}: non-finite redshift")
            continue
        lag_plot, frame = ((lag / (1 + z), f"rest frame (z={z:.4f})") if z is not None
                           else (lag, "OBSERVED frame -- NOT comparable"))
        rlag, rsf, rlo, rhi, n_zero = ref_sf(t, i, a.band, a.err_absolute, a.lag_min)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.errorbar(lag_plot, sf, xerr=half / (1 + z) if z is not None else half,
                    yerr=(elo, ehi), fmt="o", c="crimson", capsize=2,
                    label=f"ZTF (this work), {frame}")
        ax.errorbar(rlag, rsf, yerr=(rlo, rhi), fmt="s", mfc="none", c="k",
                    capsize=2, label=f"SDSS S82 ref, {a.band} (rest)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Delta t$ [days]")
        ax.set_ylabel("SF [mag]")
        ax.set_title(f"{name}  (sep {sep:.2f}\")", fontsize=10)
        ax.legend(fontsize=8)
        if z is None:
            ax.text(0.02, 0.02, "no redshift: frames differ by (1+z)",
                    transform=ax.transAxes, color="crimson", fontsize=8)
        fig.tight_layout()
        out = outdir / f"{name}_sf_vs_sdss_{a.band}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"{name}: {len(sf)} own bins, {len(rsf)} ref bins "
              f"({n_zero} with zero-width errors), z={z}, sep {sep:.2f}\" -> {out}")


if __name__ == "__main__":
    main()
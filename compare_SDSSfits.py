#!/usr/bin/env python
"""Compare A_365 and gamma between two sets of SF_linmix fits.

Default: ZTF fits (<RUN>/sf_out/*_linmixfit.pkl) vs reference fits
(ref_linmix/*_ref_*_linmixfit.pkl). Both sides are matched on sky position.

A_365 is only comparable if both fits used the SAME frame -- run fit_ref_sf.py
with --frame obs if your ZTF SF lags are observed frame. gamma is frame
invariant for a power law, so the right panel is safe either way.
"""
import argparse
import os
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

RA_DEC = re.compile(r"(\d+\.\d+)_([+-]?\d+\.\d+)")
KEYS = {"A": ("A_365_spl", "A_365_maxerr_spl", "A_365_minerr_spl"),
        "g": ("gamma_spl", "gamma_maxerr_spl", "gamma_minerr_spl")}


def unwrap(obj):
    """Accept either a bare fit dict or the run_txt_sf payload wrapper."""
    if isinstance(obj, dict) and "SF_dict" in obj:
        inner = obj["SF_dict"]
        return inner[0] if isinstance(inner, list) else inner
    if isinstance(obj, list):
        return obj[0]
    return obj


def load_set(pattern):
    """-> list of (SkyCoord, fitdict, name); skips files with no fit keys."""
    out = []
    for f in sorted(Path("/").glob(pattern.lstrip("/"))):
        m = RA_DEC.search(f.name)
        if not m:
            continue
        try:
            d = unwrap(pickle.load(open(f, "rb")))
        except Exception as e:
            print(f"  unreadable {f.name}: {e}")
            continue
        if KEYS["A"][0] not in d or KEYS["g"][0] not in d:
            print(f"  no fit keys in {f.name}")
            continue
        out.append((SkyCoord(float(m.group(1)) * u.deg, float(m.group(2)) * u.deg),
                    d, f.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.environ["HOME"] +
                    "/SDSS_S82_QSO/*/ztfphot_stars_*/sf_out/*_linmixfit.pkl",
                    help="glob for set A (default: ZTF fits)")
    ap.add_argument("--b", default=os.environ["HOME"] +
                    "/results/sf_vs_sdss/ref_linmix/*_linmixfit.pkl",
                    help="glob for set B (default: reference fits)")
    ap.add_argument("--label-a", default="ZTF (this work)")
    ap.add_argument("--label-b", default="SDSS S82 reference")
    ap.add_argument("--tol", type=float, default=1.0, help="match radius, arcsec")
    ap.add_argument("--out", default=os.environ["HOME"] +
                    "/results/sf_vs_sdss/SDSS_A365_gamma_comparison.png")
    ap.add_argument("--csv", help="write the matched table here")
    ap.add_argument("--a-max", type=float, default=1.0,
                    help="upper limit on both A_365 axes (0 = auto)")
    ap.add_argument("--g-min", type=float, default=-0.5,
                    help="lower limit on both gamma axes (None-like: use 'nan' for auto)")
    ap.add_argument("--clip", action="store_true",
                    help="honour --a-max/--g-min even if points fall outside "
                         "(they are hidden; summary stats still use every object)")
    args = ap.parse_args()

    print(f"set A: {args.a}")
    A = load_set(args.a)
    print(f"  {len(A)} fits")
    print(f"set B: {args.b}")
    B = load_set(args.b)
    print(f"  {len(B)} fits")
    if not A or not B:
        raise SystemExit("one side is empty -- check the globs")

    cb = SkyCoord([c for c, _, _ in B])
    rows = []
    for ca, da, na in A:
        j, sep, _ = ca.match_to_catalog_sky(cb)
        if sep.arcsec.item() > args.tol:
            continue
        db = B[int(j)][1]
        r = {"ra": ca.ra.deg, "dec": ca.dec.deg, "sep": sep.arcsec.item()}
        for tag, (v, hi, lo) in KEYS.items():
            r[f"{tag}_a"], r[f"{tag}_a_hi"], r[f"{tag}_a_lo"] = da[v], da[hi], da[lo]
            r[f"{tag}_b"], r[f"{tag}_b_hi"], r[f"{tag}_b_lo"] = db[v], db[hi], db[lo]
        rows.append(r)

    if not rows:
        raise SystemExit(f"no positional matches within {args.tol}\"")
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} matched objects")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    fixed = {"A": (None, args.a_max or None), "g": (args.g_min, None)}
    for ax, tag, name, logscale in [(axes[0], "A", r"$A_{365}$ [mag]", True),
                                    (axes[1], "g", r"$\gamma$", False)]:
        xa, xb = df[f"{tag}_a"].values, df[f"{tag}_b"].values
        ax.errorbar(xb, xa,
                    xerr=(df[f"{tag}_b_lo"], df[f"{tag}_b_hi"]),
                    yerr=(df[f"{tag}_a_lo"], df[f"{tag}_a_hi"]),
                    fmt="o", ms=5, capsize=2, lw=0.8, alpha=0.6, c="mediumseagreen")
        both = np.concatenate([xa, xb])
        both = both[np.isfinite(both)]
        if logscale:
            both = both[both > 0]
        auto = ((both.min() * 0.8, both.max() * 1.2) if logscale
                else (both.min() - 0.15, both.max() + 0.15))
        f_lo, f_hi = fixed[tag]
        lim = (auto[0] if f_lo is None or not np.isfinite(f_lo) else f_lo,
               auto[1] if f_hi is None or not np.isfinite(f_hi) else f_hi)
        clipped = int(((both < lim[0]) | (both > lim[1])).sum())
        if clipped and args.clip:
            print(f"{tag}: {clipped} value(s) HIDDEN outside {lim[0]:.4g}, "
                  f"{lim[1]:.4g} (still counted in the statistics)")
        elif clipped:
            lim = (min(lim[0], both.min() * (0.8 if logscale else 1) -
                       (0 if logscale else 0.15)),
                   max(lim[1], both.max() * (1.2 if logscale else 1) +
                       (0 if logscale else 0.15)))
            print(f"{tag}: {clipped} value(s) outside the requested limits -> "
                  f"axes widened to {lim[0]:.4g}, {lim[1]:.4g}")
        ax.plot(lim, lim, "k--", lw=1, label="1:1")
        if logscale:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(f"{name}  —  {args.label_b}")
        ax.set_ylabel(f"{name}  —  {args.label_a}")

        # summary: ratio for amplitude, difference for slope
        # --- pull: z = (a-b)/sqrt(sa^2+sb^2), asymmetric errors taken on the
        #     side facing the other point.  A_365 in dex, gamma linear.
        a_hi = df[f"{tag}_a_hi"].values.astype(float)
        a_lo = df[f"{tag}_a_lo"].values.astype(float)
        b_hi = df[f"{tag}_b_hi"].values.astype(float)
        b_lo = df[f"{tag}_b_lo"].values.astype(float)

        if logscale:
            d  = np.log10(xa) - np.log10(xb)
            sa = np.where(d < 0, a_hi, a_lo) / (xa * np.log(10))
            sb = np.where(d < 0, b_lo, b_hi) / (xb * np.log(10))
        else:
            d  = xa - xb
            sa = np.where(d < 0, a_hi, a_lo)
            sb = np.where(d < 0, b_lo, b_hi)

        z    = d / np.sqrt(sa ** 2 + sb ** 2)
        ok   = np.isfinite(z)
        chi2 = float(np.sum(z[ok] ** 2))
        dof  = max(int(ok.sum()) - 1, 1)

        if logscale:
            txt = (f"median ratio {np.nanmedian(xa / xb):.3f}\n"
                   f"scatter {np.nanstd(d, ddof=1):.3f} dex\n")
        else:
            txt = (f"median offset {np.nanmedian(d):+.3f}\n"
                   f"scatter {np.nanstd(d, ddof=1):.3f}\n")
        
        txt += (f"$\\chi^2$/dof = {chi2 / dof:.2f}\n"
                f"std(pull) = {np.std(z[ok], ddof=1):.2f}\n"
                f"N = {len(df)}")
            

        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))
        ax.legend(loc="lower right", fontsize=9)
        print(f"{tag}: {txt.splitlines()[0]}, {txt.splitlines()[1]}")

    fig.tight_layout()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\n-> {out}")
    if args.csv:
        df.to_csv(os.path.expanduser(args.csv), index=False)
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
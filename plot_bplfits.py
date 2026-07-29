#!/usr/bin/env python3
"""
plot_bplfits_flat.py
Regenerate broken-power-law fit plots from already-saved outputs, with no refit.
Flat-model only version: imports and draws broken_power_law_flat.

Pairs each  <stem>_bplfit.pkl  (fit params: A, gamma, dt_break)
with its input <stem>.pkl        (SF data: SF, SFmaxerr, SFminerr)
and overlays the median fit on the structure-function points.

Usage:
    python plot_bplfits_flat.py                 # save a PNG next to every _bplfit.pkl
    python plot_bplfits_flat.py --show          # show interactively instead of saving
    python plot_bplfits_flat.py PATTERN         # only fits whose path contains PATTERN
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# reuse the exact model so the drawn curve matches the fit
from newSF import broken_power_law_flat

X          = 10
FIT_GLOB   = os.environ["HOME"] + f"/results/partials/*/*_{X}cs_bplfit.pkl"
FIT_SUFFIX = "_bplfit.pkl"


def input_pkl(fit_pkl):
    """<stem>_bplfit.pkl -> <stem>.pkl"""
    return fit_pkl[:-len(FIT_SUFFIX)] + ".pkl"


def plot_one(fit_pkl, show=False):
    src = input_pkl(fit_pkl)
    if not os.path.exists(src):
        print(f"  [skip] no input SF pkl for {os.path.basename(fit_pkl)}")
        return

    with open(fit_pkl, "rb") as f:
        fit = pickle.load(f)
    with open(src, "rb") as f:
        d = pickle.load(f)

    # SF data in the fit window, same mask bpl_mcmc used
    ii   = pd.IntervalIndex(d["SF"].index)
    mask = (ii.left >= 1) & (ii.left <= 365) & (d["SF"] != 0)
    sf   = d["SF"][mask].dropna()
    if len(sf) == 0:
        print(f"  [skip] no SF points in window for {os.path.basename(fit_pkl)}")
        return
    dt   = sf.index.categories[sf.index.codes].mid
    err  = ((d["SFmaxerr"].reindex(sf.index) + d["SFminerr"].reindex(sf.index)) / 2)

    A, g, b = fit["A"], fit["gamma"], fit["dt_break"]
    dt_fit  = np.logspace(np.log10(min(dt)), np.log10(max(dt)), 100)
    y_fit   = broken_power_law_flat(dt_fit, A, g, b)

    plt.figure(figsize=(8, 6))
    plt.errorbar(dt, sf, yerr=err, fmt="o", capsize=3, color="blue", label="SF data")
    plt.plot(dt_fit, y_fit, "r-", lw=2,
             label=f"fit: γ={g:.2f} +{fit['gamma_uperr']:.2f}/-{fit['gamma_loerr']:.2f}")
    plt.axvline(b, color="black", ls=":", label=f"break {b:.0f} d")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("time lag [days]"); plt.ylabel("SF [mag]")
    plt.title(os.path.basename(src)); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        out = fit_pkl[:-len(".pkl")] + ".png"
        plt.savefig(out); plt.close()
        print(f"  saved {out}")


def main():
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]
    show    = "--show" in sys.argv
    pattern = args[0] if args else ""

    fits = sorted(f for f in glob.glob(FIT_GLOB) if pattern in f)
    print(f"{len(fits)} bpl fits found" + (f" matching '{pattern}'" if pattern else ""))
    for f in fits:
        plot_one(f, show=show)


if __name__ == "__main__":
    main()
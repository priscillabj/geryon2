#!/usr/bin/env python3
"""
Plot-only companion to run_drw_sf.py: redraws each per-band SF figure from the
saved drw_{ra}_{dec}_z{band}.pkl without refitting, and overlays the joint g+r
DRW from runDRW_lag.py (drwlag_{ra}_{dec}.pkl). The lag pkl keeps no kernel
draws, so the joint fit is shown as its median-parameter curve only.

DRW SF = sqrt(2) sigma sqrt(1 - exp(-dt/tau)), identical to eztaox 0.2.0
gpStat2.sf for the Exp kernel; checked once against the stored DRW_SF_ts.
No jax, no MPI: runs on the login node.
    python plot_drw_sf.py --spl
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HOME = Path(os.environ["HOME"])


def sf_drw(t, tau, sigma):
    return np.sqrt(2.0) * sigma * np.sqrt(-np.expm1(-t / tau))


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def fit_params(e, d, base, tail, sfx, args):
    """(A_365, gamma) from the run's sf_fit dict, else the fit pkl of the LC that was fitted."""
    f = e.get("sf_fit", {})
    if f.get(f"A_365{sfx}") is None or f.get(f"gamma{sfx}") is None:
        p = d / f"{base}_{args.x}cs_{tail}.pkl" if base else None
        if p is None or not p.exists():
            return None
        # lstrip: legacy linmix pkls use _A_365_spl / _gamma_spl
        f = {k.lstrip("_"): v for k, v in load(p).items() if k.endswith(sfx)}
    a, g = f.get(f"A_365{sfx}"), f.get(f"gamma{sfx}")
    return None if a is None or g is None else (a, g)


def resolve_lc(e, src_dir, band, x):
    """
    Basename of the LC the DRW was fitted on. drw pkls predating lc_file don't
    record it, but they store the optSF SF they were compared to, so the right
    LC is the candidate whose _{x}cs.pkl reproduces that SF. If a merged file
    and a CCD file give the same SF (same epochs), the merged one is taken.
    """
    if e.get("lc_file"):
        return e["lc_file"]
    sf, tail, hits = e["SFdata"], f"_{x}cs.pkl", []
    for p in sorted(src_dir.glob(f"*_z{band}_*.parquet{tail}")):
        c = load(p)["SF"]
        c = c[c.index.isin(sf.index)]
        if len(c) == len(sf) and np.allclose(np.asarray(c, float), np.asarray(sf, float)):
            hits.append(p.name[: -len(tail)])
    hits = [h for h in hits if "_merged" in h] or hits
    return hits[0] if len(hits) == 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sf-dir", default=str(HOME / "results" / "drw_sf"))
    p.add_argument("--lag-dir", default=str(HOME / "results" / "drw_lag"))
    p.add_argument("--partials-root", default=str(HOME / "results" / "partials"))
    p.add_argument("--outdir", default=None, help="default: <sf-dir>/plots")
    p.add_argument("--band", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--x", type=int, default=10, help="calstar count in pkl names")
    p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--spl", action="store_true", help="overlay the linmix SPL fit")
    args = p.parse_args()
    outdir = args.outdir or os.path.join(args.sf_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    # _z?.pkl excludes drw_sf_merged.pkl, which sits in the same directory
    files = sorted(glob.glob(os.path.join(args.sf_dir, f"drw_*_z{args.band or '?'}.pkl")))
    files = files[:args.limit] if args.limit else files

    unresolved = []
    checked = False
    n = {"plotted": 0, "joint": 0, "spl": 0, "boot": 0}
    for path in files:
        e = load(path)
        if not e.get("valid"):
            continue
        ra, dec, band = e["RA"], e["DEC"], e["band"]
        ts, dt = np.asarray(e["ts"]), np.asarray(e["dt"])

        if not checked:
            mine, ref = sf_drw(ts, e["tau_med"], e["amp_med"]), np.asarray(e["DRW_SF_ts"])
            assert np.allclose(mine, ref, rtol=1e-4), \
                f"sf_drw != stored gpStat2 SF (median ratio {np.median(mine / ref):.4f})"
            checked = True

        d = np.exp(np.asarray(e["log_kernel_draws"], float))     # columns: tau, sigma
        d = d[:: max(1, len(d) // args.n_draws)]

        joint = None
        lp = Path(args.lag_dir) / f"drwlag_{ra}_{dec}.pkl"
        if band in ("g", "r") and lp.exists():
            L = load(lp)
            if L.get("valid"):
                # r-band sigma = g-band sigma * exp(log_amp_scale)
                joint = (L["tau_med"],
                         L["amp_med"] * (L["amp_scale_med"] if band == "r" else 1.0))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(dt, np.asarray(e["SFdata"]),
                    yerr=(np.asarray(e["SFloerr"]), np.asarray(e["SFuperr"])),
                    fmt="o", c="k", capsize=3, label="empirical SF")
        # pkl RA/DEC are radec_filename floats: they name the partials directory,
        # but the fit pkls inside keep the LC basename (merged or per-CCD).
        src_dir = Path(args.partials_root) / f"{ra}_{dec}"
        base = resolve_lc(e, src_dir, band, args.x)
        if base is None:
            unresolved.append(os.path.basename(path))
        bp = fit_params(e, src_dir, base, "boot", "_boot", args)
        if bp:
            ax.plot(dt, bp[0] * (dt / 365.0) ** bp[1], ls="--", label="power law (boot)")
            n["boot"] += 1
        sp = fit_params(e, src_dir, base, "linmixfit", "_spl", args) if args.spl else None
        if sp:
            ax.plot(dt, sp[0] * (dt / 365.0) ** sp[1], ls=":",
                    label="single power law (linmix)")
            n["spl"] += 1
        for c in sf_drw(ts[None, :], d[:, :1], d[:, 1:]):
            ax.plot(ts, c, c="tab:green", alpha=0.15, lw=0.8, zorder=1)
        ax.plot(ts, sf_drw(ts, e["tau_med"], e["amp_med"]), c="tab:green", lw=2,
                label=rf"DRW {band} only ($\tau$={e['tau_med']:.0f} d)")
        if joint:
            ax.plot(ts, sf_drw(ts, *joint), c="tab:purple", lw=2, ls="-.",
                    label=rf"DRW joint g+r ($\tau$={joint[0]:.0f} d)")
            n["joint"] += 1
        ax.set(xscale="log", yscale="log", xlabel="Time diff (days)",
               ylabel="Structure Function (mag)", title=f"{ra} {dec}  z{band}")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(outdir, f"sf_{ra}_{dec}_z{band}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        n["plotted"] += 1

    print(f"{len(files)} pkls -> {n['plotted']} plots in {outdir}; with boot "
          f"{n['boot']}, spl {n['spl']}, joint g+r {n['joint']}; "
          f"LC file unresolved {len(unresolved)}")
    for u in unresolved:
        print("  unresolved:", u)


if __name__ == "__main__":
    main()
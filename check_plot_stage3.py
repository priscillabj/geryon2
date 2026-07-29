"""
plot_stage3.py — visual check of cached SFs and LinMix fits, straight from the
stage-3 cache. No LC reads, no SF recomputation, no MCMC (the power-law
overlay uses the cached _A_365_spl / _gamma_spl, same as newSF.plot_SF).

Usage (serial, compute node, conda env with newSF importable):
    python plot_stage3.py <ra> <dec> <band>                # grid: first 6 sims, both cadences
    python plot_stage3.py <ra> <dec> <band> <oi>           # one sim, both cadences
    python plot_stage3.py <ra> <dec> <band> <oi> --posterior
        # additionally re-runs the seeded LinMix fit for this sim and saves the
        # six-panel diagnostic via newSF.plot_linmix machinery (needs linmix).
        # Byte-identical chains only for fits produced with parallelize=False.

    python plot_stage3.py --all                            # grid, first 6 sims, EVERY cached source
    python plot_stage3.py --all <oi>                       # one sim, every cached source

Output PNGs -> ~/results/sf_cache/plots/
"""

import os
import re
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from newSF import parse_to_dict

CACHE    = os.environ["HOME"] + "/results/20yr_lc/sf_cache/"
PLOTS    = CACHE + "plots/"
# CADENCES = ("full", "crop")
CADENCES = ("full", "crop", "ztf_dur")
FIT_WINDOW = (1.0, 365.0)          # SF_linmix fitting window [days]

os.makedirs(PLOTS, exist_ok=True)

def all_sources():
    """(ra, dec, band) for every source with a stage-3 cache here — same
    sentinel discovery stage4_merge.py uses, so --all covers exactly what
    stage4 would be able to merge."""
    out = []
    for p in glob.glob(CACHE + "*.stage3.done"):
        m = re.search(r"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.stage3\.done",
                      os.path.basename(p))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(out)

def load_fits(ra, dec, band):
    rows = []
    for p in sorted(glob.glob(CACHE + f"fit_{ra}_{dec}_z{band}_rank*.jsonl")):
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return pd.DataFrame(rows) if rows else None


def load_sf(ra, dec, band):
    files = sorted(glob.glob(CACHE + f"sf_{ra}_{dec}_z{band}_rank*.parquet"))
    if not files:
        return None
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def sf_points(sf_str):
    """Cached SF string -> (mid, half_width, values) arrays via parse_to_dict,
    so the reconstruction is identical to what --refit would fit."""
    s = parse_to_dict(sf_str)["SF"]
    s = s.dropna()
    cats = s.index.categories[s.index.codes]
    return cats.mid.values, (cats.length / 2).values, s.values


def plot_one(ax, sf_row, fit_row, cadence, color):
    mid, half, sfv = sf_points(sf_row["SF"])
    _, _, maxe = sf_points(sf_row["SFmaxerr"])
    _, _, mine = sf_points(sf_row["SFminerr"])
    # error series can have different NaN patterns; align by reindexing on SF
    n = min(len(sfv), len(maxe), len(mine))

    ax.errorbar(mid[:n], sfv[:n], xerr=half[:n], yerr=(mine[:n], maxe[:n]),
                fmt="o", ms=4, capsize=2, c=color, label=f"SF ({cadence})")

    if fit_row is not None and bool(fit_row["valid"]):
        A365, gam = fit_row["_A_365_spl"], fit_row["_gamma_spl"]
        in_win = (mid >= FIT_WINDOW[0]) & (mid <= FIT_WINDOW[1])
        dtl = np.logspace(np.log10(max(mid[in_win].min(), FIT_WINDOW[0])),
                          np.log10(min(mid[in_win].max(), FIT_WINDOW[1])), 100) \
              if in_win.any() else np.logspace(0, np.log10(365), 100)
        ax.plot(dtl, A365 * (dtl / 365.0) ** gam, "-", lw=2, c="crimson",
                label=f"fit: γ={gam:.2f}, A₃₆₅={A365:.3f}")
        ax.axvspan(*FIT_WINDOW, alpha=0.06, color="grey")
    elif fit_row is not None:
        ax.set_title(ax.get_title() + f"  [INVALID: {fit_row['fail_reason']}]",
                     fontsize=8, color="crimson")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Δt [days]"); ax.set_ylabel("SF [mag]")
    ax.legend(fontsize=7)


def grid(ra, dec, band, ois):
    fits = load_fits(ra, dec, band)
    sf   = load_sf(ra, dec, band)
    tag  = f"{ra}_{dec}_z{band}"
    if fits is None or sf is None:
        print(f"no cache for {tag}")
        return
    
    cad_colors = {"full": "#436BAD", "crop": "orange", "ztf_dur": "#2CA02C"}
 
    fig, axes = plt.subplots(len(ois), len(CADENCES),
                             figsize=(5.5 * len(CADENCES), 3.2 * len(ois)),
                             squeeze=False)
    # fig, axes = plt.subplots(len(ois), 2, figsize=(11, 3.2 * len(ois)),
    #                          squeeze=False)
    for r, oi in enumerate(ois):
        for c, cad in enumerate(CADENCES):
            ax = axes[r][c]
            srow = sf[(sf.object_index == oi) & (sf.cadence == cad)]
            frow = fits[(fits.object_index == oi) & (fits.cadence == cad)]
            ax.set_title(f"sim {oi} — {cad}", fontsize=9)
            if len(srow) == 0:
                ax.text(0.5, 0.5, "no cached SF", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            plot_one(ax, srow.iloc[0],
                     frow.iloc[0] if len(frow) else None,
                     cad, color=cad_colors.get(cad, "grey"))
    fig.suptitle(f"{tag}: cached SF + cached LinMix fit")
    fig.tight_layout()
    out = PLOTS + f"sffit_{tag}_sims{ois[0]}-{ois[-1]}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"plot -> {out}")


def posterior(ra, dec, band, oi):
    """Seeded re-fit of one sim (both cadences) with diagnostic plots.
    Chains match the cache only for fits produced with parallelize=False."""
    from newSF import SF_linmix

    sf = load_sf(ra, dec, band)
    tag = f"{ra}_{dec}_z{band}"
    for cad in CADENCES:
        srow = sf[(sf.object_index == oi) & (sf.cadence == cad)]
        if len(srow) == 0:
            print(f"{tag} sim {oi} {cad}: no cached SF")
            continue
        d = {k: parse_to_dict(srow.iloc[0][k])["SF"]
             for k in ("SF", "SFmaxerr", "SFminerr")}
        seed = int(hashlib.md5(
            f"{ra}_{dec}_{band}_{oi}_{cad}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        np.random.seed(seed)
        out_prefix = PLOTS + f"posterior_{tag}_{oi}_{cad}_"
        res = SF_linmix(d, save_plot=True, path=out_prefix)
        print(f"{tag} sim {oi} {cad}: "
              f"{'ok, plots at ' + out_prefix + '*' if res else 'fit skipped/failed'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--all":
        srcs = all_sources()
        ois = [int(a[1])] if len(a) >= 2 else [0, 1, 2, 3, 4, 5]
        print(f"{len(srcs)} cached sources, sims {ois}")
        for ra, dec, band in srcs:
            grid(ra, dec, band, ois)
        sys.exit(0)
    
    ra, dec, band = float(a[0]), float(a[1]), a[2]
    if len(a) >= 4 and a[3] != "--posterior":
        oi = int(a[3])
        if "--posterior" in a:
            posterior(ra, dec, band, oi)
        else:
            grid(ra, dec, band, [oi])
    else:
        grid(ra, dec, band, [0, 1, 2, 3, 4, 5])
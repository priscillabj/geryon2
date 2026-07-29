"""
stage5_average.py — final stage: per-source central value (median or mean) of
_gamma_spl per cadence from the stage-4 merged fit parquets, with 16/84
percentile errors, plotted full-cadence vs crop-cadence per source.

Also exports a pickle in the OLD run_simulations.py format
(list of dicts with '1day_gamma' / 'orgcad_gamma' / 'ref_band' ...) so the
existing plotting script runs on it unchanged:  full -> 1day_gamma,
crop -> orgcad_gamma.

Usage (serial):
    python stage5_average.py                 # median (default) + plot + old-format pkl
    python stage5_average.py --mean          # mean instead of median as central value
    python stage5_average.py --no-pkl        # skip the compatibility pickle

Input : ~/results/stage4/fit_{ra}_{dec}_z{band}.parquet
Output: ~/results/stage5/gamma_summary.csv          per-source table
        ~/results/stage5/gamma_full_vs_crop.png     the plot
        ~/results/stage5/newstage_simfit_results.pkl   old-format export

Only valid fits enter the statistics; per-source valid counts are reported
and stored so thin distributions are visible, not silently averaged.
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S4  = os.environ["HOME"] + "/results/stage4/"
S5  = os.environ["HOME"] + "/results/stage5/"
CADENCES = ("full", "crop")

USE_MEAN = "--mean" in sys.argv
MAKE_PKL = "--no-pkl" not in sys.argv
CENTRAL  = "mean" if USE_MEAN else "median"
mode = 'spl'

os.makedirs(S5, exist_ok=True)


def find_sources():
    out = []
    for p in glob.glob(S4 + "fit_*.parquet"):
        m = re.search(r"fit_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.parquet",
                      os.path.basename(p))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), m.group(3), p))
    return sorted(out)


def central(x):
    return np.mean(x) if USE_MEAN else np.median(x)


def summarize():
    rows, old_format = [], []
    for ra, dec, band, path in find_sources():
        fit = pd.read_parquet(path)
        v = fit[fit["valid"]]
        rec = {"RA": ra, "DEC": dec, "ref_band": band}
        gammas = {}
        for cad in CADENCES:
            g = v.loc[v["cadence"] == cad, f"gamma_{mode}"].values
            gammas[cad] = g
            if len(g):
                p16, p84 = np.percentile(g, [16, 84])
                rec.update({f"{cad}_n_valid": len(g),
                            f"{cad}_gamma": central(g),
                            f"{cad}_p16": p16, f"{cad}_p84": p84})
            else:
                rec.update({f"{cad}_n_valid": 0,
                            f"{cad}_gamma": np.nan,
                            f"{cad}_p16": np.nan, f"{cad}_p84": np.nan})
        rows.append(rec)

        # old run_simulations.py format: full -> 1day_gamma, crop -> orgcad_gamma
        old_format.append({
            "RA": ra, "DEC": dec, "ref_band": band,
            "n_sims": int(min(len(gammas["full"]), len(gammas["crop"]))),
            "1day_gamma":   np.asarray(gammas["full"], dtype=np.float32),
            "orgcad_gamma": np.asarray(gammas["crop"], dtype=np.float32),
        })

        print(f"{ra}_{dec}_z{band}: "
              f"full {CENTRAL}={rec['full_gamma']:.3f} (n={rec['full_n_valid']}), "
              f"crop {CENTRAL}={rec['crop_gamma']:.3f} (n={rec['crop_n_valid']})")
        for cad in CADENCES:
            if rec[f"{cad}_n_valid"] < 0.5 * 120:
                print(f"  [warn] {cad}: only {rec[f'{cad}_n_valid']}/120 valid "
                      f"— statistics are thin")

    return pd.DataFrame(rows), old_format


def plot(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"g": "mediumseagreen", "r": "firebrick", "i": "gold"}
    for b in df["ref_band"].unique():
        m = df["ref_band"] == b
        x, y = df.loc[m, "full_gamma"], df.loc[m, "crop_gamma"]
        ax.errorbar(
            x, y,
            xerr=[x - df.loc[m, "full_p16"], df.loc[m, "full_p84"] - x],
            yerr=[y - df.loc[m, "crop_p16"], df.loc[m, "crop_p84"] - y],
            fmt="o", ms=4, alpha=0.6, elinewidth=0.5, capsize=0,
            color=colors.get(b, "gray"), label=b)
    ax.axline((0, 0), slope=1, lw=1, alpha=0.7, label="1:1")
    ax.set_xlabel(f"{CENTRAL} γ (full cadence)")
    ax.set_ylabel(f"{CENTRAL} γ (crop cadence)")
    ax.legend()
    ax.set_title(f"{CENTRAL.capitalize()} of gamma_{mode} per source, "
                 f"16–84% errors")
    out = S5 + "gamma_full_vs_crop.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"plot -> {out}")


if __name__ == "__main__":
    df, old = summarize()
    if df.empty:
        raise SystemExit("no merged fit parquets in " + S4)

    csv = S5 + "gamma_summary.csv"
    df.to_csv(csv, index=False)
    print(f"table -> {csv}")

    plot(df)

    if MAKE_PKL:
        pkl = S5 + "newstage_simfit_results.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(old, f)
        print(f"old-format pkl -> {pkl}  "
              f"(compatible with the existing 1day/orgcad plotting script)")
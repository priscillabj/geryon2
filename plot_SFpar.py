#!/usr/bin/env python
"""
Plot SF parameters from linmix_merged_props.parquet.

  python plot_SFpar.py                         # scatter, band g, grouped by clasf (default)
  python plot_SFpar.py --scatter --band r --par clasf
  python plot_SFpar.py --gamma posterior       # gamma histogram
  python plot_SFpar.py --amp   posterior       # amp   histogram
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PROPS   = os.path.expanduser("~/results/linmix_merged_props.parquet")
PROPS   = os.path.expanduser("~/results/bat_master_zmadflux2.parquet")
RESULTS = os.path.expanduser("~/results")

# clasf groups (reconcile with df['clasf'].value_counts()):
TP1 = ["Sy1", "Sy1.2"]            # Type 1        -> black, hollow
SY  = ["Sy1.5"]                   # intermediate  -> C2
TP2 = ["Sy1.8", "Sy1.9", "Sy2"]  # Type 2        -> C1

def load(band, fit_model):
    df = pd.read_parquet(PROPS)
    sub = df[(df["band"] == band) & np.isfinite(df[f"gamma_{fit_model}"]) &
             np.isfinite(df[f"A_365_{fit_model}"]) & df[f"valid_{fit_model}"]]
    return sub

def scatter(band, par, fit_model):
    sub = load(band)
    print(f"scatter z{band}: {len(sub)} sources grouped by {par}")
    fig, ax = plt.subplots(figsize=(7, 5))
    for types, color, label in [(TP1, "#9F9F9F", "Type 1"),
                                (SY,  "#E69F00",    "Sy1.5"),
                                (TP2, "#56B4E9",    "Type 2")]:
        agn = sub[sub[par].isin(types)]
        if len(agn) == 0:
            continue
        xerr = np.nan_to_num(np.abs(np.vstack([agn[f"gamma_minerr_{fit_model}"], agn[f"gamma_maxerr_{fit_model}"]])))
        yerr = np.nan_to_num(np.abs(np.vstack([agn[f"A_365_minerr_{fit_model}"], agn[f"A_365_maxerr_{fit_model}"]])))
        kw = dict(fmt="o", alpha=0.3, color=color, label=f"{label} (n={len(agn)})",
                  xerr=xerr, yerr=yerr, elinewidth=0.6, capsize=0)
        if color == "black":
            kw.update(markerfacecolor="none", ecolor="k")
        ax.errorbar(agn["gamma"], agn["A_365"], **kw)
    ax.set_xlabel(r"$\gamma$  (SF slope)")
    ax.set_ylabel(r"$A_{365}$")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5e-3, 1e0); ax.set_ylim(5e-3, 1e0)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"z{band}")
    ax.legend()
    out = os.path.join(RESULTS, f"{band}band_amp_vs_gamma_by{par}_{fit_model}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)

def posterior(band, which, fit_model):
    sub = load(band)
    col, xlabel, color, bins, xlim = {
        "gamma": (f"gamma_{fit_model}", r"$\gamma$ (slope)",        "green", np.linspace(-0.2, 1, 30), (-0.2, 0.875)),
        "amp":   (f"A_365_{fit_model}", "SF amplitude at 365 days", "blue",  np.linspace(-0.1, 2, 30), (-0.1, 2.0)),
    }[which]
    print(f"{which} posterior z{band}: {len(sub)} sources")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sub[col], bins=bins, alpha=0.7, color=color, edgecolor="black")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"z{band}")
    out = os.path.join(RESULTS, f"{band}band_{which}_distribution_{fit_model}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band", default="g", choices=["g", "r", "i"])
    p.add_argument("--par", default="clasf", help="clasf column to group the scatter by")
    p.add_argument("--hist", choices=["gamma,amp"], help="gamma or amp posterior histogram")
    p.add_argument("--fit_model", default="spl", choices=["spl", "bpl"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scatter", action="store_true", help="amp vs gamma scatter (default)")
    # g.add_argument("--gamma", choices=["posterior"], help="gamma posterior histogram")
    # g.add_argument("--amp",   choices=["posterior"], help="amp posterior histogram")
    a = p.parse_args()

    if a.hist:
        posterior(a.band, a.hist, a.fit_model)
    # elif a.amp:
    #     posterior(a.band, "amp", a.fit_model)
    else:
        scatter(a.band, a.par, a.fit_model)

# # plot: finite + valid only (the pipeline has historically stored
# # non-finite _A_365_spl from 10**alpha overflow with valid=True, so filter both)
# m = np.isfinite(df["gamma"]) & np.isfinite(df["A_365"]) & df["valid"]
# pdf = df[m]
# n_bad = (~m).sum()
# print(f"plotting {len(pdf)}/{len(df)} (dropped {n_bad}: non-finite or invalid)")

# colors = {"g": "tab:green", "r": "tab:red", "i": "tab:purple"}
# fig, ax = plt.subplots(figsize=(7, 5))
# for b, sub in pdf.groupby("band"):
#     ax.scatter(sub["gamma"], sub["A_365"], s=12, alpha=0.6,
#                c=colors.get(b, "gray"), label=f"z{b} (n={len(sub)})")
# ax.set_xlabel(r"$\gamma$  (SF slope, _gamma_spl)")
# ax.set_ylabel(r"$A_{365}$  (_A_365_spl)")
# ax.tick_params(axis='both', labelsize=12)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(5e-3,1e0)
# ax.set_ylim(5e-3,1e0)
# ax.grid(True, alpha=0.3)
# ax.legend()
# fig.tight_layout()
# fig.savefig(OUT_PNG, dpi=150)
# print(f"wrote {OUT_PNG}")
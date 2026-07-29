#!/usr/bin/env python
"""plot_no_target.py — diagnostic plots for files where the AGN wasn't found."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless, for geryon2
import matplotlib.pyplot as plt

mis = pd.read_parquet(os.path.expanduser("~/missing_target_ref.parquet"))
OUTDIR = os.path.expanduser("~/results/no_target_plots")
os.makedirs(OUTDIR, exist_ok=True)

for _, row in mis.iloc[:5].iterrows():
    ztf = pd.read_parquet(row["path"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # left: sky positions of everything in the file vs. the target coord
    ax1.scatter(ztf["ALPHAWIN_REF"], ztf["DELTAWIN_REF"], s=3, alpha=0.5, label="detections")
    ax1.scatter(row["ra"], row["dec"], marker="*", s=200, c="red", label="target")
    ax1.set_xlabel("RA"); ax1.set_ylabel("DEC"); ax1.legend()

    # right: LC of the detections nearest the target (crude 5" cone)
    sep = np.hypot((ztf["ALPHAWIN_REF"] - row["ra"]) * np.cos(np.radians(row["dec"])),
                   ztf["DELTAWIN_REF"] - row["dec"]) * 3600.0
    near = ztf[sep < 5.0]
    if len(near):
        ax2.errorbar(near["OBSMJD"], near["MAG_4_TOT_AB"], yerr=near["MERR_4_TOT_AB"],
                     fmt=".", ms=4, elinewidth=0.7)
        ax2.invert_yaxis()
        ax2.set_title(f"{len(near)} pts within 5\"")
    else:
        ax2.set_title(f"no detections within 5\" (min sep {sep.min():.1f}\")"
                      if len(sep) else "empty file")
    ax2.set_xlabel("MJD"); ax2.set_ylabel("mag")

    fig.suptitle(f"{row.ra:.5f} {row.dec:+.5f} z{row.band}")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{row.ra:.5f}_{row.dec:+.5f}_z{row.band}.png", dpi=120)
    plt.close(fig)

print(f"done: {len(mis)} plots in {OUTDIR}")
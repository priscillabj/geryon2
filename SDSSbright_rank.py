#!/usr/bin/env python
"""Rank objects by median AB magnitude of the target light curve."""
import argparse, os, re
from pathlib import Path
import numpy as np, pandas as pd

ZP_UJY, RA_DEC = 23.9, re.compile(r"^\d+\.\d+_[+-]?\d+\.\d+$")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="~/SDSS_S82_QSO")
ap.add_argument("--max-magerr", type=float, default=0.5)
ap.add_argument("--n", type=int, default=20, help="how many to show (0 = all)")
ap.add_argument("--csv", help="write the full table here")
a = ap.parse_args()

rows = []
for d in sorted(p for p in Path(os.path.expanduser(a.root)).iterdir()
                if p.is_dir() and RA_DEC.match(p.name)):
    for f in d.glob("ztfphot_stars_*/psd_out/mhps_input/*.txt"):
        lc = pd.read_csv(f, sep=r"\s+", header=None, comment="#",
                         names=["mjd", "flux", "flux_err"])
        lc = lc[(lc["flux"] > 0) & (lc["flux_err"] > 0)]
        if a.max_magerr:
            lc = lc[lc["flux"] / lc["flux_err"] >= (2.5 / np.log(10)) / a.max_magerr]
        if lc.empty:
            continue
        mag = ZP_UJY - 2.5 * np.log10(lc["flux"].values)
        rows.append({"object": d.name, "n_epochs": len(lc),
                     "mag_med": np.median(mag), "mag_min": mag.min(),
                     "mag_max": mag.max(), "mag_rms": mag.std(),
                     "baseline_d": lc["mjd"].max() - lc["mjd"].min()})

df = pd.DataFrame(rows).sort_values("mag_med").reset_index(drop=True)
if a.csv:
    df.to_csv(os.path.expanduser(a.csv), index=False)
    print(f"wrote {a.csv}")
with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
    print(df if a.n == 0 else df.head(a.n))
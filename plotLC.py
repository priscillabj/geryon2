#!/usr/bin/env python
"""plot_no_target.py — diagnostic plots for files where the AGN wasn't found."""
import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from astropy.coordinates import SkyCoord
from VarTools import _find_target_obj, radec_filename
matplotlib.use("Agg")          # headless, for geryon2
import matplotlib.pyplot as plt

# mis = pd.read_parquet(os.path.expanduser("~/missing_target_ref.parquet"))
# if sigma_filter and not df.empty:
#     df_clean = filter_calstars_sigma(df, mag_column,file=file)

# OUTDIR = os.path.expanduser("~/results/no_target_plots")
# os.makedirs(OUTDIR, exist_ok=True)

# for _, row in mis.iloc[:5].iterrows():
#     ztf = pd.read_parquet(row["path"])

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

#     # left: sky positions of everything in the file vs. the target coord
#     ax1.scatter(ztf["ALPHAWIN_REF"], ztf["DELTAWIN_REF"], s=3, alpha=0.5, label="detections")
#     ax1.scatter(row["ra"], row["dec"], marker="*", s=200, c="red", label="target")
#     ax1.set_xlabel("RA"); ax1.set_ylabel("DEC"); ax1.legend()

#     # right: LC of the detections nearest the target (crude 5" cone)
#     sep = np.hypot((ztf["ALPHAWIN_REF"] - row["ra"]) * np.cos(np.radians(row["dec"])),
#                    ztf["DELTAWIN_REF"] - row["dec"]) * 3600.0
#     near = ztf[sep < 5.0]
#     if len(near):
#         ax2.errorbar(near["OBSMJD"], near["MAG_4_TOT_AB"], yerr=near["MERR_4_TOT_AB"],
#                      fmt=".", ms=4, elinewidth=0.7)
#         ax2.invert_yaxis()
#         ax2.set_title(f"{len(near)} pts within 5\"")
#     else:
#         ax2.set_title(f"no detections within 5\" (min sep {sep.min():.1f}\")"
#                       if len(sep) else "empty file")
#     ax2.set_xlabel("MJD"); ax2.set_ylabel("mag")

#     fig.suptitle(f"{row.ra:.5f} {row.dec:+.5f} z{row.band}")
#     fig.tight_layout()
#     fig.savefig(f"{OUTDIR}/{row.ra:.5f}_{row.dec:+.5f}_z{row.band}.png", dpi=120)
#     plt.close(fig)

# print(f"done: {len(mis)} plots in {OUTDIR}")

BAND_COLOR = {"g": "mediumseagreen", "r": "firebrick", "i": "goldenrod"}

def target_chip(df):
    """Modal (qid, ccdid, field) of the target rows, or None if not available."""
    if df.empty or "qid" not in df.columns:
        return None
    return tuple(df[c].mode().iloc[0] for c in ("qid", "ccdid", "field"))
 
 
def quality_mask(df, mag_err_col, chip=None):
    mask = (df[mag_err_col] < 0.5) & (df["MAGLIM"] > 20.5) & (df["SEEING"] < 3)
    if chip is not None:
        qid, ccdid, field = chip
        mask &= (df["qid"] == qid) & (df["ccdid"] == ccdid) & (df["field"] == field)
    return mask
 
 
def load(path, coord, mag_err_col, with_calstars=False):
    """Return (target, calstars). Either may be None."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
 
    lc = pd.read_parquet(path)
    idx = _find_target_obj(path, coord)
    if idx is None:
        print(f"no target found in {path.name}, skipping")
        return None, None
 
    tgt = lc[lc["object_index"] == idx]
    chip = target_chip(tgt)                       # derived from the target only
    tgt = tgt[quality_mask(tgt, mag_err_col, chip)]
    if tgt.empty:
        print(f"quality cuts left 0 target epochs in {path.name}")
        return None, None
 
    if not with_calstars:
        return tgt, None
 
    cs = lc[lc["object_index"] != idx]
    cs = cs[quality_mask(cs, mag_err_col, chip)]  # same mask, same chip
    if cs.empty:
        print("quality cuts left 0 calibration stars")
        cs = None
    return tgt, cs
 
 
def plot_lightcurve(tgt, cs=None, *, mag_col, mag_err_col,
                    color="xkcd:blue", clip=False, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
 
    if cs is not None:
        ax.errorbar(cs["OBSMJD"], cs[mag_col], yerr=cs[mag_err_col],
                    fmt="o", ms=3, alpha=0.1, c="grey", label="cal stars")
 
    ax.errorbar(tgt["OBSMJD"], tgt[mag_col], yerr=tgt[mag_err_col],
                fmt="o", alpha=0.7, c=color, label="source")
 
    # if clip and "block_id" in tgt.columns:
    #     for i, b in enumerate(tgt.groupby("block_id")["OBSMJD"].first()[1:]):
    #         ax.axvline(b, color="xkcd:grey green", linestyle="--", linewidth=1,
    #                    label="epochs window" if i == 0 else None)
 
    ax.invert_yaxis()
    ax.set_xlabel("MJD [days]", size=25)
    ax.set_ylabel("mag", size=25)
    ax.tick_params(labelsize=20)
    ax.legend()
    return ax
 
 
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    p.add_argument("--calstars", action="store_true",
                   help="overlay calibration stars (off by default)")
    p.add_argument("--clip", action="store_true", help="mark block_id boundaries")
    p.add_argument("--mag-col", default="MAG_4_TOT_AB")
    p.add_argument("--mag-err-col", default="MERR_4_TOT_AB")
    # p.add_argument("--color", default="xkcd:blue")
    p.add_argument("-o", "--outdir", default=None)
    args = p.parse_args(argv)
 
    ra, dec, band = radec_filename(args.path, band=True)
    coord = SkyCoord(ra, dec, unit="deg")
 
    tgt, cs = load(args.path, coord, args.mag_err_col, with_calstars=args.calstars)
    if tgt is None:
        return 1
    
    color = BAND_COLOR.get(band, 'gray')
    ax = plot_lightcurve(tgt, cs, mag_col=args.mag_col, mag_err_col=args.mag_err_col,
                         color=color, clip=args.clip)
 
    outdir = Path(args.outdir) if args.outdir else Path.home() / "results/partials" / f"{ra}_{dec}"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{Path(args.path).stem}_lc.png"
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(out)
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
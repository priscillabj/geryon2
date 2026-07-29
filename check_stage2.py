"""
check_stage2.py — validate stage 2 cropped-LC output against stage 1.

Usage:
    python check_stage2.py                       # validate all cropped sources
    python check_stage2.py 288.4 -12.1 g         # detail check for one source
    python check_stage2.py --plot-sims 288.4 -12.1 g [n]  # A: n sims, one source
    python check_stage2.py --plot-sources [n]    # B: sim 0 across n sources

For each source it INDEPENDENTLY recomputes the nearest-neighbour crop mapping
from (stage-1 t_even, original observation times) and confirms the stored
cropped file matches — i.e. it doesn't just check the crop is plausible, it
checks it's *correct*. Catches silent indexing/ordering bugs before stage 3.

Checks per source:
  - sim count == full-LC sim count, contiguous object_index
  - cropped epoch count == number of original observations
  - identical time grid across all cropped sims
  - cropped times are a subset of the full even-cadence grid
  - cropped mag/err == full-LC mag/err at the recomputed nearest-neighbour idx
    (exact match, per sim)
  - no NaN/inf
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord

from VarTools import _find_target_obj, radec_filename

DATA_PATH   = os.environ["HOME"] + "/BAT_results/"
FULL_PATH   = os.environ["HOME"] + "/results/20yr_lc/full_lc/"
# CROP_PATH   = os.environ["HOME"] + "/results/20yr_lc/ZTF_dur/"
CROP_PATH   = os.environ["HOME"] + "/results/20yr_lc/cropped_lc/"


# def radec_filename(filename):
#     match  = re.search(r"(\d+\.\d+)_([-+]?\d+\.\d+)", filename)
#     match2 = re.search(r"_z(\w)_", filename)
#     band   = match2.group(1)
#     if match:
#         return float(match.group(1)), float(match.group(2)), band
#     raise ValueError(f"Cannot parse RA/DEC from filename: {filename}")


def load_original_times(ra, dec, path):
    """Identical masking to stage 1 / stage 2."""
    ztf = pd.read_parquet(path)
    coord = SkyCoord(ra, dec, unit="deg")
    tgt = _find_target_obj(Path(path), coord)
    if tgt is None:
        return None
    ztf1 = ztf[ztf["object_index"] == tgt]
    mask = ztf1["MERR_4_TOT_AB"] < 0.5
    if "qid" in ztf1.columns:
        qid   = ztf1["qid"].mode().iloc[0]
        ccdid = ztf1["ccdid"].mode().iloc[0]
        field = ztf1["field"].mode().iloc[0]
        mask &= ((ztf1["qid"] == qid) & (ztf1["ccdid"] == ccdid) & (ztf1["field"] == field))
    ztf1 = ztf1[mask]
    if len(ztf1) == 0:
        return None
    t = ztf1.OBSMJD.values.copy()
    t -= t[0]
    return t


def original_file_for(ra, dec, band):
    for f in glob.glob(DATA_PATH + "*z[gri]_merged.parquet"):
        try:
            r, d, b = radec_filename(f, band=True)
        except ValueError:
            continue
        if r == ra and d == dec and b == band:
            return f
    return None


def load_original_lc_full(ra, dec, band):
    """Original (time, mag, err) with identical masking to load_original_times.
    For plotting only — returns None if unavailable."""
    path = original_file_for(ra, dec, band)
    if path is None:
        return None
    ztf = pd.read_parquet(path)
    coord = SkyCoord(ra, dec, unit="deg")
    tgt = _find_target_obj(Path(path), coord)
    if tgt is None:
        return None
    ztf1 = ztf[ztf["object_index"] == tgt]
    mask = ztf1["MERR_4_TOT_AB"] < 0.5
    if "qid" in ztf1.columns:
        qid   = ztf1["qid"].mode().iloc[0]
        ccdid = ztf1["ccdid"].mode().iloc[0]
        field = ztf1["field"].mode().iloc[0]
        mask &= ((ztf1["qid"] == qid) & (ztf1["ccdid"] == ccdid) & (ztf1["field"] == field))
    ztf1 = ztf1[mask]
    if len(ztf1) == 0:
        return None
    t = ztf1.OBSMJD.values.copy()
    t -= t[0]
    return t, ztf1["MAG_4_TOT_AB"].values, ztf1["MERR_4_TOT_AB"].values


def load_full_source(ra, dec, band):
    files = sorted(glob.glob(FULL_PATH + f"sim_{ra}_{dec}_z{band}_rank*_full.parquet"))
    if not files:
        return None
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def find_cropped_sources():
    out = []
    for p in glob.glob(CROP_PATH + "*.stage2.done"):
        m = re.search(r"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.stage2\.done",
                      os.path.basename(p))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(out)


def check_source(ra, dec, band, verbose=False):
    tag = f"{ra}_{dec}_z{band}"
    cpath = CROP_PATH + f"sim_{ra}_{dec}_z{band}_cropped.parquet"
    if not os.path.exists(cpath):
        print(f"[FAIL] {tag}: no cropped parquet")
        return False
    crop = pd.read_parquet(cpath)

    full = load_full_source(ra, dec, band)
    if full is None:
        print(f"[FAIL] {tag}: no stage-1 data to verify against")
        return False

    orig_file = original_file_for(ra, dec, band)
    time_orig = load_original_times(ra, dec, orig_file) if orig_file else None
    if time_orig is None:
        print(f"[FAIL] {tag}: cannot recover original times")
        return False

    ok = True

    # sim sets match
    full_ids = np.sort(full["object_index"].unique())
    crop_ids = np.sort(crop["object_index"].unique())
    if not np.array_equal(full_ids, crop_ids):
        print(f"[FAIL] {tag}: sim id sets differ "
              f"(full {len(full_ids)}, crop {len(crop_ids)})")
        ok = False

    # recompute the crop mapping independently
    idx0   = full["object_index"].min()
    t_even = full[full.object_index == idx0].sort_values("time")["time"].values
    tree   = cKDTree(t_even.reshape(-1, 1))
    _, cropi = tree.query(time_orig.reshape(-1, 1))
    t_snap = t_even[cropi]

    # cropped epoch count == n original obs
    per = crop.groupby("object_index").size()
    if per.nunique() != 1:
        print(f"[FAIL] {tag}: cropped epoch count varies "
              f"({per.min()}..{per.max()})")
        ok = False
    elif per.iloc[0] != len(time_orig):
        print(f"[FAIL] {tag}: cropped epochs {per.iloc[0]} != "
              f"n original obs {len(time_orig)}")
        ok = False

    # times: subset of grid + identical across sims + match recomputed snap
    grid = set(np.round(t_even, 9))
    c0 = crop[crop.object_index == crop_ids[0]].sort_values("object_index")  # keep file order
    # note: stage 2 wrote in cropi order (not sorted by time); compare in that order
    for oi in crop_ids:
        c = crop[crop.object_index == oi]
        if not np.array_equal(c["time"].values, t_snap):
            print(f"[FAIL] {tag}: sim {oi} cropped times != recomputed snap")
            ok = False
            break

    # mag/err exactly match full LC at recomputed indices, per sim
    mism = 0
    for oi in crop_ids:
        f = full[full.object_index == oi].sort_values("time")
        c = crop[crop.object_index == oi]
        exp_mag = f["mag"].values[cropi]
        exp_err = f["err"].values[cropi]
        if not (np.array_equal(c["mag"].values, exp_mag) and
                np.array_equal(c["err"].values, exp_err)):
            mism += 1
    if mism:
        print(f"[FAIL] {tag}: {mism}/{len(crop_ids)} sims' mag/err "
              f"don't match full LC at nearest-neighbour indices")
        ok = False

    # finiteness
    for col in ("mag", "err"):
        bad = (~np.isfinite(crop[col])).sum()
        if bad:
            print(f"[FAIL] {tag}: {bad} non-finite in cropped {col}")
            ok = False

    if ok:
        print(f"[ OK ] {tag}: {len(crop_ids)} sims, {per.iloc[0]} epochs "
              f"(from {len(t_even)} full), mag/err verified against full LC")
    if verbose:
        print(f"       full grid: {len(t_even)} pts, original obs: {len(time_orig)}")
        print(f"       cropped time span: [{t_snap.min():.1f}, {t_snap.max():.1f}] d")
    return ok


def _load_for_plot(ra, dec, band):
    """Return (full_df, crop_df) or (None, None). Shared by both plot modes."""
    full = load_full_source(ra, dec, band)
    cpath = CROP_PATH + f"sim_{ra}_{dec}_z{band}_cropped.parquet"
    if full is None or not os.path.exists(cpath):
        return None, None
    return full, pd.read_parquet(cpath)


def plot_source_sims(ra, dec, band, n_sims=5):
    """Option A: overlay full LC (line) + cropped points (markers) for a few
    sims of ONE source. One subplot per sim (stacked)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full, crop = _load_for_plot(ra, dec, band)
    tag = f"{ra}_{dec}_z{band}"
    if full is None:
        print(f"[skip] {tag}: missing full or cropped data")
        return

    ids = np.sort(full["object_index"].unique())[:n_sims]
    orig = load_original_lc_full(ra, dec, band)   # (t, mag, err) or None
    fig, axes = plt.subplots(len(ids), 1, figsize=(11, 2.2 * len(ids)),
                             sharex=True, squeeze=False)
    for ax, oi in zip(axes[:, 0], ids):
        f = full[full.object_index == oi].sort_values("time")
        c = crop[crop.object_index == oi]
        ax.plot(f["time"], f["mag"], lw=0.6, color="#436BAD",
                alpha=0.8, label="full sim")
        ax.scatter(c["time"], c["mag"], s=14, color="orange",
                   zorder=3, label="cropped sim")
        if orig is not None:
            ax.errorbar(orig[0], orig[1], yerr=orig[2], fmt="k.", ms=4,
                        elinewidth=0.5, capsize=0, zorder=4, alpha=0.8,
                        label="original ZTF")
        ax.invert_yaxis()
        ax.set_ylabel(f"sim {oi}\nmag")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time (d)")
    fig.suptitle(f"{tag}: full LC vs cropped cadence, {len(ids)} sims")
    out = CROP_PATH + f"cmp_{tag}_sims.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"plot -> {out}")


def plot_sim0_sources(sources, n_sources=6):
    """Option B: overlay full LC (line) + cropped points (markers) for sim 0
    of several SOURCES. One subplot per source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sources[:n_sources]
    fig, axes = plt.subplots(len(sources), 1, figsize=(11, 2.2 * len(sources)),
                             squeeze=False)
    for ax, (ra, dec, band) in zip(axes[:, 0], sources):
        tag = f"{ra}_{dec}_z{band}"
        full, crop = _load_for_plot(ra, dec, band)
        if full is None:
            ax.set_title(f"{tag}: missing", fontsize=8)
            continue
        oi = full["object_index"].min()
        f = full[full.object_index == oi].sort_values("time")
        c = crop[crop.object_index == oi]
        ax.plot(f["time"], f["mag"], lw=0.6, color="#436BAD",
                alpha=0.8, label="full sim")
        ax.scatter(c["time"], c["mag"], s=14, color="orange",
                   zorder=3, label="cropped sim")
        orig = load_original_lc_full(ra, dec, band)
        if orig is not None:
            ax.errorbar(orig[0], orig[1], yerr=orig[2], fmt="k.", ms=4,
                        elinewidth=0.5, capsize=0, zorder=4, alpha=0.8,
                        label="original ZTF")
        ax.invert_yaxis()
        ax.set_ylabel("mag")
        ax.set_title(f"{tag} (sim {oi})", fontsize=9)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time (d)")
    fig.suptitle("Full LC vs cropped cadence, sim 0 across sources")
    out = CROP_PATH + "cmp_sim0_sources.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"plot -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]

    # Option A: plot a few sims for one source
    #   python check_stage2.py --plot-sims 288.4 -12.1 g [n_sims]
    if args and args[0] == "--plot-sims":
        ra, dec, band = float(args[1]), float(args[2]), args[3]
        n = int(args[4]) if len(args) > 4 else 5
        plot_source_sims(ra, dec, band, n_sims=n)

    # Option B: plot sim 0 for several sources
    #   python check_stage2.py --plot-sources [n_sources]
    elif args and args[0] == "--plot-sources":
        n = int(args[1]) if len(args) > 1 else 6
        plot_sim0_sources(find_cropped_sources(), n_sources=n)

    # detail check for one source
    elif len(args) == 3:
        check_source(float(args[0]), float(args[1]), args[2], verbose=True)

    # default: validate all sources
    else:
        srcs = find_cropped_sources()
        print(f"found {len(srcs)} cropped sources\n")
        n_ok = sum(check_source(*s) for s in srcs)
        print(f"\n{n_ok}/{len(srcs)} sources passed")
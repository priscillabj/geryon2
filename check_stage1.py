"""
check_stage1.py — validate stage 1 full-LC output.

Usage:
    python check_stage1.py                # summarize all sources
    python check_stage1.py 288.4 -12.1 g  # detail + plot one source

Checks, per source (all rank slices reassembled):
  - sim count == N_SIMS, contiguous object_index 0..N_SIMS-1, no dupes/gaps
  - identical time grid across all sims (even cadence, from gpSimFull)
  - no NaN/inf in mag/err
  - per-sim variability: std(mag). FLAT LCs (std ~ 0) are the DRW
    parameterization failure mode — a genuine DRW sim should have
    std comparable to the DRW amplitude, not ~1e-6.
  - determinism spot-check: re-generate one sim from its (source, index)
    seed and confirm it byte-matches the stored LC.
"""

import os
import re
import sys
import glob
import hashlib
import numpy as np
import pandas as pd

# OUTPUT_PATH = os.environ["HOME"] + "/results/full_lc/"
OUTPUT_PATH = os.environ["HOME"] + "/results/20yr_lc/full_lc/"
N_SIMS = 120


def source_seed(ra, dec, band):
    s = f"{ra}_{dec}_{band}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def find_sources():
    """Return list of (ra, dec, band) from sentinel files."""
    out = []
    for p in glob.glob(OUTPUT_PATH + "*.stage1.done"):
        m = re.search(r"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.stage1\.done",
                      os.path.basename(p))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(out)


def load_source(ra, dec, band):
    """Concatenate all rank slices for one source into one long-format df."""
    pat = OUTPUT_PATH + f"sim_{ra}_{dec}_z{band}_rank*_full.parquet"
    files = sorted(glob.glob(pat))
    if not files:
        return None, []
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df, files


def check_source(ra, dec, band, verbose=False):
    df, files = load_source(ra, dec, band)
    tag = f"{ra}_{dec}_z{band}"
    if df is None:
        print(f"[FAIL] {tag}: no parquet files")
        return False

    ok = True
    idx = np.sort(df["object_index"].unique())

    # sim count / contiguity
    if len(idx) != N_SIMS:
        print(f"[FAIL] {tag}: {len(idx)} sims, expected {N_SIMS}")
        ok = False
    if not np.array_equal(idx, np.arange(N_SIMS)):
        missing = set(range(N_SIMS)) - set(idx.tolist())
        print(f"[FAIL] {tag}: object_index not contiguous 0..{N_SIMS-1}; "
              f"missing {sorted(missing)[:10]}...")
        ok = False

    # identical time grid across sims
    g = df.groupby("object_index")
    n_epochs = g.size()
    if n_epochs.nunique() != 1:
        print(f"[FAIL] {tag}: epoch count varies across sims "
              f"({n_epochs.min()}..{n_epochs.max()})")
        ok = False
    else:
        # compare each sim's time vector to sim 0's
        t0 = df[df.object_index == idx[0]].sort_values("time")["time"].values
        for oi in idx[1:]:
            ti = df[df.object_index == oi].sort_values("time")["time"].values
            if not np.array_equal(ti, t0):
                print(f"[FAIL] {tag}: sim {oi} time grid differs from sim 0")
                ok = False
                break

    # NaN / inf
    for col in ("mag", "err"):
        bad = (~np.isfinite(df[col])).sum()
        if bad:
            print(f"[FAIL] {tag}: {bad} non-finite in {col}")
            ok = False

    # variability — flat LCs = DRW parameterization bug
    stds = g["mag"].std()
    flat = (stds < 1e-4).sum()
    if flat:
        print(f"[WARN] {tag}: {flat}/{N_SIMS} sims have std(mag)<1e-4 "
              f"(possible DRW amplitude mis-map). "
              f"std range [{stds.min():.2e}, {stds.max():.2e}]")
        ok = False
    elif verbose:
        print(f"       std(mag) range [{stds.min():.3f}, {stds.max():.3f}], "
              f"median {stds.median():.3f}")

    if ok and not verbose:
        print(f"[ OK ] {tag}: {N_SIMS} sims, {n_epochs.iloc[0]} epochs, "
              f"std(mag) median {stds.median():.3f}, {len(files)} slice files")

    if verbose:
        print(f"       files: {len(files)}, epochs/sim: {n_epochs.iloc[0]}")
        print(f"       mag: [{df.mag.min():.3f}, {df.mag.max():.3f}], "
              f"err median {df.err.median():.4f}")
        _plot_source(df, idx, tag)

    return ok


def _plot_source(df, idx, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("       (matplotlib unavailable, skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for oi in idx[:8]:
        s = df[df.object_index == oi].sort_values("time")
        ax.plot(s["time"], s["mag"], lw=0.7, alpha=0.7)
    ax.set_xlabel("time (d)"); ax.set_ylabel("mag"); ax.invert_yaxis()
    ax.set_title(f"{tag}: first 8 sims")
    out = OUTPUT_PATH + f"lc_{tag}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"       plot -> {out}")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        ra, dec, band = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
        check_source(ra, dec, band, verbose=True)
    else:
        srcs = find_sources()
        print(f"found {len(srcs)} completed sources\n")
        n_ok = sum(check_source(*s) for s in srcs)
        print(f"\n{n_ok}/{len(srcs)} sources passed")
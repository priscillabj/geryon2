"""
merge_simlc.py

Merge the per-rank parquet files written by run_simulations.py (SAVE_LC=True)
into one file per source per cadence, then delete the rank files.

Usage:
    python merge_simlc.py

Reads and writes one rank file at a time so memory usage stays proportional
to one rank's slice, not the full simulation set.
"""

import os
import glob
import re
import pyarrow.parquet as pq

# ── configuration (must match run_simulations.py) ─────────────────────────────
import os
DATA_PATH = os.environ["HOME"] + "/LC/"
OUTPUT_PATH = os.environ["HOME"] + f"/results/partials/"
# LC_PATH   = DATA_PATH + "sim_lc/"


def find_sources(lc_path):
    """Return a set of (ra, dec, band, cadence) tuples from existing rank files."""
    pattern = os.path.join(lc_path, "sim_*_rank*_*.parquet")
    files   = glob.glob(pattern)
    keys    = set()
    for fp in files:
        m = re.search(r"sim_(\d+\.\d+)_(-?\d+\.\d+)_rank\d+_(\w+)_(\w+)\.parquet", fp)
        if m:
            keys.add((m.group(1), m.group(2), m.group(3), m.group(4)))
    return keys


def merge_one(ra, dec, band, cadence, lc_path):
    """
    Merge all rank files for one (source, cadence) into a single parquet,
    reading one file at a time to keep memory low.
    """
    pattern  = os.path.join(lc_path, f"sim_{ra}_{dec}_rank*_{band}_{cadence}.parquet")
    files    = sorted(glob.glob(pattern))

    if not files:
        print(f"  no files found for {ra} {dec} {band} {cadence}")
        return

    out_path = os.path.join(lc_path, f"sim_{ra}_{dec}_z{band}_{cadence}_merged.parquet")
    writer   = None

    for fp in files:
        table = pq.read_table(fp)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    n_rows = pq.read_metadata(out_path).num_rows
    print(f"  merged {len(files)} rank files → {out_path} ({n_rows} rows)")

    for fp in files:
        os.remove(fp)


def main():
    keys = find_sources(OUTPUT_PATH)
    if not keys:
        print(f"No rank parquet files found in {OUTPUT_PATH}")
        return

    print(f"Found {len(keys)} (source, cadence) combinations to merge")
    for ra, dec, band, cadence in sorted(keys):
        print(f"Merging RA={ra} DEC={dec} band={band} cadence={cadence}")
        merge_one(ra, dec, band, cadence, OUTPUT_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
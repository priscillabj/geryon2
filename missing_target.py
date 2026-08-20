#!/usr/bin/env python
"""check_no_target.py — list sources where _find_target_obj returns None."""
import os, re, glob
import random
import pandas as pd
from VarTools import _find_target_obj, radec_filename

DATA_PATH = os.environ["HOME"] + "/BAT_results/"
SHUFFLE_SEED = 42

# # >>> paste the exact file-list + subsample block from stage1_gen_full_lc_v2.py
# # (glob + random.Random(42).shuffle + [300:1018] slice) so you scan the same set
# all_files = sorted(glob.glob(DATA_PATH + "*sci_merged.parquet"))
# random.Random(SHUFFLE_SEED).shuffle(all_files)
# files = all_files

# missing = []
# for path in files:
#     ra, dec, band = radec_filename(os.path.basename(path), band=True)
#     ztf = pd.read_parquet(path)
#     if _find_target_obj(ra, dec, ztf) is None:
#         missing.append((ra, dec, band))
#         print(f"NO_TARGET {ra} {dec} z{band}", flush=True)

# print(f"\ntotal: {len(missing)}")
# mis_tgt = pd.DataFrame(missing, columns=["ra", "dec", "band"])
# mis_tgt.to_parquet('~/missing_target.parquet')
# with open("no_target_sources.txt", "w") as f:
#     for ra, dec, band, path in missing:
#         f.write(f"{ra} {dec} {band} {path}\n")

#executado com qsub check_stage1.sh

# ------------- Parallel Process ----------------
"""check_no_target.py — list sources where _find_target_obj returns None."""
from multiprocessing import Pool

NPROC = int(os.environ.get("PBS_NP", os.cpu_count()))

def check(path):
    from pathlib import Path
    from astropy.coordinates import SkyCoord
    ra, dec, band = radec_filename(os.path.basename(path), band=True)
    # ztf = pd.read_parquet(path)
    coord = SkyCoord(ra, dec, unit='deg')
    tgt = _find_target_obj(Path(path), coord)
    if tgt is None:
        print(f"NO_TARGET {ra} {dec} z{band}", flush=True)
        return (ra, dec, band, path)
    return None

if __name__ == "__main__":
    all_files = sorted(glob.glob(DATA_PATH + "*_z[gri]_merged.parquet"))
    random.Random(SHUFFLE_SEED).shuffle(all_files)

    with Pool(NPROC) as pool:
        results = pool.map(check, all_files, chunksize=16)

    missing = [r for r in results if r is not None]
    print(f"\ntotal: {len(missing)}")
    mis_tgt = pd.DataFrame(missing, columns=["ra", "dec", "band", "path"])
    mis_tgt.to_parquet(os.path.expanduser("~/BAT_results/missing_targetBAT.parquet"))

#executado com qsub missing_target.sh
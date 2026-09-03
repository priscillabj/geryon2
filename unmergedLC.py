#!/usr/bin/env python
"""Build a JSON manifest of ZTF light-curve files from ~/BAT_results."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BAND = re.compile(r"z[gri]$")


def parse_name(stem):
    """-> ((ra, dec, band, is_sci), is_merged), or None if name breaks convention."""
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    tail = parts[2:]
    if "sci" in tail:
        return None
    band = next((p for p in tail if BAND.fullmatch(p)), None)
    if band is None:
        return None
    # return (parts[0], parts[1], band, "sci" in tail), "merged" in tail
    return (parts[0], parts[1], band), "merged" in tail

def scan(data_dir):
    groups = defaultdict(lambda: {"merged": [], "partial": []})
    unparsed, skipped = [], 0
    for f in data_dir.glob("*.parquet"):
        if "sci" in f.stem.split("_")[2:]:
            skipped += 1
            continue
        parsed = parse_name(f.stem)
        if parsed is None:
            unparsed.append(f.name)
            continue
        key, is_merged = parsed
        groups[key]["merged" if is_merged else "partial"].append(f.name)
    return groups, unparsed, skipped


def select(groups, unmerged_only):
    files = []
    n_merged = n_partial = n_keys_unmerged = 0
    for key, g in sorted(groups.items()):
        if g["merged"]:
            if len(g["merged"]) > 1:
                print(f"multiple merged for {key}: {g['merged']}", file=sys.stderr)
            if not unmerged_only:
                files.extend(g["merged"])
                n_merged += len(g["merged"])
        elif g["partial"]:
            files.extend(g["partial"])
            n_partial += len(g["partial"])
            n_keys_unmerged += 1
    return sorted(files), (n_merged, n_partial, n_keys_unmerged)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unmerged", action="store_true",
                   help="write only per-CCD files whose (RA, DEC, band) has no merged file")
    p.add_argument("--data-dir", type=Path, default=Path.home() / "BAT_results")
    p.add_argument("-o", "--out", type=Path,
                   help="output JSON (default: <data-dir>/{input,unmerged}_files.json)")
    args = p.parse_args()

    out = args.out or args.data_dir / (
        "unmerged_files.json" if args.unmerged else "input_files.json")
    
    import pandas as pd    
    d = pd.read_parquet(args.data_dir/'zmad_all_flux.parquet')
    
    # files = d.loc[(d.ok == True) & (d.sigma >= 10), 'file'].drop_duplicates().sort_values().tolist()
    files = d.loc[(d.sigma >= 10), 'file'].drop_duplicates().sort_values().tolist()
    # json.dump(files, open(out, 'w'), indent=1)

    # groups, unparsed, skipped = scan(args.data_dir)
    # files, (n_merged, n_partial, n_keys) = select(groups, args.unmerged)
    out.write_text(json.dumps(files, indent=2))

    print(f"{len(files)} files -> {out}")
    # print(f"  {n_merged} merged + {n_partial} partial from {n_keys} unmerged keys")
    # print(f"  {len(groups)} keys total")
    # print(f"  {skipped} sci files skipped")
    # if unparsed:
    #     print(f"  {len(unparsed)} unparsed, e.g. {unparsed[:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
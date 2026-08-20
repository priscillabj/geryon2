#!/usr/bin/env python3
import os, sys, time
from pathlib import Path

root  = Path.home() / "results/partials"
lst   = sys.argv[1]
since = time.mktime(time.strptime(sys.argv[2], "%Y-%m-%d"))
n     = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def coords(name):
    ra, dec = name.split("_")[:2]
    return (float(ra), float(dec))


names = [l.strip() for l in open(lst) if l.strip()]
if n:
    names = names[:n]

# one traversal: coordinate pair -> list of paths
index = {}
for dirpath, _, files in os.walk(root):
    for fn in files:
        try:
            key = coords(fn)
        except (ValueError, IndexError):
            continue                      # name doesn't start with ra_dec
        index.setdefault(key, []).append(os.path.join(dirpath, fn))

for f in names:
    try:
        key = coords(f)
    except (ValueError, IndexError):
        print(f"unparseable: {f}", file=sys.stderr)
        continue

    hits = index.get(key, [])
    if not hits:
        print(f"absent: {f}", file=sys.stderr)
        continue

    recent = [p for p in hits if os.stat(p).st_mtime >= since]
    if recent:
        # print(*recent, sep="\n")
        print(f"{f} - {len(recent)} files already existed")
    else:
        # print(f"stale: {f}", file=sys.stderr)
        print(f"{f} was not processed")

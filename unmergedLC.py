import re
from collections import defaultdict
from pathlib import Path
import json

d = Path(".")
BAND = re.compile(r"z[a-z]+$")
OUT = Path("unmerged_files.json")

groups = defaultdict(list)         # (ra, dec, band) -> [(suffix, path), ...]
for f in d.glob("*.parquet"):
    parts = f.stem.split("_")
    band = next((p for p in parts[2:] if BAND.fullmatch(p)), None)
    if band is None:
        print(f"unparsed: {f.name}")
        continue
    groups[(parts[0], parts[1], band)].append((parts[-1], f))


# missing = sorted(k for k, sufs in groups.items() if "merged" not in sufs)
# for ra, dec, band in missing:
#     print(f"{ra}_{dec}_{band}")

unmerged = sorted(
    f for v in groups.values()
    if not any(s == "merged" for s, _ in v)
    for _, f in v
)

# OUT.write_text("".join(f"{f}\n" for f in unmerged))

OUT.write_text(json.dumps([str(f) for f in unmerged]))
print(f"{len(unmerged)} files from {sum(1 for v in groups.values() if not any(s == 'merged' for s, _ in v))} keys -> {OUT}")
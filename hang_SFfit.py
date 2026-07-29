import numpy as np, pandas as pd, hashlib
from newSF import parse_to_dict, SF_linmix
import signal, traceback, sys

def _dump(signum, frame):
    print("=== HANG: stack at 300s ===", flush=True)
    traceback.print_stack(frame)
    sys.exit(2)

signal.signal(signal.SIGALRM, _dump)
signal.alarm(300)

ra, dec, band, OI, CAD = 263.89065, 20.7964, "r", 42, "full"

df = pd.read_parquet(f"/home/pjorge/results/sf_cache/sf_{ra}_{dec}_z{band}_rank8.parquet")
srow = df.iloc[-1]
d = {k: parse_to_dict(srow[k])["SF"] for k in ("SF", "SFmaxerr", "SFminerr")}

seed = int(hashlib.md5(f"{ra}_{dec}_{band}_{OI}_{CAD}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
np.random.seed(seed)
print(f"seed={seed}; running SF_linmix — if no 'LinMix complete!' within ~2 min, Ctrl-C and save the FULL traceback")
res = SF_linmix(d)
print("completed:", None if res is None else res["_gamma_spl"])
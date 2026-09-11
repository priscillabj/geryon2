"""
runSF.py
run optSF(more efficient and modular version
with filename as input parameter) in parallel 
in parquet files LCs

Usage on the cluster:
    mpirun -n <nranks> python runSF.py                                # all parquets
    mpirun -n 64 python runSF.py --list ~/results/input_files.json    # subsample
    mpirun -n 1  python runSF.py --file 0.20323_-7.15322_zg_merged.parquet
    mpirun -n 64 python runSF.py --force                              # overwrite existing

Requirements:
    mpi4py, numpy, pandas, scipy, linmix, astropy, matplotlib

Parallelisation strategy
────────────────────────
optSF is embarrassingly parallel — each file is fully independent.
Rank 0 builds the file list and broadcasts it. Each rank takes every
nrank-th file (round-robin), processes it with optSF, and saves its
own pkl directly via save=True. No gathering needed.
"""

import os
import glob
import time
import pickle
import traceback
from mpi4py import MPI
from newSF import optSF

import argparse
import json
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
# INPUT_GLOB = os.environ["HOME"] + "/BAT_results/*z[gri]_merged.parquet"
# INPUT_GLOB = os.environ["HOME"] + "/BAT_results/139.80500_+55.46528_zg_merged.parquet"

# CUTOFF = time.mktime(time.strptime("2026-08-17", "%Y-%m-%d"))  # rerun anything older
CUTOFF = None
DATA_DIR = Path(os.environ["HOME"]) / "BAT_results"
OUTPUT_ROOT = os.path.join(os.environ["HOME"], "results", "partials")

# N_SOURCES = 10
N_SOURCES = None
OPTsf_KWARGS = dict(
    save        = True,
    clip        = True,
    showallcs   = True,
    save_plt    = True,
    sigma_filter= True,
    x           = 10,
    # plot_sigma  = True
)

# ── MPI setup ─────────────────────────────────────────────────────────────────
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()

# ── main ──────────────────────────────────────────────────────────────────────
def output_exists(f):
    base = os.path.basename(f)                                  # 0.20323_-7.15322_zg_merged.parquet

    # --------- skip all src that already exists ------------
    # src = re.sub(r"_z[gri]_merged\.parquet$", "", base)         # 0.20323_-7.15322
    # out = os.path.join(OUTPUT_ROOT, src, f"{base}_{OPTsf_KWARGS['x']}cs.pkl")
    # return os.path.exists(out)
    # if not os.path.exists(out):
    #     return False
    # try:
    #     with open(out, "rb") as fh:
    #         pickle.load(fh)
    #     return True
    # except Exception:
    #     os.remove(out)   # corrupt; delete so it re-runs
    #     return False
    
    # --------- skip new out that already exists ------------
    # --------- rerun old (stale) ones ------------
    out  = f"{base}_{OPTsf_KWARGS['x']}cs.pkl"
    hits = glob.glob(os.path.join(OUTPUT_ROOT, "*", out))  # subdir-agnostic
    if not hits:
        return False
    p = hits[0]
    if os.path.getmtime(p) < CUTOFF:      # stale — rerun
        return False
    if not os.path.exists(p):  #True if file does NOT exist, False if it does
        return False
    try:
        with open(p, "rb") as fh:
            pickle.load(fh)
        return True
    except Exception:
        os.remove(p)
        return False

def build_file_list(args):
    """Resolve the CLI options into a list of absolute parquet paths."""
    if args.file:
        p = Path(args.file)
        if not p.exists():
            p = DATA_DIR / p                       # bare basename
        if not p.exists():
            raise FileNotFoundError(args.file)
        return [str(p)]

    if args.list:
        list_path = Path(args.list)
        suffix = list_path.suffix.lower()

        if suffix == ".txt":
            names = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]
        elif suffix == ".json":
            names = json.loads(list_path.read_text())
        elif suffix == ".jsonl":
            names = [json.loads(l)["file"] for l in list_path.read_text().splitlines() if l.strip()]
        # elif suffix == ".csv":
        #     df = pd.read_csv(list_path)
        #     col = "file" if "file" in df.columns else "lc_file"
        #     names = df[col].tolist()
        # elif suffix == ".parquet":
        #     df = pd.read_parquet(list_path)
        #     col = "file" if "file" in df.columns else "lc_file"
        #     names = df[col].tolist()
        else:
            raise ValueError(f"Unsupported --list extension: {suffix}")

        return [str(DATA_DIR / p) for p in names][:N_SOURCES]

    return sorted(glob.glob(str(DATA_DIR / "*.parquet")))[:N_SOURCES]


def main():
    if rank == 0:
        job_start = time.time()
        parser = argparse.ArgumentParser()
        src = parser.add_mutually_exclusive_group()
        src.add_argument("--file", help="single parquet: basename (resolved against "
                                        "DATA_DIR) or a full path")
        src.add_argument("--list", help='subsample: .txt (one per line), .json, .jsonl, '
                                  '.csv or .parquet with a file/lc_file column')
        parser.add_argument("--force", action="store_true",
                            help="reprocess and overwrite existing pkl")
        args = parser.parse_args()

        all_files = build_file_list(args)
        force = args.force

        print(f"[rank 0] {len(all_files)} files to process across {nrank} ranks"
              f"{' (--force: overwriting)' if force else ''}", flush=True)
    else:
        all_files = None
        job_start = None
        force = None

    all_files = comm.bcast(all_files, root=0)
    job_start = comm.bcast(job_start, root=0)
    force     = comm.bcast(force, root=0)

    # round-robin distribution — rank r processes files r, r+nrank, r+2*nrank, ...
    my_files = all_files[rank::nrank]
    print(f"[rank {rank}] assigned {len(my_files)} files", flush=True)

    for f in my_files:
        if not force and output_exists(f):
            print(f"[rank {rank}] skip (exists) — {f}", flush=True)
            continue
        t0 = time.time()
        try:
            result = optSF(f, **OPTsf_KWARGS)
            status = "done" if result is not None else "skipped (returned None)"
            print(f"[rank {rank}] {status} in {time.time()-t0:.1f}s — {f}",
                  flush=True)
        except Exception:
            print(f"[rank {rank}] ERROR on {f}:\n{traceback.format_exc()}",
                  flush=True)

    # wait for all ranks before printing total time
    comm.Barrier()
    if rank == 0:
        total = time.time() - job_start
        hours, rem = divmod(total, 3600)
        mins, secs = divmod(rem, 60)
        print(f"[rank 0] all done. "
              f"walltime: {int(hours):02d}h {int(mins):02d}m {secs:04.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
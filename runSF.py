"""
run_optSF.py
run optSF(more efficient and modular version
with filename as input parameter) in parallel 
in parquet files LCs

Usage on the cluster:
    mpirun -n <nranks> python run_optSF.py

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

# ── configuration ─────────────────────────────────────────────────────────────
INPUT_GLOB = os.environ["HOME"] + "/BAT_results/*z[gri]_merged.parquet"
# INPUT_GLOB = os.environ["HOME"] + "/BAT_results/139.80500_+55.46528_zg_merged.parquet"
OUTPUT_ROOT = os.path.join(os.environ["HOME"], "results", "partials")
# CUTOFF = time.mktime(time.strptime("2026-06-25", "%Y-%m-%d"))  # rerun anything older

N_SOURCES = 10
OPTsf_KWARGS = dict(
    save        = True,
    # clip        = True,
    showallcs   = True,
    save_plt    = True,
    sigma_filter= True,
    x           = 10,
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
    # if os.path.getmtime(p) < CUTOFF:      # stale — rerun
    #     return False
    try:
        with open(p, "rb") as fh:
            pickle.load(fh)
        return True
    except Exception:
        os.remove(p)
        return False

def main():
    if rank == 0:
        job_start = time.time()
        all_files = sorted(glob.glob(INPUT_GLOB))[:N_SOURCES]
        # subset = all_files[:N_SOURCES]
        print(f"[rank 0] {len(all_files)} files to process across {nrank} ranks",
              flush=True)
    else:
        all_files = None
        job_start = None

    all_files = comm.bcast(all_files, root=0)
    job_start = comm.bcast(job_start, root=0)

    # round-robin distribution — rank r processes files r, r+nrank, r+2*nrank, ...
    my_files = all_files[rank::nrank]
    print(f"[rank {rank}] assigned {len(my_files)} files", flush=True)

    for f in my_files:
        # if output_exists(f):
            # print(f"[rank {rank}] skip (exists) — {f}", flush=True)
            # continue
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
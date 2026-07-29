
from newSF import *
import os
import glob
import pandas as pd
from VarTools import radec_filename

# INPUT_GLOB = os.environ["HOME"] + "/BAT_results/clean_*.parquet"
# all_files = sorted(glob.glob(INPUT_GLOB))
# inp_ra,inp_dec = radec_filename(all_files[0])
# df = pd.read_parquet(all_files[0])

# # print(inp_ra,inp_dec)
# ztf     = filter_agn(df, ra=inp_ra, dec=inp_dec)

# mag_column='MAG_4_TOT_AB'
# cs = quality_cuts(df, ztf,mag_column,x=10)
# # cs_clean = filter_calstars_sigma(cs, mag_column, file=all_files[0],ra=inp_ra,dec=inp_dec)
# # print(cs['object_index'].unique())
# cs = filter_calstars_sigma(df, mag_column, file=all_files[0],
#                             cs_list=cs['object_index'].unique(),
#                             ra=inp_ra,dec=inp_dec)

from mpi4py import MPI


# ── per-file processing ───────────────────────────────────────────────────────

def process_one_file(file, mag_column='MAG_4_TOT_AB', x=10):
    """Read one parquet file, select calstars, return result dict.

    Reads the file exactly once. Returns None on any unrecoverable issue.
    """
    inp_ra, inp_dec = radec_filename(file)
    df              = pd.read_parquet(file)
    ztf             = filter_agn(df, ra=inp_ra, dec=inp_dec)

    if ztf.empty:
        print(f'skipping {os.path.basename(file)} — no AGN found')
        return None

    cs = quality_cuts(df, ztf, mag_column, x=x)
    if cs is None or cs.empty:
        print(f'skipping {os.path.basename(file)} — no calstars after quality cuts')
        return None

    cs = filter_calstars_sigma(df, mag_column,
                               file=file,
                               cs_list=cs['object_index'].unique(),
                               ra=inp_ra, dec=inp_dec,
                               plot=False,
                               save_plt=True)
    if cs is None or cs.empty:
        print(f'skipping {os.path.basename(file)} — no calstars after sigma filtering')
        return None

    return {'file': file, 'ztf': ztf, 'cs': cs}


# ── MPI main ──────────────────────────────────────────────────────────────────

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ── rank 0 builds the file list ───────────────────────────────────────────
    if rank == 0:
        INPUT_GLOB = os.environ['HOME'] + '/BAT_results/270.80581_+4.85428_*.parquet'
        all_files  = sorted(glob.glob(INPUT_GLOB))
        print(f'{len(all_files)} files found, distributing across {size} ranks')
    else:
        all_files = None

    # broadcast full list to all ranks before scatter
    all_files = comm.bcast(all_files, root=0)

    # ── scatter: each rank gets a roughly equal slice ─────────────────────────
    local_files = np.array_split(all_files, size)[rank].tolist()

    if not local_files:
        print(f'rank {rank} — no files assigned, waiting at barrier')

    # ── each rank processes its slice ─────────────────────────────────────────
    # local_results = []
    for file in local_files:
        print(f'rank {rank} — processing {os.path.basename(file)}')
        try:
            # result = process_one_file(file)
            process_one_file(file)
            # if result is not None:
            #     local_results.append(result)
        except Exception as e:
            print(f'rank {rank} — ERROR on {os.path.basename(file)}: {e}')

    # # ── each rank saves its own results independently ─────────────────────────
    # rank_file = os.environ['HOME'] + f'/BAT_results/var_cs_rank{rank}.pkl'
    # with open(rank_file, 'wb') as f:
    #     pickle.dump(local_results, f)
    # print(f'rank {rank} — saved {len(local_results)} results to {rank_file}')

    # ── wait for all ranks to finish writing ──────────────────────────────────
    comm.Barrier()

    # ── rank 0 merges all per-rank files ──────────────────────────────────────
    if rank == 0:
        # results = []
        # for r in range(size):
        #     r_file = os.environ['HOME'] + f'/BAT_results/var_cs_rank{r}.pkl'
        #     with open(r_file, 'rb') as f:
        #         results.extend(pickle.load(f))
        #     os.remove(r_file)

        print(f'done: {len(results)} / {len(all_files)} files processed successfully')

        # output_file = os.environ['HOME'] + '/BAT_results/var_cs_results.pkl'
        # with open(output_file, 'wb') as f:
        #     pickle.dump(results, f)
        # print(f'results saved to {output_file}')


if __name__ == '__main__':
    main()
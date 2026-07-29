# from newSF import plot_SF   # or whatever the file is called
from newSF import SF_wnoise, SF_linmix
import pickle                   # or however you load your data
import pandas as pd
import pyarrow
import os

PKL_PATH = '/home/pjorge/results/partials/'

# ------------------ PICKLE FIT DICT ------------------------
# -----------------------------------------------------------
'''
plot SF and linmix SF fit from pickle generated in 
SF_wnoise AND SF_linmix
'''
# FILE = '12simfit_results.pkl'

# load your data
# with open(PKL_PATH+FILE, 'rb') as f:
#     SF_dict = pickle.load(f)

# call the function
# plot_SF(SF_dict[0], band='g', model='single')

# ------------------ PARQUET LC DF --------------------------
# -----------------------------------------------------------
'''SF calculation (SF_wnoise) from parquet LC, identifying 
mag, magerr and MJD cols'''

cadence = 'daily'
# cadence = 'orgcad'
RA = 0.610105
DEC = 3.3519118
band = 'g'
# FILE = f'sim_{RA}_{DEC}_z{band}_{cadence}_merged.parquet'

# load your data
# df = pd.read_parquet(PKL_PATH+FILE)

# call the function
# SF_wnoise(mag_col, time_col, mag_err, cs_all=None, clip=False, weight=False,
#               color='red', showallcs=False, plot=False, save_plt=False, save=False,
#               logbin_min=np.log10(0.5), logbin_max=np.log10(2500))

# SF_wnoise(df['mag'], df['time'], df['err'],save_plt=True, save=True)

# results = []

# for obj_idx, group in df.groupby('object_index'):
#     print(f'\n── object_index {obj_idx} ──')
#     subpath= PKL_PATH + f'{RA}_{DEC}/'
#     # if not os.path.exists(subpath):
#     os.makedirs(subpath, exist_ok=True)

#     path = subpath + f'{obj_idx}_{cadence}'
#     SF_dict = SF_wnoise(
#         mag_col  = group['mag'],
#         time_col = group['time'],
#         mag_err  = group['err'],
#         cs_all   = None,   # or pass your calstar DataFrame
#         plot     = False,
#         path     = path,
#         save_plt = True,
#         save     = True,
#     )

#     SF_dict[0].update({'object_index': obj_idx})
#     results.append(SF_dict[0])

# with open(PKL_PATH + f'simulated_{RA}_{DEC}_z{band}_{cadence}SF.pkl', 'wb') as f:
#     pickle.dump(results, f)
# print(f'pkl file successfully saved in {PKL_PATH}\n')


# ------------- parallel ----------------------

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # ── rank 0 prepares the work list ─────────────────────────────────────────────
# if rank == 0:
#     obj_indices = df['object_index'].unique()
# else:
#     obj_indices = None

# # broadcast the full index list to all ranks
# obj_indices = comm.bcast(obj_indices, root=0)

# # ── scatter: each rank gets a roughly equal slice ─────────────────────────────
# local_indices = np.array_split(obj_indices, size)[rank]

# # ── each rank processes its own slice ─────────────────────────────────────────
# local_results = []
# for obj_idx in local_indices:
#     group = df[df['object_index'] == obj_idx]
#     print(f'rank {rank} processing object_index {obj_idx}')
#     SF_dict = SF_wnoise(
#         mag_col  = group['MAG_4_TOT_AB'].values,
#         time_col = group['OBSMJD'].values,
#         mag_err  = group['MERR_4_TOT_AB'].values,
#         cs_all   = None,
#         plot     = False,
#         save     = False,
#     )
#     SF_dict[0].update({'object_index': obj_idx})
#     local_results.append(SF_dict[0])

# # ── gather all results to rank 0 ──────────────────────────────────────────────
# all_results = comm.gather(local_results, root=0)

# if rank == 0:
#     # flatten the list of lists from each rank
#     results = [item for sublist in all_results for item in sublist]
#     print(f'done: {len(results)} objects processed')

# ------------------ LINMIX PKL FIT -------------------------
# -----------------------------------------------------------
'''Fit SF with SF_linmix'''
FILE = f'simulated_{RA}_{DEC}_z{band}_{cadence}SF.pkl'
SF_dict = pd.read_pickle(PKL_PATH+FILE)

subpath= PKL_PATH + f'{RA}_{DEC}/'
os.makedirs(subpath, exist_ok=True)

fitted = []
for i, old_dict in enumerate(SF_dict):
        obj_id = old_dict.get('object_index', old_dict.get('RA', f'item_{i}'))
        print(f'fitting {obj_id}')

        path = subpath + f'v2_{obj_id}_{cadence}'

        result = SF_linmix(
        old_dict,
        save_plot = True,
        save_pkl  = True,
        path      = path
        )
        if result is not None:
            fitted.append(result)

with open(PKL_PATH + f'v2_simulated_{RA}_{DEC}_z{band}_{cadence}SFfit.pkl', 'wb') as f:
    pickle.dump(fitted, f)
print(f'pkl file successfully saved in {PKL_PATH}\n')
print(f'done: {len(fitted)} / {len(SF_dict)} objects fitted successfully')

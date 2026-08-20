# from newSF import plot_SF   # or whatever the file is called
# from newSF import SF_wnoise, SF_linmix
# import pickle                   # or however you load your data
import pandas as pd
# import pyarrow
import os

# PKL_PATH = '/home/pjorge/results/partials/'

# ------------------ PICKLE FIT DICT ------------------------
# -----------------------------------------------------------
# '''
# plot SF and linmix SF fit from pickle generated in 
# SF_wnoise AND SF_linmix
# '''
# FILE = '12simfit_results.pkl'

# load your data
# with open(PKL_PATH+FILE, 'rb') as f:
#     SF_dict = pickle.load(f)

# call the function
# plot_SF(SF_dict[0], band='g', model='single')

# ------------------ PARQUET LC DF --------------------------
# -----------------------------------------------------------
# '''SF calculation (SF_wnoise) from parquet LC, identifying 
# mag, magerr and MJD cols'''

# cadence = 'daily'
# # cadence = 'orgcad'
# RA = 0.610105
# DEC = 3.3519118
# band = 'g'
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
# '''Fit SF with SF_linmix'''
# FILE = f'simulated_{RA}_{DEC}_z{band}_{cadence}SF.pkl'
# SF_dict = pd.read_pickle(PKL_PATH+FILE)

# subpath= PKL_PATH + f'{RA}_{DEC}/'
# os.makedirs(subpath, exist_ok=True)

# fitted = []
# for i, old_dict in enumerate(SF_dict):
#         obj_id = old_dict.get('object_index', old_dict.get('RA', f'item_{i}'))
#         print(f'fitting {obj_id}')

#         path = subpath + f'v2_{obj_id}_{cadence}'

#         result = SF_linmix(
#         old_dict,
#         save_plot = True,
#         save_pkl  = True,
#         path      = path
#         )
#         if result is not None:
#             fitted.append(result)

# with open(PKL_PATH + f'v2_simulated_{RA}_{DEC}_z{band}_{cadence}SFfit.pkl', 'wb') as f:
#     pickle.dump(fitted, f)
# print(f'pkl file successfully saved in {PKL_PATH}\n')
# print(f'done: {len(fitted)} / {len(SF_dict)} objects fitted successfully')


from zmad_plots import zmad_type_hist, zmad_type_fraction, zmad_summary

df = pd.read_parquet('~/results/bat_master_zmadflux.parquet')

zmad_type_hist(df, type_col='clasf', prefix='zmad_', cut=10, save='~/results/zmad_g/types_hist_flux=TRUE.png')
zmad_type_fraction(df, type_col='clasf', prefix='zmad_', cut=10, save='~/results/zmad_g/type_frac_flux=TRUE.png')
# zmad_type_fraction(df, type_col='clasf', prefix='zmad_', cut=10, mode='rate', save='~/results/zmad_g/type_frac_rate.png')
zmad_summary(df, prefix='zmad_', save='~/results/zmad_g/summary_g_flux=TRUE.png')

# print(df['clasf'].value_counts(dropna=False))

# from zmad_plots import zmad_census
# zmad_census(df, type_col='clasf', prefix='zmad_')

# groups = {'Sy1-1.2': ['Sy1.0','Sy1.2'], 'Sy1.5': ['Sy1.5'],
#           'Sy1.8-1.9-2': ['Sy1.8','Sy1.9','Sy2.0']}
# colors = {'Sy1.0': '#9F9F9F', 'Sy2.0': '#4A0E2E'}
# zmad_type_fraction(d, type_col='clasf', prefix='zmad_', groups=groups, colors=colors)

# import numpy as np
# import matplotlib.pyplot as plt

# # --- plot config: band + classification groups ----------------------------
# BAND = "g"                         # <-- band to plot: 'g', 'r', or 'i'
# MODEL = "spl"                      # 'spl' (plots A_365) or 'bpl' (plots A_break)

# # reconcile these with the printed clasf.value_counts() labels before trusting the plot:
# TP1 = ["Sy1", "Sy1.2"]             # Type 1        -> black, hollow markers
# SY  = ["Sy1.5"]                    # intermediate  -> C2
# TP2 = ["Sy1.8", "Sy1.9", "Sy2"]   # Type 2        -> C1

# AMP   = {"spl": "A_365", "bpl": "A_break"}

# OUT_PNG    = os.path.expanduser("~/results/amp_vs_gamma_byclasf_sig>10.png")

# # --- 5. plot: one band + one model, coloured by classification --------------
# amp  = AMP[MODEL]
# gcol, ycol = f"gamma_{MODEL}", f"{amp}_{MODEL}"
# xerr_cols  = (f"gamma_minerr_{MODEL}", f"gamma_maxerr_{MODEL}")
# yerr_cols  = (f"{amp}_minerr_{MODEL}", f"{amp}_maxerr_{MODEL}")
 
# _missing = [c for c in (gcol, ycol, *xerr_cols, *yerr_cols) if c not in df.columns]
# if _missing:
#     raise SystemExit(f"plot columns absent: {_missing}\n"
#                      f"available: {[c for c in df.columns if c.endswith(('_spl', '_bpl'))]}")
 
# sub = df[(df["band"] == BAND) & (df[f"zmad_sigma"] >= 10) & (df[f"valid_{MODEL}"] == True) &
#          np.isfinite(df[gcol]) & np.isfinite(df[ycol])]
# print(f"plotting {MODEL} band z{BAND}: {len(sub)} sources")
 
# fig, ax = plt.subplots(figsize=(7, 5))
# for types, color, label in [(TP1, "black", "Type 1"),
#                             (SY,  "C2",    "Sy1.5"),
#                             (TP2, "C1",    "Type 2")]:
#     agn = sub[sub["clasf"].isin(types)]
#     if len(agn) == 0:
#         continue
#     xerr = np.nan_to_num(np.abs(np.vstack([agn[c] for c in xerr_cols])))
#     yerr = np.nan_to_num(np.abs(np.vstack([agn[c] for c in yerr_cols])))
#     kw = dict(fmt="o", alpha=0.3, color=color, label=f"{label} (n={len(agn)})",
#               xerr=xerr, yerr=yerr, elinewidth=0.6, capsize=0)
#     if color == "black":
#         kw.update(markerfacecolor="none", ecolor="k")
#     ax.errorbar(agn[gcol], agn[ycol], **kw)
 
# ax.set_xlabel(r"$\gamma$  (SF slope)")
# ax.set_ylabel(rf"${amp}$")
# ax.tick_params(axis="both", labelsize=12)
# ax.set_xscale("log"); ax.set_yscale("log")
# ax.set_xlim(5e-3, 1e0); ax.set_ylim(5e-3, 1e0)
# ax.grid(True, alpha=0.3)
# ax.set_title(f"z{BAND}  ({MODEL})")
# ax.legend()
# fig.tight_layout()
# fig.savefig(OUT_PNG, dpi=150)
# print("wrote", OUT_PNG)
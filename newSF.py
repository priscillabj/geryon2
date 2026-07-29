#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

import os, math, time,re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from linmix import linmix
from VarTools import *
from VarTools import _find_target_obj
#import psutil
path= os.environ['HOME']+'/Nextcloud/Doutorado/Forced_Phot/' 


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized SFarray  (drop-in replacement, identical outputs)
# ─────────────────────────────────────────────────────────────────────────────
def SFarray(jd, mag, err):
    """
    Vectorized version of the original SFarray.
    Outputs are identical: (tauarray, sfarray, errarray)
      tauarray : jd[j] - jd[i]          for all i < j  (always positive)
      sfarray  : |mag[i] - mag[j]|
      errarray : err[i]^2 + err[j]^2
    Safe for both list and NumPy array inputs.
    """
    jd  = np.asarray(jd,  dtype=float)
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)

    i_idx, j_idx = np.triu_indices(len(mag), k=1)

    tauarray = jd[j_idx]  - jd[i_idx]
    sfarray  = np.abs(mag[i_idx] - mag[j_idx])
    errarray = err[i_idx] ** 2 + err[j_idx] ** 2

    return tauarray, sfarray, errarray

def err_prop(dmag, prnt=False, plot=False):
    
    if len(dmag)>0:
        #sqrt_avg_dmag = np.sqrt(sq_avg_dmag)
        #plt.hist(sqrt_avg_dmag_corr)
        #plt.xlabel('sqrt mag diff')
        #plt.ylabel('frequency')
        # Calculate the quantiles
        median = np.percentile(dmag, 50)
        lower_bound = np.percentile(dmag, 16)
        upper_bound = np.percentile(dmag, 84)

        # 1-sigma errors (34% quantiles)
        lower_error = median - lower_bound
        upper_error = upper_bound - median

        if prnt:
            print(f"Median: {median:.2f}")
            print(f"Lower 1-sigma error: {lower_error:.3f}")
            print(f"Upper 1-sigma error: {upper_error:.3f}")
        if plot:
            plt.plot(dmag)
            plt.xlabel('Δmag')
            plt.ylabel('number of sources per bin')

        return lower_error,upper_error


# --- Helper utilities moved to module level (no nested defs) ---

def clip_block(group, mag_column):
    """Sigma-clip a group's magnitude column and return only non-clipped rows."""
    clipped = sigma_clip(group[mag_column], sigma=2.5)
    good_points = ~clipped.mask
    return group.loc[good_points]


def group_stats(group, mag_col_name, norm=True):
    """Estimate error statistics for a binned group.
    Returns a pd.Series with index ['max_err','min_err','count']
    The function expects an err_prop() available in the namespace.
    """
    dmag_vals = group[mag_col_name].dropna()
    if len(dmag_vals) == 0:
        return pd.Series([np.nan, np.nan, 0], index=['max_err', 'min_err', 'count'])
    err = err_prop(dmag_vals)
    if norm is False:
        return pd.Series([err[1], err[0], len(dmag_vals)], index=['max_err', 'min_err', 'count'])
    return pd.Series([err[1] / np.sqrt(len(dmag_vals)), err[0] / np.sqrt(len(dmag_vals)), len(dmag_vals)],
                     index=['max_err', 'min_err', 'count'])


def weighted_mean_mag(group, phot, mag_column):
    total_weight = group['weight'].sum()
    if total_weight <= 1e-10:
        return np.nan
    return np.average(group[mag_column], weights=group['weight'])


def weighted_mean_sigma(group, phot, sq_dmag_column):
    total_weight = group['weight'].sum()
    if total_weight <= 1e-10:
        return np.nan
    return np.average(group[sq_dmag_column], weights=group['weight'])

def check_memory_usage(threshold=0.9):
    """Check if memory usage is too high."""
    process = psutil.Process(os.getpid())
    memory_percent = process.memory_percent() / 100
    if memory_percent > threshold:
        print(f"Memory usage high: {memory_percent:.1%}, forcing cleanup...")
        gc.collect()
        return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# optSF — modular breakdown
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. bin setup ──────────────────────────────────────────────────────────────

def make_log_bins(logbin_min=np.log10(0.5), logbin_max=np.log10(2500)):
    """Return the log-spaced bin edges used throughout the SF computation."""
    num_bins = max(int(np.ceil((logbin_max - logbin_min) / 0.1)) + 1, 10)
    return np.logspace(logbin_min, logbin_max, num=num_bins)


# ── 2. calibration star selection ─────────────────────────────────────────────

# def apply_mag_mask(pool, ztf, mag_column, margin=0.0):
#     med  = pool.groupby('object_index')[mag_column].median()
#     mask = (
#         (med > ztf[mag_column].min() - margin) &
#         (med < ztf[mag_column].max() + margin)
#     )
#     return mask[mask].index

def apply_mag_mask(pool, ztf, mag_column, margin=0.0):
    p05 = np.percentile(ztf[mag_column].dropna(), 5)
    p95 = np.percentile(ztf[mag_column].dropna(), 95)
    med = pool.groupby('object_index')[mag_column].median()
    mask = (
        (med > p05 - margin) &
        (med < p95 + margin)
    )
    return mask[mask].index


def apply_coverage(pool, ztf,verbose=False):
    ztf_mjd_set        = set(ztf['OBSMJD'].values)
    covered            = pool.sort_values('OBSMJD').copy()
    covered['covered'] = covered['OBSMJD'].isin(ztf_mjd_set)
    cov                = covered.groupby('object_index')['covered'].agg(
        coverage_pct='mean', n_epochs='count')
    cov['coverage_pct'] *= 100
    if verbose:
        print(cov.describe())
    return cov[cov['coverage_pct'] >= 90].index


def apply_class_star(pool, ztf, class_win=0.3):
    """Keep object_index whose CLASS_STAR_OBJ is within class_win of the target."""
    target_class = ztf['CLASS_STAR_OBJ'].mode().iloc[0]
    dist = (
        target_class - pool.groupby('object_index')['CLASS_STAR_OBJ'].first()
    ).abs()
    return dist[dist <= class_win].index


def apply_mag_proximity(pool, ztf, mag_column, n_keep):
    """Return the n_keep object_index values closest to the ztf median magnitude."""
    target_med = ztf[mag_column].median()
    dist       = (
        pool.groupby('object_index')[mag_column].median() - target_med
    ).abs().sort_values()
    return dist.head(n_keep).index

def select_calstars(df, ztf, mag_column, x=10):
    """Filter calibration stars from the full parquet DataFrame.

    Steps applied in order:
      1. exclude the target's own rows by RA/Dec
      2. apply MAGLIM, SEEING, qid, ccdid quality cuts
      3. magnitude range match to the target
      4. co-observation coverage >= 90%
      5. keep the x stars closest in CLASS_STAR_OBJ to the target

    Returns
    -------
    cs1 : DataFrame — stars after quality cuts, before coverage/class filter
    cs  : DataFrame — final filtered stars
    """
    cs_all = df[
        (~df['ALPHAWIN_REF'].isin(ztf['ALPHAWIN_REF'])) &
        (~df['DELTAWIN_REF'].isin(ztf['DELTAWIN_REF']))
    ]
    print(f'{cs_all["object_index"].nunique()} cs in parquet')

    if cs_all.empty:
        print('empty cs')
        return pd.DataFrame(), pd.DataFrame()

    # quality cuts
    mask = (cs_all['MAGLIM'] > 20.5) & (cs_all['SEEING'] < 3)
    if all(col in df.columns for col in ['qid', 'ccdid']):
        qid, ccdid, probstar = ztf[['qid', 'ccdid', 'CLASS_STAR_OBJ']].mode().iloc[0]
        mask &= (cs_all['qid'] == qid) & (cs_all['ccdid'] == ccdid)

    cs1 = cs_all[mask]
    if cs1.empty:
        print('quality cuts left 0 cs')
        return pd.DataFrame(), pd.DataFrame()

    cs = cs1.copy()

    # magnitude range
    mag_mask = (
        (cs.groupby('object_index')[mag_column].median() > np.min(ztf[mag_column])) &
        (cs.groupby('object_index')[mag_column].median() < np.max(ztf[mag_column]))
    )
    cs = cs[cs['object_index'].isin(mag_mask[mag_mask].index)]

    # co-observation coverage
    ztf_mjd_set = set(ztf['OBSMJD'].values)
    cs_sorted   = cs.sort_values('OBSMJD')
    cs_sorted['covered'] = cs_sorted['OBSMJD'].isin(ztf_mjd_set)
    coverage    = cs_sorted.groupby('object_index')['covered'].agg(
        coverage_pct=lambda x: 100 * x.mean(),
        n_epochs='count'
    )
    good_stars = coverage[coverage['coverage_pct'] >= 90].index
    cs = cs[cs['object_index'].isin(good_stars)]

    # CLASS_STAR_OBJ proximity
    target_class    = ztf['CLASS_STAR_OBJ'].mode().iloc[0]
    star_class_dist = (
        target_class - cs.groupby('object_index')['CLASS_STAR_OBJ'].first()
    ).abs().sort_values()
    closest_stars = star_class_dist.head(x).index
    cs = cs[cs['object_index'].isin(closest_stars)]

    n = cs['object_index'].nunique()
    print(f'{n} cs after mag, coverage and CLASS_STAR_OBJ filters: {len(cs)} epochs')
    # if len(cs) > 0:
    # n = cs['object_index'].nunique()
    if n < x:
        # how many extra stars we need
        n_extra = x - n

        # stars not already selected
        already_selected = cs['object_index'].unique()
        cs_remaining = cs1[~cs1['object_index'].isin(already_selected)]

        if not cs_remaining.empty:
            target_med = ztf[mag_column].median()
            mag_dist = (
                cs_remaining.groupby('object_index')[mag_column].median() - target_med
            ).abs().sort_values()
            extra_stars = mag_dist.head(n_extra).index
            cs_extra = cs_all[cs_all['object_index'].isin(extra_stars)]
            cs = pd.concat([cs, cs_extra], ignore_index=True)
            print(f'added {cs_extra["object_index"].nunique()} extra stars by median mag proximity (n was {n})')

    return cs1, cs

def quality_flags(df, ztf):
    """Filter calibration stars from the full parquet DataFrame.

    Steps applied in order:
      1. exclude the target's own rows by RA/Dec
      2. apply MAGLIM, SEEING, qid, ccdid quality cuts         → cs1
      3. magnitude range match to the target
      4. co-observation coverage >= 90%
      5. keep the x stars closest in CLASS_STAR_OBJ to target  → cs
      6. if n < x, top up from cs1 by median mag proximity,
         re-applying coverage and CLASS_STAR_OBJ checks

    Returns
    -------
    cs1 : DataFrame — stars after quality cuts only
    cs  : DataFrame — final selected stars
    """

    # ── step 1: exclude target rows ───────────────────────────────────────────
    # cs_all = df[
    #     (~df['ALPHAWIN_REF'].isin(ztf['ALPHAWIN_REF'])) &
    #     (~df['DELTAWIN_REF'].isin(ztf['DELTAWIN_REF']))
    # ]
    cs_all = df
    print(f'{cs_all["object_index"].nunique()} object_index in parquet')
    if cs_all.empty:
        print('empty cs')
        return pd.DataFrame()

    # ── step 2: quality cuts ──────────────────────────────────────────────────
    mask = (cs_all['MAGLIM'] > 20.5) & (cs_all['SEEING'] < 3)
    if all(col in df.columns for col in ['qid', 'ccdid']):
        qid, ccdid, _ = ztf[['qid', 'ccdid', 'CLASS_STAR_OBJ']].mode().iloc[0]
        mask &= (cs_all['qid'] == qid) & (cs_all['ccdid'] == ccdid)

    cs1 = cs_all[mask]

    return cs1

def quality_cuts(cs1, ztf, mag_column, x=10,class_win=0.1, mag_margin=0.0):
    # ── step 1: exclude target rows ───────────────────────────────────────────
    cs_all = cs1[
        (~cs1['ALPHAWIN_REF'].isin(ztf['ALPHAWIN_REF'])) &
        (~cs1['DELTAWIN_REF'].isin(ztf['DELTAWIN_REF']))
    ]

    if cs_all.empty:
        print('quality cuts left 0 cs')
        return pd.DataFrame(), pd.DataFrame()

    # # ── steps 3–5: mag range → coverage → CLASS_STAR_OBJ ─────────────────────
    # cs = cs_all[cs_all['object_index'].isin(apply_mag_mask(cs_all, ztf, mag_column))]
    # print('mag filter', len(cs))
    # cs = cs[cs['object_index'].isin(apply_coverage(cs, ztf))]
    # print('coverage check', len(cs))
    # cs = cs[cs['object_index'].isin(apply_class_star(cs, ztf, x))]
    # print('CLASS_OBJ_ID selection',len(cs))

    # n = cs['object_index'].nunique()
    # print(f'{n} cs after mag, coverage and CLASS_STAR_OBJ filters: {len(cs)} epochs')

    # # ── step 6: top up if n < x ───────────────────────────────────────────────
    # if n < x:
    #     n_extra      = x - n
    #     cs_remaining = cs_all[~cs_all['object_index'].isin(cs['object_index'].unique())]

    #     if not cs_remaining.empty:
    #         cs_extra = cs_remaining[cs_remaining['object_index'].isin(
    #             apply_mag_proximity(cs_remaining, ztf, mag_column, n_extra))]
    #         cs_extra = cs_extra[cs_extra['object_index'].isin(
    #             apply_coverage(cs_extra, ztf))]
    #         cs_extra = cs_extra[cs_extra['object_index'].isin(
    #             apply_class_star(cs_extra, ztf, n_extra))]

    #         if not cs_extra.empty:
    #             cs = pd.concat([cs, cs_extra], ignore_index=True)
    #             print(f'added {cs_extra["object_index"].nunique()} extra stars '
    #                   f'by median mag proximity (n was {n})')
    #         else:
    #             print(f'no extra stars found after filters (n stays {n})')
    # coverage: hard cut, NEVER relaxed (biases the noise floor otherwise)

    cov_ok = cs_all[cs_all['object_index'].isin(apply_coverage(cs_all, ztf))]
    print('coverage check', cov_ok['object_index'].nunique())

    # relax class window first, then mag margin, until x stars survive
    cs, n, cw = pd.DataFrame(), 0, class_win
    for cw, mm in [(class_win,   mag_margin),
                   (2*class_win, mag_margin),
                   (2*class_win, mag_margin + 0.1),
                   (4*class_win, mag_margin + 0.1),
                   (6*class_win, mag_margin + 0.1),
                   (9*class_win, mag_margin + 0.1),
                   (np.inf,      mag_margin + 0.2)]:
                #    (np.inf,      mag_margin + 0.2)]:
        gated = cov_ok[cov_ok['object_index'].isin(
            apply_mag_mask(cov_ok, ztf, mag_column, mm))]
        gated = gated[gated['object_index'].isin(
            apply_class_star(gated, ztf, cw))]
        # rank survivors by |Δmag|, take the x nearest — magnitude is the ONLY ranker
        cs = gated[gated['object_index'].isin(
            apply_mag_proximity(gated, ztf, mag_column, x))]
        n = cs['object_index'].nunique()
        if n >= x:
            break

    print(f'{n} cs after coverage + class(±{cw}) + mag(±{mm}) gates: {len(cs)} epochs')

    return cs

# # ── 3. optional sigma filtering of calstars ───────────────────────────────────

# def filter_calstars_sigma(cs, mag_column,file=None,cs_list=None,ra=None,dec=None,
#                             plot=False, save_plt=False):
#     """Run sigma filtering on the calstar pool and return the cleaned DataFrame.

#     Calls run_sigma_filtering() from VarTools — must be available in scope.
#     """
#     # print(cs_list,ra,dec)
#     cs_clean = run_sigma_filtering(cs.rename(columns={mag_column: 'MAG_4_TOT_AB'}),
#                                     # save_file=True, 
#                                     plot=plot,
#                                     save_plt=save_plt,
#                                     filename=file,
#                                     cs_list=cs_list,
#                                     ra=ra,
#                                     dec=dec)
#     # rename back if needed
#     if mag_column != 'MAG_4_TOT_AB':
#         cs_clean = cs_clean.rename(columns={'MAG_4_TOT_AB': mag_column})
#     return cs_clean


# ── 4. optional sigma clipping by temporal block ──────────────────────────────

def clip_by_blocks(lc, mag_column, gap_threshold=90):
    """Apply clip_block() per temporal block. Returns reset-index DataFrame."""
    time_diffs     = lc['OBSMJD'].diff()
    lc['block_id'] = (time_diffs > gap_threshold).cumsum()
    cleaned        = lc.groupby('block_id').apply(lambda g: clip_block(g, mag_column),include_groups=False)
    cleaned = cleaned.reset_index(drop=True)
    # restore block_id since include_groups=False drops it from the output
    time_diffs_clean     = cleaned['OBSMJD'].diff()
    cleaned['block_id']  = (time_diffs_clean > gap_threshold).cumsum()
    
    return cleaned


# ── 5. SF pair computation ─────────────────────────────────────────────────────

def compute_sf_pairs(ztf, mag_column, mag_err, log_bins):
    """Compute SF pairs for the target light curve.

    Returns
    -------
    corr_sf     : DataFrame with columns [dt, dmag, err, log_bin]
    grouped_corr: groupby object on log_bin
    """
    sf_corrLC = SFarray(
        ztf.OBSMJD.values,
        ztf[mag_column].values,
        ztf[mag_err].values
    )
    corr_sf = pd.DataFrame(sf_corrLC).transpose()
    corr_sf.columns = ['dt', 'dmag', 'err']
    corr_sf['dmag']    = np.abs(corr_sf['dmag'])
    corr_sf['log_bin'] = pd.cut(corr_sf['dt'], bins=log_bins)
    grouped_corr       = corr_sf.groupby('log_bin', observed=True)
    return corr_sf, grouped_corr


def compute_target_stats(grouped_corr):
    """Vectorised error stats for the target (replaces group_stats apply).

    Returns
    -------
    binned_corr : Series — mean |dmag| per bin
    minerr_corr : Series — lower 1-sigma / sqrt(n)
    maxerr_corr : Series — upper 1-sigma / sqrt(n)
    ndmag       : Series — count per bin
    """
    g_corr      = grouped_corr['dmag']
    n_corr      = g_corr.count()
    sqrt_n      = np.sqrt(n_corr.clip(lower=1))
    cp16        = g_corr.quantile(0.16)
    cp50        = g_corr.quantile(0.50)
    cp84        = g_corr.quantile(0.84)
    binned_corr = g_corr.mean()
    minerr_corr = ((cp50 - cp16) / sqrt_n).dropna()
    maxerr_corr = ((cp84 - cp50) / sqrt_n).dropna()
    ndmag       = n_corr.dropna()
    return binned_corr, minerr_corr, maxerr_corr, ndmag


def compute_calstar_sf(cs, mag_column, mag_err, log_bins):
    """Compute SF pairs for all calstars and return vectorised stats.

    Returns
    -------
    df_cs          : DataFrame of all calstar SF pairs
    grouped_cs     : groupby on [object_index, log_bin]
    avg_sq_dmag_cs : Series — median of mean sq_dmag across stars per bin
    binminerr_cs   : Series — lower 1-sigma of per-star RMS per bin
    binmaxerr_cs   : Series — upper 1-sigma of per-star RMS per bin
    minerr_cs      : Series — lower 1-sigma of |dmag| per bin (norm=False)
    maxerr_cs      : Series — upper 1-sigma of |dmag| per bin (norm=False)
    """
    sf_csLC = cs.groupby('object_index').apply(
        lambda group: SFarray(
            group.OBSMJD.values,
            group[mag_column].values,
            group[mag_err].values
        ),
        include_groups=False
    )

    tau_list, sf_list, err_list = zip(*sf_csLC.values)
    df_cs = pd.DataFrame({
        'dt':           np.concatenate(tau_list),
        'dmag':         np.concatenate(sf_list),
        'err':          np.concatenate(err_list),
        'object_index': np.repeat(sf_csLC.index, [len(t) for t in tau_list])
    })

    df_cs               = df_cs.loc[df_cs['dt'] != 0]
    df_cs['log_bin']    = pd.cut(df_cs['dt'], bins=log_bins)
    df_cs['dmag']       = df_cs['dmag'].astype('float')
    df_cs['sq_dmag_cs'] = df_cs['dmag'] ** 2

    grouped_cs = df_cs.groupby(['object_index', 'log_bin'], observed=True)

    # pass 1: mean sq_dmag → median across stars
    mean_sq        = grouped_cs['sq_dmag_cs'].mean()
    avg_sq_dmag_cs = mean_sq.unstack(level='object_index').median(axis=1)

    # pass 1 continued: per-star RMS → err_prop across stars
    rms_per_star_bin = np.sqrt(mean_sq)
    g_rms_bin        = rms_per_star_bin.groupby(level='log_bin', observed=True)
    rp16 = g_rms_bin.quantile(0.16)
    rp50 = g_rms_bin.quantile(0.50)
    rp84 = g_rms_bin.quantile(0.84)
    binminerr_cs = (rp50 - rp16).dropna()
    binmaxerr_cs = (rp84 - rp50).dropna()

    # pass 2: err_prop on raw |dmag| per bin
    g_dmag_bin = df_cs.groupby('log_bin', observed=True)['dmag']
    dp16 = g_dmag_bin.quantile(0.16)
    dp50 = g_dmag_bin.quantile(0.50)
    dp84 = g_dmag_bin.quantile(0.84)
    minerr_cs = (dp50 - dp16).dropna()
    maxerr_cs = (dp84 - dp50).dropna()

    return df_cs, grouped_cs, mean_sq, avg_sq_dmag_cs, binminerr_cs, binmaxerr_cs, minerr_cs, maxerr_cs


# ── 6. SF combination (target − calstars) ────────────────────────────────────

def combine_sf(binned_corr, avg_sq_dmag_cs,
               minerr_corr, maxerr_corr,
               minerr_cs, maxerr_cs):
    """Subtract calstar noise from target SF.

    Returns
    -------
    SF        : Series — corrected structure function
    sf        : Series — uncorrected sqrt SF (for plotting)
    sf_cs     : Series — calstar SF (for plotting)
    dif       : Series — signed difference (pi/2 * <dmag²> - <dmag²_cs>)
    maxerr_sf : Series
    minerr_sf : Series
    dt_midbin_cs, dt_lenbin_cs : arrays for plotting calstar error bars
    """
    sq_avg_dmag_corr = binned_corr ** 2
    dt               = sq_avg_dmag_corr.dropna()

    dif = (np.pi / 2) * sq_avg_dmag_corr - avg_sq_dmag_cs
    SF  = np.sqrt(dif)

    maxerr_sf = np.sqrt(maxerr_corr ** 2 + maxerr_cs ** 2)
    minerr_sf = np.sqrt(minerr_corr ** 2 + minerr_cs ** 2)

    sf_cs        = np.sqrt(avg_sq_dmag_cs.dropna())
    dt_midbin_cs = sf_cs.index.categories[sf_cs.index.codes].mid
    dt_lenbin_cs = sf_cs.index.categories[sf_cs.index.codes].length / 2

    sf = np.sqrt((np.pi / 2) * dt)

    # zero out bins where calstar noise exceeds target signal
    neg_idx = dif[dif < 0].index
    SF[SF.index.isin(neg_idx)]               = 0
    maxerr_sf[maxerr_sf.index.isin(neg_idx)] = 0
    sf_dif = sf - sf_cs
    minerr_sf[maxerr_sf.index.isin(neg_idx)] = np.abs(sf_dif[sf_dif < 0])

    return SF, sf, sf_cs, dif, maxerr_sf, minerr_sf, dt_midbin_cs, dt_lenbin_cs


# ── 7. plotting ───────────────────────────────────────────────────────────────

def plot_sf(ztf, cs, cs1, corr_sf, sf, sf_cs, SF, dif,
            dt_midbin, dt_lenbin,
            dt_midbin_cs, dt_lenbin_cs,
            minerr_corr, maxerr_corr,
            binminerr_cs, binmaxerr_cs,
            minerr_sf2, maxerr_sf2,
            final_dt, final_dt_midbin, final_dt_lenbin,
            df_cs, grouped_cs, mean_sq,
            mag_column, mag_err, color, color2, band,
            ra, dec, clip, calstars, n,
            showallcs=False, save_plt=False, plot=False,file=None,x=None):
    """Build and optionally show/save the two-panel SF plot."""

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    # left panel: light curves
    if calstars and (not cs1.empty) and n > 1:
        try:
            ax[0].errorbar(cs['OBSMJD'], cs[mag_column], yerr=cs[mag_err],
                           fmt='o', alpha=0.1, c='grey', label='cs')
        except Exception:
            pass

    ax[0].errorbar(ztf['OBSMJD'], ztf[mag_column], yerr=ztf[mag_err],
                   fmt='o', alpha=0.7, c=color, label='source')

    if clip:
        block_boundaries = ztf.groupby('block_id')['OBSMJD'].first()[1:]
        for i, boundary in enumerate(block_boundaries):
            label = 'epochs window' if i == 0 else None
            ax[0].axvline(boundary, color='xkcd:grey green',
                          linestyle='--', linewidth=1, label=label)

    ax[0].invert_yaxis()
    ax[0].set_xlabel('MJD [days]', size=25)
    ax[0].set_ylabel('mag', size=25)
    ax[0].tick_params(labelsize=20)
    ax[0].legend()

    # right panel: structure function
    ax[1].errorbar(dt_midbin, sf, xerr=dt_lenbin,
                   yerr=(minerr_corr, maxerr_corr),
                   label='source', fmt='o', c=color)

    if calstars and (not cs1.empty) and n > 1:
        target_intervals = final_dt.index

        ax[1].errorbar(dt_midbin_cs, sf_cs, xerr=dt_lenbin_cs,
                       yerr=(binminerr_cs, binmaxerr_cs),
                       label='calib stars', fmt='+', c='lightgrey')

        pos_mask = dif.loc[target_intervals] > 0
        ax[1].errorbar(final_dt_midbin[pos_mask], final_dt[pos_mask],
                       xerr=final_dt_lenbin[pos_mask],
                       yerr=(minerr_sf2[pos_mask], maxerr_sf2[pos_mask]),
                       fmt='o', c=color2, label='source-stars', capsize=3)

        neg_mask = dif.loc[target_intervals] < 0
        ax[1].errorbar(final_dt_midbin[neg_mask], final_dt[neg_mask],
                       xerr=final_dt_lenbin[neg_mask],
                       yerr=(minerr_sf2[neg_mask], minerr_sf2[neg_mask]),
                       fmt='o', linestyle='none', c=color2, capsize=3)

        if showallcs and (not df_cs.empty):
            binned_df_cs = mean_sq.unstack(level='object_index')
            dt_cs        = binned_df_cs.index.categories[binned_df_cs.index.codes].mid
            err_cs       = grouped_cs.apply(
                lambda g: group_stats(g, 'dmag'),
                include_groups=False).unstack(level='object_index')
            for obj_col in binned_df_cs.columns:
                ax[1].errorbar(dt_cs, np.sqrt(binned_df_cs[obj_col]),
                               yerr=(err_cs['min_err'][obj_col],
                                     err_cs['max_err'][obj_col]),
                               fmt='o', alpha=0.2)

    ax[1].axhline(y=0, color='xkcd:mocha', linestyle='-.')
    ax[1].set_xlabel('time lag [days]', size=25)
    ax[1].set_ylabel('SF [mag]', size=25)
    ax[1].tick_params(labelsize=20)
    ax[1].set_xscale('log')
    ax[1].set_xlim(4e-1, 4e3)
    ax[1].legend()
    plt.tight_layout()

    if save_plt:
        # prefix    = 'sci' if 'sci' in file else 'ref'
        # save_name = f'{prefix}{mag_column}_SF_clip{clip}_{ra}_{dec}_{band}_{x}cs.png'
        prefix    = os.path.basename(file)#.split("_",2)[2]
        save_name = f'{prefix}_{x}cs.png'
        # plt.savefig(os.environ['HOME'] + '/BAT_SF/' + save_name)
        subpath = os.environ['HOME'] + f'/results/partials/{ra}_{dec}/'
        os.makedirs(subpath, exist_ok=True)
        plt.savefig(subpath + save_name)
        print(f"Plot saved to /results/partials/{ra}_{dec}/ as {save_name}")

    if plot:
        plt.show()
    else:
        plt.close()


# ── 8. main orchestrator ──────────────────────────────────────────────────────

def optSF(file, calstars=True, weight=False,
          clip=False, showallcs=False, plot=False, save_plt=False, save=False,
          x=10, gap_threshold=90, mag_column='MAG_4_TOT_AB', mag_err='MERR_4_TOT_AB',
          logbin_min=np.log10(0.5), logbin_max=np.log10(2500),
          sigma_filter=False):
    """
    Structure function pipeline.

    New parameter
    -------------
    sigma_filter : bool
        If True, run run_sigma_filtering() on the calstar pool before SF
        computation to remove photometrically variable calibrators.
    """

    tic      = time.perf_counter()
    log_bins = make_log_bins(logbin_min, logbin_max)
    SF_dict  = []

    #for file in filenames:
    match = re.search(r'_z(\w)_', file)
    # print(file)
    band  = match.group(1)

    color,  color2 = {
        'g': ('mediumseagreen', 'seagreen'),
        'r': ('firebrick',      'maroon'),
        'i': ('gold',           'goldenrod'),
    }[band]

    print('target', file)
    print(f'band {band}')

    df   = pd.read_parquet(file)
    ra, dec = radec_filename(file)
    # if sigma_filter and not df.empty:
    #     df_clean = filter_calstars_sigma(df, mag_column,file=file)
    coord = SkyCoord(ra, dec, unit='deg')
    tgt_obj_idx = _find_target_obj(Path(file), coord)
    # tgt_obj_idx = _find_target_obj(ra, dec, df)

    if tgt_obj_idx is None:
        print(f'no target found in {os.path.basename(file)}, skipping')
        return None

    # obj_id = df.loc[tgt_obj_idx, 'object_index']
    # ztf1 = df[df['object_index'] == obj_id]
    ztf1 = df[df['object_index'] == tgt_obj_idx]

    mask = ztf1[mag_err] < 0.5

    if 'qid' in ztf1.columns:
        qid    = ztf1['qid'].mode().iloc[0]
        ccdid  = ztf1['ccdid'].mode().iloc[0]
        field  = ztf1['field'].mode().iloc[0]
        mask &= (
        (ztf1['qid']   == qid)  &
        (ztf1['ccdid'] == ccdid) &
        (ztf1['field'] == field)
    )

    ztf1 = ztf1[mask]

    if len(ztf1) == 0:
        print('empty target')
        return None
        # continue

    path_file = os.environ['HOME'] + f'/results/partials/{ra}_{dec}/'
    # name_file = path_file+f'SF_clip{clip}_{band}band.pkl'
    # prefix    = 'sci' if 'sci' in file else 'ref'
    prefix    = os.path.basename(file)#.split("_",2)[2]
    name_file = path_file+f'{prefix}_{x}cs.pkl'
    # name_file = path_file+f'SFdict_{prefix}_clip{clip}_{ra}_{dec}_{band}_{x}cs.pkl'
    
    # if append and os.path.exists(name_file):
    #     with open(name_file, 'rb') as f:
    #         SF_dict = pickle.load(f)

    if mag_column is None or mag_column not in ztf1.columns:
        print('no mag column, skipping file')
        return None
        # continue

    mztf   = ztf1[mag_column].median()
    minztf = np.min(ztf1[mag_column])
    maxztf = np.max(ztf1[mag_column])
    print(f'median mag = {mztf:.2f}, max = {maxztf:.2f} and min = {minztf:.2f}')

    # ── clip target ───────────────────────────────────────────────────────
    ztf = clip_by_blocks(ztf1, mag_column, gap_threshold) if clip else ztf1

    # ── select calstars ───────────────────────────────────────────────────
    if calstars:
        # cs1, cs = select_calstars(df_clean, ztf, mag_column, x=x)
        # df   = pd.read_parquet(file)
        cs1 = quality_flags(df, ztf)
        if cs1.empty:# or cs.empty:
            print('empty cs')
            return None

        # optional sigma filtering of calstar pool
        if sigma_filter and not cs1.empty:
            # cs = filter_calstars_sigma(cs1, mag_column, file=file)
            cs = run_sigma_filtering(cs1.rename(columns={mag_column: 'MAG_4_TOT_AB'}),
                                    agn_indices = tgt_obj_idx,
                                    save_file=True, 
                                    plot=False,
                                    save_plt=True,
                                    filename=file,
                                    # cs_list=None,
                                    ra=ra,
                                    dec=dec)

        cs = quality_cuts(cs, ztf,mag_column,x=x)
        # print(f"selected calib stars: {cs['object_index'].unique()}")
        # optional clipping of calstars by block
        if clip and not cs.empty:
            # print(f"Objects before clipping: {cs['object_index'].unique()}")
            cs = clip_by_blocks(cs, mag_column, gap_threshold)
            # print(f"calib stars after clipping: {cs['object_index'].unique()}")

    else:
        cs1 = pd.DataFrame()
        cs  = pd.DataFrame()

    n = cs['object_index'].nunique() if not cs.empty else 0
    print(f'source: {len(ztf)} epochs')
    print(f'{n} calib stars: {len(cs)} epochs')

    # ── target SF pairs and stats ─────────────────────────────────────────
    corr_sf, grouped_corr = compute_sf_pairs(ztf, mag_column, mag_err, log_bins)
    binned_corr, minerr_corr, maxerr_corr, ndmag = compute_target_stats(grouped_corr)
    print(f'    +{np.round(time.perf_counter() - tic, 2)} s: source errors estimate')

    # ── calstar SF pairs and stats ────────────────────────────────────────
    df_cs = pd.DataFrame()
    grouped_cs = mean_sq = None
    avg_sq_dmag_cs = binmaxerr_cs = binminerr_cs = pd.Series(dtype=float)
    minerr_cs = maxerr_cs = pd.Series(dtype=float)

    if n > 1:
        (df_cs, grouped_cs, mean_sq,
            avg_sq_dmag_cs,
            binminerr_cs, binmaxerr_cs,
            minerr_cs, maxerr_cs) = compute_calstar_sf(cs, mag_column, mag_err, log_bins)
    else:
        print('not enough calib stars')
        return None
        # continue

    # ── weighted mean override ────────────────────────────────────────────
    if weight:
        corr_sf['weight'] = 1 / (corr_sf['err'] ** 2)
        binned_corr = grouped_corr.apply(
            lambda g: weighted_mean_mag(g, '', 'dmag'))
        if n > 1:
            df_cs['weight']  = 1 / (df_cs['err'] ** 2)
            avg_sq_dmag_cs   = grouped_cs.apply(
                lambda g: weighted_mean_sigma(g, '', 'sq_dmag_cs'))

    # ── combine into SF ───────────────────────────────────────────────────
    sq_avg_dmag_corr = binned_corr ** 2
    dt               = sq_avg_dmag_corr.dropna()
    dt_midbin        = dt.index.categories[dt.index.codes].mid
    dt_lenbin        = dt.index.categories[dt.index.codes].length / 2

    if calstars and (not cs1.empty) and n > 1:
        (SF, sf, sf_cs, dif,
            maxerr_sf, minerr_sf,
            dt_midbin_cs, dt_lenbin_cs) = combine_sf(
            binned_corr, avg_sq_dmag_cs,
            minerr_corr, maxerr_corr,
            minerr_cs, maxerr_cs)
        print(f'    +{np.round(time.perf_counter() - tic, 2)} s: stars errors estimate')
    else:
        SF        = np.sqrt(sq_avg_dmag_corr)
        sf        = SF.dropna()
        maxerr_sf = maxerr_corr
        minerr_sf = minerr_corr
        dif = sf_cs = dt_midbin_cs = dt_lenbin_cs = None

    final_dt         = SF.dropna()
    final_dt_midbin  = final_dt.index.categories[final_dt.index.codes].mid
    final_dt_lenbin  = final_dt.index.categories[final_dt.index.codes].length / 2
    target_intervals = final_dt.index

    maxerr_sf2 = maxerr_sf[maxerr_sf.index.isin(target_intervals)]
    minerr_sf2 = minerr_sf[maxerr_sf.index.isin(target_intervals)]

    # ── plot ──────────────────────────────────────────────────────────────
    if plot or save_plt:
        plot_sf(ztf, cs, cs1, corr_sf, sf, sf_cs, SF, dif,
                dt_midbin, dt_lenbin,
                dt_midbin_cs, dt_lenbin_cs,
                minerr_corr, maxerr_corr,
                binminerr_cs, binmaxerr_cs,
                minerr_sf2, maxerr_sf2,
                final_dt, final_dt_midbin, final_dt_lenbin,
                df_cs, grouped_cs, mean_sq,
                mag_column, mag_err, color, color2, band,
                ra, dec, clip, calstars, n,
                showallcs=showallcs, save_plt=save_plt, plot=plot,file=file,x=x)

    # ── store ─────────────────────────────────────────────────────────────
    # SF_dict.append({
    #     'SF':        SF,
    #     'SFmaxerr':  maxerr_sf,
    #     'SFminerr':  minerr_sf,
    #     '#elements': ndmag,
    #     'RA':        ra,
    #     'mag':       mztf,
    #     'band':      band
    # })
    SF_dict = {
        'SF':        SF,
        'SFmaxerr':  maxerr_sf,
        'SFminerr':  minerr_sf,
        '#elements': ndmag,
        'RA':        ra,
        'mag':       mztf,
        'band':      band
    }

    if save:
        os.makedirs(path_file, exist_ok=True)
        with open(name_file, 'wb') as f:
            pickle.dump(SF_dict, f)
        print(f'pkl file successfully saved in {name_file}\n')

    print('\n')
    return SF_dict

def SF_wnoise(mag_col, time_col, mag_err, cs_all=None, clip=False, weight=False,
              color='red', showallcs=False, plot=False,path=None, save_plt=False, save=False,
              logbin_min=np.log10(0.5), logbin_max=np.log10(2500),verbose=False):
    """Structure function for simulated / array-input light curves.

    Accepts raw arrays instead of parquet files. Shares all computation
    modules with optSF (compute_sf_pairs, compute_target_stats,
    compute_calstar_sf, combine_sf, make_log_bins).

    Parameters
    ----------
    mag_col   : array-like  — target magnitudes
    time_col  : array-like  — target MJDs
    mag_err   : array-like  — target magnitude errors
    cs_all    : DataFrame or None — calibration star pool; if None, no
                noise subtraction is performed
    """

    tic      = time.perf_counter()
    log_bins = make_log_bins(logbin_min, logbin_max)

    gap_threshold = 90
    # mag_column    = 'mag'
    mag_column    = 'MAG_4_TOT_AB'
    err_column    = 'MERR_4_TOT_AB'

    # ── build target DataFrame ────────────────────────────────────────────────
    ztf1 = pd.DataFrame({
        mag_column:  mag_col,
        err_column:   mag_err,
        # mag_column:  mag_col,
        # 'mag_err':   mag_err,
        'OBSMJD':    time_col       # rename to OBSMJD so shared modules work
    })

    mztf   = ztf1[mag_column].median()
    minztf = ztf1[mag_column].min()
    maxztf = ztf1[mag_column].max()
    if verbose:
        print(f'median mag = {mztf:.2f}, max = {maxztf:.2f} and min = {minztf:.2f}')

    # ── optional sigma clipping ───────────────────────────────────────────────
    ztf = clip_by_blocks(ztf1, mag_column, gap_threshold) if clip else ztf1

    # ── calibration stars ─────────────────────────────────────────────────────
    cs1 = pd.DataFrame()
    cs  = pd.DataFrame()
    n   = 0

    if cs_all is not None and not cs_all.empty:

        # filter by band and CCD
        cs1 = cs_all[
            cs_all['CCDquadID'] == ztf['CCDquadID'].mode()[0]
        ].reset_index(drop=True)

        # flux → mag conversion if needed
        if not cs1.empty and mag_column not in cs1.columns:
            flux_col = 'flux'
            fxunc    = 'flux_err'
            cs1[mag_column] = -2.5 * np.log10(cs1[flux_col].values * 1e-6) + 8.90
            cs1[err_column]  = 2.5 / np.log(10) * (cs1[fxunc].values / cs1[flux_col].values)
            cs1.dropna(subset=[mag_column], inplace=True)

        if cs1.empty:
            if verbose:
                print('band/CCD filter left 0 cs')
        else:
            # rename mjd → OBSMJD so shared modules work
            cs1 = cs1.rename(columns={'mjd': 'OBSMJD'})

            # optional clipping — groupby on 'ra' instead of 'object_index'
            if clip:
                cs1 = cs1.sort_values('OBSMJD')
                cstime_diffs    = cs1['OBSMJD'].diff()
                cs1['block_id'] = (cstime_diffs > gap_threshold).cumsum()
                cleaned         = cs1.groupby(['object_index', 'block_id']).apply(
                    lambda g: clip_block(g, mag_column), include_groups=False)
                cs = cleaned.reset_index(drop=True)
            else:
                cs = cs1

            # magnitude range filter  (grouped by 'ra' — no object_index here)
            cs_med   = cs.groupby('object_index')[mag_column].median()
            mag_mask = (
                (cs_med > ztf[mag_column].min()) &
                (cs_med < ztf[mag_column].max())
            )
            cs = cs[cs['object_index'].isin(mag_mask[mag_mask].index)]

        n = cs['object_index'].nunique() if not cs.empty else 0
        if verbose:
            print(f'{n} calib stars: {len(cs)} epochs')

    else:
        if verbose:
            print('no calibration stars used')
    
    if verbose:
        print(f'source: {len(ztf)} epochs')

    # ── target SF pairs and stats ─────────────────────────────────────────────
    # compute_sf_pairs / compute_target_stats expect 'OBSMJD' and mag_err col
    # named as passed — wrap to match the shared module signatures
    corr_sf, grouped_corr = compute_sf_pairs(ztf, mag_column, err_column, log_bins)
    binned_corr, minerr_corr, maxerr_corr, ndmag = compute_target_stats(grouped_corr)
    if verbose:
        print(f'    +{np.round(time.perf_counter() - tic, 2)} s: source errors estimate')

    # ── calstar SF pairs and stats ────────────────────────────────────────────
    df_cs          = pd.DataFrame()
    grouped_cs     = mean_sq = None
    avg_sq_dmag_cs = binmaxerr_cs = binminerr_cs = pd.Series(dtype=float)
    minerr_cs      = maxerr_cs = pd.Series(dtype=float)

    if n > 1:
        # compute_calstar_sf groups by 'object_index' — remap 'ra' temporarily
        # cs_mapped = cs.rename(columns={'ra': 'object_index'})
        (df_cs, grouped_cs, mean_sq,
         avg_sq_dmag_cs,
         binminerr_cs, binmaxerr_cs,
         minerr_cs, maxerr_cs) = compute_calstar_sf(
            cs, mag_column, err_column, log_bins)

    # ── weighted mean override ────────────────────────────────────────────────
    if weight:
        corr_sf['weight'] = 1 / (corr_sf['err'] ** 2)
        binned_corr = grouped_corr.apply(
            lambda g: weighted_mean_mag(g, '', mag_column))
        if n > 1:
            df_cs['weight']  = 1 / (df_cs['err'] ** 2)
            avg_sq_dmag_cs   = grouped_cs.apply(
                lambda g: weighted_mean_sigma(g, '', 'sq_dmag_cs'))

    # ── combine into SF ───────────────────────────────────────────────────────
    sq_avg_dmag_corr = binned_corr ** 2
    dt               = sq_avg_dmag_corr.dropna()
    dt_midbin        = dt.index.categories[dt.index.codes].mid
    dt_lenbin        = dt.index.categories[dt.index.codes].length / 2

    if cs_all is not None and (not cs1.empty) and n > 1:
        (SF, sf, sf_cs, dif,
         maxerr_sf, minerr_sf,
         dt_midbin_cs, dt_lenbin_cs) = combine_sf(
            binned_corr, avg_sq_dmag_cs,
            minerr_corr, maxerr_corr,
            minerr_cs, maxerr_cs)
        if verbose:
            print(f'    +{np.round(time.perf_counter() - tic, 2)} s: stars errors estimate')
    else:
        SF        = np.sqrt(sq_avg_dmag_corr)
        sf        = SF.dropna()
        maxerr_sf = maxerr_corr
        minerr_sf = minerr_corr
        dif = sf_cs = dt_midbin_cs = dt_lenbin_cs = None

    final_dt         = SF.dropna()
    final_dt_midbin  = final_dt.index.categories[final_dt.index.codes].mid
    final_dt_lenbin  = final_dt.index.categories[final_dt.index.codes].length / 2
    target_intervals = final_dt.index

    maxerr_sf2 = maxerr_sf[maxerr_sf.index.isin(target_intervals)]
    minerr_sf2 = minerr_sf[maxerr_sf.index.isin(target_intervals)]

    # ── plotting ──────────────────────────────────────────────────────────────
    if plot or save_plt:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

        if cs_all is not None and (not cs1.empty) and n > 1:
            try:
                ax[0].errorbar(cs['OBSMJD'], cs[mag_column], yerr=cs[err_column],
                               fmt='o', alpha=0.1, c='grey', label='cs')
            except Exception:
                pass

        ax[0].errorbar(ztf['OBSMJD'], ztf[mag_column], yerr=ztf[err_column],
                       fmt='o', alpha=0.7, c=color, label='source')

        if clip and 'block_id' in ztf.columns:
            block_boundaries = ztf.groupby('block_id')['OBSMJD'].first()[1:]
            for i, boundary in enumerate(block_boundaries):
                label = 'epochs window' if i == 0 else None
                ax[0].axvline(boundary, color='xkcd:grey green',
                              linestyle='--', linewidth=1, label=label)

        ax[0].invert_yaxis()
        ax[0].set_xlabel('MJD [days]', size=25)
        ax[0].set_ylabel('mag', size=25)
        ax[0].tick_params(labelsize=20)
        ax[0].legend()

        ax[1].errorbar(dt_midbin, sf, xerr=dt_lenbin,
                       yerr=(minerr_corr, maxerr_corr),
                       label='source', fmt='o', c=color)

        if cs_all is not None and (not cs1.empty) and n > 1:
            ax[1].errorbar(dt_midbin_cs, sf_cs, xerr=dt_lenbin_cs,
                           yerr=(binminerr_cs, binmaxerr_cs),
                           label='calib stars', fmt='+', c='lightgrey')

            pos_mask = dif.loc[target_intervals] > 0
            ax[1].errorbar(final_dt_midbin[pos_mask], final_dt[pos_mask],
                           xerr=final_dt_lenbin[pos_mask],
                           yerr=(minerr_sf2[pos_mask], maxerr_sf2[pos_mask]),
                           fmt='o', c=color, label='source-stars', capsize=3)

            neg_mask = dif.loc[target_intervals] < 0
            ax[1].errorbar(final_dt_midbin[neg_mask], final_dt[neg_mask],
                           xerr=final_dt_lenbin[neg_mask],
                           yerr=(minerr_sf2[neg_mask], minerr_sf2[neg_mask]),
                           fmt='o', linestyle='none', c=color, capsize=3)

            if showallcs and (not df_cs.empty):
                binned_df_cs = mean_sq.unstack(level='object_index')
                dt_cs        = binned_df_cs.index.categories[binned_df_cs.index.codes].mid
                err_cs       = grouped_cs.apply(
                    lambda g: group_stats(g, mag_column),
                    include_groups=False).unstack(level='object_index')
                for col in binned_df_cs.columns:
                    ax[1].errorbar(dt_cs, np.sqrt(binned_df_cs[col]),
                                   yerr=(err_cs['min_err'][col],
                                         err_cs['max_err'][col]),
                                   fmt='o', alpha=0.2)

        ax[1].axhline(y=0, color='xkcd:mocha', linestyle='-.')
        ax[1].set_xlabel('time difference [days]', size=25)
        ax[1].set_ylabel('SF [mag]', size=25)
        ax[1].tick_params(labelsize=20)
        ax[1].set_xscale('log')
        ax[1].set_xlim(4e-1, 4e3)
        ax[1].legend()
        plt.tight_layout()

        if save_plt:
            plt.savefig(path + 'simulated_SF.png')
            print(f"Plot saved to {path} as simulated_SF.png")

        if plot:
            plt.show()
        else:
            plt.close()

    # ── store and return ──────────────────────────────────────────────────────
    SF_dict = [{'SF': SF, 'SFmaxerr': maxerr_sf, 'SFminerr': minerr_sf}]

    if save:
        name_file = path + 'simulated_SF.pkl'
        with open(name_file, 'wb') as f:
            pickle.dump(SF_dict, f)
        print(f'pkl file successfully saved in {name_file}\n')

    return SF_dict


# ── 1. scaling helpers ────────────────────────────────────────────────────────
 
def scale_log_data(log_x, log_y, xerr, yerr):
    """Standardise log-space data and propagated errors.
 
    Returns
    -------
    log_x_s, log_y_s, xerr_s, yerr_s : scaled arrays
    params : tuple (x_mean, x_std, y_mean, y_std) needed for unscaling
    """
    x_mean, x_std = np.mean(log_x), np.std(log_x)
    y_mean, y_std = np.mean(log_y), np.std(log_y)
    return (
        (log_x - x_mean) / x_std,
        (log_y - y_mean) / y_std,
        xerr / x_std,
        yerr / y_std,
        (x_mean, x_std, y_mean, y_std)
    )
 
 
def unscale_chains(chain_alpha_s, chain_beta_s, x_mean, x_std, y_mean, y_std):
    """Transform LinMix chains from scaled back to original log space.
 
    Returns
    -------
    chain_alpha, chain_beta : arrays in original log₁₀ space
    """
    chain_beta  = chain_beta_s * y_std / x_std
    chain_alpha = (chain_alpha_s * y_std + y_mean
                   - chain_beta_s * y_std * x_mean / x_std)
    return chain_alpha, chain_beta
 
 
# ── 2. posterior band ─────────────────────────────────────────────────────────
 
def posterior_band(chain_alpha, chain_beta, dt_data, n_samples=500):
    """Vectorised posterior predictive band.
 
    Returns
    -------
    dt_fit_range          : array of 500 dt values
    sf_fit_p16/med/p84    : lower, median, upper 1-sigma envelope
    """
    dt_fit_range = np.logspace(
        np.log10(dt_data.min()), np.log10(dt_data.max()), 500)
    log_dt_range = np.log10(dt_fit_range)
 
    idx        = np.random.randint(len(chain_alpha), size=n_samples)
    sf_samples = 10 ** (
        chain_alpha[idx, None] + chain_beta[idx, None] * log_dt_range[None, :]
    )
    p16, p50, p84 = np.percentile(sf_samples, [16, 50, 84], axis=0)
    return dt_fit_range, p16, p50, p84
 
 
# ── 3. plotting ───────────────────────────────────────────────────────────────
 
def plot_linmix(dt_data, sf_mag, sf_err, dt_lenbin,
                log_dt, log_sf, xerr, yerr,
                chain_alpha, chain_beta,
                alpha_med, beta_med,
                dt_fit_range, sf_fit_p16, sf_fit_med, sf_fit_p84,
                ra, save_plot=False,path=None):
    """Six-panel LinMix diagnostic plot."""
 
    log_sf_fit = alpha_med + beta_med * np.log10(dt_fit_range)
 
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat
    ax1, ax2, ax3, ax4 = axes.flat
 
    # posterior: alpha
    ax1.hist(chain_alpha, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(alpha_med, color='r', linestyle='--', linewidth=2,
                label=f'Median: {alpha_med:.4f}')
    ax1.set_xlabel('α (intercept)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Posterior: Intercept', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)
 
    # posterior: beta
    ax2.hist(chain_beta, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(beta_med, color='r', linestyle='--', linewidth=2,
                label=f'Median: {beta_med:.4f}')
    ax2.set_xlabel('β (slope)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Posterior: Slope', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)
 
    # parameter correlation
    ax3.scatter(chain_alpha, chain_beta, alpha=0.3, s=1, c='black')
    ax3.axvline(alpha_med, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(beta_med,  color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('α (intercept)', fontsize=12)
    ax3.set_ylabel('β (slope)', fontsize=12)
    ax3.set_title('Parameter Correlation', fontweight='bold')
    ax3.grid(True, alpha=0.3)
 
    # linear scale
    # ax4.plot(dt_fit_range, sf_fit_med, 'r-', linewidth=2, label='Median fit')
    # ax4.fill_between(dt_fit_range, sf_fit_p16, sf_fit_p84,
    #                  alpha=0.3, color='red', label='68% credible interval')
    # ax4.errorbar(dt_data, sf_mag, xerr=dt_lenbin, yerr=sf_err,
    #              fmt='o', markersize=8, color='blue', capsize=4, label='Data', zorder=5)
    # ax4.set_xlabel('dt', fontsize=12); ax4.set_ylabel('SF', fontsize=12)
    # ax4.set_title('Linear Scale', fontweight='bold')
    # ax4.legend(); ax4.grid(True, alpha=0.3)
 
    # # log-log scale
    # ax5.loglog(dt_fit_range, sf_fit_med, 'r-', linewidth=2, label='Median fit')
    # ax5.fill_between(dt_fit_range, sf_fit_p16, sf_fit_p84,
    #                  alpha=0.3, color='red', label='68% credible interval')
    # ax5.errorbar(dt_data, sf_mag, xerr=dt_lenbin, yerr=sf_err,
    #              fmt='o', markersize=8, color='blue', capsize=4, label='Data', zorder=5)
    # ax5.set_xlabel('dt (log)', fontsize=12); ax5.set_ylabel('SF (log)', fontsize=12)
    # ax5.set_title(f'Log-Log (β = {beta_med:.3f})', fontweight='bold')
    # ax5.legend(); ax5.grid(True, alpha=0.3, which='both')
 
    # direct log space
    ax4.plot(np.log10(dt_fit_range), log_sf_fit, 'r-', linewidth=2,
             label=f'y = {alpha_med:.3f} + {beta_med:.3f}×x')
    ax4.fill_between(np.log10(dt_fit_range),
                 np.log10(sf_fit_p16),
                 np.log10(sf_fit_p84),
                 alpha=0.3, color='red', label='68% credible interval')
    ax4.errorbar(log_dt, log_sf, xerr=xerr, yerr=yerr,
                 fmt='o', markersize=8, color='blue', capsize=4, label='Data', zorder=5)
    ax4.set_xlabel('log₁₀(dt)', fontsize=12); ax4.set_ylabel('log₁₀(SF)', fontsize=12)
    ax4.set_title('Log-Log Space', fontweight='bold')
    ax4.legend(); ax4.grid(True, alpha=0.3)
 
    plt.tight_layout()
 
    if save_plot:
        plt.savefig(path+f'SF_fit_{ra}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
 
 
# ── 4. main function ──────────────────────────────────────────────────────────
 
def SF_linmix(old_dict, verbose=False, amp_at=365,
              plot=False, save_plot=False, save_pkl=False,path=None):
    """LinMix MCMC power-law fit to the structure function in log-log space.
 
    Parameters
    ----------
    old_dict : dict — SF result dict from optSF / SF_wnoise
    phot     : bool — if True, extract photometry key from old_dict
    verbose  : bool — print detailed fit summary
    amp_at   : float — reference timescale in days for amplitude normalisation
    plot     : bool — show diagnostic plots interactively
    save_plot: bool — save diagnostic plots to disk
    save_pkl : bool — save updated old_dict to pkl
 
    Returns
    -------
    old_dict updated with fitted parameters, or None if skipped
    """
 
    # ── extract SF in fitting window [1, 365] days ────────────────────────────
    interval_index = pd.IntervalIndex(old_dict['SF'].index)
    window_mask    = (
        (interval_index.left  >= 1)   &
        (interval_index.right <= 365) &
        (old_dict['SF'] != 0)
    )
    sf_mag = old_dict['SF'][window_mask].dropna()
 
    if len(sf_mag) <= 3:
        print(f'too few data points to fit: {len(sf_mag)} — SKIPPING')
        return None
 
    dt_lenbin = sf_mag.index.categories[sf_mag.index.codes].length / 2
    dt_data   = sf_mag.index.categories[sf_mag.index.codes].mid
 
    maxerr_sf2 = old_dict['SFmaxerr'].reindex(sf_mag.index)
    minerr_sf2 = old_dict['SFminerr'].reindex(sf_mag.index)
    sf_err     = (maxerr_sf2 + minerr_sf2) / 2
 
    log_dt = np.log10(dt_data)
    log_sf = np.log10(sf_mag)
    xerr   = dt_lenbin / (dt_data * np.log(10))
    yerr   = sf_err    / (sf_mag  * np.log(10))
 
    # ── photometry key ────────────────────────────────────────────────────────
    if phot:
        mag_key = list(old_dict.keys())[5]
        if mag_key.endswith('_mag'):
            phot = mag_key.split('_')[0]
        else:
            print('photometry is not defined')
            return
    else:
        phot = ''
 
    # ── data quality checks ───────────────────────────────────────────────────
    checks = {
        'NaN/Inf in data':     not (np.all(np.isfinite(log_dt)) and np.all(np.isfinite(log_sf))),
        'NaN/Inf in errors':   not (np.all(np.isfinite(xerr))   and np.all(np.isfinite(yerr))),
        'non-positive errors': np.any(xerr <= 0) or np.any(yerr <= 0),
    }
    for reason, failed in checks.items():
        if failed:
            print(f'SKIPPING: {reason}')
            return None
 
    # ── scale ─────────────────────────────────────────────────────────────────
    log_dt_s, log_sf_s, xerr_s, yerr_s, scale_params = scale_log_data(
        log_dt, log_sf, xerr, yerr)
    x_mean, x_std, y_mean, y_std = scale_params
 
    if verbose:
        print('=' * 70)
        print('LINMIX LOG-LOG FIT WITH ERRORS (SCALED DATA)')
        print('=' * 70)
        print(f'Data points: {len(dt_data)}')
        print(f'  log(dt): [{log_dt.min():.4f}, {log_dt.max():.4f}]')
        print(f'  log(sf): [{log_sf.min():.4f}, {log_sf.max():.4f}]')
        print(f'  sf_mag fractional error: {np.mean(sf_err / sf_mag) * 100:.1f}%')
        print(f'  log(dt) error: {np.mean(xerr):.4f} ± {np.std(xerr):.4f}')
        print(f'  log(sf) error: {np.mean(yerr):.4f} ± {np.std(yerr):.4f}')
 
    print(f'Scaling — log(dt): mean={x_mean:.4f}, std={x_std:.4f} | '
          f'log(sf): mean={y_mean:.4f}, std={y_std:.4f}')
    print('Running LinMix MCMC on scaled data...')
 
    # ── LinMix fit ────────────────────────────────────────────────────────────
    try:
        print(f'fitting photometry')
        lm = linmix.LinMix(log_dt_s, log_sf_s,
                           xsig=xerr_s, ysig=yerr_s, nchains=2,parallelize=False)
        lm.run_mcmc(miniter=1000, maxiter=3000, silent=True)
        print('LinMix complete!')
    except Exception as e:
        print(f'ERROR: LinMix failed: {e} — SKIPPING')
        return None
 
    # ── unscale chains ────────────────────────────────────────────────────────
    chain_alpha, chain_beta = unscale_chains(
        lm.chain['alpha'], lm.chain['beta'], x_mean, x_std, y_mean, y_std)
 
    # ── batch percentiles ─────────────────────────────────────────────────────
    alpha_p16, alpha_med, alpha_p84 = np.percentile(chain_alpha, [16, 50, 84])
    beta_p16,  beta_med,  beta_p84  = np.percentile(chain_beta,  [16, 50, 84])
 
    A_at_1        = 10 ** alpha_med
    # A_ref_samples = 10 ** chain_alpha * amp_at ** chain_beta
    # A_ref_p16, A_ref_med, A_ref_p84 = np.percentile(A_ref_samples, [16, 50, 84])
    log_A_ref = chain_alpha + chain_beta * np.log10(amp_at)
    A_ref_p16, A_ref_med, A_ref_p84 = 10 ** np.percentile(log_A_ref, [16, 50, 84])

    # ── update dict ───────────────────────────────────────────────────────────
    old_dict.update({
        f'A_1_spl':              A_at_1,
        f'A_{amp_at}_spl':   A_ref_med,
        f'A_maxerr_spl':     A_ref_p84  - A_ref_med,
        f'A_minerr_spl':     A_ref_med  - A_ref_p16,
        f'gamma_spl':        beta_med,
        f'gamma_maxerr_spl': beta_p84   - beta_med,
        f'gamma_minerr_spl': beta_med   - beta_p16,
    })
 
    if verbose:
        print('\n' + '=' * 70)
        print('RESULTS (TRANSFORMED BACK TO ORIGINAL SCALE)')
        print('=' * 70)
        print(f'  α: {alpha_med:.6f} +{alpha_p84 - alpha_med:.6f} -{alpha_med - alpha_p16:.6f}')
        print(f'  β: {beta_med:.6f}  +{beta_p84  - beta_med:.6f}  -{beta_med  - beta_p16:.6f}')
        print(f'  A(dt=1)      = {A_at_1:.6e}')
        print(f'  A(dt={amp_at}) = {A_ref_med:.6e} '
              f'+{A_ref_p84 - A_ref_med:.6e} -{A_ref_med - A_ref_p16:.6e}')
        print(f'Power law: SF ∝ dt^{beta_med:.3f} ± {(beta_p84 - beta_p16) / 2:.3f}')
        print('=' * 70)
 
    # ── plot ──────────────────────────────────────────────────────────────────
    if plot or save_plot:
        dt_fit_range, sf_p16, sf_p50, sf_p84 = posterior_band(
            chain_alpha, chain_beta, dt_data)
        ra = old_dict.get('RA', 'simulatedLC')
        plot_linmix(dt_data, sf_mag, sf_err, dt_lenbin,
                    log_dt, log_sf, xerr, yerr,
                    chain_alpha, chain_beta,
                    alpha_med, beta_med,
                    dt_fit_range, sf_p16, sf_p50, sf_p84,
                    ra, save_plot=save_plot,path=path)
 
    # ── save ──────────────────────────────────────────────────────────────────
    if save_pkl:
        ra        = old_dict.get('RA', 'simulatedLC')
        name_file = path + f'SFdict_fit_linmix_{ra}.pkl'
        with open(name_file, 'wb') as f:
            pickle.dump(old_dict, f)
        print(f'{name_file} successfully saved')
 
    return old_dict

import emcee
import corner
from scipy.optimize import curve_fit

# Define the broken power law model
def broken_power_law_flat(dt, A, gamma1, dt_break):
    """Broken power law with flat slope after break."""
    return np.where(
        dt <= dt_break,
        A * (dt ** gamma1),
        A * (dt_break ** gamma1)  # Flat after break
    )

def broken_power_law_at_break(dt, A_break, gamma1, dt_break):
    return np.where(dt <= dt_break, A_break * (dt / dt_break) ** gamma1, A_break)

# 2. Define log-likelihood (Gaussian)
def log_likelihood(theta, dt, sf, sf_err):
    """Gaussian log-likelihood"""
    A, gamma1, dt_break = theta
    
    # Model prediction
    model = broken_power_law_flat(dt, A, gamma1, dt_break)
    
    # Chi-squared term
    chi2 = np.sum(((sf - model) / sf_err) ** 2)
    
    # Normalization term
    norm = np.sum(np.log(2 * np.pi * sf_err ** 2))
    
    return -0.5 * (chi2 + norm)

# 3. Define log-prior (physically motivated)
def log_prior(theta, dt):
    """Log-prior probability"""
    A, gamma1, dt_break = theta
    
    # Physical constraints
    if A <= 0:  # Amplitude must be positive
        return -np.inf
    if gamma1 < 0 or gamma1 > 5:  # Reasonable power-law slope
        return -np.inf
    if dt_break < min(dt) or dt_break > max(dt):  # Break within data range
        return -np.inf
    
    # Uniform prior in log-space for A (scale-invariant)
    # This is equivalent to prior ∝ 1/A
    return -np.log(A)  # log of 1/A prior

# 4. Define log-posterior
def log_posterior(theta, dt, sf, sf_err):
    """Log-posterior probability"""
    lp = log_prior(theta, dt)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dt, sf, sf_err)

def bpl_mcmc(old_dict, initial_guess=[0.5, 0.5, 100], model_check=False,
             mcmc_check=False, plot=False, save_plot=False, path=None,
             progress=True, verbose=False):
    # initial_guess = [A, gamma, dt_break]
    # model check: plot model and data. checks the fit is right
    _p = print if verbose else (lambda *a, **k: None)

    interval_index = pd.IntervalIndex(old_dict['SF'].index)
    sf_mag = old_dict['SF'][(interval_index.left >= 1) & (interval_index.left <= 365) & (old_dict['SF'] != 0)].dropna()
    dt = sf_mag.index.categories[sf_mag.index.codes].mid
    dt_lenbin = sf_mag.index.categories[sf_mag.index.codes].length / 2
    maxerr_sf2 = old_dict['SFmaxerr'][old_dict['SFmaxerr'].index.isin(sf_mag.index)]
    minerr_sf2 = old_dict['SFminerr'][old_dict['SFminerr'].index.isin(sf_mag.index)]

    err = (maxerr_sf2 + minerr_sf2) / 2

    if len(sf_mag) < 3:
        _p(f'not enough data points ({len(sf_mag)})')
        return None

    # 5. starting values from an error-weighted curve_fit
    try:
        popt, pcov = curve_fit(broken_power_law_flat, dt, sf_mag,
                               p0=initial_guess, sigma=err, absolute_sigma=True)
        A_guess, gamma_guess, dt_break_guess = popt
    except Exception:
        A_guess, gamma_guess, dt_break_guess = 0.5, 0.5, 100
        _p("curve_fit failed, using initial guesses")

    _p(f"Initial guesses: A={A_guess:.3f}, gamma={gamma_guess:.3f}, dt_break={dt_break_guess:.1f}")

    if model_check:
        y_broken_pl = broken_power_law_flat(dt, A_guess, gamma_guess, dt_break_guess)
        plt.errorbar(dt, sf_mag, yerr=err, fmt='o', capsize=5, label='Data')
        plt.plot(dt, y_broken_pl, label='broken power law')
        plt.axvline(dt_break_guess, color='black', linestyle=':')
        plt.xscale('log'); plt.yscale('log'); plt.legend()

    # non-variable sources give a flat/declining SF -> gamma <= 0, rejected by design
    if gamma_guess <= 0:
        _p(f'gamma seed {gamma_guess:.3f} <= 0 (non-variable) - skipping')
        return None

    # 6. MCMC setup
    ndim = 3  # [A, gamma1, dt_break]
    nwalkers = 32
    nsteps = 3000
    burnin = 500

    # clip the seed into the prior's support so walkers don't all start at -inf
    b_lo, b_hi = float(np.min(dt)), float(np.max(dt))
    A_seed = A_guess if A_guess > 0 else 0.1
    g_seed = min(gamma_guess, 5.0)
    b_seed = min(max(dt_break_guess, b_lo), b_hi)
    seed = np.array([A_seed, g_seed, b_seed])

    # absolute per-parameter scatter (proportional scatter collapses when a guess ~ 0)
    scatter = np.array([max(0.05 * A_seed, 1e-3), 0.05, 0.05 * (b_hi - b_lo) + 1e-6])

    # draw walkers, resampling any that land outside the prior's support
    pos = np.empty((nwalkers, ndim))
    for k in range(nwalkers):
        for _ in range(200):
            trial = seed + scatter * np.random.randn(ndim)
            if np.isfinite(log_prior(trial, dt)):
                pos[k] = trial
                break
        else:
            _p('could not seed walkers inside the prior - skipping')
            return None

    # 7. run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(dt, sf_mag, err))
    _p("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, progress=progress)

    # 8. analyse
    samples = sampler.get_chain(discard=burnin, flat=True)
    A_mcmc, gamma_mcmc, dt_break_mcmc = np.median(samples, axis=0)
    A_err = np.percentile(samples[:, 0], [16, 84])
    gamma_err = np.percentile(samples[:, 1], [16, 84])
    dt_break_err = np.percentile(samples[:, 2], [16, 84])

    # amplitude at the break = plateau level of the flat-after-break model,
    # i.e. the saturation SF (comparable to the DRW SF_inf). Computed PER
    # SAMPLE so the (A, gamma, dt_break) correlations propagate correctly —
    # same pattern as SF_linmix's A_ref_samples.
    plateau_samples = samples[:, 0] * samples[:, 2] ** samples[:, 1]
    pl_p16, pl_med, pl_p84 = np.percentile(plateau_samples, [16, 50, 84])

    _p("\nMCMC Results:")
    _p(f"A = {A_mcmc:.3f} +{A_err[1]-A_mcmc:.3f} -{A_mcmc-A_err[0]:.3f}")
    _p(f"A_break = {pl_med:.3f} +{pl_p84-pl_med:.3f} -{pl_med-pl_p16:.3f}")
    _p(f"gamma = {gamma_mcmc:.3f} +{gamma_err[1]-gamma_mcmc:.3f} -{gamma_mcmc-gamma_err[0]:.3f}")
    _p(f"dt_break = {dt_break_mcmc:.1f} +{dt_break_err[1]-dt_break_mcmc:.1f} -{dt_break_mcmc-dt_break_err[0]:.1f}")

    SF_dict = {'RA': old_dict['RA'], 'A_1_bpl': A_mcmc, 
               'A_maxerr_bpl': A_err[1]-A_mcmc, 
               'A_minerr_bpl': A_mcmc-A_err[0],
               'A_break_bpl': pl_med,
               'A_break_maxerr_bpl': pl_p84 - pl_med,
               'A_break_minerr_bpl': pl_med - pl_p16,
               'gamma_bpl': gamma_mcmc, 
               'gamma_maxerr_bpl': gamma_err[1]-gamma_mcmc,
               'gamma_minerr_bpl': gamma_mcmc-gamma_err[0],
               'dt_break_bpl': dt_break_mcmc, 
               'dt_break_maxerr_bpl': dt_break_err[1]-dt_break_mcmc,
               'dt_break_minerr_bpl': dt_break_mcmc-dt_break_err[0]}

    if mcmc_check:
        # 9. trace plots
        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        samples_chain = sampler.get_chain()
        labels = ["A_1_bpl", "gamma_bpl", "dt_break_bpl"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples_chain[:, :, i], "k", alpha=0.3)
            ax.set_ylabel(labels[i])
            ax.axvline(burnin, color='red', linestyle='--')
        axes[-1].set_xlabel("Step number")
        plt.tight_layout()
        plt.show()

    if plot or save_plot:
        # 11. best fit with data + posterior draws
        dt_fit = np.logspace(np.log10(min(dt)), np.log10(max(dt)), 100)
        y_fit = broken_power_law_flat(dt_fit, A_mcmc, gamma_mcmc, dt_break_mcmc)

        fig = plt.figure(figsize=(10, 6))
        plt.errorbar(dt, sf_mag, yerr=err, fmt='o', capsize=3,
                     alpha=0.7, label='Data', color='blue')
        inds = np.random.randint(len(samples), size=100)
        for ind in inds:
            A_samp, gamma_samp, dt_break_samp = samples[ind]
            plt.plot(dt_fit, broken_power_law_flat(dt_fit, A_samp, gamma_samp, dt_break_samp),
                     'gray', alpha=0.1, linewidth=0.5)
        plt.plot(dt_fit, y_fit, 'r-', linewidth=2, label='Median fit')
        plt.axvline(dt_break_mcmc, color='black', linestyle=':',
                    label=f'Break at {dt_break_mcmc:.1f} days')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Time Delay (days)'); plt.ylabel('Structure Function')
        plt.legend(); plt.grid(True, alpha=0.3)

        if save_plot:
            ra = old_dict.get('RA', 'sim'); band = old_dict.get('band', '')
            prefix = '' if path is None else str(path)
            fname = prefix if prefix.endswith('.png') else os.path.join(prefix, f'bpl_fit_{ra}_{band}.png')
            plt.savefig(fname)
            _p(f'plot saved to {fname}')
        if plot:
            plt.show()
        else:
            plt.close(fig)

    # 12. convergence — tol=0 returns an estimate for short chains instead of raising
    try:
        tau = sampler.get_autocorr_time(tol=0)
        _p(f"\nAutocorrelation time: {tau}")
        _p(f"Effective sample size: {len(samples) / tau}")
    except Exception as e:
        _p(f"Could not compute autocorrelation time: {e}")

    return SF_dict 

def parse_to_dict(series_str):
    """
    Parse the string and return a dictionary with 'SF' as key.
    Returns: {'SF': parsed_series} or original value if parsing fails
    """
    if not isinstance(series_str, str):
        # If it's already a dictionary or something else
        if isinstance(series_str, dict) and 'SF' in series_str:
            return series_str
        # If it's already a Series, wrap it
        elif isinstance(series_str, pd.Series):
            return {'SF': series_str}
        else:
            return {'SF': series_str}  # Keep as is
    
    try:
        # Parse the string to create Series
        lines = [line.strip() for line in series_str.split('\n') 
                if line.strip() and 'dtype:' not in line]
        
        if not lines:
            return {'SF': None}
        
        # Skip header if present
        if lines[0] == 'log_bin' or '(' not in lines[0]:
            name = lines[0]
            lines = lines[1:]
        else:
            name = 'log_bin'
        
        intervals = []
        values = []
        
        for line in lines:
            if not line or ']' not in line:
                continue
            
            # Find interval end
            bracket_pos = line.find(']')
            interval_str = line[:bracket_pos + 1]
            value_str = line[bracket_pos + 1:].strip()
            
            # Parse interval
            clean_interval = interval_str.replace('(', '').replace('[', '').replace(']', '')
            left, right = map(float, clean_interval.split(','))
            intervals.append((left, right))
            
            # Parse value
            if not value_str or value_str.lower() in ['nan', 'null', 'na', 'none']:
                values.append(np.nan)
            else:
                try:
                    values.append(float(value_str))
                except:
                    values.append(np.nan)
        
        if intervals:
            interval_index = pd.IntervalIndex.from_tuples(intervals, closed='right')
            series = pd.Series(values, index=interval_index, name=name)
            # Convert the index
            series_categorical = series.copy()
            series_categorical.index = pd.CategoricalIndex(
            [idx for idx in series.index],
            categories=[idx for idx in series.index]
)
            return {'SF': series_categorical}
        else:
            return {'SF': None}
            
    except Exception as e:
        print(f"Error parsing: {e}")
        return {'SF': series_str}  # Return original as fallback


# Function to recreate old_dict format from parsed data
def format_SFdf(row, columns=['RA_x','DEC','SF','SFmaxerr',
                            'SFminerr', 'PSF_A_365', 'PSF_gamma', 
                                'PSF_A_minerr', 'PSF_A_maxerr']):
    """
    Create the old dictionary format from a DataFrame row.
    Assumes row has: SF_series, and other columns like SFmaxerr, SFminerr, etc.
    """
    old_dict = {}
    
    # Add SF Series
    #if 'SF_dict' in row and isinstance(row['SF_dict']['SF'], pd.Series):
        #print('yes')
    #    old_dict['SF'] = row['SF_dict']['SF']
    #print('oi')
    # Add other parameters if they exist
    for key in  columns:# ['RA_x','DEC','SF','SFmaxerr', 'SFminerr', 'PSF_A_365', 'PSF_gamma', 
                #'PSF_A_minerr', 'PSF_A_maxerr']:
        if key in row:
            #print(key)
            # Check if it's also a Series that needs parsing
            if isinstance(row[key], str) and '\n' in row[key] and ']' in row[key]:# and 'IntervalIndex' in row[key]:
                # Parse it
                parsed = parse_to_dict(row[key])
                old_dict[key] = parsed['SF'] if 'SF' in parsed else row[key]
            else:
                old_dict[key] = row[key]
    
    return old_dict


def plot_SF(old_dict,band, model,use_all_points=True,label=True,color_data='blue'):
    if band == 'g':
        color = 'mediumseagreen'
        #color2 = 'darkgreen'
        color2 = 'seagreen'
        # color2 = 'blue'
    elif band == 'r':
        color = 'firebrick'
        # color2 = 'maroon'
        # color2 = 'red'
        color2 = '#E17701'
    elif band =='i':
        color = 'gold'
        color2 = 'goldenrod'

    #old_dict = SF_dict[2]
    interval_index = pd.IntervalIndex(old_dict['SF'].index)
    sf_mag_lim=old_dict['SF'][(interval_index.left >= 1)&(interval_index.right < 365)&(old_dict['SF']!=0)].dropna()
    
    if not use_all_points:
        print(sf_mag_lim[-1:],f'excluded from fit')
        sf_mag = sf_mag_lim[:-1]
    else:
        sf_mag = sf_mag_lim
    
    dt_data = sf_mag.index.categories[sf_mag.index.codes].mid
    dt_lenbin = sf_mag.index.categories[sf_mag.index.codes].length/2

    maxerr_sf2 = old_dict['SFmaxerr'][old_dict['SFmaxerr'].index.isin(sf_mag.index)]
    minerr_sf2 = old_dict['SFminerr'][old_dict['SFminerr'].index.isin(sf_mag.index)]
    #err = (maxerr_sf2+minerr_sf2)/2
    #ax2.scatter(dt_data, sf_mag, s=100, c='blue', marker='o', 
    #        label='Data', zorder=5)

    sf = old_dict['SF'].dropna()
    dt = sf.index.categories[sf.index.codes].mid
    
    # print(old_dict['SF'])
    # print(dt)
    if label:
        lab1 = 'Data'
        lab2 = 'fitted Data'
    else:
        lab1 = None
        lab2 = None
    
    plt.errorbar(dt,sf, yerr=(old_dict['SFminerr'].values,old_dict['SFmaxerr'].values),
                    fmt='o',color=color_data,capsize=3,label=lab1)
    
    plt.errorbar(dt_data,sf_mag, yerr=(minerr_sf2,maxerr_sf2),
                    xerr=dt_lenbin,fmt='o',color=color2,capsize=3,label=lab2)

    if model=='single':
        mod = '_spl'
        # mod = ''
        # xlin = np.linspace(1, 2000, 50)

        # CORRECTED: A_365 is amplitude at 365 days
        # y_lm = old_dict['PSF_A_365'] * (xlin/365)**old_dict['PSF_gamma']
        y_fit = old_dict[f'A_365{mod}'] * (dt_data/365)**old_dict[f'gamma{mod}']

        # plt.plot(dt_data, y_fit, label='single power law', linewidth=2, c='#CC4F1B')
        plt.plot(dt_data, y_fit, label='single power law', linewidth=2, c=color2)
        #plt.fill_between(xlin, y_lm/np.abs(old_dict['PSF_A_minerr']), y_lm*np.abs(old_dict['PSF_A_maxerr']),
        #            alpha=0.3, facecolor='#FF9848')
        # print(len(dt_data),len(sf_mag),len(minerr_sf2),len(maxerr_sf2))
        #print(minerr_sf2.max(),maxerr_sf2.max())
        #plt.fill_between(xlin, y_lm+minerr_sf2.max(), y_lm-maxerr_sf2.max(),
        #               alpha=0.3, facecolor='#FF9848')
        # plt.fill_between(dt_data, sf_mag-(minerr_sf2.max()), sf_mag+(maxerr_sf2.max()),
        # alpha=0.3, facecolor='#FF9848')

        #if len(dt_data) > 5:
        #    ax2.scatter(dt_data[5:], sf_mag[5:], s=100, c='gray', marker='x', 
        #                label='Data (not used)', zorder=5)
        #ax2.axvline(x_break, color='green', linestyle='--', alpha=0.5, 
        #            label=f'Break at x={x_break:.2f}')

    elif model=='broken':
        mod = '_bpl'
        # xlog = np.logspace(np.log10(min(dt)), np.log10(max(dt)), 100)

        y_fit = broken_power_law_flat(dt_data, old_dict[f'A_1{mod}'], old_dict[f'gamma{mod}'], old_dict[f'dt_break{mod}'])
        # Plot median fit
        bk = old_dict[f'dt_break{mod}']
        plt.plot(dt_data, y_fit, c='#FF958F',linewidth=2, label='broken power law')
        plt.axvline(old_dict[f'dt_break{mod}'], color='black', linestyle=':', 
                    label=f'Break at {bk:.1f} days')

    plt.xlabel('Time diff (days)', fontsize=15)
    plt.ylabel('Structure Function (mag)', fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tick_params(axis='both', labelsize=12)
    # plt.show()

    plt.savefig(f'plot_SF_{band}_{model}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return None
    # return sf_mag,maxerr_sf2,minerr_sf2,y_fit

from scipy.stats import chi2

# CHI-SQUARE MODEL COMPARISON FUNCTION
def chi2_stats(y_data, y_pred, n_params, model_name,sigma=1):
    """Calculate chi-square statistics for a model."""
    n_points = len(y_data)
    
    # Chi-square calculation
    chi2_val = np.sum(((y_data - y_pred) / sigma) ** 2)

    # Degrees of freedom
    dof = n_points - n_params
    
    # Reduced chi-square
    reduced_chi2 = chi2_val / dof if dof > 0 else np.inf
    
    # p-value (probability that chi-square could be this large by chance)
    p_value = 1 - chi2.cdf(chi2_val, dof) if dof > 0 else 0
    
    print(f"\n--- {model_name} ---")
    print(f"Parameters: {n_params}")
    print(f"χ² = {chi2_val:.3f}")
    print(f"Degrees of freedom = {dof}")
    print(f"Reduced χ² = {reduced_chi2:.3f}")
    print(f"p-value = {p_value:.3f}")
    
    return chi2_val, reduced_chi2, p_value
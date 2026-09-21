#!/usr/bin/env python3
"""
trace_pairs.py
Map dmag values in an SF bin back to the epoch pairs that produced them.

compute_sf_pairs applies no filtering — corr_sf is built straight from
SFarray with no dropna and no dt!=0 cut — so corr_sf row k corresponds
exactly to element k of np.triu_indices(len(ztf), k=1). That makes the
inverse mapping exact rather than approximate.

IMPORTANT: ztf must be the same frame that was passed to compute_sf_pairs,
i.e. AFTER clip_by_blocks / sigma filtering. optSF does

    ztf = clip_by_blocks(ztf1, mag_column, gap_threshold) if clip else ztf1

so passing ztf1 when clip=True silently gives wrong epochs.
"""

import numpy as np
import pandas as pd


def trace_bin(ztf, corr_sf, bin_label, mag_column, mag_err=None):
    """Epoch pairs contributing to one log_bin.

    Parameters
    ----------
    ztf        : DataFrame — the light curve handed to compute_sf_pairs
    corr_sf    : DataFrame — first return value of compute_sf_pairs
    bin_label  : the bin, as a string '(1.256, 1.581]', a pd.Interval, or a
                 (lo, hi) tuple
    mag_column : str
    mag_err    : str, optional — adds per-epoch errors and their quadrature sum

    Returns
    -------
    DataFrame, one row per pair, sorted by dt
    """
    if isinstance(bin_label, tuple):
        lo, hi = bin_label
        mask = (corr_sf['dt'].values > lo) & (corr_sf['dt'].values <= hi)
    else:
        mask = corr_sf['log_bin'].astype(str).values == str(bin_label)

    pos = np.flatnonzero(mask)
    if pos.size == 0:
        raise ValueError(f'no pairs matched {bin_label!r}')

    z = ztf.reset_index(drop=True)
    if len(corr_sf) != len(z) * (len(z) - 1) // 2:
        raise ValueError(
            f'corr_sf has {len(corr_sf)} rows but ztf has {len(z)} epochs '
            f'(expected {len(z)*(len(z)-1)//2}). Wrong ztf — pass the clipped one.')

    i_idx, j_idx = np.triu_indices(len(z), k=1)
    i, j = i_idx[pos], j_idx[pos]

    out = pd.DataFrame({
        'pair_row': pos,
        'i': i, 'j': j,
        'mjd_i': z['OBSMJD'].values[i],
        'mjd_j': z['OBSMJD'].values[j],
        'dt':    corr_sf['dt'].values[pos],
        'mag_i': z[mag_column].values[i],
        'mag_j': z[mag_column].values[j],
        'dmag':  corr_sf['dmag'].values[pos],
    })
    if mag_err is not None:
        out['err_i'] = z[mag_err].values[i]
        out['err_j'] = z[mag_err].values[j]
        out['sigma_pair'] = np.sqrt(out.err_i**2 + out.err_j**2)
        out['dmag_over_sigma'] = out.dmag / out.sigma_pair
    if 'block_id' in z.columns:
        out['blk_i'] = z['block_id'].values[i]
        out['blk_j'] = z['block_id'].values[j]
    return out.sort_values('dt').reset_index(drop=True)


def sparse_bins(corr_sf, max_n=3):
    """Bins with max_n or fewer pairs — the ones whose error bars collapse.

    With n=1 all three quantiles coincide, so (p84-p50)=0 and the bin is
    plotted with a zero error bar and gets near-infinite weight in the fit.
    """
    g = corr_sf.groupby('log_bin', observed=True)['dmag']
    n = g.count()
    return pd.DataFrame({
        'n_pairs': n,
        'dmag_min': g.min(),
        'dmag_max': g.max(),
        'quantile_spread': g.quantile(.84) - g.quantile(.50),
    })[n <= max_n]
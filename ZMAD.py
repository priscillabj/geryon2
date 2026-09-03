"""ZMAD variability metric for BAT_results single-file parquets (magnitudes).

One file = one AGN + the calibration-star field, one band:
    object_index   : source id; AGN found by coordinate match (index 0 is NOT the target)
    OBSMJD         -> mjd
    ALPHAWIN_REF / DELTAWIN_REF -> ra/dec (constant per object_index)
    field/ccdid/qid -> CCDquadID
    filtercode     -> 'zg' | 'zr' | 'zi'
    MAG_{3,4,6,10}_TOT_AB / MERR_{3,4,6,10}_TOT_AB : aperture magnitudes (AB)

Selection follows optSF: MERR < 0.5 + modal field/ccdid/qid on the target,
then quality_flags -> run_sigma_filtering -> quality_cuts on the star pool.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

from VarTools import (radec_filename, _find_target_obj,run_sigma_filtering)
from newSF import quality_flags, quality_cuts

BAT_ROOT = os.path.expanduser('~/BAT_results')
AB_ZP = 23.9                    # AB mag -> uJy


def mag2flux(m, zp=AB_ZP):
    return 10 ** (-0.4 * (m - zp))


def bat_file(ra, dec, band, root=BAT_ROOT):
    return os.path.join(root, f'{ra}_{dec}_z{band}_merged.parquet')


QUAD_KEYS = ('field', 'ccdid', 'qid')

# Why the last zmad_metric call returned None.  A single source per process
# (subprocess engine) or one at a time per worker, so a module global is safe.
# The driver reads this instead of recording a bare 'no result'.
FAIL = None


def _prep(df):
    """Mutates in place: only ever called on a fresh read.

    CCDquadID is an integer code, not a string.  These files run to ~2e7 rows
    and a Python-object string column at that length costs several GB on its
    own; the value is only ever compared for equality, so the encoding is free.
    Files missing field/ccdid/qid (single-quadrant products, where the field is
    in the filename) get a constant code rather than a KeyError.
    """
    out = df
    out['mjd'] = out['OBSMJD'].astype('float64', copy=False)

    have = [k for k in QUAD_KEYS if k in out.columns]
    if len(have) == len(QUAD_KEYS):
        out['CCDquadID'] = (out['field'].astype('int64') * 10_000
                            + out['ccdid'].astype('int64') * 10
                            + out['qid'].astype('int64'))
    else:
        # nothing to group on: the whole file is one quadrant by construction
        out['CCDquadID'] = np.int64(0)
    return out


def _target_lc(t, mag_err):
    """optSF target cut: MERR < 0.5 and modal field/ccdid/qid."""
    mask = t[mag_err] < 0.5
    if 'qid' in t.columns:
        qid = t['qid'].mode().iloc[0]
        ccdid = t['ccdid'].mode().iloc[0]
        field = t['field'].mode().iloc[0]
        mask &= (t['qid'] == qid) & (t['ccdid'] == ccdid) & (t['field'] == field)
    return t[mask]


def _tag(col):
    m = re.search(r'MAG_(\d+)', col)
    return m.group(1) if m else col


def _star_stats(s, col, tag):
    """Per-star median mag -> |dmag| -> per-epoch median -> residual -> per-epoch MAD -> ZMAD."""
    s[f'median_mag_{tag}'] = s.groupby('object_index')[col].transform('median')
    s[f'dmag_{tag}'] = (s[col] - s[f'median_mag_{tag}']).abs()

    s[f'median_per_epoch_{tag}'] = s.groupby('mjd')[f'dmag_{tag}'].transform('median')
    s[f'residual_{tag}'] = (s[f'dmag_{tag}'] - s[f'median_per_epoch_{tag}']).abs()
    s[f'MAD_{tag}'] = s.groupby('mjd')[f'residual_{tag}'].transform('median')
    s.loc[s[f'MAD_{tag}'] <= 0, f'MAD_{tag}'] = np.nan     # epochs with degenerate MAD
    s[f'ZMAD_{tag}'] = s[f'residual_{tag}'] / s[f'MAD_{tag}']
    return s


def zmad_metric(file, band=None, mag_columns=('MAG_4_TOT_AB',),
                agn_index=None, match_radius_arcsec=3.0, mag_pad=None,
                sigma_filter=True, save_sigma=True, x=100,
                flux=False, zp=AB_ZP,
                agg='mean', sigma=3.0, min_for_clip=5, min_stars=3, min_epochs=5,
                verbose=False):
    """
    Returns (agn, stars, out):
        agn, stars : dict {tag: DataFrame} carrying the per-aperture stat columns
        out        : dict with RA/DEC/band/CCDquadID/object_index and out['metrics'][tag]

    metrics[tag]: n_stars, n_stars_preclip, n_epochs, agn, cs_mean, cs_std,
    cs_median, cs_max, sigma, perc, star_stat (clipped Series), star_stat_all.

    mag_pad : half-width of the extra brightness bracket, in mag; None to rely
              entirely on quality_cuts.
    flux    : compute the statistic on uJy = 10**(-0.4*(mag - zp)) instead of on
              magnitudes (the original AGNProp behaviour).  Deviations are then
              absolute rather than fractional, so the brightness bracket is doing
              real work and mag_pad=None is a bad idea.  Selection and the bracket
              always happen in magnitudes; only the statistic changes space.

    On failure returns (None, None, None).
    """
    global FAIL
    FAIL = None

    file = Path(file)
    name = file.name
    ra, dec, band_file = radec_filename(name, band=True)
    if band is None:
        band = band_file
    elif band != band_file:
        raise ValueError(f'band={band!r} but filename says {band_file!r}: {name}')

    df = _prep(pd.read_parquet(file))
    if 'filtercode' in df.columns:
        df = df[df['filtercode'] == f'z{band}']
    # else: single-quadrant products carry no filtercode; the band is in the
    # filename and the file holds exactly that band by construction
    if df.empty:
        if verbose:
            print(f'{name}: no z{band} rows')
        FAIL = f'no z{band} rows in file'
        return None, None, None

    if agn_index is None:
        tgt = SkyCoord(ra=ra, dec=dec, unit='deg')     # scalar: _find_target_obj does int(idx)
        agn_index = _find_target_obj(file, tgt, match_radius_arcsec)
        if agn_index is None:
            FAIL = f'no object within {match_radius_arcsec}" of {ra},{dec}'
            return None, None, None

    ccd = int(df.loc[df['object_index'] == agn_index, 'CCDquadID'].mode()[0])

    reject = {}                  # tag -> why this aperture produced nothing
    agn_out, stars_out, metrics = {}, {}, {}
    aggf = {'mean': 'mean', 'sum': 'sum', 'median': 'median'}[agg]
    sigma_keep = None          # object_index surviving sigma filtering; computed once

    for col in mag_columns:
        tag = _tag(col)
        mag_err = col.replace('MAG_', 'MERR_', 1)

        # --- target light curve ---
        a = _target_lc(df[df['object_index'] == agn_index], mag_err)
        a = a.dropna(subset=[col]).copy()
        if len(a) < min_epochs:
            if verbose:
                print(f'{name} {tag}: {len(a)} epochs after quality cut')
            reject[tag] = f'{len(a)} epochs < min_epochs {min_epochs}'
            metrics[tag] = None
            continue

        # --- comparison-star pool ---
        s = quality_flags(df, a)
        if s.empty:
            if verbose:
                print(f'{name} {tag}: empty cs')
            reject[tag] = 'quality_flags returned no stars'
            metrics[tag] = None
            continue

        # sigma filtering is run once (on mag_columns[0]) and reused: it renames
        # the mag column to MAG_4_TOT_AB, which collides for any other aperture
        if sigma_filter and sigma_keep is None:
            filtered, sdiag = run_sigma_filtering(
                s.rename(columns={col: 'MAG_4_TOT_AB'}),
                agn_indices=agn_index, save_file=save_sigma, plot=False,
                save_plt=save_sigma, filename=str(file), ra=ra, dec=dec)
            sigma_keep = pd.unique(filtered['object_index'])
        if sigma_keep is not None:
            s = s[s['object_index'].isin(sigma_keep)]

        s = quality_cuts(s, a, col, x=x)
        s = s[(s['object_index'] != agn_index) &
              (s['CCDquadID'] == ccd) &
              (s['mjd'].isin(a['mjd']))].dropna(subset=[col]).copy()

        if mag_pad is not None:
            med = s.groupby('object_index')[col].median()
            lo, hi = a[col].min() - mag_pad, a[col].max() + mag_pad
            s = s[s['object_index'].isin(med[(med >= lo) & (med <= hi)].index)].copy()

        if s['object_index'].nunique() < min_stars:
            if verbose:
                print(f'{name} {tag}: only {s["object_index"].nunique()} stars left')
            reject[tag] = f'{s["object_index"].nunique()} stars < min_stars {min_stars}'
            metrics[tag] = None
            continue

        val = col
        if flux:
            val = f'FLUX_{tag}_uJy'
            a[val] = mag2flux(a[col], zp)
            s[val] = mag2flux(s[col], zp)

        s = _star_stats(s, val, tag)

        # one row per epoch carries the reference level and scale for the target
        ep = s.drop_duplicates('mjd')[['mjd', f'median_per_epoch_{tag}', f'MAD_{tag}']]
        a = a.merge(ep, on='mjd', how='left')
        a[f'median_mag_{tag}'] = a[val].median()
        a[f'dmag_{tag}'] = (a[val] - a[f'median_mag_{tag}']).abs()
        a[f'residual_{tag}'] = (a[f'dmag_{tag}'] - a[f'median_per_epoch_{tag}']).abs()
        a[f'ZMAD_{tag}'] = a[f'residual_{tag}'] / a[f'MAD_{tag}']

        # compare target and stars on the same epochs only
        good = a.loc[a[f'ZMAD_{tag}'].notna(), 'mjd']
        sg = s[s['mjd'].isin(good) & s[f'ZMAD_{tag}'].notna()]
        star_stat = sg.groupby('object_index')[f'ZMAD_{tag}'].agg(aggf)
        if star_stat.empty:
            metrics[tag] = None
            continue
        agn_stat = float(a.loc[a['mjd'].isin(good), f'ZMAD_{tag}'].agg(aggf))

        v = star_stat.values
        mask = (~sigma_clip(v, sigma=sigma, maxiters=5).mask
                if len(v) > min_for_clip else np.ones(len(v), bool))
        clipped = star_stat[mask]

        mu = clipped.mean()
        sd = clipped.std(ddof=1) if len(clipped) > 1 else np.nan
        metrics[tag] = {
            'col': col,
            'val': val,
            'n_stars': int(len(clipped)),
            'n_stars_preclip': int(len(star_stat)),
            'n_epochs': int(len(good)),
            'agn': agn_stat,
            'cs_mean': mu,
            'cs_std': sd,
            'cs_median': float(clipped.median()),
            'cs_max': float(clipped.max()),
            'sigma': (agn_stat - mu) / sd if sd and np.isfinite(sd) else np.nan,
            'perc': 100.0 * np.mean(clipped.values < agn_stat),
            'star_stat': clipped,
            'star_stat_all': star_stat,
        }
        agn_out[tag], stars_out[tag] = a, s

    if not agn_out:
        FAIL = ('every aperture rejected: ' +
                '; '.join(f'{t}={r}' for t, r in reject.items()))
        return None, None, None
    out = {'RA': ra, 'DEC': dec, 'band': band, 'CCDquadID': ccd,
           'object_index': agn_index, 'agg': agg,
           'space': 'flux' if flux else 'mag', 'metrics': metrics}
    return agn_out, stars_out, out


def zmad_batch(files, **kw):
    """Tidy one-row-per-(file, aperture) summary."""
    rows = []
    for f in files:
        _, _, out = zmad_metric(f, **kw)
        if out is None:
            continue
        for tag, m in out['metrics'].items():
            if m is None:
                continue
            rows.append({'file': os.path.basename(f), 'RA': out['RA'], 'DEC': out['DEC'],
                         'band': out['band'], 'CCDquadID': out['CCDquadID'],
                         'object_index': out['object_index'], 'aperture': tag,
                         'space': out['space'],
                         **{k: v for k, v in m.items() if not isinstance(v, pd.Series)}})
    return pd.DataFrame(rows)


def zmad_null(file, n=20, seed=0, tag=None, **kw):
    """Empirical null: re-run the metric with n comparison stars standing in as
    the target.  Each star is excluded from its own pool, so a well-behaved
    metric returns sigma scattered around 0.  Costs n+1 full passes over the
    file, so use it on a few sources, not the whole sample.

        null = zmad_null(f, n=20)
        null['sigma'].describe()
    """
    agn, stars, out = zmad_metric(file, **kw)
    if out is None:
        return pd.DataFrame()
    tag = tag or next(t for t, m in out['metrics'].items() if m is not None)

    pool = pd.unique(stars[tag]['object_index'])
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, size=min(n, len(pool)), replace=False)

    rows = [{'object_index': int(out['object_index']), 'is_agn': True,
             'sigma': out['metrics'][tag]['sigma'], 'perc': out['metrics'][tag]['perc'],
             'n_stars': out['metrics'][tag]['n_stars']}]
    for oi in pick:
        _, _, o = zmad_metric(file, agn_index=oi, **kw)
        if o is None or o['metrics'].get(tag) is None:
            continue
        m = o['metrics'][tag]
        rows.append({'object_index': int(oi), 'is_agn': False, 'sigma': m['sigma'],
                     'perc': m['perc'], 'n_stars': m['n_stars']})
    return pd.DataFrame(rows)
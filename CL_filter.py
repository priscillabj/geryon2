#!/usr/bin/env python
"""Shortlist CL transitions from cl_transitions.py output and annotate them.

Stage 1  per-row quality: every failed cut is recorded in `reject`, nothing is
         silently dropped. Thin windows are flagged, not rejected.
Stage 2  one row per object-band: the highest-SNR file (several lc_files in one
         band are largely the same photons, not independent evidence).
Stage 3  cross-band corroboration with epoch matching: bands agree only if
         their transition spans overlap (within COINC) and they move the same way.
         Tiers: A   >=2 bands agree, no conflict, |dg/dr| >= 1
                A?  >=2 bands agree, but a conflicting band or |dg/dr| < 1
                B   one band, no conflict, >= MIN_NIGHTS_SIDE nights each side
                B?  one band with a conflict or a thin window
Stage 4  annotate with bat_index / clasf from the candidates table, and write
         the plot_SFpar --CL overlay (master-schema rows, every band of each
         shortlisted object; plot_SFpar's own band filter picks the right ones).

Outputs, next to the input (A_365_spl shown; a _limN input keeps its suffix):
  A_365_spl_CL_rows_flagged.parquet  every transitions row + `reject` reasons
  A_365_spl_CL_shortlist.parquet     one row per surviving object, with tier
  A_365_spl_CL_overlay.parquet       candidate rows of shortlisted objects, each
                                     with its own results and the object's tier
"""
import os
import re
import argparse

import numpy as np
import pandas as pd

MIN_NIGHTS_SIDE = 10    # nights in each compared window
FAINT_LIMIT = 20.5      # fainter window median than this -> near ZTF single-epoch limit
COINC = 30.0            # d of slack when matching transition spans across bands
KINDS = ('sustained', 'fast_transition', 'both_separate')  # sustained stat passed

SL_COLS = ['object', 'ref_band', 'ref_lc_file', 'snr', 'd', 'mjd_start', 'mjd_end',
           'min_side', 'corr_bands', 'n_corr', 'n_conflict', 'g_over_r']


def derive_paths(trans, cand=None):
    m = re.match(r'(.*)_CL_transitions(_lim\d+)?\.parquet$', trans)
    if not m:
        raise SystemExit(f'unexpected transitions filename: {trans}')
    base, lim = m.group(1), m.group(2) or ''
    out = f'{base}_CL{lim}'
    return (cand or f'{base}_CL_candidates.parquet',
            f'{out}_rows_flagged.parquet',
            f'{out}_shortlist.parquet',
            f'{out}_overlay.parquet')


# ---- stage 1 -------------------------------------------------------------------
def flag_rows(tr):
    """Hard cuts go in `reject`. Thin windows are only flagged here: they demote
    single-band objects in stage 3, but corroborated objects keep their tier."""
    cuts = {
        'kind':       ~tr.kind.isin(KINDS),             # jump_only usually reverts
        'unresolved': ~tr.sus_resolved.astype(bool),    # levels not reached in window
        'faint':      tr[['m_before_sus', 'm_after_sus']].max(axis=1) > FAINT_LIMIT,
    }
    tr = tr.copy()
    tr['reject'] = ''
    for name, m in cuts.items():
        tr.loc[m.fillna(True), 'reject'] += name + ';'
    tr['min_side'] = tr[['n_before_sus', 'n_after_sus']].min(axis=1)
    tr['thin'] = tr.min_side < MIN_NIGHTS_SIDE
    nfail = {k: int(v.fillna(True).sum()) for k, v in cuts.items()}
    nfail['(thin flag)'] = int(tr.thin.sum())
    return tr, nfail


# ---- stage 3 -------------------------------------------------------------------
def _agree(g, ref):
    lo, hi = ref.mjd_sus_start - COINC, ref.mjd_sus_end + COINC
    return ((g.mjd_sus_start <= hi) & (g.mjd_sus_end >= lo)
            & (np.sign(g.d_sus) == np.sign(ref.d_sus)))


def xband(g):
    # reference = the band the most bands agree with (epoch overlap + same sign);
    # g arrives sorted by SNR, so argmax breaks ties in favour of the higher SNR.
    # Highest-SNR-alone would let one discrepant band veto two agreeing ones.
    counts = [int(_agree(g, row).sum()) for _, row in g.iterrows()]
    ref = g.iloc[int(np.argmax(counts))]
    same = _agree(g, ref)
    d = g[same].set_index('band').d_sus
    return pd.Series({
        'ref_band': ref.band, 'ref_lc_file': ref.lc_file,
        'snr': ref.snr_sus, 'd': ref.d_sus,
        'mjd_start': ref.mjd_sus_start, 'mjd_end': ref.mjd_sus_end,
        'min_side': int(ref.min_side),
        'corr_bands': ''.join(sorted(d.index)),
        'n_corr': len(d),
        'n_conflict': int((~same).sum()),              # passed, but other epoch or sign
        'g_over_r': abs(d['g'] / d['r']) if {'g', 'r'} <= set(d.index) else np.nan,
    })


def make_shortlist(good):
    best = (good.sort_values('snr_sus', ascending=False)        # stage 2
                .drop_duplicates(['object', 'band']))
    if best.empty:
        return best, pd.DataFrame(columns=SL_COLS + ['tier'])
    obj = (best.groupby('object', sort=False)[best.columns.drop('object')]
               .apply(xband).reset_index())
    obj['tier'] = np.select(
        [(obj.n_corr >= 2) & (obj.n_conflict == 0) & ~(obj.g_over_r < 1),
         obj.n_corr >= 2,
         (obj.n_conflict == 0) & (obj.min_side >= MIN_NIGHTS_SIDE)],
        ['A', 'A?', 'B'], default='B?')
    return best, obj.sort_values(['tier', 'snr'], ascending=[True, False])


# ---- stage 4 -------------------------------------------------------------------
ROW_COLS = ['lc_file', 'kind', 'reject', 'd_sus', 'snr_sus', 'm_before_sus', 'm_after_sus',
            'mjd_sus_start', 'mjd_sus_end', 'min_side', 'thin', 'd_jmp', 'snr_jmp']


def annotate(sl, cl, tr):
    cl = cl.copy()
    cl['object'] = cl.lc_file.str.split('_').str[:2].str.join('_')
    if cl.groupby('object').bat_index.nunique().max() > 1:
        raise SystemExit('bat_index differs across band files of one object')
    meta = cl.drop_duplicates('object').set_index('object')[['bat_index', 'clasf']]
    sl = sl.join(meta, on='object')

    # overlay: master-schema candidate rows of shortlisted objects, each with
    # only its OWN results (NaN where this lc_file did not fire) and the object's
    # tier -- an object property like clasf. Other files' results stay in the
    # shortlist (join on bat_index if needed).
    ov = (cl[cl.object.isin(sl.object)]
            .merge(tr[ROW_COLS], on='lc_file', how='left', validate='one_to_one')
            .merge(sl[['object', 'tier']], on='object', how='left', validate='many_to_one')
            .drop(columns='object'))
    return sl, ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trans', default=os.path.expanduser(
        '~/results/A_365_spl_CL_transitions.parquet'))
    ap.add_argument('--cand', default=None,
                    help='candidates table (default: derived from --trans)')
    args = ap.parse_args()
    trans = os.path.expanduser(args.trans)
    cand, f_rows, f_sl, f_ov = derive_paths(
        trans, os.path.expanduser(args.cand) if args.cand else None)

    tr, nfail = flag_rows(pd.read_parquet(trans))
    good = tr[tr.reject == '']
    best, sl = make_shortlist(good)
    sl, ov = annotate(sl, pd.read_parquet(cand), tr)

    print(f'{len(tr)} rows, {tr.object.nunique()} objects in {os.path.basename(trans)}')
    for name, n in nfail.items():
        print(f'  {name:13s}: {n:4d} rows')
    print(f'{len(good)} rows pass -> {len(best)} object-bands -> {len(sl)} objects')
    print(sl.tier.value_counts().sort_index().to_string())
    print()
    print(sl[['tier', 'object', 'bat_index', 'clasf', 'ref_band', 'snr', 'd', 'min_side',
              'corr_bands', 'g_over_r', 'ref_lc_file']]
          .to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print()
    print('overlay rows by tier x band:')
    print(ov.groupby(['tier', 'band']).size().to_string())

    tr.to_parquet(f_rows)
    sl.to_parquet(f_sl)
    ov.to_parquet(f_ov)
    print(f'\nwrote {os.path.basename(f_rows)}, {os.path.basename(f_sl)}, '
          f'{os.path.basename(f_ov)}')


if __name__ == '__main__':
    main()
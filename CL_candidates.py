#!/usr/bin/env python
"""
Vet CLAGN candidates by looking for changes in the ZTF continuum level.

Two statistics, both from the same data-adaptive changepoint scan:
  sustained : wide window (+/- W_SUS d), the expected CLAGN morphology
  jump      : narrow window (+/- W_JMP d) with a max-gap constraint, a fast step

Both compare medians of nightly-binned points either side of a candidate split,
normalised by the local scatter -- i.e. the null hypothesis is "ordinary AGN
variability", not "photometric noise".

Objects are independent, so the object loop is run over a process pool.
"""
import os
import re
import sys
import zlib
import glob
import argparse
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from VarTools import _find_target_obj, radec_filename

# ---- detection parameters -------------------------------------------------
W_SUS, NMIN_SUS, NSIG_SUS = 180.0, 5, 5.0   # sustained: half-window, min pts/side, sigma
W_JMP, NMIN_JMP, NSIG_JMP = 45.0, 3, 6.0    # jump
DTMAX = 25.0      # max gap across a jump split (d); keeps season gaps out
DMAG = 0.4        # min |delta mag| for either statistic
COINC = 30.0      # |t_jump - t_sus| below which the two are the same event (d)
MIN_NIGHTS = 30   # min nightly points per band to attempt a scan
ERRFLOOR = 0.01   # systematic added in quadrature to the magnitude error
SMOOTH = 15.0     # half-width (d) of the rolling median used to trace a transition
F_LO, F_HI = 0.1, 0.9   # transition = 10%..90% of the way from old to new level
MIN_SHADE_FRAC = 0.006  # plot-only floor on shaded width, as a fraction of the x-range

BANDS = {'g': 'mediumseagreen',
         'r': 'firebrick',
         'i': 'darkgoldenrod'}

MAG_COL = 'MAG_4_TOT_AB'
ERR_COL = 'MERR_4_TOT_AB'
MJD_CANDIDATES = ('mjd', 'MJD', 'obsmjd', 'OBSMJD', 'mjd_obs', 'MJD_OBS', 'jd', 'JD')


# ---------------------------------------------------------------- file index
def build_index(root):
    paths = glob.glob(os.path.join(root, '**', '*.parquet'), recursive=True)
    dups = [b for b, c in Counter(map(os.path.basename, paths)).items() if c > 1]
    if dups:
        raise RuntimeError(f'{len(dups)} non-unique basenames, e.g. {dups[:5]}')
    return {os.path.basename(p): p for p in paths}


def parse_name(fn):
    """'0.61008_+3.35193_000498_zi_ccd05_q3.parquet'
       -> ('0.61008_+3.35193', 0.61008, 3.35193, 'i')"""
    parts = os.path.basename(fn)[:-len('.parquet')].split('_')
    band = next(x for x in parts if re.fullmatch(r'z[gri]', x))[-1]
    return f'{parts[0]}_{parts[1]}', float(parts[0]), float(parts[1]), band


def assemble(cand_file, index):
    """Exactly the files in cand_file's lc_file column, grouped by object.

    Nothing is added, preferred, substituted or merged: every listed file that
    exists under the root is scanned as its own light curve. The index is used
    only to resolve basenames to paths.
    Returns {object_key: [(lc_file, path, band), ...]}, {object_key: (ra, dec)}.
    """
    cl = pd.read_parquet(cand_file)
    files = cl.lc_file.dropna().unique()
    missing = [f for f in files if f not in index]
    if missing:
        print(f'WARNING: {len(missing)}/{len(files)} lc_file entries not found '
              f'under root, skipped, e.g. {missing[:3]}', file=sys.stderr)

    keys, objects, skipped = {}, defaultdict(list), []
    for f in files:
        try:
            k, ra, dec, band = parse_name(f)
        except (StopIteration, ValueError, IndexError):
            print(f'WARNING: unparsable name {f}', file=sys.stderr)
            continue
        if band not in BANDS:
            skipped.append(f)
            continue
        keys[k] = (ra, dec)
        if f in index:
            objects[k].append((f, index[f], band))
    if skipped:
        print(f'WARNING: {len(skipped)} files in a band outside {list(BANDS)}, '
              f'skipped, e.g. {skipped[:3]}', file=sys.stderr)
    n = sum(len(v) for v in objects.values())
    print(f'{n} lc_files across {len(objects)} objects')
    return {k: sorted(v) for k, v in objects.items()}, keys


# ------------------------------------------------------------------ lightcurve
def load_band(paths, mag_column=MAG_COL, mag_err=ERR_COL):

    df   = pd.read_parquet(paths)
    ra, dec, band = radec_filename(paths, band=True)
    coord = SkyCoord(ra, dec, unit='deg')
    tgt_obj_idx = _find_target_obj(Path(paths), coord)

    if tgt_obj_idx is None:
        print(f'no target found in {os.path.basename(paths)}, skipping')
        return None

    ztf1 = df[df['object_index'] == tgt_obj_idx]

    mask = (ztf1[mag_err] < 0.5) & (ztf1['MAGLIM'] > 20.5) & (ztf1['SEEING'] < 3)

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
    else:
        return ztf1


def get_mjd(d):
    for c in MJD_CANDIDATES:
        if c in d.columns:
            v = np.asarray(d[c].values, dtype=float)
            return v - 2400000.5 if np.nanmedian(v) > 2400000 else v
    raise KeyError(f'no MJD column among {MJD_CANDIDATES}; '
                   f'columns are {list(d.columns)[:25]}')


def nightly(mjd, mag, err):
    """One point per night: kills single-epoch outliers before any differencing."""
    ok = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(err)
    t = pd.DataFrame({'night': np.floor(mjd[ok] - 0.5).astype('int64'),
                      'mjd': mjd[ok], 'mag': mag[ok], 'err': err[ok]})
    o = (t.groupby('night')
          .agg(mjd=('mjd', 'mean'), mag=('mag', 'median'),
               err=('err', lambda e: np.sqrt((e ** 2).sum()) / len(e)),
               n=('mag', 'size'))
          .sort_values('mjd'))
    return o.mjd.values, o.mag.values, o.err.values


# ------------------------------------------------------------------ detection
class Change(NamedTuple):
    d: float      # median(after) - median(before), mag; > 0 means fainter
    snr: float    # |d| / combined SE of the two medians
    t_lo: float   # last night before the split
    t_hi: float   # first night after the split
    nb: int       # nights in the before-window
    na: int       # nights in the after-window
    mb: float     # before-window median
    ma: float     # after-window median
    k: int        # index of t_lo; -1 if no valid split
    i0: int       # first index of the before-window
    i1: int       # one past the last index of the after-window


NO_CHANGE = Change(0.0, 0.0, np.nan, np.nan, 0, 0, np.nan, np.nan, -1, 0, 0)


def best_change(t, y, e, W, nmin, max_gap=np.inf):
    """Max-significance split point.

    t must be sorted ascending (nightly() guarantees it), so window bounds come
    from searchsorted and the statistics run on contiguous slices.
    """
    n = len(t)
    best = NO_CHANGE
    if n < 2 * nmin:
        return best
    gap = np.diff(t)
    lo = np.searchsorted(t, 0.5 * (t[:-1] + t[1:]) - W, side='left')
    hi = np.searchsorted(t, 0.5 * (t[:-1] + t[1:]) + W, side='right')
    for k in range(nmin - 1, n - nmin):
        if gap[k] > max_gap:
            continue
        i0, i1 = lo[k], hi[k]
        nb, na = k + 1 - i0, i1 - (k + 1)
        if nb < nmin or na < nmin:
            continue
        yb, ya = y[i0:k + 1], y[k + 1:i1]
        eb, ea = e[i0:k + 1], e[k + 1:i1]
        mb, ma = np.median(yb), np.median(ya)
        d = ma - mb
        # scatter-based SE of the median, floored at the propagated photometric
        # error so a quiet stretch cannot manufacture runaway significance
        sb = max(1.253 * yb.std(ddof=1) / np.sqrt(nb), np.sqrt((eb ** 2).sum()) / nb)
        sa = max(1.253 * ya.std(ddof=1) / np.sqrt(na), np.sqrt((ea ** 2).sum()) / na)
        s = np.hypot(sb, sa)
        snr = abs(d) / s if s > 0 else np.inf
        if snr > best.snr:
            best = Change(d, snr, t[k], t[k + 1], int(nb), int(na),
                          mb, ma, k, int(i0), int(i1))
    return best


def rolling_median(t, y, hw):
    lo = np.searchsorted(t, t - hw, side='left')
    hi = np.searchsorted(t, t + hw, side='right')
    return np.fromiter((np.median(y[a:b]) for a, b in zip(lo, hi)), float, len(t))


def transition_extent(t, y, c, hw=SMOOTH):
    """Where the light curve actually makes the move, as a 10-90% span.

    f = fractional progress from the before-median (f=0) to the after-median
    (f=1). Walking back from the split, the transition starts after the last
    night still at the old level (smoothed f <= F_LO); walking forward, it ends
    at the first night at the new level (smoothed f >= F_HI).

    Each side is smoothed on its own, so a sharp step is not smeared across the
    split by the smoothing window. resolved=False means one of the two levels
    was never reached inside its window and that edge is clipped to the window
    boundary -- typically a transition slower than W.
    """
    if c.k < 0 or c.d == 0:
        return np.nan, np.nan, False
    tb, ta = t[c.i0:c.k + 1], t[c.k + 1:c.i1]
    fb = (rolling_median(tb, y[c.i0:c.k + 1], hw) - c.mb) / c.d
    fa = (rolling_median(ta, y[c.k + 1:c.i1], hw) - c.mb) / c.d
    back = np.flatnonzero(fb <= F_LO)
    fwd = np.flatnonzero(fa >= F_HI)
    t0 = tb[back[-1]] if back.size else tb[0]
    t1 = ta[fwd[0]] if fwd.size else ta[-1]
    return float(t0), float(t1), bool(back.size and fwd.size)


def null_pvalue(t, y, e, W, nmin, max_gap, obs_snr, nperm, rng):
    """Empirical p for the max-SNR statistic: shuffle magnitudes, keep cadence."""
    if nperm <= 0 or not np.isfinite(obs_snr) or obs_snr <= 0:
        return np.nan
    ge = 0
    for _ in range(nperm):
        p = rng.permutation(len(y))
        if best_change(t, y[p], e[p], W, nmin, max_gap).snr >= obs_snr:
            ge += 1
    return (1 + ge) / (1 + nperm)


SUS_KINDS = ('sustained', 'fast_transition', 'both_separate')
JMP_KINDS = ('jump_only', 'fast_transition', 'both_separate')


def classify(sus, jmp):
    hit_s = abs(sus.d) >= DMAG and sus.snr >= NSIG_SUS
    hit_j = abs(jmp.d) >= DMAG and jmp.snr >= NSIG_JMP
    if not (hit_s or hit_j):
        return None
    if hit_s and hit_j:
        return ('fast_transition'
                if abs(jmp.t_lo - sus.t_lo) < COINC else 'both_separate')
    return 'sustained' if hit_s else 'jump_only'


# -------------------------------------------------------------------- plotting
MARKERS = ('.', 'x', '+', '^', 's', 'v')


def file_tag(fname, key):
    """'0.61008_+3.35193_000498_zi_ccd05_q3.parquet' -> '000498_zi_ccd05_q3'"""
    return fname[len(key) + 1:-len('.parquet')]


def plot_object(key, series, hits, outdir):
    """One line per lc_file. Shading = 10-90% transition span of each statistic
    that fired; dashed segments = the two window medians that were compared."""
    fig, ax = plt.subplots(figsize=(11, 5))
    nth = defaultdict(int)
    for fname, (band, t, y, e) in sorted(series.items()):
        m = MARKERS[nth[band] % len(MARKERS)]
        nth[band] += 1
        ax.errorbar(t, y, yerr=e, fmt=m, ms=4, c=BANDS[band],
                    label=file_tag(fname, key), alpha=0.8, zorder=2)
    x0, x1 = ax.get_xlim()
    minw = MIN_SHADE_FRAC * (x1 - x0)

    for fname, h in hits.items():
        t, col = series[fname][1], BANDS[h['band']]
        for tag, kinds, style in (('sus', SUS_KINDS, dict(alpha=0.20)),
                                  ('jmp', JMP_KINDS, dict(alpha=0.35, hatch='///'))):
            if h['kind'] not in kinds:
                continue
            c = h[tag]
            s0, s1, _ = h[f'ext_{tag}']
            if s1 - s0 < minw:                      # visual floor only
                mid = 0.5 * (s0 + s1)
                s0, s1 = mid - minw / 2, mid + minw / 2
            ax.axvspan(s0, s1, facecolor=col, edgecolor=col, lw=0, zorder=1, **style)
            for lvl, a, b in ((c.mb, t[c.i0], t[c.k]), (c.ma, t[c.k + 1], t[c.i1 - 1])):
                ax.hlines(lvl, a, b, colors='k', lw=3.2, zorder=3)
                ax.hlines(lvl, a, b, colors=col, lw=1.8, linestyles='--', zorder=4)
    ax.set_xlim(x0, x1)

    handles, _ = ax.get_legend_handles_labels()
    kinds = {h['kind'] for h in hits.values()}
    if kinds & set(SUS_KINDS):
        handles.append(Patch(facecolor='grey', alpha=0.20,
                             label='sustained: 10–90% transition'))
    if kinds & set(JMP_KINDS):
        handles.append(Patch(facecolor='grey', edgecolor='grey', alpha=0.35,
                             hatch='///', label='jump: 10–90% transition'))
    if hits:
        handles.append(Line2D([], [], color='grey', ls='--', lw=1.8,
                              label='compared window medians'))
    ax.legend(handles=handles, loc='best', fontsize=8)

    ax.set_xlabel('MJD')
    ax.set_ylabel('apparent magnitude (AB)')
    ax.invert_yaxis()
    tag = ', '.join(f"{file_tag(f, key)}:{h['kind']}"
                    for f, h in sorted(hits.items())) or 'no detection'
    ax.set_title(f'{key}  [{tag}]', fontsize=10)
    fig.savefig(os.path.join(outdir, f'{key}.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- worker unit
def scan_object(task):
    """One object: every one of its lc_files scanned independently.
    Returns (key, rows, traceback_or_None)."""
    key, files, (ra, dec), cfg = task
    try:
        # crc32, not hash(): str hashing is salted per process, so hash() would
        # make the permutation p-values irreproducible across runs
        rng = np.random.default_rng([cfg.seed, zlib.crc32(key.encode())])
        rows, series, hits = [], {}, {}
        for fname, path, band in files:
            try:
                d = load_band(path)
            except Exception as exc:
                print(f'WARNING: load_band failed on {fname}: {exc}', file=sys.stderr)
                continue
            if d is None or not len(d):
                continue
            t, y, e = nightly(get_mjd(d),
                              np.asarray(d[MAG_COL].values, dtype=float),
                              np.hypot(ERRFLOOR,
                                       np.asarray(d[ERR_COL].values, dtype=float)))
            if not len(t):
                continue
            series[fname] = (band, t, y, e)
            if len(t) < MIN_NIGHTS:
                continue
            sus = best_change(t, y, e, W_SUS, NMIN_SUS)
            jmp = best_change(t, y, e, W_JMP, NMIN_JMP, max_gap=DTMAX)
            kind = classify(sus, jmp)
            if kind is None:
                continue
            es = transition_extent(t, y, sus)
            ej = transition_extent(t, y, jmp)
            hits[fname] = {'band': band, 'kind': kind, 'sus': sus, 'jmp': jmp,
                           'ext_sus': es, 'ext_jmp': ej}
            rows.append({
                'object': key, 'RA': ra, 'DEC': dec,
                'lc_file': fname, 'band': band, 'kind': kind,
                'partial': not fname.endswith('_merged.parquet'),
                'n_nights': len(t),
                'd_sus': sus.d, 'snr_sus': sus.snr,
                'm_before_sus': sus.mb, 'm_after_sus': sus.ma,
                'mjd_sus_lo': sus.t_lo, 'mjd_sus_hi': sus.t_hi,
                'mjd_sus_start': es[0], 'mjd_sus_end': es[1],
                'sus_resolved': es[2],
                'n_before_sus': sus.nb, 'n_after_sus': sus.na,
                'd_jmp': jmp.d, 'snr_jmp': jmp.snr,
                'm_before_jmp': jmp.mb, 'm_after_jmp': jmp.ma,
                'mjd_jmp_lo': jmp.t_lo, 'mjd_jmp_hi': jmp.t_hi,
                'mjd_jmp_start': ej[0], 'mjd_jmp_end': ej[1],
                'jmp_resolved': ej[2],
                'n_before_jmp': jmp.nb, 'n_after_jmp': jmp.na,
                'p_sus': null_pvalue(t, y, e, W_SUS, NMIN_SUS, np.inf,
                                     sus.snr, cfg.nperm, rng),
                'p_jmp': null_pvalue(t, y, e, W_JMP, NMIN_JMP, DTMAX,
                                     jmp.snr, cfg.nperm, rng)})
        if series and (hits or cfg.plot_all):
            plot_object(key, series, hits, cfg.plotdir)
        return key, rows, None
    except Exception:
        return key, [], traceback.format_exc()


def _init_worker():
    """Each worker is already a core; stop pyarrow/BLAS spawning more inside it."""
    try:
        import pyarrow as pa
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    except Exception:
        pass


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', default=os.path.expanduser(
        '~/results/A_365_spl_CL_candidates.parquet'))
    ap.add_argument('--root', default=os.path.expanduser('~/BAT_results'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--plotdir', default=os.path.expanduser('~/results/CL_plots'))
    ap.add_argument('--plot-all', action='store_true',
                    help='plot every object with usable photometry, not only hits')
    ap.add_argument('--nperm', type=int, default=0,
                    help='permutation trials for empirical p-values (0 = off)')
    ap.add_argument('--limit', type=int, default=0,
                    help='scan only N objects (0 = all); output name gets _limN')
    ap.add_argument('--pick', choices=('first', 'random'), default='random',
                    help='how --limit selects objects')
    ap.add_argument('--objects', nargs='+', default=None,
                    help='scan only these object keys, e.g. 0.61008_+3.35193')
    ap.add_argument('--nproc', type=int,
                    default=int(os.environ.get('PBS_NP', 4)),
                    help='worker processes (default: $PBS_NP, else 4; 1 = serial)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    suffix = f'_lim{args.limit}' if args.limit and not args.out else ''
    out = args.out or args.cand.replace(
        '_CL_candidates.parquet', f'_CL_transitions{suffix}.parquet')
    os.makedirs(args.plotdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    index = build_index(args.root)
    objects, coords = assemble(args.cand, index)
    total = len(objects)

    items = sorted(objects.items())
    if args.objects:
        want = set(args.objects)
        unknown = want - {k for k, _ in items}
        if unknown:
            print(f'WARNING: not in the candidate set: {sorted(unknown)}',
                  file=sys.stderr)
        items = [x for x in items if x[0] in want]
    if args.limit and args.limit < len(items):
        if args.pick == 'random':
            idx = sorted(rng.choice(len(items), args.limit, replace=False))
            items = [items[i] for i in idx]
        else:
            items = items[:args.limit]

    nproc = max(1, min(args.nproc, len(items)))
    print(f'{len(items)} of {total} objects on {nproc} process(es) -> {out}',
          flush=True)
    tasks = [(k, bf, coords[k], args) for k, bf in items]

    rows, failed = [], []
    if nproc == 1:
        results = map(scan_object, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker)
        # chunksize=1: per-object cost varies by an order of magnitude with
        # epoch count, so static chunking would leave workers idle
        results = pool.map(scan_object, tasks, chunksize=1)

    for n, (key, r, err) in enumerate(results, 1):
        if err:
            failed.append(key)
            print(f'--- {key}\n{err}', file=sys.stderr)
        rows.extend(r)
        if n % (5 if len(tasks) <= 50 else 25) == 0:
            print(f'  {n}/{len(tasks)} scanned, {len(rows)} detections',
                  flush=True)
    if nproc > 1:
        pool.shutdown()

    res = pd.DataFrame(rows)
    res.to_parquet(out)
    print(f'\n{len(res)} detections over '
          f'{res.object.nunique() if len(res) else 0} objects -> {out}')
    if len(res):
        print(res.kind.value_counts().to_string())
    if failed:
        print(f'{len(failed)} objects failed: {failed[:10]}', file=sys.stderr)


if __name__ == '__main__':
    main()
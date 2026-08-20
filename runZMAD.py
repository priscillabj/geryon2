#!/usr/bin/env python3
"""Run zmad_metric over a subsample of ~/BAT_results in parallel on one node.

Embarrassingly parallel per file: no MPI, no collectives, no shared state.
The parent is the only writer, so the JSONL needs no lock.

  python run_zmad.py --limit 50 --band g --ncores 20
  python run_zmad.py --list subsample.txt --out ~/results/zmad_sub.parquet
"""

import argparse
import json
import os
import time
import traceback
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from ZMAD import zmad_metric

BAT_ROOT = Path(os.path.expanduser('~/BAT_results'))
KW = {}          # filled from CLI in main(), inherited by workers through fork
PLOT = {}        # {'dir': path, 'dpi': int} when --plot is given


def _one(path):
    """One source file -> list of row dicts (one per aperture, or one failure row)."""
    t0 = time.time()
    base = os.path.basename(path)
    try:
        agn, stars, out = zmad_metric(path, verbose=False, **KW)
    except Exception as exc:
        return [{'file': base, 'ok': False, 'fail_reason': f'{type(exc).__name__}: {exc}',
                 'traceback': traceback.format_exc(limit=-5), 'secs': time.time() - t0}]
    if out is None:
        return [{'file': base, 'ok': False, 'fail_reason': 'no result',
                 'secs': time.time() - t0}]

    png = {}
    if PLOT:
        from zmad_plots import save_all         # imported only in plotting runs
        for tag in [t for t, m in out['metrics'].items() if m is not None]:
            png[tag] = save_all(agn, stars, out, tag=tag,
                                outdir=PLOT['dir'], dpi=PLOT['dpi'])

    rows, dt = [], time.time() - t0
    for tag, m in out['metrics'].items():
        if m is None:
            rows.append({'file': base, 'aperture': tag, 'ok': False,
                         'fail_reason': 'aperture rejected', 'secs': dt})
            continue
        rows.append({'file': base, 'ok': True, 'aperture': tag, 'secs': dt,
                     'RA': out['RA'], 'DEC': out['DEC'], 'band': out['band'],
                     'CCDquadID': out['CCDquadID'], 'object_index': int(out['object_index']),
                     'space': out['space'], 'agg': out['agg'], 'png': png.get(tag),
                     **{k: (float(v) if hasattr(v, 'item') else v)
                        for k, v in m.items() if not isinstance(v, pd.Series)}})
    return rows


def select_files(args):
    if args.list:
        files = [l.strip() for l in open(args.list) if l.strip() and not l.startswith('#')]
        files = [f if os.path.isabs(f) else str(BAT_ROOT / f) for f in files]
    else:
        files = sorted(glob(str(BAT_ROOT / f'*_z{args.band}_merged.parquet')))
    if args.limit:
        files = files[args.offset:args.offset + args.limit]
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', help='file of basenames or paths, one per line')
    p.add_argument('--band', default='g')
    p.add_argument('--limit', type=int) #N_SOURCES
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--ncores', type=int,
                   default=int(os.environ.get('PBS_NP', os.cpu_count())))
    p.add_argument('--out', default=os.path.expanduser('~/results/zmad.parquet'))
    p.add_argument('--apertures', default='MAG_4_TOT_AB')
    p.add_argument('--flux', action='store_true')
    p.add_argument('--no-sigma-filter', action='store_true')
    p.add_argument('--save-sigma', action='store_true',
                   help='let run_sigma_filtering write its pkl/png per source')
    p.add_argument('--x', type=int, default=100,
                   help='passed to quality_cuts; omit to use the ZMAD.py default')
    p.add_argument('--plot', metavar='DIR', help='write a PNG per source into DIR')
    p.add_argument('--dpi', type=int, default=110)
    p.add_argument('--restart', action='store_true', help='ignore the existing JSONL')
    p.add_argument('--retry-failed', action='store_true',
                   help='re-run files that produced no ok row')
    args = p.parse_args()

    KW.update(mag_columns=tuple(args.apertures.split(',')),
              flux=args.flux,
              sigma_filter=not args.no_sigma_filter,
              save_sigma=args.save_sigma)
    if args.x is not None:          # otherwise zmad_metric's own default applies
        KW['x'] = args.x

    if args.plot:
        os.makedirs(os.path.expanduser(args.plot), exist_ok=True)
        PLOT.update(dir=os.path.expanduser(args.plot), dpi=args.dpi)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out_path.with_suffix('.jsonl')

    files = select_files(args)
    if args.restart and jsonl.exists():
        jsonl.unlink()          # the jsonl is the source of truth, not the parquet
        
    done = set()
    if jsonl.exists() and not args.restart:
        with open(jsonl) as fh:
            rec = [json.loads(l) for l in fh if l.strip()]
        if args.retry_failed:
            done = {r['file'] for r in rec if r.get('ok')}
        else:
            done = {r['file'] for r in rec}
    todo = [f for f in files if os.path.basename(f) not in done]

    print(f'{len(files)} selected, {len(done)} already done, {len(todo)} to run '
          f'on {args.ncores} cores', flush=True)
    if not todo:
        return

    t0 = time.time()
    with open(jsonl, 'a') as fh, Pool(args.ncores, maxtasksperchild=8) as pool:
        for i, rows in enumerate(pool.imap_unordered(_one, todo, chunksize=1), 1):
            for r in rows:
                fh.write(json.dumps(r) + '\n')
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f'{i}/{len(todo)}  {el/60:.1f} min  '
                      f'eta {el/i*(len(todo)-i)/60:.1f} min', flush=True)

    with open(jsonl) as fh:
        df = pd.DataFrame([json.loads(l) for l in fh if l.strip()])
    if 'aperture' in df:            # a retried file appends new rows; keep the newest
        df = df.drop_duplicates(subset=['file', 'aperture'], keep='last')
    tmp = out_path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)

    ok = df.get('ok', pd.Series(dtype=bool)).fillna(False)
    print(f'wrote {out_path}  ({int(ok.sum())} ok rows, {int((~ok).sum())} failed)')
    if (~ok).any() and 'fail_reason' in df:
        print(df.loc[~ok, 'fail_reason'].value_counts().head(10))


if __name__ == '__main__':
    main()
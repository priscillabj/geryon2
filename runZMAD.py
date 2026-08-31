#!/usr/bin/env python3
"""Run zmad_metric over a subsample of ~/BAT_results in parallel on one node.
 
Embarrassingly parallel per file: no MPI, no collectives, no shared state.
The parent is the only writer, so the JSONL needs no lock.
 
  python run_zmad.py --limit 50 --band g --ncores 20
  python run_zmad.py --list subsample.txt --out ~/results/zmad_sub.parquet
"""
 
import argparse
import json
import re
import resource
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
 
 
def _rss_mb():
    """Peak RSS of this worker, MB. ru_maxrss is KB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
 
 
def _one(path):
    """One source file -> list of row dicts (one per aperture, or one failure row)."""
    t0 = time.time()
    base = os.path.basename(path)
    try:
        mb_file = os.path.getsize(path) / 1024.0 ** 2
    except OSError:
        mb_file = None
    try:
        agn, stars, out = zmad_metric(path, verbose=False, **KW)
    except Exception as exc:
        return [{'file': base, 'ok': False, 'fail_reason': f'{type(exc).__name__}: {exc}',
                 'traceback': traceback.format_exc(limit=-5), 'secs': time.time() - t0,
                 'mb_file': mb_file, 'rss_mb': _rss_mb()}]
    if out is None:
        return [{'file': base, 'ok': False, 'fail_reason': 'no result',
                 'secs': time.time() - t0, 'mb_file': mb_file, 'rss_mb': _rss_mb()}]
 
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
                         'fail_reason': 'aperture rejected', 'secs': dt,
                         'mb_file': mb_file, 'rss_mb': _rss_mb()})
            continue
        rows.append({'file': base, 'ok': True, 'aperture': tag, 'secs': dt,
                     'mb_file': mb_file, 'rss_mb': _rss_mb(),
                     'RA': out['RA'], 'DEC': out['DEC'], 'band': out['band'],
                     'CCDquadID': out['CCDquadID'], 'object_index': int(out['object_index']),
                     'space': out['space'], 'agg': out['agg'], 'png': png.get(tag),
                     **{k: (float(v) if hasattr(v, 'item') else v)
                        for k, v in m.items() if not isinstance(v, pd.Series)}})
    return rows
 
 
FILE_KEYS = ('file', 'lc_file', 'path', 'filename')      # accepted column/key names
 
 
def _from_records(recs, src):
    """Pull filenames out of a list of strings or a list of dicts."""
    if not recs:
        return []
    if isinstance(recs[0], str):
        return list(recs)
    for k in FILE_KEYS:
        if k in recs[0]:
            return [r[k] for r in recs]
    raise KeyError(f'{src}: no {FILE_KEYS} key; got {list(recs[0])[:8]}')
 
 
def read_list(path):
    """Read a subsample list from .json, .csv, .parquet, or plain text.
 
    JSON may be a list of names, a list of records, or {"files": [...]}.
    Tabular formats need one of FILE_KEYS as a column.  Duplicates are dropped,
    order is preserved.
    """
    path = os.path.expanduser(path)
    ext = os.path.splitext(path)[1].lower()
 
    if ext == '.json':
        with open(path) as fh:
            obj = json.load(fh)
        if isinstance(obj, dict):
            obj = obj.get('files', next(iter(obj.values())))
        files = _from_records(obj, path)
    elif ext in ('.csv', '.tsv'):
        t = pd.read_csv(path, sep='\t' if ext == '.tsv' else ',')
        files = _from_records(t.to_dict('records'), path)
    elif ext == '.parquet':
        files = _from_records(pd.read_parquet(path).to_dict('records'), path)
    elif ext == '.jsonl':
        with open(path) as fh:
            files = _from_records([json.loads(l) for l in fh if l.strip()], path)
    else:
        with open(path) as fh:
            files = [l.strip() for l in fh if l.strip() and not l.startswith('#')]
 
    return list(dict.fromkeys(str(f) for f in files if f and str(f) != 'nan'))
 
 
def select_files(args):
    if args.list:
        files = read_list(args.list)
        files = [f if os.path.isabs(f) else str(BAT_ROOT / f) for f in files]
        if args.band:                 # a list may span bands; keep only one
            files = [f for f in files
                     if re.search(rf'_z{args.band}_', os.path.basename(f))]
    else:
        files = sorted(glob(str(BAT_ROOT / f'*_z{args.band or "g"}_merged.parquet')))
    if args.limit:
        files = files[args.offset:args.offset + args.limit]
    return files

def write_parquet(jsonl, out_path):
    with open(jsonl) as fh:
        df = pd.DataFrame([json.loads(l) for l in fh if l.strip()])
    if 'aperture' in df:            # a retried file appends new rows; keep the newest
        df = df.drop_duplicates(subset=['file', 'aperture'], keep='last')
        # a file-level failure row (aperture is null) is stale once the file
        # has produced any real row -- otherwise it survives the dedup forever
        good = set(df.loc[df['ok'] == True, 'file'])                    # noqa: E712
        df = df[~(df['aperture'].isna() & df['file'].isin(good))]
    tmp = out_path.with_suffix('.parquet.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
 
    ok = df.get('ok', pd.Series(dtype=bool)).fillna(False)
    print(f'wrote {out_path}  ({int(ok.sum())} ok rows, {int((~ok).sum())} failed)')
    for c in ('secs', 'mb_file', 'rss_mb'):
        if c in df and df[c].notna().any():
            q = df[c].describe(percentiles=[0.5, 0.95])
            print(f'  {c:8s} median {q["50%"]:.1f}  p95 {q["95%"]:.1f}  max {q["max"]:.1f}')
    if (~ok).any() and 'fail_reason' in df:
        print(df.loc[~ok, 'fail_reason'].value_counts().head(10))
 
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', help='subsample: .txt (one per line), .json, .jsonl, '
                                  '.csv or .parquet with a file/lc_file column')
    p.add_argument('--band', default=None,
                   help='glob mode: which band to select (default g). '
                        'list mode: keep only this band; omit to run all bands '
                        'present, each read from its own filename.')
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
    p.add_argument('--order', choices=('size', 'list', 'shuffle'), default='size',
                   help="'size' runs the largest files first (default): the cost "
                        "distribution has a long tail, and starting the heavy ones "
                        "last strands the whole node waiting on them")
    p.add_argument('--rebuild', action='store_true',
                   help='rebuild the parquet from the JSONL and exit (no compute); '
                        'use after a walltime kill, which skips the final write')
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

    if args.rebuild:
        write_parquet(jsonl, out_path)
        return

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
    if args.order == 'size':
        todo.sort(key=lambda f: -(os.path.getsize(f) if os.path.exists(f) else 0))
    elif args.order == 'shuffle':
        import random
        random.Random(0).shuffle(todo)
 
    mb = [os.path.getsize(f) / 1024.0**2 for f in todo if os.path.exists(f)]
    print(f'{len(files)} selected, {len(done)} already done, {len(todo)} to run '
          f'on {args.ncores} cores  (order={args.order})', flush=True)
    if mb:
        print(f'  input size: total {sum(mb)/1024:.1f} GB, largest {max(mb):.0f} MB, '
              f'median {sorted(mb)[len(mb)//2]:.0f} MB', flush=True)
        
    print(f'{len(files)} selected, {len(done)} already done, {len(todo)} to run '
          f'on {args.ncores} cores', flush=True)
    if not todo:
        return

    every = max(1, min(25, len(todo) // 20))
    t0 = time.time()
    with open(jsonl, 'a') as fh, Pool(args.ncores, maxtasksperchild=8) as pool:
        for i, rows in enumerate(pool.imap_unordered(_one, todo, chunksize=1), 1):
            for r in rows:
                fh.write(json.dumps(r) + '\n')
            fh.flush()
            if i % every == 0 or i == len(todo):
                el = time.time() - t0
                print(f'{i}/{len(todo)}  {el/60:.1f} min  '
                      f'eta {el/i*(len(todo)-i)/60:.1f} min  '
                      f'parent_rss {_rss_mb():.0f} MB  '
                      f'last {rows[0].get("secs", 0):.0f}s '
                      f'{rows[0].get("mb_file") or 0:.0f} MB file', flush=True)
 
    write_parquet(jsonl, out_path)
 
if __name__ == '__main__':
    main()
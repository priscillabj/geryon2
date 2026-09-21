"""Plots for zmad_bat.zmad_metric output.

Kept out of zmad_bat so the batch driver never imports matplotlib.
All three take the (agn, stars, out) triple returned by zmad_metric; nothing is
recomputed here, so the figures cannot disagree with out['metrics'].

    agn, stars, out = zmad_metric(f)
    zmad_hist(out)                    # one ZMAD histogram per aperture
    zmad_plots(agn, stars, out, '4')  # light curve + histogram for one aperture
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAND_COLOR = {'g': 'mediumseagreen', 'r': 'firebrick', 'i': 'goldenrod'}
STAR_COLOR = '#9F9F9F'


def _save(fig, save, dpi=150):
    """Write fig to `save` (creating parent dirs) and close it; return the path."""
    if not save:
        return None
    import os
    path = os.path.expanduser(save)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')
    return path


def _tags(out, tags):
    """Apertures that actually produced a result, in mag_columns order."""
    ok = [t for t, m in out['metrics'].items() if m is not None]
    if tags is None:
        return ok
    tags = [tags] if isinstance(tags, str) else list(tags)
    missing = [t for t in tags if t not in ok]
    if missing:
        raise KeyError(f'no metrics for aperture(s) {missing}; have {ok}')
    return tags


def _title(out, tag):
    return (f"{out['RA']:.5f} {out['DEC']:+.5f}  z{out['band']}  "
            f"CCD {out['CCDquadID']}  ap {tag}")


def _err(frame, m):
    """Error bars in the plotted space (mag, or uJy via sigma_f = 0.921 f sigma_m)."""
    merr = m['col'].replace('MAG_', 'MERR_', 1)
    if merr not in frame:
        return None
    if m['val'] == m['col']:
        return frame[merr]
    return 0.4 * np.log(10) * frame[m['val']] * frame[merr]


def _stats_box(ax, m, x=0.02, y=0.98):
    ax.text(x, y,
            f"stars: {m['cs_mean']:.2f} ± {m['cs_std']:.2f}\n"
            f"max  : {m['cs_max']:.2f}\n"
            f"AGN  : {m['agn']:.2f}  ({m['sigma']:.1f}σ, {m['perc']:.0f}%)",
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _hist_panel(ax, out, tag, color):
    """Star ZMAD distribution with the AGN marked."""
    m = out['metrics'][tag]
    v = m['star_stat'].values
    n_pre = m.get('n_stars_preclip', len(v))
    clip_note = f'; {n_pre - len(v)} clipped' if n_pre > len(v) else ''
    ax.hist(v, bins=min(30, max(10, len(v) // 5)), alpha=0.7,
            color=STAR_COLOR, edgecolor='black',
            label=f'stars in stats: {len(v)}/{n_pre}{clip_note}')

    mu, sd = m['cs_mean'], m['cs_std']
    ax.axvline(mu, color='black', linestyle='-.', linewidth=2)
    for k, c in ((1, 'black'), (3, 'red')):
        ax.axvline(mu + k * sd, color=c, linestyle='dotted', linewidth=2)
        ax.axvline(mu - k * sd, color=c, linestyle='dotted', linewidth=2)
    ax.axvline(m['agn'], color=color, linestyle='--', linewidth=2,
               label=f"AGN {m['agn']:.2f} ({m['sigma']:.1f}σ)")

    ax.set_xlabel(f"ZMAD ({out['agg']} over {m['n_epochs']} epochs)", fontsize=11)
    ax.set_ylabel('number of sources', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    _stats_box(ax, m)


def _lc_panel(ax, a, s, out, tag, color):
    """Target light curve over the comparison-star cloud."""
    m = out['metrics'][tag]
    val, flux = m['val'], out['space'] == 'flux'

    ax.errorbar(s['mjd'], s[val], yerr=_err(s, m), fmt='o', ms=3,
                c=STAR_COLOR, alpha=0.15, zorder=1,
                label=f"stars plotted: {s['object_index'].nunique()}")
    ax.errorbar(a['mjd'], a[val], yerr=_err(a, m), fmt='o', ms=4,
                color=color, zorder=3, label=f"AGN (z{out['band']})")

    ax.set_xlabel('MJD', fontsize=11)
    ax.set_ylabel(f'flux [µJy] (ap {tag})' if flux else f'{m["col"]}', fontsize=11)
    if not flux:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def zmad_hist(out, tags=None, figsize=(5.5, 4)):
    """One ZMAD histogram per aperture."""
    tags = _tags(out, tags)
    color = BAND_COLOR.get(out['band'], 'mediumseagreen')
    fig, axes = plt.subplots(len(tags), 1,
                             figsize=(figsize[0], figsize[1] * len(tags)),
                             squeeze=False)
    axes = axes.flatten()
    for ax, tag in zip(axes, tags):
        _hist_panel(ax, out, tag, color)
        ax.set_title(_title(out, tag), fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig, axes


def zmad_plots(agn, stars, out, tag=None, figsize=(12, 5), save=None, dpi=150):
    """Light curve + ZMAD histogram for one aperture (default: the first)."""
    tag = _tags(out, tag)[0]
    color = BAND_COLOR.get(out['band'], 'mediumseagreen')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _lc_panel(axes[0], agn[tag], stars[tag], out, tag, color)
    _hist_panel(axes[1], out, tag, color)
    fig.suptitle(f"{_title(out, tag)}   obj {out['object_index']}",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, save, dpi)
    return fig, axes


# plot_results was the PSF-vs-AP 2x2; with no PSF it is just zmad_plots.
plot_results = zmad_plots


def save_all(agn, stars, out, tag=None, outdir='.', dpi=120):
    """zmad_plots -> <outdir>/zmad_<ra>_<dec>_z<band>_ap<tag>.png"""
    import os
    fig, _ = zmad_plots(agn, stars, out, tag)
    tag = _tags(out, tag)[0]
    path = os.path.join(outdir,
                        f"zmad_{out['RA']}_{out['DEC']}_z{out['band']}_ap{tag}.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------- population


def zmad_summary(df, aperture=None, prefix='', sigma_cut=3.0, clip=99.5,
                 figsize=(12, 9), save=None, dpi=150):
    """Population view of a zmad results parquet (no recompute needed).

        df = pd.read_parquet('~/results/zmad_sub_g.parquet')
        zmad_summary(df)

    Panels: sigma distribution, percentile distribution, and sigma against
    n_stars and n_epochs -- the last two are the ones that matter, because a
    trend there means the metric is tracking the comparison sample rather than
    the source.
    """
    sig, perc = f'{prefix}sigma', f'{prefix}perc'
    keys = (f'{prefix}n_stars', f'{prefix}n_epochs')
    d = _ok(df, f'{prefix}ok').copy()
    if 'aperture' in d:
        aperture = aperture if aperture is not None else sorted(d['aperture'].dropna())[0]
        d = d[d['aperture'] == aperture]
    d = d[np.isfinite(d[sig])]
    if d.empty:
        raise ValueError('no finite sigma rows')

    hi = np.percentile(d[sig], clip)
    n_over = int((d[sig] > hi).sum())
    n_det = int((d[sig] > sigma_cut).sum())

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    a = ax[0, 0]
    a.hist(d[sig].clip(upper=hi), bins=50, color=STAR_COLOR, edgecolor='black')
    a.axvline(sigma_cut, color='red', linestyle='--', lw=2,
              label=f'{sigma_cut:g}σ  ({n_det}/{len(d)} = {100*n_det/len(d):.0f}%)')
    a.set(xlabel=f'σ (clipped at {hi:.1f}; {n_over} above)',
          ylabel='sources', yscale='log')
    a.legend(fontsize=9)

    a = ax[0, 1]
    a.hist(d[perc], bins=50, color=STAR_COLOR, edgecolor='black')
    a.set(xlabel='percentile of AGN within its star pool', ylabel='sources')

    for a, key in zip((ax[1, 0], ax[1, 1]), keys):
        if key not in d:
            a.set_axis_off()
            continue
        a.scatter(d[key], d[sig].clip(upper=hi), s=8, alpha=0.4, color='black')
        a.axhline(sigma_cut, color='red', linestyle='--', lw=1)
        a.set(xlabel=key, ylabel='σ', yscale='symlog')
        if d[key].nunique() > 2:
            r = np.corrcoef(d[key], d[sig])[0, 1]
            a.set_title(f'r = {r:+.2f}', fontsize=10,
                        color='red' if abs(r) > 0.3 else 'black')

    fig.suptitle(f'ZMAD population  |  aperture {aperture}  |  N = {len(d)}',
                 fontsize=13)
    fig.tight_layout()
    _save(fig, save, dpi)
    return fig, ax


# ------------------------------------------------------------ by AGN type

# Seyfert palette; extend or override with the `colors` argument.
TYPE_COLOR = {
    'Sy1': '#9F9F9F', 'Sy1.2': 'white', 'Sy1.5': '#E69F00',
    'Sy1.8': '#04D9FF', 'Sy1.9': '#56B4E9', 'Sy2': '#0072B2',
}
TYPE_HATCH = {'Sy1': '', 'Sy1.2': '///'}

# coarse bins, in plotting order; shared by the histogram and the fraction plot
TYPE_GROUPS = {
    'Type 1 (Sy1-1.2)': ['Sy1', 'Sy1.2'],
    'Sy1.5': ['Sy1.5'],
    'Type 2 (Sy1.8-1.9-2)': ['Sy1.8', 'Sy1.9', 'Sy2'],
}
GROUP_COLOR = {'Type 1 (Sy1-1.2)': 'black', 'Sy1.5': '#E69F00',
               'Type 2 (Sy1.8-1.9-2)': '#56B4E9'}
# linestyle for groups drawn as unfilled steps (see `outline`)
GROUP_STYLE = {'Type 1 (Sy1-1.2)': '--', 'Type 2 (Sy1.8-1.9-2)': '-'}


def _ok(d, ok_col):
    """Drop failed rows when an ok column exists; pass everything through if not."""
    return d[d[ok_col] == True] if ok_col in d else d                  # noqa: E712


def zmad_census(d, type_col='type', prefix='', sigma_col=None, ok_col=None,
                fail_col=None, groups=None, log=True, verbose=True):
    """Where rows go between the input table and the histogram.

    Returns (funnel, fails): `funnel` is a Series of surviving counts after each
    filter, `fails` is the fail_reason breakdown of the rows dropped by the ok
    cut (empty if there is no fail column).
    """
    sigma_col = sigma_col or f'{prefix}sigma'
    ok_col = ok_col or f'{prefix}ok'
    fail_col = fail_col or f'{prefix}fail_reason'
    groups = groups or TYPE_GROUPS
    lut = {t: g for g, m in groups.items() for t in m}

    step = {'rows in table': len(d)}
    fails = pd.Series(dtype='int64')
    if ok_col in d:
        bad = d[d[ok_col] != True]                                     # noqa: E712
        if fail_col in d and len(bad):
            fails = bad[fail_col].fillna('(no reason recorded)').value_counts()
        d = d[d[ok_col] == True]                                       # noqa: E712
        step[f'{ok_col} is True'] = len(d)
    if type_col in d:
        d = d[d[type_col].notna()]
        step[f'{type_col} present'] = len(d)
    d = d[np.isfinite(d[sigma_col])]
    step[f'{sigma_col} finite'] = len(d)
    d = d[d[type_col].map(lut).notna()]
    step['class in groups'] = len(d)
    if log:
        d = d[d[sigma_col] > 0]
        step[f'{sigma_col} > 0 (log)'] = len(d)

    funnel = pd.Series(step)
    if verbose:
        out = pd.DataFrame({'kept': funnel, 'lost': -funnel.diff().fillna(0).astype(int)})
        print(out.to_string())
        if len(fails):
            print('\nreasons for not-ok rows:')
            print(fails.to_string())
    return funnel, fails


def _typed(d, type_col, sigma_col, ok_col):
    """Rows with a usable type and a finite statistic."""
    for c in (type_col, sigma_col):
        if c not in d:
            raise KeyError(f'no {c!r} column; columns are {list(d.columns)[:12]}...')
    d = _ok(d, ok_col)
    return d[d[type_col].notna() & np.isfinite(d[sigma_col])]


def zmad_type_hist(d, type_col='type', prefix='', sigma_col=None, ok_col=None,
                   cut=10.0, log=True, groups=None, colors=None,
                   outline=('Type 1 (Sy1-1.2)', 'Type 2 (Sy1.8-1.9-2)'),
                   linestyles=None, lw=1.8, bins=20, figsize=(7, 5),
                   ax=None, save=None, dpi=150):
    """Sigma distribution by coarse AGN type, with an optional cut line.

    Fine classes are collapsed into `groups` (default: Type 1 / Sy1.5 / Type 2),
    so the panel carries three histograms rather than one per subclass.  Groups
    named in `outline` are drawn as dashed steps instead of filled.

    Any class label not listed in `groups` is dropped and reported -- worth
    reading, since BASS spellings vary ('Sy1' vs 'Sy1.0').
    """
    sigma_col = sigma_col or f'{prefix}sigma'
    ok_col = ok_col or f'{prefix}ok'
    groups = groups or TYPE_GROUPS
    colors = {**GROUP_COLOR, **(colors or {})}
    linestyles = {**GROUP_STYLE, **(linestyles or {})}
    d_in = d
    d = _typed(d, type_col, sigma_col, ok_col)

    lut = {t: g for g, members in groups.items() for t in members}
    d = d.assign(_grp=d[type_col].map(lut))
    lost = d.loc[d['_grp'].isna(), type_col].value_counts()
    d = d[d['_grp'].notna()]
    if d.empty:
        raise ValueError(f'no rows matched {list(groups)}; labels present: '
                         f'{list(lost.index)[:10]}')

    v = d[sigma_col]
    if log:
        n_bad = int((v <= 0).sum())
        d = d[v > 0]
        d = d.assign(_x=np.log10(d[sigma_col]))
        xlabel = f'log {sigma_col}'
    else:
        n_bad, xlabel = 0, sigma_col
        d = d.assign(_x=d[sigma_col])

    edges = np.histogram_bin_edges(d['_x'], bins=bins)
    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    ax = ax or fig.add_subplot(111)

    for g in groups:
        xs = d.loc[d['_grp'] == g, '_x']
        if xs.empty:
            continue
        if g in outline:
            ax.hist(xs, bins=edges, histtype='step',
                    linestyle=linestyles.get(g, '--'),
                    color=colors.get(g, 'black'), lw=lw,
                    label=f'{g}  n={len(xs)}')
        else:
            ax.hist(xs, bins=edges, color=colors.get(g), alpha=0.65,
                    edgecolor='black', lw=0.4, label=f'{g}  n={len(xs)}')

    if cut:
        ax.axvline(np.log10(cut) if log else cut, color='black',
                   linestyle='--', lw=2.5)
        n = int((d[sigma_col] > cut).sum())
        ax.set_title(f'{sigma_col} > {cut:g}: {n}/{len(d)} '
                     f'({100*n/len(d):.0f}%)', fontsize=10)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('#sources', fontsize=12)
    funnel, fails = zmad_census(d_in, type_col=type_col, sigma_col=sigma_col,
                                ok_col=ok_col, groups=groups, log=log, verbose=False)
    n_in, n_plot = int(funnel.iloc[0]), int(funnel.iloc[-1])
    notes = [f'{n_plot} of {n_in} rows plotted']
    for k, v in (-funnel.diff().dropna()).items():
        if v:
            notes.append(f'  -{int(v)} failed: {k}')
    if len(lost):
        notes.append('unmatched labels: ' + ', '.join(
            f'{k} x{v}' for k, v in lost.items()))
    # ax.annotate('\n'.join(notes), (0.02, 0.02), xycoords='axes fraction',
    #             fontsize=7.5, color='firebrick', va='bottom')
    print('\n'.join(notes))
    if len(fails):
        print('reasons for not-ok rows:\n' + fails.to_string())
    ax.legend(fontsize=9, loc='upper left')
    _save(fig, save, dpi)
    return fig, ax


def _annot(kind, n, denom, pct):
    """Bar label: 'pct' -> 41%, 'n' -> n=215, 'both' -> 41% (88/215), None -> nothing."""
    if kind == 'pct':
        return f'{pct:.0f}%'
    if kind == 'n':
        return f'n={denom}'
    if kind == 'both':
        return f'{pct:.0f}%\n({n}/{denom})'
    return None


def zmad_type_fraction(d, type_col='type', prefix='', sigma_col=None, ok_col=None,
                       cut=10.0, groups=None, mode='composition', colors=None,
                       annot='pct', figsize=(7, 5), ax=None, save=None, dpi=150):
    """Type make-up of the sources passing the cut, stacked by fine type.

    mode='composition' : each bar is a share of the above-cut sample; bars sum
                         to 100%.  This is the published figure.
    mode='rate'        : each group's own sample is 100%; the stacked segments
                         split that group rate by fine type, and n= is the group
                         denominator.
    mode='type_rate'   : one unstacked bar per fine type, each as a fraction of
                         that type's OWN sample.  Answers 'what fraction of
                         Sy1.9 vary', which 'rate' does not.

    annot : bar label for the rate modes -- 'pct' (default), 'n', 'both', None.
    """
    sigma_col = sigma_col or f'{prefix}sigma'
    ok_col = ok_col or f'{prefix}ok'
    colors = {**TYPE_COLOR, **(colors or {})}
    groups = groups or TYPE_GROUPS
    d = _typed(d, type_col, sigma_col, ok_col)
    above = d[d[sigma_col] > cut]

    fig = ax.figure if ax is not None else plt.figure(figsize=figsize)
    ax = ax or fig.add_subplot(111)

    seen = set()
    if mode == 'type_rate':
        # one bar per fine type, each against its OWN sample size.  Independent
        # rates, so they must not be stacked: two types at 20% would otherwise
        # read as 40%.
        for i, (label, members) in enumerate(groups.items()):
            present = [t for t in members if (d[type_col] == t).any()]
            if not present:
                continue
            w = 0.76 / len(present)
            for j, t in enumerate(present):
                denom = int((d[type_col] == t).sum())
                n = int((above[type_col] == t).sum())
                h = 100.0 * n / denom
                x = i - 0.38 + w * (j + 0.5)
                ax.bar(x, h, width=w * 0.88, color=colors.get(t),
                       edgecolor='black', hatch=TYPE_HATCH.get(t, ''),
                       label=None if t in seen else t)
                seen.add(t)
                lab = _annot(annot, n, denom, h)
                if lab:
                    ax.annotate(lab, (x, h), ha='center', va='bottom',
                                fontsize=7.5, color='dimgray')
    else:
        for i, (label, members) in enumerate(groups.items()):
            denom = len(above) if mode == 'composition' else \
                len(d[d[type_col].isin(members)])
            bottom = 0.0
            for t in members:
                n = int((above[type_col] == t).sum())
                if not n or not denom:
                    continue
                h = 100.0 * n / denom
                ax.bar(i, h, bottom=bottom, width=0.72, color=colors.get(t),
                       edgecolor='black', linestyle='--' if t == 'Sy1' else '-',
                       hatch=TYPE_HATCH.get(t, ''),
                       label=None if t in seen else t)
                seen.add(t)
                bottom += h
            if mode == 'rate' and denom:
                lab = _annot(annot, int(round(bottom * denom / 100)), denom, bottom)
                if lab:
                    ax.annotate(lab, (i, bottom), ha='center', va='bottom',
                                fontsize=8, color='dimgray')

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.keys())
    ax.set_ylabel({'composition': 'Fraction of sources above cut',
                   'rate': 'Fraction of each group above cut',
                   'type_rate': 'Fraction of each type above cut'}[mode],
                  fontsize=12)
    ax.yaxis.set_major_formatter(lambda y, _: f'{y:.0f}%')
    ax.set_title(f'{sigma_col} > {cut:g}  ({len(above)} sources, mode={mode})',
                 fontsize=10)
    ax.legend(title='Type', fontsize=9, ncol=2)
    _save(fig, save, dpi)
    return fig, ax


# ---------------------------------------------------------------------- CLI


def _main():
    import argparse
    import os

    import pandas as pd

    p = argparse.ArgumentParser(
        description='Plot ZMAD results. Population panels need only the parquet; '
                    'per-source figures re-run zmad_metric on the chosen files.')
    p.add_argument('parquet', help='output of run_zmad.py')
    p.add_argument('--aperture', default=None)
    p.add_argument('--band', help="keep only this band ('g', 'r', 'i'; 'zg' also "
                                  "accepted). A run over all bands mixes them in "
                                  "one table and sigma is not comparable across "
                                  "them, so a per-band figure is usually what you "
                                  "want. The band is appended to output filenames.")
    p.add_argument('--prefix', default='', help="e.g. 'zmad_' for bat_master.parquet")
    p.add_argument('--types', metavar='COL',
                   help='class column (e.g. clasf) -> also write the type figures')
    p.add_argument('--cut', type=float, default=10.0, help='sigma cut for the type figures')
    p.add_argument('--only', metavar='LABEL',
                   help='restrict the type histogram to the group whose label '
                        "contains LABEL (e.g. 'Type 2'), or to a bare class name "
                        "(e.g. 'Sy2'). Use --subclasses to split it further.")
    p.add_argument('--subclasses', action='store_true',
                   help='one contour per fine class instead of per coarse group')
    p.add_argument('--filled', action='store_true',
                   help='fill the histograms instead of drawing dashed contours')
    p.add_argument('--census', action='store_true', help='print the row accounting only')
    p.add_argument('--summary', metavar='PNG', help='write the population figure here')
    p.add_argument('--top', type=int, metavar='N',
                   help='also re-run the N highest-sigma sources and plot each')
    p.add_argument('--outdir', default='.', help='where --top figures go')
    p.add_argument('--sigma-cut', type=float, default=3.0)
    p.add_argument('--dpi', type=int, default=110)
    args = p.parse_args()

    plt.switch_backend('Agg')
    d = pd.read_parquet(os.path.expanduser(args.parquet))
    outdir = os.path.expanduser(args.outdir)

    band = args.band[1:] if args.band and args.band.startswith('z') else args.band
    tag_b = f'_{band}' if band else ''
    if band:
        col = f'{args.prefix}band' if f'{args.prefix}band' in d else 'band'
        if col not in d:
            raise KeyError(f'--band given but no {col!r} column; '
                           f'columns are {list(d.columns)[:12]}...')
        n0 = len(d)
        d = d[d[col] == band]
        if d.empty:
            raise ValueError(f'no rows with {col}={band!r}; '
                             f'present: {sorted(pd.read_parquet(os.path.expanduser(args.parquet))[col].dropna().unique())}')
        print(f'band {band}: {len(d)} of {n0} rows')

    if args.census:
        zmad_census(d, type_col=args.types or 'clasf', prefix=args.prefix)
        return

    if args.types:
        zmad_census(d, type_col=args.types, prefix=args.prefix)

        groups = dict(TYPE_GROUPS)
        if args.only:
            hit = {k: v for k, v in groups.items()
                   if args.only.lower() in k.lower() or args.only in v}
            if not hit:
                raise ValueError(f'--only {args.only!r} matched no group; '
                                 f'labels are {list(groups)} '
                                 f'and classes {sorted(sum(groups.values(), []))}')
            groups = hit
        if args.subclasses:
            groups = {c: [c] for c in sum(groups.values(), [])}
        outline = () if args.filled else tuple(groups)

        zmad_type_hist(d, type_col=args.types, prefix=args.prefix, cut=args.cut,
                       groups=groups, outline=outline,
                       save=os.path.join(outdir, f'zmad_type_hist{tag_b}.png'),
                       dpi=args.dpi)
        for mode in ('composition', 'rate', 'type_rate'):
            zmad_type_fraction(d, type_col=args.types, prefix=args.prefix,
                               cut=args.cut, mode=mode, dpi=args.dpi,
                               save=os.path.join(outdir, f'zmad_type_{mode}{tag_b}.png'))

    fig, _ = zmad_summary(d, aperture=args.aperture, prefix=args.prefix,
                          sigma_cut=args.sigma_cut)
    dest = os.path.expanduser(args.summary or f'zmad_summary{tag_b}.png')
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    fig.savefig(dest, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {dest}')

    if not args.top:
        return

    from ZMAD import BAT_ROOT, zmad_metric
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    sel = _ok(d, 'ok').sort_values('sigma', ascending=False)
    for f in sel['file'].drop_duplicates().head(args.top):
        path = os.path.join(BAT_ROOT, f)
        agn, stars, out = zmad_metric(path, verbose=False)
        if out is None:
            print(f'{f}: no result on re-run')
            continue
        for tag, m in out['metrics'].items():
            if m is not None:
                print('wrote', save_all(agn, stars, out, tag=tag,
                                        outdir=outdir, dpi=args.dpi))


if __name__ == '__main__':
    _main()
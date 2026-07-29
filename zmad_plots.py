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

BAND_COLOR = {'g': 'mediumseagreen', 'r': 'firebrick', 'i': 'goldenrod'}
STAR_COLOR = '#9F9F9F'


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
    ax.hist(v, bins=min(30, max(10, len(v) // 5)), alpha=0.7,
            color=STAR_COLOR, edgecolor='black', label=f'stars (N={len(v)})')

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
                label=f"stars (N={s['object_index'].nunique()})")
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


def zmad_plots(agn, stars, out, tag=None, figsize=(12, 5)):
    """Light curve + ZMAD histogram for one aperture (default: the first)."""
    tag = _tags(out, tag)[0]
    color = BAND_COLOR.get(out['band'], 'mediumseagreen')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _lc_panel(axes[0], agn[tag], stars[tag], out, tag, color)
    _hist_panel(axes[1], out, tag, color)
    fig.suptitle(f"{_title(out, tag)}   obj {out['object_index']}",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
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
 
 
def zmad_summary(df, aperture=None, sigma_cut=3.0, clip=99.5, figsize=(12, 9)):
    """Population view of a zmad results parquet (no recompute needed).
 
        df = pd.read_parquet('~/results/zmad_sub_g.parquet')
        zmad_summary(df)
 
    Panels: sigma distribution, percentile distribution, and sigma against
    n_stars and n_epochs -- the last two are the ones that matter, because a
    trend there means the metric is tracking the comparison sample rather than
    the source.
    """
    d = df[df.get('ok', True) == True].copy()                      # noqa: E712
    if 'aperture' in d:
        aperture = aperture if aperture is not None else sorted(d['aperture'].dropna())[0]
        d = d[d['aperture'] == aperture]
    d = d[np.isfinite(d['sigma'])]
    if d.empty:
        raise ValueError('no finite sigma rows')
 
    hi = np.percentile(d['sigma'], clip)
    n_over = int((d['sigma'] > hi).sum())
    n_det = int((d['sigma'] > sigma_cut).sum())
 
    fig, ax = plt.subplots(2, 2, figsize=figsize)
 
    a = ax[0, 0]
    a.hist(d['sigma'].clip(upper=hi), bins=50, color=STAR_COLOR, edgecolor='black')
    a.axvline(sigma_cut, color='red', linestyle='--', lw=2,
              label=f'{sigma_cut:g}σ  ({n_det}/{len(d)} = {100*n_det/len(d):.0f}%)')
    a.set(xlabel=f'σ (clipped at {hi:.1f}; {n_over} above)',
          ylabel='sources', yscale='log')
    a.legend(fontsize=9)
 
    a = ax[0, 1]
    a.hist(d['perc'], bins=50, color=STAR_COLOR, edgecolor='black')
    a.set(xlabel='percentile of AGN within its star pool', ylabel='sources')
 
    for a, key in ((ax[1, 0], 'n_stars'), (ax[1, 1], 'n_epochs')):
        a.scatter(d[key], d['sigma'].clip(upper=hi), s=8, alpha=0.4, color='black')
        a.axhline(sigma_cut, color='red', linestyle='--', lw=1)
        a.set(xlabel=key, ylabel='σ', yscale='symlog')
        if d[key].nunique() > 2:
            r = np.corrcoef(d[key], d['sigma'])[0, 1]
            a.set_title(f'r = {r:+.2f}', fontsize=10,
                        color='red' if abs(r) > 0.3 else 'black')
 
    fig.suptitle(f'ZMAD population  |  aperture {aperture}  |  N = {len(d)}',
                 fontsize=13)
    fig.tight_layout()
    return fig, ax
 
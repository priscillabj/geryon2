import os, sys, io, contextlib
import matplotlib; matplotlib.use('Agg')
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.ticker import MaxNLocator
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import Table
from astropy.coordinates import SkyCoord

sys.path.insert(0, os.path.expanduser('~/git'))   # dir containing newSF.py / VarTools.py
from newSF import SF_wnoise
from VarTools import radec_filename, _find_target_obj

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 17,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 13})

COLORS = ['#4c72b0', '#dd8452','#c44e52','mediumseagreen']   # DR, FPS, Zubercal, Our Photometry

LC  = os.path.expanduser('~/LC/')                 # SET THIS
# OUT = os.path.expanduser('~/results/SF_phot_comparison_g.png')
RES = os.path.expanduser('~/results')

dr_f = [LC+'DR/'+f for f in [
    'ZTF_DR24_37.8949549_36.1807395.fits', 'ZTF_DR24_63.41809_11.20411.fits',
    'ZTF_DR24_179.48389_55.45359.fits', 'ZTF_DR24_189.91435_-5.34418.fits']]
fp_f = [LC+'batch/calibrated/'+f for f in [
    'corrlc_37.8949549_36.1807395.csv', 'corrlc_63.4180893_11.2041342.csv',
    'corrlc_179.483894_55.4536137.csv', 'corrlc_189.914336_-5.3441634.csv']]
zb_f = [LC+'Zubercal/'+f for f in [
    'zubercal_37.8949549_36.1807395.csv', 'zubercal_63.41809_11.20411.csv',
    'zubercal_179.48389_55.45359.csv', 'zubercal_189.91435_-5.34418.csv']]
np_f = [os.path.expanduser('~/BAT_results/')+f for f in [
    '37.89498_+36.18072_zg_merged.parquet', '63.41809_+11.20411_zg_merged.parquet',
    '179.48389_+55.45359_zg_merged.parquet',
    '189.91435_-5.34418_000423_zg_ccd04_q1.parquet']]

def single_quad(df, f='field', c='ccdid', q='qid'):
    if {f, c, q} <= set(df.columns):
        m = df[[f, c, q]].mode().iloc[0]
        df = df[(df[f] == m[f]) & (df[c] == m[c]) & (df[q] == m[q])]
    return df

def _read_dr(p, c):
    d = Table.read(p).to_pandas()
    return d[d.catflags == 0]

def _read_csv(p, c):
    return pd.read_csv(p)

def _read_newphot(p, c):
    d = pd.read_parquet(p)
    d = d[d.object_index == _find_target_obj(Path(p), c)]
    return d[(d.MAGLIM > 20.5) & (d.SEEING < 3)]

SOURCES = [
    ('DR',             dr_f, _read_dr,      'mjd',    'mag',          'magerr',        ('filtercode', b'zg')),
    ('FPS',            fp_f, _read_csv,     'mjd',    'mag_tot',      'magunc_tot',    ('filter', 'ZTF_g')),
    ('Zubercal',       zb_f, _read_csv,     'MJD',    'Mag',          'Magerr',        ('Filter', 'g')),
    ('This work',      np_f, _read_newphot, 'OBSMJD', 'MAG_4_TOT_AB', 'MERR_4_TOT_AB', None),
]


def load(fn, reader, t, m, e, band):
    d = reader(fn, SkyCoord(*radec_filename(fn), unit='deg'))
    if band:
        d = d[d[band[0]] == band[1]]
        print(len(d))
    # d = d.dropna(subset=[t, m, e])
    # print(len(d))
    # d = single_quad(d[d[e] < 0.5])
    d = single_quad(d)
    print(len(d))
    return d[t].values, d[m].values, d[e].values

def sci(v, _):
    k = int(np.floor(np.log10(v))); m = round(v / 10**k, 2)
    return rf'$10^{{{k}}}$' if m == 1 else rf'${m:g}\times10^{{{k}}}$'

def log_ticks(axis, lim, subs):
    lo, hi = lim
    t = [s * 10.**k for k in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
         for s in subs]
    axis.set_major_locator(FixedLocator([v for v in t if lo <= v <= hi]))
    axis.set_major_formatter(FuncFormatter(sci))
    axis.set_minor_formatter(NullFormatter())

def plot_target(k, lcs, ylim=None, savedir=None):
    fig, axes = plt.subplots(len(SOURCES), 1, figsize=(9, 8), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0)

    for a, (name, *_), col in zip(axes, SOURCES, COLORS):
        a.set_ylabel(r'$\Delta$ mag', fontsize=16)
        a.tick_params(direction='in', top=True, right=True, which='both',
                      length=7, width=1.2, labelsize=14)
        a.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        t, m, e = lcs.get((k, name), ([], [], []))
        if len(t) == 0:
            a.text(0.5, 0.5, f'{name}: no epochs', transform=a.transAxes,
                   ha='center', va='center', color='r')
            continue
        a.errorbar(t, m - np.median(m), yerr=e, fmt='o', ms=3, elinewidth=0.8,
                   capsize=0, color=col)
        a.text(0.01, 0.95, f'{name}  (N={len(t)})', transform=a.transAxes, va='top', fontsize=12)

    axes[-1].set_xlabel('MJD', fontsize=16)
    axes[0].set_ylim(ylim if ylim else axes[0].get_ylim()[::-1])
    fig.suptitle(os.path.basename(np_f[k]).split('_z')[0], y=0.92)
    if savedir:
        ra, dec, b = radec_filename(np_f[k], band=True)
        fig.savefig(os.path.join(savedir, f'LC_phot_comparison_{ra}_{dec}_{b}.png'), bbox_inches='tight')
        # fig.savefig(os.path.join(savedir, f'lc_{k}.png'), bbox_inches='tight')
    return fig


LCS = {}   # (k, source name) -> (t, m, e)
for i in range(len(np_f)):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    plotted = False

    for (name, files, reader, tc, mc, ec, band), col in zip(SOURCES, COLORS):
        t, m, e = load(files[i], reader, tc, mc, ec, band)
        LCS[i, name] = (t, m, e)
        if len(t) < 2:
            print(f'{name} {os.path.basename(files[i])}: {len(t)} epochs after cuts, skipped')
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            SF = SF_wnoise(m, t, e, cs_all=None)[0]
        s  = SF['SF'].dropna()
        iv = s.index.categories[s.index.codes]
        ax.errorbar(iv.mid, s.values, xerr=iv.length / 2,
                    yerr=(SF['SFminerr'].loc[s.index], SF['SFmaxerr'].loc[s.index]),
                    fmt='o', ms=4, capsize=0, color=col, label=f'{name} (N={len(t)})')
        plotted = True
        print(f'{name} {os.path.basename(files[i])}: {len(t)} epochs, {len(s)} bins')

    ra, dec, b = radec_filename(np_f[i], band=True)
    ax.set(xscale='log', yscale='log', xlabel=r'$\Delta t$ [d]', ylabel='SF [mag]',
           title=os.path.basename(np_f[i]).split('_z')[0])
    if plotted:
        lo, hi = ax.get_ylim()
        log_ticks(ax.yaxis, (lo, hi),
                  (1, 1.5, 2, 3, 5, 7) if np.log10(hi / lo) < 1.2 else (1, 2, 3, 5, 7))
        ax.legend()
    else:
        print(f'{ra}_{dec}: nothing plotted')

    out = os.path.join(RES, f'SF_phot_comparison_{ra}_{dec}_{b}.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('saved', out)

    plt.close(plot_target(i, LCS, savedir=RES))

# plt.tight_layout()
# plt.savefig(OUT, dpi=150)
# print('saved', OUT)
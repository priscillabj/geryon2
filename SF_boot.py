#!/usr/bin/env python3
"""
starplaw.py — structure function of a source fitted as
"comparison-star noise floor + intrinsic power law", built the same way as
sf_boot.py except for the |dm| estimator.

    SF_obs(dt) = sqrt( SF_p^2 (dt/pivot)^(2 gamma) + (N SF_star(dt))^2 )

  estimator   SF = sqrt((pi/2) <|dm|>^2) per lag bin, for the target AND for each
              comparison star; the noise template SF_star = median over stars.
  star epochs stars are evaluated on the TARGET's epochs (same pairs, same lags).
  bootstrap   resample nights (or epochs) with replacement — the SAME draw for the
              target and the stars (joint=True) — and resample the star list.
              -> Cov(SF_obs), Cov(SF_star) and the cross-covariance Cov(SF_obs, SF_star).
  fit space   log10 SF. The template enters through the linearised residual covariance
                  C = C_obs + J C_star J - J C_x^T - C_x J,  J = dlog m / dlog SF_star = (N SF_star/m)^2
  method      'boot' (default): MAP with per-bin (diagonal) errors, so the model is
              centred on the data; uncertainties = 16/50/84 percentiles of the same MAP
              fit repeated on every bootstrap realisation (each with its own SF_obs and
              SF_star, and the N-prior centre redrawn), which carries the bin-to-bin and
              target-template correlations.
              'gls': full-covariance (Hartlap-corrected) likelihood + emcee.
  fit range   bins whose GEOMETRIC centre lies in [fit_min, fit_max], with
              >= min_pairs target pairs and finite values in every bootstrap.

Target and comparison-star selection call newSF's functions exactly as optSF does.
Caveat (as in sf_boot.py): the bootstrap captures measurement noise and pair-sharing
between bins, not the stochastic variance of the AGN process itself.

Main entry point:  starplaw_from_parquet(parquet_file, ...)  -> result dict (+ pkl/png)
"""
import os
import re
import glob
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.coordinates import SkyCoord

from newSF import make_log_bins, quality_flags, quality_cuts, clip_by_blocks
from VarTools import (radec_filename, _find_target_obj, run_sigma_filtering,
                      aggregate_lightcurves, build_sigma_locus)
from sf_keys import keys_for, gamma_for

SQRT_PI_2 = np.sqrt(np.pi / 2)


# ══════════════════════════════════════════════════════════════════════════════
# 1. target + comparison stars  (same steps as optSF)
# ══════════════════════════════════════════════════════════════════════════════
def select_target_and_stars(file, x=10, mag_column='MAG_4_TOT_AB', mag_err='MERR_4_TOT_AB',
                            clip=False, gap_threshold=90, sigma_filter=True):
    """optSF's target cuts and comparison-star selection.
    Returns (ztf, cs, meta) or None if the source cannot be used."""
    band = re.search(r'_z(\w)_', os.path.basename(file)).group(1)
    df = pd.read_parquet(file)
    ra, dec = radec_filename(file)
    tgt = _find_target_obj(Path(file), SkyCoord(ra, dec, unit='deg'))
    if tgt is None:
        print(f'no target found in {os.path.basename(file)}, skipping')
        return None
 
    ztf1 = df[df['object_index'] == tgt]
    if mag_column not in ztf1.columns:
        print('no mag column, skipping file')
        return None
    mask = (ztf1[mag_err] < 0.5) & (ztf1['MAGLIM'] > 20.5) & (ztf1['SEEING'] < 3)
    if 'qid' in ztf1.columns:                       # dominant quadrant, as in optSF
        for col in ('qid', 'ccdid', 'field'):
            mask &= ztf1[col] == ztf1[col].mode().iloc[0]
    ztf1 = ztf1[mask]
    if len(ztf1) == 0:
        print('empty target')
        return None
    ztf = clip_by_blocks(ztf1, mag_column, gap_threshold) if clip else ztf1
 
    cs1 = quality_flags(df, ztf)
    if cs1.empty:
        print('empty cs')
        return None
    if sigma_filter:
        # VarTools' scatter-magnitude locus needs >= 2 of these 0.5-mag bins with >= 5
        # stars, else interp1d crashes (in optSF too). Same edges as its default.
        edges = np.arange(13, 23.5, 0.5)
        n_bins = len(build_sigma_locus(aggregate_lightcurves(cs1), edges)[0])
        if n_bins < 2:
            print(f'too few candidate stars for the sigma-filter locus '
                  f'({n_bins} mag bins with >= 5 stars)')
            return None
        cs, _ = run_sigma_filtering(cs1, edges=edges, agn_indices=tgt, save_file=False,
                                    plot=False, save_plt=False, filename=file, ra=ra, dec=dec)
    else:
        cs = cs1
    cs = quality_cuts(cs, ztf, mag_column, x=x)
    if clip and not cs.empty:
        cs = clip_by_blocks(cs, mag_column, gap_threshold)
    if cs.empty or cs['object_index'].nunique() < 2:
        print('not enough calib stars')
        return None
 
    meta = {'file': file, 'RA': ra, 'DEC': dec, 'band': band, 'tgt_obj_idx': tgt,
            'mag': float(ztf[mag_column].median()), 'epochs': int(ztf['OBSMJD'].nunique()),
            'n_stars': int(cs['object_index'].nunique())}
    return ztf, cs, meta



# ══════════════════════════════════════════════════════════════════════════════
# 2. pairs on the target's epochs, SF per object
# ══════════════════════════════════════════════════════════════════════════════
def _bin_index(dt, log_bins):
    """bin of each lag for right-closed bins (a, b] — the pd.cut convention."""
    k = np.searchsorted(log_bins, dt, side='left') - 1
    return k, (k >= 0) & (k < len(log_bins) - 1)


def build_pairs(ztf, cs, mag_column, log_bins):
    """Light curves of target + stars on the TARGET's epochs (stars NaN where they
    miss an epoch) and the list of epoch pairs falling in a lag bin.

    Returns mjd (nep), i, j (epoch indices of each pair), k (bin of each pair),
    adm (npair x nobj |dm|, 0 where missing), fin (npair x nobj, 1 if usable),
    star_ids. Object 0 is the target."""
    tgt  = ztf.drop_duplicates('OBSMJD').sort_values('OBSMJD')
    mjd  = tgt['OBSMJD'].to_numpy(float)
    star_ids = np.sort(cs['object_index'].unique())
    piv = (cs.drop_duplicates(['object_index', 'OBSMJD'])
             .pivot(index='OBSMJD', columns='object_index', values=mag_column)
             .reindex(mjd)[star_ids])
    mags = np.vstack([tgt[mag_column].to_numpy(float), piv.to_numpy(float).T])   # (nobj, nep)

    i, j = np.triu_indices(len(mjd), k=1)          # every pair of target epochs once
    k, ok = _bin_index(mjd[j] - mjd[i], log_bins)
    i, j, k = i[ok], j[ok], k[ok]
    adm = np.abs(mags[:, j] - mags[:, i]).T.astype(np.float32)   # (npair, nobj)
    fin = np.isfinite(adm)
    adm[~fin] = 0.0
    return mjd, i, j, k, np.ascontiguousarray(adm), fin.astype(np.float32), star_ids


def nightpairs_per_bin(mjd, i, j, k, nb):
    """number of distinct (night_i, night_j) pairs feeding each lag bin."""
    night = np.unique(np.floor(mjd), return_inverse=True)[1]
    return np.bincount(np.unique(np.c_[k, night[i], night[j]], axis=0)[:, 0], minlength=nb)


def sf_all(w_ep, i, j, k, adm, fin, nb):
    """SF = sqrt(pi/2) * weighted mean |dm| per object and bin, all objects at once.
    w_ep = per-epoch multiplicity (ones = the data; bootstrap draws otherwise).
    Returns (nobj, nb); NaN where a bin has no usable pair."""
    w = (w_ep[i] * w_ep[j]).astype(np.float32)               # copies of each pair
    W = sparse.csr_matrix((w, (k, np.arange(len(k)))), shape=(nb, len(k)))
    num, den = W @ adm, W @ fin                               # (nb, nobj)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (SQRT_PI_2 * num / den).T


def joint_bootstrap(mjd, i, j, k, adm, fin, nb, n_boot=200, unit='night',
                    joint=True, rng=None):
    """Bootstrap realisations of SF_obs and of the star template.

    unit  : 'night' resamples observing nights (floor(MJD)) — all exposures of a
            night stay together; 'epoch' resamples single epochs.
    joint : True  -> the SAME night/epoch draw for target and stars (keeps the
                     common-mode noise of a bad night correlated -> cross-covariance)
            False -> independent draws for target and stars (for comparison).
    The star list is resampled in both cases. Returns boot_obs, boot_star (n_boot, nb)."""
    rng = rng if rng is not None else np.random.default_rng()
    if unit == 'night':
        uid = np.unique(np.floor(mjd).astype(int), return_inverse=True)[1]
    elif unit == 'epoch':
        uid = np.arange(len(mjd))
    else:
        raise ValueError("unit must be 'night' or 'epoch'")
    nu, nstar = uid.max() + 1, adm.shape[1] - 1

    def draw():                                               # per-epoch multiplicity
        return np.bincount(rng.integers(0, nu, nu), minlength=nu).astype(float)[uid]

    boot_obs = np.full((n_boot, nb), np.nan)
    boot_st  = np.full((n_boot, nb), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN medians in empty bins
        for b in range(n_boot):
            sfb = sf_all(draw(), i, j, k, adm, fin, nb)
            sfs = sfb if joint else sf_all(draw(), i, j, k, adm, fin, nb)
            boot_obs[b] = sfb[0]
            boot_st[b]  = np.nanmedian(sfs[1 + rng.integers(0, nstar, nstar)], axis=0)
    return boot_obs, boot_st


# ══════════════════════════════════════════════════════════════════════════════
# 3. model, prior, likelihood (log10 SF space)
# ══════════════════════════════════════════════════════════════════════════════
def _unpack(theta, free_N):
    return (theta[0], theta[1], theta[2]) if free_N else (theta[0], theta[1], 1.0)


def sf_model(theta, x, sf_star, free_N=True):
    """theta = [log10 SF_p, gamma, N];  x = dt / pivot."""
    lsf, g, N = _unpack(theta, free_N)
    pl = 10 ** lsf * x ** g
    return np.sqrt(pl ** 2 + (N * sf_star) ** 2), N, pl


def _log_prior(theta, free_N, pri):
    n_mu, n_sig, n_bounds, gamma_max, lsf_bounds = pri
    lsf, g, N = _unpack(theta, free_N)
    if not (lsf_bounds[0] < lsf < lsf_bounds[1] and 0.0 < g < gamma_max
            and n_bounds[0] < N < n_bounds[1]):
        return -np.inf
    return -0.5 * ((N - n_mu) / n_sig) ** 2 if free_N else 0.0   # flat in log10 SF_p


def _logpost_diag(theta, x, y, s, v_obs, v_st, free_N, pri):
    """per-bin (diagonal) errors: var = v_obs + J^2 v_star (v_* = variances of log10 SF)."""
    lp = _log_prior(theta, free_N, pri)
    if not np.isfinite(lp):
        return -np.inf
    m, N, _ = sf_model(theta, x, s, free_N)
    r = y - np.log10(m)
    J = (N * s / m) ** 2
    var = v_obs + J ** 2 * v_st
    return lp - 0.5 * np.sum(r * r / var + np.log(var))


def _resid_cov(theta, x, y, s, C_obs, C_st, C_x, free_N):
    m, N, _ = sf_model(theta, x, s, free_N)
    r = y - np.log10(m)
    J = (N * s / m) ** 2
    C = C_obs + J[:, None] * C_st * J[None, :] - J[:, None] * C_x.T - C_x * J[None, :]
    return r, C


def _logpost_full(theta, x, y, s, C_obs, C_st, C_x, hartlap, free_N, pri):
    """full bootstrap covariance incl. the target-template cross term."""
    lp = _log_prior(theta, free_N, pri)
    if not np.isfinite(lp):
        return -np.inf
    r, C = _resid_cov(theta, x, y, s, C_obs, C_st, C_x, free_N)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return -np.inf
    z = np.linalg.solve(L, r)
    return lp - 0.5 * hartlap * z @ z - np.sum(np.log(np.diag(L)))


def _map(logpost, starts):
    """MAP = maximum of the posterior, by Nelder-Mead from several starts."""
    best = None
    for x0 in starts:
        r = minimize(lambda t: -logpost(t), x0, method='Nelder-Mead',
                     options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 20000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


# ══════════════════════════════════════════════════════════════════════════════
# 4. fit
# ══════════════════════════════════════════════════════════════════════════════
def fit_starplaw(tau, sf_obs, sf_star, npair, boot_obs, boot_star, method='boot',
                 fit_min=1.0, fit_max=365.0, min_pairs=3, pivot=None,
                 report_at=(1, 100, 300, 365), free_N=True, n_prior=(1.0, 0.01),
                 n_bounds=(0.5, 2.0), gamma_max=2.0, lsf_bounds=(-4.0, 1.0),
                 nwalkers=32, nsteps=3000, rng=None, verbose=False):
    """Fit log10 SF_obs = log10 sqrt(PL^2 + (N SF_star)^2). Returns a dict of
    `_boot` keys (incl. the bin selection `sel` and `samples_boot`: bootstrap refits
    or emcee chain, columns [log10 SF_pivot, gamma, N], N only if free_N),
    or None if too few bins."""
    _p = print if verbose else (lambda *a, **k: None)
    rng = rng if rng is not None else np.random.default_rng()
    nboot = len(boot_obs)

    sel = ((npair >= min_pairs) & (tau >= fit_min) & (tau <= fit_max)
           & np.isfinite(sf_obs) & np.isfinite(sf_star) & (sf_obs > 0) & (sf_star > 0)
           & np.all(np.isfinite(boot_obs) & (boot_obs > 0), axis=0)
           & np.all(np.isfinite(boot_star) & (boot_star > 0), axis=0))
    p = int(sel.sum())
    ndim = 3 if free_N else 2
    if p < ndim + 1:
        print(f'SF_starplaw: too few usable bins ({p}) — SKIPPING')
        return None

    t, y, s = tau[sel], np.log10(sf_obs[sel]), sf_star[sel]
    if pivot is None:
        pivot = float(10 ** np.mean(np.log10(t)))           # log-centre of the fitted lags
    x = t / pivot

    Lo, Ls = np.log10(boot_obs[:, sel]), np.log10(boot_star[:, sel])
    Cfull = np.cov(np.hstack([Lo, Ls]), rowvar=False)
    C_obs, C_st, C_x = Cfull[:p, :p], Cfull[p:, p:], Cfull[:p, p:]
    hartlap = (nboot - p - 2) / (nboot - 1)
    v_obs, v_st = np.diag(C_obs).copy(), np.diag(C_st).copy()
    corr = C_obs / np.sqrt(np.outer(v_obs, v_obs))
    pri = lambda mu: (mu, n_prior[1], n_bounds, gamma_max, lsf_bounds)

    starts = [[l0, g0] + ([1.0] if free_N else [])
              for g0 in (0.1, 0.3, 0.6, 1.0) for l0 in (-2.5, -2.0, -1.3, -0.8)]

    if method == 'boot':
        lp = lambda th, yy=y, ss=s, mu=n_prior[0]: _logpost_diag(th, x, yy, ss, v_obs, v_st,
                                                                   free_N, pri(mu))
        theta = _map(lp, starts)
        samples = np.empty((nboot, ndim))
        for b in range(nboot):                     # refit every bootstrap realisation
            mu_b = n_prior[0] + n_prior[1] * rng.standard_normal() if free_N else n_prior[0]
            samples[b] = _map(lambda th: lp(th, Lo[b], boot_star[b, sel], mu_b), [theta])
    elif method == 'gls':
        if nboot <= p + 2:
            raise ValueError(f'gls needs n_boot > n_bins + 2 ({nboot} vs {p})')
        lp = lambda th: _logpost_full(th, x, y, s, C_obs, C_st, C_x, hartlap, free_N, pri(n_prior[0]))
        theta = _map(lp, starts)
        pos = theta + 1e-3 * rng.standard_normal((nwalkers, ndim))
        pos[:, 1] = np.clip(pos[:, 1], 1e-3, gamma_max - 1e-3)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp)
        sampler.run_mcmc(pos, nsteps, progress=False)
        samples = sampler.get_chain(discard=nsteps // 4, thin=10, flat=True)
    else:
        raise ValueError("method must be 'boot' or 'gls'")

    def pct(a):
        lo, med, hi = np.percentile(a, [16, 50, 84])
        return med, hi - med, med - lo

    out = {'method_boot': method, 'fit_space_boot': 'log10SF', 'pivot_boot': pivot,
           'n_bins_boot': p, 'sel': sel, 'samples_boot': samples}
    SFp = 10 ** samples[:, 0]
    N_s = samples[:, 2] if free_N else np.ones(len(samples))
    for name, a in (('SF_pivot', SFp), ('gamma', samples[:, 1]), ('N', N_s)):
        out[f'{name}_boot'], out[f'{name}_maxerr_boot'], out[f'{name}_minerr_boot'] = pct(a)
    for L in report_at:
        out[f'A_{L}_boot'], out[f'A_{L}_maxerr_boot'], out[f'A_{L}_minerr_boot'] = \
            pct(SFp * (L / pivot) ** samples[:, 1])
    lsf_m, g_m, N_m = _unpack(theta, free_N)
    out.update({'SF_pivot_map_boot': 10 ** lsf_m, 'gamma_map_boot': g_m, 'N_map_boot': N_m})

    # diagnostics at the MAP
    m, _, pl = sf_model(theta, x, s, free_N)
    r = y - np.log10(m)
    J = (N_m * s / m) ** 2
    chi2_diag = float(np.sum(r ** 2 / (v_obs + J ** 2 * v_st)))
    chi2_full = np.nan
    if nboot > p + 2:
        _, C = _resid_cov(theta, x, y, s, C_obs, C_st, C_x, free_N)
        chi2_full = float(hartlap * r @ np.linalg.solve(C, r))
    intr = SFp * (x[-1]) ** samples[:, 1]                    # intrinsic SF at the longest lag
    im, _, il = pct(intr)
    out.update({
        'chi2_full_boot': chi2_full, 'chi2_diag_boot': chi2_diag, 'dof_boot': p - ndim,
        'chi2red_boot': chi2_full / (p - ndim) if p > ndim else np.nan,
        'frac_above_model_boot': float(np.mean(r > 0)),
        'adjcorr_boot': float(np.median(np.abs(np.diag(corr, 1)))) if p > 1 else np.nan,
        'frac_intr_maxlag_boot': float(pl[-1] ** 2 / m[-1] ** 2),
        'hartlap_boot': hartlap,
        'flag_boot': 'non-variable (intrinsic SF consistent with 0; gamma unconstrained)'
                      if im < 3 * il else 'ok',
        'N_prior_dominated_boot': bool(free_N and (out['N_maxerr_boot'] + out['N_minerr_boot'])
                                        > 1.6 * n_prior[1]),
    })
    _p(f"[{method}] gamma={out['gamma_boot']:.3f} +{out['gamma_maxerr_boot']:.3f}"
       f"/-{out['gamma_minerr_boot']:.3f} | A_300={out.get('A_300_boot', np.nan):.4f} | "
       f"N={out['N_boot']:.2f} | chi2/dof(full)={chi2_full:.1f}/{p - ndim} | {out['flag_boot']}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. plot
# ══════════════════════════════════════════════════════════════════════════════
def _linmix_ref(res):
    """MPI_linmixfit output for the same parquet as (dict, sf_keys model), or None.
    Looked up next to our pkl, else in any partials dir (as runSF). The dict is
    SF_dict[0] or the top level, whichever carries a complete sf_keys key set:
    'spl' (current names) or 'spl_legacy' (the '_'-prefixed legacy campaign). A
    pkl without either gives None, i.e. no overlay."""
    name = f"{os.path.basename(res['file'])}_{res['x']}cs_linmixfit.pkl"
    p = os.path.join(os.path.dirname(output_path(res['file'], x=res['x'])), name)
    hits = [p] if os.path.exists(p) else \
        glob.glob(os.path.join(os.environ['HOME'], 'results', 'partials', '*', name))
    if not hits:
        return None
    with open(hits[0], 'rb') as f:
        d = pickle.load(f)
    for c in ((d['SF_dict'][0],) if 'SF_dict' in d else ()) + (d,):
        for model in ('spl', 'spl_legacy'):
            if 'SF' in c and set(keys_for(model)) <= c.keys():
                return c, model
    print(f'linmix pkl without a fit, not drawn: {hits[0]}')
    return None


def plot_starplaw(res, fname=None, show=False, samples=None):
    """Drawn as fit_sf.plot_fit: SF_obs (fitted / excluded), median star template SF_cs,
    sqrt(SF_obs^2 - SF_cs^2) reference (N = 1, as sf_boot), total model, power law and
    N x SF_cs, all over the full lag range; plus, if it exists, the old linmix fit
    (optSF's noise-subtracted SF, and A_365 (dt/365)^gamma over its fitted bins). With samples (argument, else
    res['samples_boot'] if stored): 100 sample curves, 16-84% bands of both components,
    per-lag posterior medians and median parameters in the legend; without, the MAP."""
    tau, sel = res['tau'], res['sel']
    so, ss, eo, es = res['sf_obs'], res['sf_star'], res['sf_obs_err'], res['sf_star_err']
    piv = res['pivot_boot']
    ok = np.isfinite(so) & np.isfinite(ss)
    samples = samples if samples is not None else res.get('samples_boot')

    fig, ax = plt.subplots(figsize=(7, 5))
    for m, alpha, lab in ((sel & ok, 1.0, 'SF_obs (fit)'), (~sel & ok, 0.3, 'SF_obs (excluded)')):
        if m.any():
            ax.errorbar(tau[m], so[m], yerr=eo[m], fmt='o', ms=4, color='k', alpha=alpha, label=lab)
    ax.errorbar(tau[ok], ss[ok], yerr=es[ok], fmt='s', ms=3, color='tab:gray', alpha=0.5,
                label='SF_cs (median comp. stars)')
    ref = np.sqrt(np.clip(so ** 2 - ss ** 2, 0, None))
    pos = ok & (ref > 0)
    ax.plot(tau[pos], ref[pos], 'x', color='tab:green', alpha=0.6,
            label='SF (noise-subtracted, for reference)')

    lm, model = _linmix_ref(res) or (None, None)
    if lm is not None:
        leg = '' if model == 'spl' else ' (legacy)'
        iv = pd.IntervalIndex(lm['SF'].index)
        tl, v = np.sqrt(iv.left * iv.right), lm['SF'].to_numpy(float)
        lo, hi = np.asarray(lm['SFminerr'], float), np.asarray(lm['SFmaxerr'], float)
        win = ((iv.left >= 1) & (iv.right <= 365) & (v != 0)            # SF_linmix's window
               & (np.asarray(lm['#elements'], float) >= 3))
        # above the green crosses, which they coincide with where SF_obs >> SF_cs
        for m, alpha, lab in ((win & (v > 0), 0.9, f'linmix SF{leg} (fit)'),
                              (~win & (v > 0), 0.5, f'linmix SF{leg} (excluded)')):
            if m.any():
                ax.errorbar(tl[m], v[m], yerr=[lo[m], hi[m]], fmt='D', ms=4, zorder=3,
                            color='tab:purple', alpha=alpha, label=lab)
        z = v == 0                        # optSF clips SF_obs^2 - SF_cs^2 < 0 to 0: not on a log axis
        if z.any():                       # -> marked at the bottom edge (x in data, y in axes units)
            ax.plot(tl[z], np.full(z.sum(), 0.02), 'v', ms=4, color='tab:purple', alpha=0.5,
                    transform=ax.get_xaxis_transform(), label=f'linmix SF{leg} = 0 (excluded)')
        tw = tl[win & (v > 0)]
        a = lm[next(k for k in keys_for(model) if k.endswith('A_365_spl'))]
        gl = lm[gamma_for(model)]
        if len(tw) and np.isfinite(a) and np.isfinite(gl):       # failed fits store NaN
            tt = np.geomspace(tw.min(), tw.max(), 50)
            ax.plot(tt, a * (tt / 365) ** gl, '-.', color='tab:purple', lw=1.5,
                    label=rf'linmix{leg}: A$_{{365}}$={a:.3f}, $\gamma$={gl:.2f}')

    t = np.geomspace(tau[ok].min(), tau[ok].max(), 200)
    cs_t = np.interp(np.log(t), np.log(tau[ok]), ss[ok])
    if samples is None:
        lsf, g, N = np.log10(res['SF_pivot_map_boot']), res['gamma_map_boot'], res['N_map_boot']
        pl_mid, nz_mid = 10 ** lsf * (t / piv) ** g, N * cs_t
        tot, a365 = np.hypot(pl_mid, nz_mid), 10 ** lsf * (365 / piv) ** g
    else:
        lsf, g = samples[:, 0:1], samples[:, 1:2]
        N = samples[:, 2:3] if samples.shape[1] == 3 else np.ones_like(g)
        pl, nz = 10 ** lsf * (t / piv) ** g, N * cs_t
        tot_s = np.hypot(pl, nz)
        idx = np.random.default_rng(0).choice(len(samples), min(100, len(samples)), replace=False)
        for s in tot_s[idx]:
            ax.plot(t, s, color='tab:red', alpha=0.03, lw=1)
        # 1-sigma bands of each component; the noise band reflects N only, the
        # template's own uncertainty is in the SF_cs error bars
        for comp, col in ((pl, 'tab:blue'), (nz, 'tab:orange')):
            lo, hi = np.percentile(comp, [16, 84], axis=0)
            ax.fill_between(t, lo, hi, color=col, alpha=0.15, lw=0)
        tot, pl_mid, nz_mid = (np.median(a, axis=0) for a in (tot_s, pl, nz))
        a365, g, N = np.median(10 ** lsf * (365 / piv) ** g), np.median(g), np.median(N)
    ax.plot(t, tot, color='tab:red', lw=2, label='total model')
    ax.plot(t, pl_mid, '--', color='tab:blue', lw=1.5,
            label=rf'power law: A$_{{365}}$={a365:.3f}, $\gamma$={g:.2f}')
    ax.plot(t, nz_mid, ':', color='tab:orange', lw=1.5, label=f'noise: {N:.2f} × SF_cs')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 3, 5)))   # 0.02, 0.03, 0.05, 0.1 ...
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(r'$\tau$ [days]'); ax.set_ylabel('SF [mag]')
    name = os.path.basename(str(res.get('file', ''))).split('_z')[0]
    ax.set_title(f"{name}  {res.get('band', '')}-band  mag={float(res.get('mag', np.nan)):.2f}")
    # keep the axes on the data: the power law at short lags can fall far below it
    y = np.r_[so[ok], ss[ok]]
    ylo = np.r_[y, v[v > 0]] if lm is not None else y           # NaN > 0 is False
    ax.set_ylim(0.5 * ylo[ylo > 0].min(), 2.0 * y.max())
    ax.legend(fontsize=7, loc='best', ncol=2)          # inside, in the emptiest corner
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 6. one source end to end
# ══════════════════════════════════════════════════════════════════════════════
def output_path(file, x=10, tag='', outdir=None):
    """~/results/partials/{ra}_{dec}/<parquet_basename>_{x}cs_boot{tag}.pkl"""
    ra, dec = radec_filename(file)
    d = outdir or os.path.join(os.environ['HOME'], 'results', 'partials', f'{ra}_{dec}')
    return os.path.join(d, f'{os.path.basename(file)}_{x}cs_boot{tag}.pkl')


def starplaw_from_parquet(file, x=10, mag_column='MAG_4_TOT_AB', mag_err='MERR_4_TOT_AB',
                          clip=False, gap_threshold=90, sigma_filter=True,
                          n_boot=200, unit='night', joint=True, method='boot', seed=None,
                          save=True, save_plot=False, outdir=None, tag='', store_boot=False,
                          store_samples=False, verbose=False, **fit_kwargs):
    """Select target + stars (as optSF), build the SF on the target's epochs, run the
    joint bootstrap, fit. Saves the result pkl (+ png) next to the other partials.
    store_boot / store_samples keep the bootstrap SFs / the posterior samples in it.
    fit_kwargs go to fit_starplaw (fit_min, fit_max, min_pairs, n_prior, ...).
    sf_obs_err is the bootstrap std, except in bins that some realisations leave empty
    (too few independent night pairs for the bootstrap to show scatter: a single night
    pair gives ~0): there the larger of it and the Gaussian small-sample error of
    sqrt(pi/2)<|dm|> from n distinct night pairs, sf_obs sqrt((pi/2 - 1)/n). The fit
    never uses those bins (it needs every realisation finite)."""
    got = select_target_and_stars(file, x=x, mag_column=mag_column, mag_err=mag_err,
                                  clip=clip, gap_threshold=gap_threshold,
                                  sigma_filter=sigma_filter)
    if got is None:
        return None
    ztf, cs, meta = got

    log_bins = make_log_bins()
    nb = len(log_bins) - 1
    mjd, i, j, k, adm, fin, star_ids = build_pairs(ztf, cs, mag_column, log_bins)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sf0 = sf_all(np.ones(len(mjd)), i, j, k, adm, fin, nb)
        sf_obs, sf_star = sf0[0], np.nanmedian(sf0[1:], axis=0)
    npair = np.bincount(k, weights=fin[:, 0], minlength=nb)
    n_np = nightpairs_per_bin(mjd, i, j, k, nb)

    rng = np.random.default_rng(seed)
    boot_obs, boot_star = joint_bootstrap(mjd, i, j, k, adm, fin, nb, n_boot=n_boot,
                                          unit=unit, joint=joint, rng=rng)
    tau = np.sqrt(log_bins[:-1] * log_bins[1:])                # geometric bin centres

    res = fit_starplaw(tau, sf_obs, sf_star, npair, boot_obs, boot_star, method=method,
                       rng=rng, verbose=verbose, **fit_kwargs)
    if res is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        err_obs = np.nanstd(boot_obs, axis=0, ddof=1)
        thin = ~np.all(np.isfinite(boot_obs), axis=0)
        err_obs[thin] = np.fmax(err_obs[thin], sf_obs[thin] * np.sqrt((np.pi / 2 - 1) / n_np[thin]))
        out = {**meta, 'x': x, 'nboot': n_boot, 'unit': unit, 'joint': joint,
               'log_bins': log_bins, 'tau': tau, 'npair': npair, 'n_nightpair': n_np,
               'sf_obs': sf_obs, 'sf_star': sf_star,
               'sf_obs_err': err_obs,
               'sf_star_err': np.nanstd(boot_star, axis=0, ddof=1),
               'star_ids': star_ids, **res}
    if store_boot:
        out['boot_obs'], out['boot_star'] = boot_obs, boot_star
    samples = out['samples_boot'] if store_samples else out.pop('samples_boot')   # plot uses them either way

    fname = output_path(file, x=x, tag=tag, outdir=outdir)
    if save:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(out, f)
    if save_plot:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plot_starplaw(out, fname[:-len('.pkl')] + '.png', samples=samples)
    return out
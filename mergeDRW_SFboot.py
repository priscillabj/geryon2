import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

BAT = os.path.expandvars("$HOME/results/bat_master_boot.parquet")
DRW = os.path.expandvars("$HOME/results/drw_lag/drw_lag_params.parquet")
OUT = os.path.expandvars("$HOME/results/drw_lag/bat_drw_merged.parquet")
FIG = os.path.expandvars("$HOME/results/drw_lag/log_drw_sigma_vs_A365_boot.png")
TOL = 1.0  # arcsec

bat = pd.read_parquet(BAT).dropna(subset=["RA", "DEC"])
drw = pd.read_parquet(DRW).dropna(subset=["RA", "DEC"]).rename(columns={"RA": "RA_drw", "DEC": "DEC_drw"})

# BAT: several rows per source (bands/LCs); DRW: one per source -> nearest neighbour, many-to-one, inner

idx, sep, _ = SkyCoord(bat.RA.values, bat.DEC.values, unit="deg").match_to_catalog_sky(
    SkyCoord(drw.RA_drw.values, drw.DEC_drw.values, unit="deg"))
ok = sep.arcsec < TOL
m = pd.concat([bat[ok].reset_index(drop=True), drw.iloc[idx[ok]].reset_index(drop=True)], axis=1)
m["sep_arcsec"] = sep.arcsec[ok]
print(f"BAT rows matched: {ok.sum()}/{len(bat)} | DRW sources matched: {np.unique(idx[ok]).size}/{len(drw)}")
m.to_parquet(OUT, index=False)

q = m.valid.eq(True) & m.converged.eq(True) & m.valid_boot.eq(True)
bands = sorted(m.band.dropna().unique())
fig, axs = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4.5),
                        sharex=True, sharey=True, squeeze=False)
BAND_COLOR = {'g': 'mediumseagreen', 'r': 'firebrick', 'i': 'goldenrod'}

for ax, b in zip(axs[0], bands):
    s = m[q & m.band.eq(b)]
    res = s.tau_resolved.eq(True)
    c = BAND_COLOR[b]
    for sel, mfc, lab in [(res, c, r"$\tau$ resolved"), (~res, "none", r"$\tau$ unresolved")]:
        t = s[sel]
        log_amp = t.log_drw_sigma + (t.log_amp_scale if b == "r" else 0)
        amp_drw = np.exp(log_amp)
        ax.errorbar(amp_drw, t.A_365_boot,
                    xerr=[amp_drw * (1 - np.exp(-t.sigma_loerr)),
                          amp_drw * (np.exp(t.sigma_uperr) - 1)],
                    yerr=[t.A_365_minerr_boot, t.A_365_maxerr_boot],
                    fmt="o", ms=3, elinewidth=0.6, alpha=0.7,
                    color=c, mfc=mfc, label=f"{lab} ({len(t)})")
    ax.axline((1e-2, 1e-2), (1, 1), color="k", ls="--", label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(title=f"{b}  (N={len(s)})", xlabel=r"$\sigma_{\rm DRW}$")
    ax.legend(fontsize=8)
    
axs[0, 0].set_ylabel(r"$A_{365}$ (boot)")
fig.tight_layout()
fig.savefig(FIG, dpi=150)
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re, os

def radec_filename(i, band=False):
    #path = os.environ['HOME']+r"/Nextcloud/Doutorado/Forced_Phot/"
    #calpath = path+r'calibration_sources/PS1/'
    
    match = re.search(r'(\d+\.\d+)_([-+]?\d+\.\d+)', i)
    if not match:
        raise ValueError(f"no RA/Dec found in filename: {i}")
    regex_ra = float(match.group(1))
    regex_dec = float(match.group(2))
    
    if band:
        match2 = re.search(r"_z(\w)_", i)
        # band   = match2.group(1)
        return regex_ra, regex_dec, match2.group(1)
    
    return regex_ra, regex_dec
    


# def filter_agn(file_or_df, ra=None, dec=None, pattern=r"(\d+\.\d+)_([-+]?\d+\.\d+)"):
#     """Accept either a file path or an already-loaded DataFrame.
    
#     If a path is passed, ra/dec are extracted from the filename.
#     If a DataFrame is passed, ra/dec must be provided explicitly.
#     """
#     if isinstance(file_or_df, str):
#         match = re.search(pattern, file_or_df)
#         if not match:
#             print('no re match or nan values found')
#             return pd.DataFrame()
#         ra  = float(match.group(1))
#         dec = float(match.group(2))
#         df  = pd.read_parquet(file_or_df)
#     else:
#         if ra is None or dec is None:
#             raise ValueError('ra and dec must be provided when passing a DataFrame')
#         df = file_or_df

#     ztf = df[
#         np.isclose(df['ALPHAWIN_REF'], ra, atol=3/3600) &
#         np.isclose(df['DELTAWIN_REF'], dec, atol=3/3600)
#     ].copy()

#     if len(ztf) > 0:
#         ztf.loc[:, 'ra']  = ra
#         ztf.loc[:, 'dec'] = dec
#     else:
#         print('NO AGN FOUND')

#     print(f'object_index: {ztf['object_index'].unique()} were identified as the AGN')
#     return ztf
from astropy.coordinates import SkyCoord

# def _find_target_obj(ra: float, dec: float, df: pd.DataFrame) -> int | None:
#     """Find object_index closest to (ra, dec) in a precomputed srcs DataFrame.

#     Parameters
#     ----------
#     ra, dec : float — target coordinates in degrees
#     srcs    : DataFrame with index=object_index, columns ALPHAWIN_REF, DELTAWIN_REF
#     """
#     if ("ALPHAWIN_REF" not in df.columns) or df.empty:
#         return None

#     import astropy.units as u

#     # if df.empty:
#     #     return None

#     srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()

#     cats = SkyCoord(ra=srcs['ALPHAWIN_REF'].values * u.deg,
#                     dec=srcs['DELTAWIN_REF'].values * u.deg)
#     tgt  = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
#     idx, sep, _ = tgt.match_to_catalog_sky(cats)

#     if sep[0].arcsec < 3.0:
#         print(f'idx: {srcs.index[int(idx)]} were identified as the AGN')
#         return srcs.index[int(idx)]
#     return None

from pathlib import Path
# MATCH_RADIUS_ARCSEC = 3.0

def _find_target_obj(lc_path: Path, tgt: SkyCoord, MATCH_RADIUS_ARCSEC: float = 3.0) -> int | None:
    """Returns True if the target is found within MATCH_RADIUS_ARCSEC."""
    print(f"  parquet   : {lc_path.name}", end="  ")
    if not lc_path.exists():
        print("[NOT FOUND]")
        return None
    df = pd.read_parquet(lc_path, columns=["object_index", "ALPHAWIN_REF", "DELTAWIN_REF"])
    srcs = df.groupby("object_index")[["ALPHAWIN_REF", "DELTAWIN_REF"]].first().dropna()
    if srcs.empty:
        print("NO SOURCES")
        return None
    cat = SkyCoord(ra=srcs["ALPHAWIN_REF"].values, dec=srcs["DELTAWIN_REF"].values, unit="deg")
    idx, sep, _ = tgt.match_to_catalog_sky(cat)
    d = sep[0].arcsec
    if d < MATCH_RADIUS_ARCSEC:
        obj_idx = srcs.index[int(idx)]
        n_epochs = (df["object_index"] == obj_idx).sum()
        print(f"MATCH  sep={d:.2f}\"  object_index={obj_idx}  N_epochs={n_epochs}")
        return obj_idx
    print(f"NO MATCH  (nearest={d:.2f}\")")
    return None

# ── helpers ──────────────────────────────────────────────────────────────────

def _running_median(grp: pd.DataFrame, edges: np.ndarray):
    cx, my = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = grp[(grp["med"] >= lo) & (grp["med"] < hi)]["std"] * 1000
        if len(s) >= 5:
            cx.append(0.5 * (lo + hi))
            my.append(float(np.median(s)))
    return cx, my

def _running_mad(grp: pd.DataFrame, edges: np.ndarray):
    cx, mad = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = grp[(grp["med"] >= lo) & (grp["med"] < hi)]["std"] * 1000
        if len(s) >= 5:
            cx.append(0.5 * (lo + hi))
            mad.append(float(np.median(np.abs(s - np.median(s)))))
    return cx, mad

# ── core functions ────────────────────────────────────────────────────────────

def aggregate_lightcurves(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-object statistics from the light curve DataFrame."""
    return (
        df.groupby("object_index")
        .agg(
            n_org=('MAG_4_TOT_AB', "count"),
            med=('MAG_4_TOT_AB',   "median"),
            std=('MAG_4_TOT_AB',   "std"),
        )
        .dropna(subset=["std"])
    )


def build_sigma_locus(agg_df: pd.DataFrame,
                      edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute running median + MAD locus and return arrays ready for
    interpolation and plotting.

    Returns
    -------
    cx_arr, my_arr, sigma_arr  — all 1-D numpy arrays of the same length
    """
    cx, my   = _running_median(agg_df, edges)
    cx, mady = _running_mad(agg_df, edges)
    cx_arr    = np.array(cx)
    my_arr    = np.array(my)
    sigma_arr = 1.4826 * np.array(mady)          # MAD → Gaussian σ
    return cx_arr, my_arr, sigma_arr


def assign_sigma_levels(agg_df: pd.DataFrame,
                        cx_arr: np.ndarray,
                        my_arr: np.ndarray,
                        sigma_arr: np.ndarray) -> pd.DataFrame:
    """
    Add a 'sigma_level' column (1/2/3/4) to agg_df in-place and return it.
    Level 4 means > 3σ from the running-median locus.
    """
    med_interp = interp1d(cx_arr, my_arr,    bounds_error=False, fill_value="extrapolate")
    mad_interp = interp1d(cx_arr, sigma_arr, bounds_error=False, fill_value="extrapolate")

    std_milli   = agg_df["std"] * 1000
    med_at_x    = med_interp(agg_df["med"])
    sigma_at_x  = mad_interp(agg_df["med"])
    residual    = np.abs(std_milli - med_at_x)

    agg_df["sigma_level"] = np.select(
        [residual <= sigma_at_x,
         residual <= 2 * sigma_at_x,
         residual <= 3 * sigma_at_x],
        [1, 2, 3],
        default=4
    )
    return agg_df


def plot_sigma_locus(agg_df: pd.DataFrame,
                     cx_arr: np.ndarray,
                     my_arr: np.ndarray,
                     sigma_arr: np.ndarray,
                     tgt_obj_idx: int | None = None,
                     ra: float | None = None,
                     dec: float | None = None,
                     save_plot: bool = False,
                     filename: str | None = None,
                     highlight: list | None = None) -> None:
    """Scatter plot of per-object σ coloured by sigma level + envelope bands."""
    colors = {1: "steelblue", 2: "orange", 3: "red", 4: "darkred"}
    labels = {1: "≤1σ", 2: "1–2σ", 3: "2–3σ", 4: ">3σ"}
    sizes  = {1: 8,    2: 12,     3: 18,     4: 25}

    fig, ax = plt.subplots(figsize=(16, 12))

    for level in [4, 3, 2, 1]:
        mask = agg_df["sigma_level"] == level
        ax.scatter(agg_df.loc[mask, "med"],
                   agg_df.loc[mask, "std"] * 1000,
                   s=sizes[level], c=colors[level],
                   alpha=0.6, label=labels[level], zorder=level)

    ax.plot(cx_arr, my_arr, "k-", lw=2, zorder=5, label="Median σ (calibrated)")

    for n, ls in zip([1, 2, 3], ["--", "-.", ":"]):
        lo = my_arr - n * sigma_arr
        hi = my_arr + n * sigma_arr
        ax.fill_between(cx_arr, lo, hi, alpha=0.08, color="grey")
        ax.plot(cx_arr, hi, "k", lw=0.8, ls=ls, label=f"+{n}σ")
        ax.plot(cx_arr, lo, "k", lw=0.8, ls=ls)

    # tgt_obj_idx is now resolved upstream — no file read needed here
    print(tgt_obj_idx)
    if tgt_obj_idx is not None and tgt_obj_idx in agg_df.index:
        r = agg_df.loc[tgt_obj_idx]
        ax.plot(r["med"], r["std"] * 1000, "*", ms=14, color="red",
                zorder=5, label=f"Target  σ={r['std']*1000:.1f} mmag")
    else:
        print('target not found in agg_df')

    if highlight:
        sub  = agg_df.loc[agg_df.index.isin(highlight)]
        print(sub[["sigma_level"]])
        cmap = plt.cm.get_cmap("tab10", len(sub))
        for i, (idx, row) in enumerate(sub.iterrows()):
            color = cmap(i)
            ax.scatter(row["med"], row["std"] * 1000,
                       s=150, color=color, marker="*",
                       edgecolors="black", lw=0.5,
                       zorder=7, label=f"obj {idx}")
            ax.annotate(str(idx),
                        xy=(row["med"], row["std"] * 1000),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=color))

    ax.set_xlabel("Median magnitude")
    ax.set_ylabel("σ (mmag)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_plot:
        name_list  = filename.rsplit('/', 1)
        add_dir    = f'/{ra}_{dec}/' if (ra is not None and dec is not None) else ''
        subpath    = name_list[0] + add_dir
        os.makedirs(subpath, exist_ok=True)
        name_clean = subpath + 'precision_sigma_' + name_list[1].replace('.parquet', '.png')
        print(name_clean)
        plt.savefig(name_clean)
        plt.close()
    else:
        plt.show()

def plot_outlier_lightcurves(df: pd.DataFrame, 
                            agg_df: pd.DataFrame,
                            ra: float | None = None,
                            dec: float | None = None,
                            save_plot=False, 
                            filename =None) -> None:
    
    """Scatter plot of raw light curves for all >3σ objects."""
    outliers   = agg_df[agg_df["sigma_level"] == 4].index
    df_3sigma  = df[df["object_index"].isin(outliers)]
    codes      = df_3sigma["object_index"].astype("category").cat.codes

    fig, ax = plt.subplots(figsize=(14, 6))
    sc = ax.scatter(df_3sigma["OBSMJD"], df_3sigma["MAG_4_TOT_AB"],
                    c=codes, s=10, cmap="tab10")
    plt.colorbar(sc, label="object_index")
    ax.set_xlabel("MJD")
    ax.set_ylabel("MAG_4_TOT_AB")
    plt.tight_layout()
    plt.show()

    if save_plot:
        name_list = filename.rsplit('/', 1)
        add_dir = ''
        if ra and dec:
            add_dir =f'/{ra}_{dec}'
        name_clean = name_list[0]+add_dir+'/rejected_precision_sigma_'+name_list[1].replace('.parquet', '.png')
        # print(name_clean)
        plt.savefig(name_clean)


def remove_outliers(df: pd.DataFrame, agg_df: pd.DataFrame,verbose=False) -> pd.DataFrame:
    """Return df with all >3σ objects removed."""
    outliers = agg_df[agg_df["sigma_level"] == 4].index
    if verbose:
        print(outliers)

    df_clean = df[~df["object_index"].isin(outliers)]
    print(f"Objects before: {df['object_index'].nunique()}")
    print(f"Objects after : {df_clean['object_index'].nunique()}")
    return df_clean


# ── main call ─────────────────────────────────────────────────────────────────

def run_sigma_filtering(df: pd.DataFrame,
                        edges: np.ndarray | None = None, 
                        ra: float | None = None,
                        dec: float | None = None,
                        agn_indices: np.ndarray | None = None,
                        plot: bool = False,
                        save_plt: bool = False,
                        save_file: bool = False,
                        filename: str | None = None,
                        cs_list: list | None = None) -> pd.DataFrame:
    """
    Full pipeline: aggregate → locus → classify → plot → clean.
    Returns df_clean (>3σ objects removed).
    """
    if edges is None:
        edges = np.arange(13, 23.5, 0.5)

    agg_df              = aggregate_lightcurves(df)
    cx_arr, my_arr, sigma_arr = build_sigma_locus(agg_df, edges)
    agg_df              = assign_sigma_levels(agg_df, cx_arr, my_arr, sigma_arr)

    # resolve target before remove_outliers — target may be variable/flagged
    tgt_obj_idx = agn_indices
    # if ra is not None and dec is not None:
        # srcs        = df.groupby('object_index')[['ALPHAWIN_REF', 'DELTAWIN_REF']].first().dropna()
        # tgt_obj_idx = _find_target_obj(ra, dec, df)
    if (agn_indices is None) and (ra is not None) and (dec is not None):
        tgt_obj_idx = _find_target_obj(ra, dec, df)
    # else:
    #     print('agn object_index or coordinates must be provided')

    clean_df = remove_outliers(df, agg_df)

    if plot or save_plt:
        
        plot_sigma_locus(agg_df, cx_arr, my_arr, sigma_arr,
                         tgt_obj_idx=tgt_obj_idx,
                         save_plot=save_plt,
                         filename=filename,
                         highlight=cs_list,
                         ra=ra, dec=dec)
        plot_outlier_lightcurves(df, agg_df,
                                 save_plot=save_plt,
                                 filename=filename)

    if save_file:
        if filename is None:
            print('filename is needed to save the clean file')
        else:    
            name_list = filename.rsplit('/', 1)
            name_clean = name_list[0]+'/sigma_filter/clean_'+name_list[1]
            print(f"file {name_clean} successfully saved")
            clean_df.to_parquet(name_clean)

    return clean_df


# ── usage ─────────────────────────────────────────────────────────────────────
# df_clean = run_sigma_filtering(df)
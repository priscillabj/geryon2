#!/usr/bin/env python3

import os, math, time,re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from linmix import linmix
#import psutil
path= os.environ['HOME']+'/Nextcloud/Doutorado/Forced_Phot/' 

# def SFarray(jd, mag, err):
#     """
#     Calculates an array with (m(ti)-m(tj)), with (err(t)^2+err(t+tau)^2) and
#     another with tau=dt

#     inputs:
#     jd: julian days array
#     mag: magnitudes array
#     err: error of magnitudes array

#     outputs:
#     tauarray: array with the difference in time (ti-tj)
#     sfarray: array with |m(ti)-m(tj)|
#     errarray: array with err(ti)^2+err(tj)^2
#     """
#     sfarray = []
#     tauarray = []
#     errarray = []
#     err_squared = err**2
#     len_mag = len(mag)
#     for i in range(len_mag):
#         for j in range(i+1, len_mag):
#             dm = mag[i] - mag[j]
#             sigma = err_squared[i] + err_squared[j]
#             dt = jd[j] - jd[i]
#             sfarray.append(np.abs(dm))
#             tauarray.append(dt)
#             errarray.append(sigma)
#     sfarray = np.array(sfarray)
#     tauarray = np.array(tauarray)
#     errarray = np.array(errarray)
#     return tauarray, sfarray, errarray

# ─────────────────────────────────────────────────────────────────────────────
# Vectorized SFarray  (drop-in replacement, identical outputs)
# ─────────────────────────────────────────────────────────────────────────────
def SFarray(jd, mag, err):
    """
    Vectorized version of the original SFarray.
    Outputs are identical: (tauarray, sfarray, errarray)
      tauarray : jd[j] - jd[i]          for all i < j  (always positive)
      sfarray  : |mag[i] - mag[j]|
      errarray : err[i]^2 + err[j]^2
    Safe for both list and NumPy array inputs.
    """
    jd  = np.asarray(jd,  dtype=float)
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)

    i_idx, j_idx = np.triu_indices(len(mag), k=1)

    tauarray = jd[j_idx]  - jd[i_idx]
    sfarray  = np.abs(mag[i_idx] - mag[j_idx])
    errarray = err[i_idx] ** 2 + err[j_idx] ** 2

    return tauarray, sfarray, errarray

def err_prop(dmag, prnt=False, plot=False):
    
    if len(dmag)>0:
        #sqrt_avg_dmag = np.sqrt(sq_avg_dmag)
        #plt.hist(sqrt_avg_dmag_corr)
        #plt.xlabel('sqrt mag diff')
        #plt.ylabel('frequency')
        # Calculate the quantiles
        median = np.percentile(dmag, 50)
        lower_bound = np.percentile(dmag, 16)
        upper_bound = np.percentile(dmag, 84)

        # 1-sigma errors (34% quantiles)
        lower_error = median - lower_bound
        upper_error = upper_bound - median

        if prnt:
            print(f"Median: {median:.2f}")
            print(f"Lower 1-sigma error: {lower_error:.3f}")
            print(f"Upper 1-sigma error: {upper_error:.3f}")
        if plot:
            plt.plot(dmag)
            plt.xlabel('Δmag')
            plt.ylabel('number of sources per bin')

        return lower_error,upper_error

def runSF(filenames,band ,phot='PSF', weight=False, clip=False, plot=False, save_plt=False, save=False):
    
    """
    Structure Function computatation for the ZTF DR 
    
    Parameters:
        filenames: a list of all lightcurves including their paths
        band: 'g', 'r' or 'i', set for ZTF in case there is a filter column of "ZTF_g", "ZTF_r" or "ZTF_i".
        phot: 'PSF' or 'AP' for PSF or aperture photometry 
        weight: boolean True or False. True makes a weighted average of dmags by its errors
        clip: boolean True or False. Clip in rolling windows: true divide the LC in chuncks of epochs and clip it separately 
        plot: boolean True or False. True for showing the LC and SF 
        save_plt: boolean True or False. True saves plot
        save: boolean True or False. True saves the final dictionary as pkl with the 
                                    SF for each time bin, lower and upper limit errors
    
    Returns
        List of dictionary with the binned SF and its errors, the number of data points in 
            each bin (#elements), name of the source(name) and median magnitude of the target for the
            chosen photometry

    Ex usage:
        list = [os.environ['HOME']+'/Downloads/2MASX_J19271951+6533539.csv']
        SF = runSF(list, 'g', plot=True)
    """
    
    # Rolling clip per dataframe group
    def clip_block(group):
        # Apply sigma clipping to the magnitude column
        clipped = sigma_clip(group[mag_column], sigma=2.5)
        
        # Create a mask of good (non-clipped) points
        good_points = ~clipped.mask
        
        # Return only the good points from the original group
        return group[good_points]
    
    # Estimate the errors per dataframe group
    # norm is the normalization of the error by the lenght of data points in each group
    def group_stats(group, norm=True):
        dmag_vals = group[f'dmag_{phot}'].dropna()
        if len(dmag_vals) == 0:
            return pd.Series([np.nan,np.nan, 0], index=['max_err','min_err' ,'count'])
        err = err_prop(dmag_vals)#, plot=True)#, prnt=True)
        #print(err)
        
        if norm==False:
            return pd.Series([
                #np.max(err) ,np.min(err),
                err[1] ,err[0],
                len(dmag_vals)
                ], index=['max_err','min_err' ,'count'])
        return pd.Series([
            #np.max(err) / np.sqrt(len(dmag_vals)),np.min(err) / np.sqrt(len(dmag_vals)),
            err[1] / np.sqrt(len(dmag_vals)),err[0] / np.sqrt(len(dmag_vals)),
            len(dmag_vals)
        ], index=['max_err','min_err' ,'count'])

    # weighted average, weighted by the errors
    def weighted_mean_mag(group):
        total_weight = group['weight'].sum()
        if total_weight <= 1e-10:  # Threshold for "zero"
            return np.nan  # or group['dmag'].mean() as fallback
        return np.average(group[f'dmag_{phot}'], weights=group['weight']) #<sigma_{noise}>
        #return np.average(group[f'sq_dmag_cs_{phot}'], weights=group['weight']) #<sigma_{noise}^2>
    
    def weighted_mean_sigma(group):
        total_weight = group['weight'].sum()
        if total_weight <= 1e-10:  # Threshold for "zero"
            return np.nan  # or group['dmag'].mean() as fallback
        #return np.average(group[f'dmag_{phot}'], weights=group['weight']) #<sigma_{noise}>
        return np.average(group[f'sq_dmag_cs_{phot}'], weights=group['weight']) #<sigma_{noise}^2>

    tic = time.perf_counter()

    # Define the range for the log-scale bins
    min_value = 0.0001   # Minimum value for binning
    max_value = 2500  # Maximum value for binning

    # Define the number of bins
    num_bins_required = np.ceil((np.log10(max_value) - np.log10(min_value)) / 0.5)
    num_bins = int(max(num_bins_required, 10))
    print(f'{num_bins} bins')
    #num_bins = 10

    # Calculate logarithmically spaced bin edges
    log_bins = np.logspace(np.log10(min_value), np.log10(max_value), num=num_bins)

    # gaps larger than 90 days separate LC windows
    gap_threshold = 90 
    
    # Magnitude Column
    mag_col = 'mag'
    
    # if phot=='PSF':
    #     filename = f'{phot}_{mag_col}_SF_clip{clip}.pkl' 
    # elif phot=='AP':
    #     filename = f'{phot}_AP{mag_col}_SF_clip{clip}.pkl'
    
    # filepath = os.path.join(path, filename)

    # # Load existing data (or initialize if first run)
    # if os.path.exists(filepath):
    #     with open(filepath, 'rb') as f:
    #         SF_dict = pickle.load(f)
    # else:
    #     SF_dict = []
    
    for file in filenames:
        print('target',file)
        #ztf1 = pd.read_parquet(file)
        ztf1 = pd.read_csv(file)
        ztf1 = ztf1[(ztf1['filtercode']==f'z{band}')]
        #print(len(ztf1))

        # Step 1: Replace 'mag' with 'flux_' (note the added underscore)
        modified_str = mag_col.replace('mag', 'flux_')
        # Step 2: Replace 'corr' with 'uJy'
        flux_col =  modified_str.replace('corr', 'uJy')
        
        if mag_col in ztf1.columns:
            if phot=='PSF':
                mag_column=mag_col
            elif phot=='AP':
                mag_column='AP'+mag_col
        else:
            parts=flux_col.split('_',1)
            fxunc = parts[0]+'unc_'+parts[1]+'_alt2'
            magunc = mag_col.replace('tot','unc')

            # Step 1: Replace 'mag' with 'flux_' (note the added underscore)
            modified_str = mag_col.replace('mag', 'flux_')
            # Step 2: Replace 'corr' with 'uJy'
            fx_col = modified_str.replace('corr', 'uJy')+'_corr'
            #print(fx_col)

            if phot=='PSF':
 
                ztf1.dropna(subset=[fx_col], inplace=True)

                ztf1[mag_col] = -2.5*np.log10(ztf1[fx_col].values*10**(-6))+8.90 #uJy corrected flux to AB mag
                ztf1[magunc] = 2.5/np.log(10)*(ztf1[fxunc].values/ztf1[fx_col].values)
                mag_column=mag_col
            
            elif phot=='AP':
                
                ztf1.dropna(subset=[f'AP{fx_col}'], inplace=True)

                ztf1[f'AP{mag_col}'] = -2.5*np.log10(ztf1[f'AP{fx_col}'].values*10**(-6))+8.90 #uJy corrected flux to AB mag
                ztf1[f'AP{magunc}'] = 2.5/np.log(10)*(ztf1[f'AP{fxunc}'].values/ztf1[f'AP{fx_col}'].values)
                mag_column='AP'+mag_col
            
        mztf=ztf1[mag_column].median()
        minztf = np.min(ztf1[mag_column])
        maxztf = np.max(ztf1[mag_column])

        #mag_err = mag_column.replace('tot', 'unc')
        mag_err = 'magerr'

        if (len(ztf1))>0:
            if clip:
                # Rolling sigma (per window)
                ztftime_diffs = ztf1['MJD'].diff()
                ztf1['block_id'] = (ztftime_diffs > gap_threshold).cumsum()  # This creates proper block numbering
                cleaned_ztfblocks = ztf1.groupby('block_id').apply(clip_block)
                ztf = cleaned_ztfblocks.reset_index(drop=True)

            else:
                ztf=ztf1

            print(f'median {phot} mag = {mztf:.2f}, max = {maxztf:.2f} and min = {minztf:.2f}')


            if mag_column not in ztf1.columns:
                print(f'{mag_column} not in target LC, please provide other magnitude column')
                #flux_col = 'cal'
                continue

            sf_corrLC = SFarray(ztf.mjd.values,ztf[mag_column].values,
                                ztf[mag_err].values)

            corr_sf = pd.DataFrame(sf_corrLC).transpose()
            corr_sf.columns =['dt', f'dmag_{phot}', 'err']
            corr_sf['dt']=np.abs(corr_sf['dt'])
            #corr_sf[f'dmag_{phot}'] = np.abs(corr_sf[f'dmag_{phot}'])
            corr_sf['log_bin'] = pd.cut(corr_sf['dt'], bins=log_bins)

            # Group by 'log_bin'
            grouped_corr = corr_sf.groupby('log_bin')

            if weight:
                corr_sf['weight'] = 1 / (corr_sf['err'] ** 2)
                binned_corr = grouped_corr.apply(weighted_mean_mag)

            else:
                binned_corr = grouped_corr[f'dmag_{phot}'].mean()

            sq_avg_dmag_corr = binned_corr**2

            SF = np.sqrt(sq_avg_dmag_corr)

            # Single groupby operation
            #print(f'Estimating target errors')
            stats_corr = grouped_corr.apply(group_stats).unstack()
            maxerr_sf = stats_corr['max_err'].dropna()
            minerr_sf = stats_corr['min_err'].dropna()
            ndmag = stats_corr['count'].dropna()

            print(f'    +{np.round(time.perf_counter()-tic, 2)} s: errors estimate')

            #final_dt = SF.fillna(0)
            sf = SF.dropna()
            dt_midbin = sf.index.categories[sf.index.codes].mid
            dt_lenbin = sf.index.categories[sf.index.codes].length/2

            target_intervals = sf.index
            maxerr_sf2 = maxerr_sf[maxerr_sf.index.isin(target_intervals)]
            minerr_sf2 = minerr_sf[minerr_sf.index.isin(target_intervals)]

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            ax[0].errorbar(ztf['mjd'], ztf[mag_column], yerr=ztf[mag_err], fmt='o', alpha=0.3, c='xkcd:fern',label='source')
            if clip:
                # Add vertical lines at block boundaries
                block_boundaries = ztf.groupby('block_id')['MJD'].first()[1:]  # First MJD of each block (except first)
                for i,boundary in enumerate(block_boundaries):
                    label = 'epochs window' if i == 0 else None
                    ax[0].axvline(boundary, color='xkcd:grey green', linestyle='--', linewidth=1,label=label)

            ax[0].invert_yaxis()
            ax[0].set_xlabel('MJD [days]')
            ax[0].set_ylabel('mag')
            ax[0].legend()

            ax[1].errorbar(dt_midbin,sf,xerr=dt_lenbin,yerr=(minerr_sf,maxerr_sf),
                            label='source',fmt= 'o',c='xkcd:fern')

            ax[1].set_xlabel('time lag [days]')
            ax[1].set_ylabel('SF [mag]')
            ax[1].set_xscale('log')
            #ax[1].set_ylim(-0.001, 0.05)
            ax[1].legend()

            name=os.path.basename(file).replace('.csv', '')
            if save_plt:
                plt.savefig(path+f'{phot}_{mag_column}_SF_clip{clip}_{name}_{band}.png')
                print('\n')
                print(f"Plot saved to {path} as {phot}_{mag_column}_SF_clip{clip}_{name}_{band}.png")

            if plot:
                plt.show()
            else:
                plt.close()  # Prevents display if plot=False

            SF_dict.append({'SF':SF, 'SFmaxerr':maxerr_sf2,'SFminerr':minerr_sf2,
                                '#elements': ndmag ,'name_source': name, f'{phot}_mag': mztf,'band':band})#,
        
        else:
            print('empty LC')
    
        if save:
            # Save back to the same file
            #print(f'{phot}_{mag_column}_SF_clip{clip}.pkl')
            name_file = f'{phot}_{mag_column}_SF_clip{clip}.pkl'
            #name_file = path+'test.pkl'
            with open(name_file, 'wb') as f:
                pickle.dump(SF_dict, f)
    
    return SF_dict


# --- Helper utilities moved to module level (no nested defs) ---

def clip_block(group, mag_column):
    """Sigma-clip a group's magnitude column and return only non-clipped rows."""
    clipped = sigma_clip(group[mag_column], sigma=2.5)
    good_points = ~clipped.mask
    return group.loc[good_points]


def group_stats(group, mag_col_name, norm=True):
    """Estimate error statistics for a binned group.
    Returns a pd.Series with index ['max_err','min_err','count']
    The function expects an err_prop() available in the namespace.
    """
    dmag_vals = group[mag_col_name].dropna()
    if len(dmag_vals) == 0:
        return pd.Series([np.nan, np.nan, 0], index=['max_err', 'min_err', 'count'])
    err = err_prop(dmag_vals)
    if norm is False:
        return pd.Series([err[1], err[0], len(dmag_vals)], index=['max_err', 'min_err', 'count'])
    return pd.Series([err[1] / np.sqrt(len(dmag_vals)), err[0] / np.sqrt(len(dmag_vals)), len(dmag_vals)],
                     index=['max_err', 'min_err', 'count'])


def weighted_mean_mag(group, phot, mag_column):
    total_weight = group['weight'].sum()
    if total_weight <= 1e-10:
        return np.nan
    return np.average(group[mag_column], weights=group['weight'])


def weighted_mean_sigma(group, phot, sq_dmag_column):
    total_weight = group['weight'].sum()
    if total_weight <= 1e-10:
        return np.nan
    return np.average(group[sq_dmag_column], weights=group['weight'])


# --- Main function (keeps original API) ---

def SF_wnoise_ref(filenames, band, phot='PSF', append=True, calstars=True, weight=False,
              clip=False, showallcs=False, plot=False, save_plt=False, save=False):
    """Refactored from your original function. Behavior and signature preserved.

    Key changes:
    - helper functions moved outside
    - variables that may be referenced later are initialized early with safe defaults
    - flattened conditionals and fewer repeated computations inside loops
    - same return value (SF_dict)

    NOTES / ASSUMPTIONS:
    - `path`, `SFarray`, and `err_prop` must be available in the calling scope (same as original).
    - This refactor aims to preserve exact behavior; subtle numeric differences could come from
      reordering of operations but should be negligible.
    """

    tic = time.perf_counter()

    # --- constants and configuration ---
    pattern = r"(\d+\.\d+)_(-?\d+\.\d+)"
    gap_threshold = 90

    mag_col = 'magtot_clr_corr'

    # bins for log scale
    min_value = 0.5
    max_value = 2500
    log_min = np.log10(min_value)  # -4.0
    log_max = np.log10(max_value)  # ~3.39794

    # Calculate number of bins needed for 0.1 dex spacing
    num_bins_required = int(np.ceil((log_max - log_min) / 0.1)) + 1
    #num_bins_required = np.ceil((np.log10(max_value) - np.log10(min_value)) / 0.5)
    num_bins = max(num_bins_required, 10)
    log_bins = np.logspace(log_min, log_max, num=num_bins)

    # Prepare output container
    SF_dict = []

    # Prepare some variables that were conditionally defined in the original
    cs1 = pd.DataFrame()
    cs = pd.DataFrame()

    # Determine output filename (same pattern as you used)
    if phot == 'PSF':
        filename = f'{phot}_{mag_col}_SF_clip{clip}_finebin_{band}band..pkl'
    elif phot == 'AP':
        filename = f'{phot}_AP{mag_col}_SF_clip{clip}_finebin_{band}band..pkl'
    # else:
    #     filename = f'{phot}_{mag_col}_SF_clip{clip}2.pkl'

    filepath = os.path.join(path, filename)

    # load existing SF_dict when requested
    if append and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            SF_dict = pickle.load(f)

    if band == 'g':
        color = 'mediumseagreen'
        #color2 = 'darkgreen'
        color2 = 'seagreen'
    elif band == 'r':
        color = 'firebrick'
        color2 = 'maroon'
    elif band =='i':
        color = 'gold'
        color2 = 'goldenrod'

    # loop over files (each file == target light curve)
    for file in filenames:
        print('target', file)
        # ztf1 = pd.read_csv(file)
        ztf1 = pd.read_parquet(file)
        ztf1 = ztf1[(ztf1['filter'] == f'ZTF_{band}')]

        # Build flux <-> mag column names
        modified_str = mag_col.replace('mag', 'flux_')
        flux_col = modified_str.replace('corr', 'uJy')

        # Decide mag_column and possibly convert flux->mag when mag column missing
        mag_column = None
        if mag_col in ztf1.columns:
            mag_column = mag_col if phot == 'PSF' else 'AP' + mag_col
        else:
            parts = flux_col.split('_', 1)
            fxunc = parts[0] + 'unc_' + parts[1] + '_alt2'
            magunc = mag_col.replace('tot', 'unc')
            fx_col = modified_str.replace('corr', 'uJy') + '_corr'

            if phot == 'PSF':
                ztf1.dropna(subset=[fx_col], inplace=True)
                ztf1[mag_col] = -2.5 * np.log10(ztf1[fx_col].values * 10 ** (-6)) + 8.90
                ztf1[magunc] = 2.5 / np.log(10) * (ztf1[fxunc].values / ztf1[fx_col].values)
                mag_column = mag_col
            else:  # AP
                apfx = f'AP{fx_col}'
                ztf1.dropna(subset=[apfx], inplace=True)
                ztf1[f'AP{mag_col}'] = -2.5 * np.log10(ztf1[apfx].values * 10 ** (-6)) + 8.90
                ztf1[f'AP{magunc}'] = 2.5 / np.log(10) * (ztf1[f'AP{fxunc}'].values / ztf1[apfx].values)
                mag_column = 'AP' + mag_col

        # If still no mag column, skip
        if mag_column is None or mag_column not in ztf1.columns:
            print('no mag column after attempts, skipping file')
            continue

        # quick stats
        if len(ztf1) == 0:
            print('empty LC')
            continue

        mztf = ztf1[mag_column].median()
        minztf = np.min(ztf1[mag_column])
        maxztf = np.max(ztf1[mag_column])
        mag_err = mag_column.replace('tot', 'unc')

        # optional rolling sigma clipping by block
        if clip:
            ztftime_diffs = ztf1['mjd'].diff()
            ztf1['block_id'] = (ztftime_diffs > gap_threshold).cumsum()
            cleaned_ztfblocks = ztf1.groupby('block_id').apply(lambda g: clip_block(g, mag_column))
            ztf = cleaned_ztfblocks.reset_index(drop=True)
        else:
            ztf = ztf1

        print(f'median {phot} mag = {mztf:.2f}, max = {maxztf:.2f} and min = {minztf:.2f}')

        match = re.search(pattern, file)
        if not match or math.isnan(mztf):
            print(match, mztf)
            print('no re match or nan values found')
            continue

        ra = match.group(1)
        dec = match.group(2)

        # --- calibration stars handling ---
        cs1 = pd.DataFrame()
        if calstars:
            # handle case where mag_col absent in target LC
            if mag_col not in ztf1.columns:
                print(f'{mag_col} not in target LC, using calstars_cal.parquet instead')
                flux_col_used = 'cal'
            else:
                flux_col_used = flux_col

            try:
                cs_all = pd.read_parquet(path + f'calibration_sources/PS1/{ra}_{dec}/calstars_{flux_col_used}.parquet')
                print(path + f'calibration_sources/PS1/{ra}_{dec}/calstars_{flux_col_used}.parquet')            
            except Exception:
                cs_all = pd.DataFrame()

            if not cs_all.empty:
                cs1 = cs_all[(cs_all['filter'] == f'ZTF_{band}') &
                              (cs_all['CCDquadID'] == ztf['CCDquadID'].mode()[0])].reset_index(drop=True)

                if not cs1.empty:
                    # convert flux->mag for calstars if needed
                    if mag_col not in cs1.columns:
                        if phot == 'PSF':
                            cs1[mag_col] = -2.5 * np.log10(cs1[f'{flux_col_used}_corr'].values * 10 ** (-6)) + 8.90
                            cs1[mag_col.replace('tot', 'unc')] = 2.5 / np.log(10) * (
                                cs1[fxunc].values / cs1[f'{flux_col_used}_corr'].values)
                            cs1.dropna(subset=[mag_col], inplace=True)
                        else:
                            cs1[f'AP{mag_col}'] = -2.5 * np.log10(cs1[f'AP{flux_col_used}_corr'].values * 10 ** (-6)) + 8.90
                            cs1[f'AP{mag_col}'.replace('tot', 'unc')] = 2.5 / np.log(10) * (
                                cs1[f'AP{fxunc}'].values / cs1[f'AP{flux_col_used}_corr'].values)
                            cs1.dropna(subset=[f'AP{mag_col}'], inplace=True)
                
                else:
                    print('band/CCD filter left 0 cs')
                    continue

                # apply optional clipping to calstars
                if clip and not cs1.empty:
                    cstime_diffs = cs1['mjd'].diff()
                    cs1['block_id'] = (cstime_diffs > gap_threshold).cumsum()
                    cleaned_csblocks = cs1.groupby(['ra', 'block_id']).apply(lambda g: clip_block(g, mag_column))
                    cs = cleaned_csblocks.reset_index(drop=True)
                else:
                    cs = cs1

                n = len(cs['ra'].unique()) if not cs.empty else 0
                # keep only calstars within target magnitude range
                if not cs.empty:
                    print(f'{n} cs before mag filter')
                    #print(f'using lower lim {math.floor(np.min(ztf[mag_column]))} and {math.ceil(np.max(ztf[mag_column]))} upper lim')
                    # mag_mask = ((cs.groupby('ra')[mag_column].median() > math.floor(np.min(ztf[mag_column])) ) &
                    #             (cs.groupby('ra')[mag_column].median() < math.ceil(np.max(ztf[mag_column])) ) )
                    mag_mask = ((cs.groupby('ra')[mag_column].median() > np.min(ztf[mag_column])-0.1 ) &
                                (cs.groupby('ra')[mag_column].median() < np.max(ztf[mag_column])+0.1 ) )
                    cs = cs[(cs['ra'].isin(mag_mask[mag_mask].index))]
                
                else:
                    print('clipping left 0 cs')
                    continue
            
            else:
                print('empty cs')
                continue


        n = len(cs['ra'].unique()) if not cs.empty else 0
        print(f'source: {len(ztf)} epochs')
        print(f'{n} calib stars after mag filter: {len(cs)} epochs')

        # --- compute SF for source and calstars ---
        sf_corrLC = SFarray(ztf.mjd.values, ztf[mag_column].values, ztf[mag_err].values)
        corr_sf = pd.DataFrame(sf_corrLC).transpose()
        corr_sf.columns = ['dt', f'dmag_{phot}', 'err']
        corr_sf[f'dmag_{phot}'] = np.abs(corr_sf[f'dmag_{phot}'])
        corr_sf['log_bin'] = pd.cut(corr_sf['dt'], bins=log_bins)
        grouped_corr = corr_sf.groupby('log_bin')

        df_cs = pd.DataFrame()
        grouped_cs = None
        temp_avg_sq_dmag_cs = pd.Series(dtype=float)
        avg_sq_dmag_cs = pd.Series(dtype=float)

        if n > 1:
            sf_csLC = cs.groupby('ra').apply(lambda group: SFarray(group.mjd.values, group[mag_column].values,
                                                                      group[mag_err].values))
            sf_cs = pd.DataFrame(sf_csLC.tolist(), dtype=float, index=sf_csLC.index)
            sf_cs.columns = ['dt', f'dmag_{phot}', 'err']
            df_cs = sf_cs.apply(pd.Series.explode)
            df_cs.columns = ['dt', f'dmag_{phot}', 'err']
            df_cs = df_cs.loc[(df_cs['dt'] != 0)]
            df_cs['log_bin'] = pd.cut(df_cs['dt'], bins=log_bins)
            df_cs[f'dmag_{phot}'] = df_cs[f'dmag_{phot}'].astype('float')
            df_cs[f'sq_dmag_cs_{phot}'] = df_cs[f'dmag_{phot}'] ** 2
            grouped_cs = df_cs.groupby(['ra', 'log_bin'])

        else:
            print('not enough calib stars')
            continue

        # weighting or simple mean
        if weight:
            corr_sf['weight'] = 1 / (corr_sf['err'] ** 2)
            binned_corr = grouped_corr.apply(lambda g: weighted_mean_mag(g, phot, f'dmag_{phot}'))
            if n > 1:
                df_cs['weight'] = 1 / (df_cs['err'] ** 2)
                avg_sq_dmag_cs = grouped_cs.apply(lambda g: weighted_mean_sigma(g, phot, f'sq_dmag_cs_{phot}'))
        else:
            binned_corr = grouped_corr[f'dmag_{phot}'].mean()
            if n > 1:
                temp_avg_sq_dmag_cs = grouped_cs[f'sq_dmag_cs_{phot}'].mean()
                avg_sq_dmag_cs = temp_avg_sq_dmag_cs.unstack().median()

                # compute bin-level RMS and errors for cal stars
                binned_rms_dmag = np.sqrt(temp_avg_sq_dmag_cs).reset_index(name=f'dmag_{phot}')
                binstats_cs = binned_rms_dmag.groupby('log_bin').apply(lambda g: group_stats(g, f'dmag_{phot}', norm=False)).unstack()
                binmaxerr_cs = binstats_cs['max_err'].dropna()
                binminerr_cs = binstats_cs['min_err'].dropna()
        

        # compute source statistics
        sq_avg_dmag_corr = binned_corr ** 2
        dt = sq_avg_dmag_corr.dropna()

        stats_corr = grouped_corr.apply(lambda g: group_stats(g, f'dmag_{phot}')).unstack()
        maxerr_corr = stats_corr['max_err'].dropna()
        minerr_corr = stats_corr['min_err'].dropna()
        ndmag = stats_corr['count'].dropna()
        print(f'    +{np.round(time.perf_counter() - tic, 2)} s: source errors estimate')

        # combine target and calstars into SF
        if (calstars) and (not cs1.empty) and n > 1:
            dif = (np.pi / 2) * sq_avg_dmag_corr - avg_sq_dmag_cs
            SF = np.sqrt(dif)
            stats_cs = df_cs.groupby('log_bin').apply(lambda g: group_stats(g, f'dmag_{phot}', norm=False)).unstack()
            maxerr_cs = stats_cs['max_err'].dropna()
            minerr_cs = stats_cs['min_err'].dropna()
            print(f'    +{np.round(time.perf_counter() - tic, 2)} s: stars errors estimate')

            maxerr_sf = np.sqrt(maxerr_corr ** 2 + maxerr_cs ** 2)
            minerr_sf = np.sqrt(minerr_corr ** 2 + minerr_cs ** 2)
            sf_cs = np.sqrt(avg_sq_dmag_cs.dropna())
            dt_midbin_cs = sf_cs.index.categories[sf_cs.index.codes].mid
            dt_lenbin_cs = sf_cs.index.categories[sf_cs.index.codes].length / 2

            sf = np.sqrt((np.pi / 2) * dt)
            SF[SF.index.isin(dif[dif < 0].index)] = 0

            maxerr_sf[maxerr_sf.index.isin(dif[dif < 0].index)] = 0
            sf_dif = sf - sf_cs
            minerr_sf[maxerr_sf.index.isin(dif[dif < 0].index)] = np.abs(sf_dif[sf_dif < 0])
        else:
            SF = np.sqrt(sq_avg_dmag_corr)
            sf = SF.dropna()
            maxerr_sf = maxerr_corr
            minerr_sf = minerr_corr

        # final bin mid/len calculations (for plotting)
        dt_midbin = dt.index.categories[dt.index.codes].mid
        dt_lenbin = dt.index.categories[dt.index.codes].length / 2

        final_dt = SF.dropna()
        final_dt_midbin = final_dt.index.categories[final_dt.index.codes].mid
        final_dt_lenbin = final_dt.index.categories[final_dt.index.codes].length / 2
        target_intervals = final_dt.index

        maxerr_sf2 = maxerr_sf[maxerr_sf.index.isin(target_intervals)]
        minerr_sf2 = minerr_sf[maxerr_sf.index.isin(target_intervals)]

        # --- plotting (preserved behavior) ---
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

        if (calstars) and (not cs1.empty) and n > 1:
            try:
                ax[0].errorbar(cs['mjd'], cs[mag_column], yerr=cs[mag_err], fmt='o', alpha=0.1, c='grey', label='cs')
            except Exception:
                pass

        ax[0].errorbar(ztf['mjd'], ztf[mag_column], yerr=ztf[mag_err], fmt='o', alpha=0.7, c=color, label='source')

        if clip:
            block_boundaries = ztf.groupby('block_id')['mjd'].first()[1:]
            for i, boundary in enumerate(block_boundaries):
                label = 'epochs window' if i == 0 else None
                ax[0].axvline(boundary, color='xkcd:grey green', linestyle='--', linewidth=1, label=label)

        ax[0].invert_yaxis()
        ax[0].set_xlabel('MJD [days]', size =25)
        ax[0].set_ylabel('mag', size =25)
        ax[0].tick_params(labelsize=20)
        ax[0].legend()

        ax[1].errorbar(dt_midbin, sf, xerr=dt_lenbin, yerr=(minerr_corr, maxerr_corr),
                       label='source', fmt='o', c=color)

        if (calstars) and (not cs1.empty) and n > 1:
            ax[1].errorbar(dt_midbin_cs, sf_cs, xerr=dt_lenbin_cs, yerr=(binminerr_cs, binmaxerr_cs),
                           label='calib stars', fmt='+', c='lightgrey')

            pos_mask = dif.loc[target_intervals] > 0
            ax[1].errorbar(final_dt_midbin[pos_mask], final_dt[pos_mask],
                           xerr=final_dt_lenbin[pos_mask],
                           yerr=(minerr_sf2[pos_mask], maxerr_sf2[pos_mask]),
                           fmt='o', c=color2, label='source-stars', capsize=3)

            neg_mask = dif.loc[target_intervals] < 0
            ax[1].errorbar(final_dt_midbin[neg_mask], final_dt[neg_mask],
                           xerr=final_dt_lenbin[neg_mask],
                           yerr=(minerr_sf2[neg_mask], minerr_sf2[neg_mask]),
                           fmt='o', linestyle='none', c=color2, capsize=3)

            if showallcs and (not df_cs.empty):
                binned_df_cs = temp_avg_sq_dmag_cs.unstack(level='ra')
                dt_cs = binned_df_cs.index.categories[binned_df_cs.index.codes].mid
                err_cs = grouped_cs.apply(lambda g: group_stats(g, f'dmag_{phot}')).unstack(level='ra')
                for ra_col in binned_df_cs.columns:
                    ax[1].errorbar(dt_cs, np.sqrt(binned_df_cs[ra_col]),
                                   yerr=(err_cs['min_err'][ra_col], err_cs['max_err'][ra_col]),
                                   fmt='o', alpha=0.2)

        ax[1].axhline(y=0, color='xkcd:mocha', linestyle='-.')
        ax[1].set_xlabel('time lag [days]', size =25)
        ax[1].set_ylabel('SF [mag]', size =25)
        ax[1].tick_params(labelsize=20)
        ax[1].set_xscale('log')
        ax[1].set_xlim(4e-1,4e3)
        ax[1].legend()

        plt.tight_layout()
        if save_plt:
            plt.savefig(path + f'{phot}_{mag_column}_SF_clip{clip}_{ra}_{dec}_{band}_finebin.png')
            print(f"Plot saved to {path} as {phot}_{mag_column}_SF_clip{clip}_{ra}_{dec}_{band}_finebin.png")

        if plot:
            plt.show()
        else:
            plt.close()

        # append results (keeps original dictionary structure)
        SF_dict.append({'SF': SF, 'SFmaxerr': maxerr_sf, 'SFminerr': minerr_sf,
                        '#elements': ndmag, 'RA': ra, f'{phot}_mag': mztf, 'band': band})

        # periodic save
        if save:
            name_file = path + f'{phot}_{mag_column}_SF_clip{clip}_finebin_{band}band.pkl'
            with open(name_file, 'wb') as f:
                pickle.dump(SF_dict, f)
            print(f'pkl file sucessfully saved in {name_file}')
            print('\n')
    
    print('\n')
    return SF_dict

def SF_wnoise(mag_col,time_col, mag_err, cs_all=None,clip=False, weight=False, color='red',
                showallcs=False, plot=False, save_plt=False, save=False):
    """Refactored from your original function. Behavior and signature preserved."""

    tic = time.perf_counter()

    # --- constants and configuration ---
    pattern = r"(\d+\.\d+)_(-?\d+\.\d+)"
    gap_threshold = 90

    #mag_col = 'magtot_clr_corr'

    # bins for log scale
    min_value = 0.5
    max_value = 2500
    log_min = np.log10(min_value)  # -4.0
    log_max = np.log10(max_value)  # ~3.39794

    # Calculate number of bins needed for 0.1 dex spacing
    num_bins_required = int(np.ceil((log_max - log_min) / 0.1)) + 1
    #num_bins_required = np.ceil((np.log10(max_value) - np.log10(min_value)) / 0.5)
    num_bins = max(num_bins_required, 10)
    log_bins = np.logspace(log_min, log_max, num=num_bins)

    # Prepare output container
    SF_dict = []

    # Prepare some variables that were conditionally defined in the original
    cs1 = pd.DataFrame()
    cs = pd.DataFrame()

    mag_column = 'mag'

    ztf1 = pd.DataFrame({f'{mag_column}': mag_col,'mag_err':mag_err, 'mjd':time_col})
    #print(ztf1)

    mztf = ztf1[mag_column].median()
    minztf = np.min(ztf1[mag_column])
    maxztf = np.max(ztf1[mag_column])
    # mag_err = mag_column.replace('tot', 'unc')
    n = 0 
    # optional rolling sigma clipping by block
    if clip:
        ztftime_diffs = ztf1['mjd'].diff()
        ztf1['block_id'] = (ztftime_diffs > gap_threshold).cumsum()
        cleaned_ztfblocks = ztf1.groupby('block_id').apply(lambda g: clip_block(g, mag_column))
        ztf = cleaned_ztfblocks.reset_index(drop=True)
    else:
        ztf = ztf1

    print(f'median mag = {mztf:.2f}, max = {maxztf:.2f} and min = {minztf:.2f}')

    # --- calibration stars handling ---
    cs1 = pd.DataFrame()
    if cs_all is not None:
        # handle case where mag_col absent in target LC
        if (mag_column not in ztf.columns) and (flux_col in ztf.columns):
            print(f'{mag_column} not in target LC, converting flux into mag')
            flux_col = 'flux'
            fxunc = 'flux_err'

        if not cs_all.empty:
            cs1 = cs_all[(cs_all['filter'] == f'ZTF_{band}') &
                            (cs_all['CCDquadID'] == ztf['CCDquadID'].mode()[0])].reset_index(drop=True)

            if not cs1.empty:
                # convert flux->mag for calstars if needed
                if mag_column not in cs1.columns:

                    cs1[mag_column] = -2.5 * np.log10(cs1[f'{flux_col}'].values * 10 ** (-6)) + 8.90
                    cs1['mag_err'] = 2.5 / np.log(10) * (
                        cs1[fxunc].values / cs1[f'{flux_col}'].values)
                    cs1.dropna(subset=[mag_column], inplace=True)

            # apply optional clipping to calstars
            if clip and not cs1.empty:
                cstime_diffs = cs1['mjd'].diff()
                cs1['block_id'] = (cstime_diffs > gap_threshold).cumsum()
                cleaned_csblocks = cs1.groupby(['ra', 'block_id']).apply(lambda g: clip_block(g, mag_column))
                cs = cleaned_csblocks.reset_index(drop=True)
            else:
                cs = cs1

            # keep only calstars within target magnitude range
            if not cs.empty:
                #print(f'using lower lim {math.floor(np.min(ztf[mag_column]))} and {math.ceil(np.max(ztf[mag_column]))} upper lim')
                # mag_mask = ((cs.groupby('ra')[mag_column].median() > math.floor(np.min(ztf[mag_column])) ) &
                #             (cs.groupby('ra')[mag_column].median() < math.ceil(np.max(ztf[mag_column])) ) )
                mag_mask = ((cs.groupby('ra')[mag_column].median() > np.min(ztf[mag_column]) ) &
                            (cs.groupby('ra')[mag_column].median() < np.max(ztf[mag_column]) ) )
                cs = cs[(cs['ra'].isin(mag_mask[mag_mask].index))]
        else:
            print('band/CCD filter left 0 cs')


        n = len(cs['ra'].unique()) if not cs.empty else 0
        print(f'{n} calib stars: {len(cs)} epochs')
        df_cs = pd.DataFrame()

        grouped_cs = None
        temp_avg_sq_dmag_cs = pd.Series(dtype=float)
        avg_sq_dmag_cs = pd.Series(dtype=float)

        if n > 1:
            sf_csLC = cs.groupby('ra').apply(lambda group: SFarray(group.mjd.values, group[mag_column].values,
                                                                        group['mag_err'].values))
            sf_cs = pd.DataFrame(sf_csLC.tolist(), dtype=float, index=sf_csLC.index)
            sf_cs.columns = ['dt', f'dmag', 'err']
            df_cs = sf_cs.apply(pd.Series.explode)
            df_cs.columns = ['dt', f'dmag', 'err']
            df_cs = df_cs.loc[(df_cs['dt'] != 0)]
            df_cs['log_bin'] = pd.cut(df_cs['dt'], bins=log_bins)
            df_cs[f'dmag'] = df_cs[f'dmag'].astype('float')
            df_cs[f'sq_dmag_cs'] = df_cs[f'dmag'] ** 2
            grouped_cs = df_cs.groupby(['ra', 'log_bin'])

    else:
        print('no calibration stars used')

    print(f'source: {len(ztf)} epochs')
    
    # --- compute SF for source and calstars ---
    sf_corrLC = SFarray(ztf.mjd.values, ztf[mag_column].values, ztf['mag_err'].values)
    corr_sf = pd.DataFrame(sf_corrLC).transpose()
    corr_sf.columns = ['dt', f'dmag', 'err']
    corr_sf[f'dmag'] = np.abs(corr_sf[f'dmag'])
    corr_sf['log_bin'] = pd.cut(corr_sf['dt'], bins=log_bins)
    grouped_corr = corr_sf.groupby('log_bin')


    # weighting or simple mean
    if weight:
        corr_sf['weight'] = 1 / (corr_sf['err'] ** 2)
        binned_corr = grouped_corr.apply(lambda g: weighted_mean_mag(g,'' , f'dmag'))
        if n > 1:
            df_cs['weight'] = 1 / (df_cs['err'] ** 2)
            avg_sq_dmag_cs = grouped_cs.apply(lambda g: weighted_mean_sigma(g, '', f'sq_dmag_cs'))
    else:
        binned_corr = grouped_corr[f'dmag'].mean()
        if n > 1:
            temp_avg_sq_dmag_cs = grouped_cs[f'sq_dmag_cs'].mean()
            avg_sq_dmag_cs = temp_avg_sq_dmag_cs.unstack().median()

            # compute bin-level RMS and errors for cal stars
            binned_rms_dmag = np.sqrt(temp_avg_sq_dmag_cs).reset_index(name=f'dmag')
            binstats_cs = binned_rms_dmag.groupby('log_bin').apply(lambda g: group_stats(g, f'dmag', norm=False)).unstack()
            binmaxerr_cs = binstats_cs['max_err'].dropna()
            binminerr_cs = binstats_cs['min_err'].dropna()
    

    # compute source statistics
    sq_avg_dmag_corr = binned_corr ** 2
    dt = sq_avg_dmag_corr.dropna()

    stats_corr = grouped_corr.apply(lambda g: group_stats(g, f'dmag')).unstack()
    maxerr_corr = stats_corr['max_err'].dropna()
    minerr_corr = stats_corr['min_err'].dropna()
    ndmag = stats_corr['count'].dropna()
    print(f'    +{np.round(time.perf_counter() - tic, 2)} s: source errors estimate')

    # combine target and calstars into SF
    if (cs_all is not None) and (not cs1.empty) and n > 1:
        dif = (np.pi / 2) * sq_avg_dmag_corr - avg_sq_dmag_cs
        SF = np.sqrt(dif)
        stats_cs = df_cs.groupby('log_bin').apply(lambda g: group_stats(g, f'dmag', norm=False)).unstack()
        maxerr_cs = stats_cs['max_err'].dropna()
        minerr_cs = stats_cs['min_err'].dropna()
        print(f'    +{np.round(time.perf_counter() - tic, 2)} s: stars errors estimate')

        maxerr_sf = np.sqrt(maxerr_corr ** 2 + maxerr_cs ** 2)
        minerr_sf = np.sqrt(minerr_corr ** 2 + minerr_cs ** 2)
        sf_cs = np.sqrt(avg_sq_dmag_cs.dropna())
        dt_midbin_cs = sf_cs.index.categories[sf_cs.index.codes].mid
        dt_lenbin_cs = sf_cs.index.categories[sf_cs.index.codes].length / 2

        sf = np.sqrt((np.pi / 2) * dt)
        SF[SF.index.isin(dif[dif < 0].index)] = 0

        maxerr_sf[maxerr_sf.index.isin(dif[dif < 0].index)] = 0
        sf_dif = sf - sf_cs
        minerr_sf[maxerr_sf.index.isin(dif[dif < 0].index)] = np.abs(sf_dif[sf_dif < 0])
    else:
        SF = np.sqrt(sq_avg_dmag_corr)
        sf = SF.dropna()
        maxerr_sf = maxerr_corr
        minerr_sf = minerr_corr

    # final bin mid/len calculations (for plotting)
    dt_midbin = dt.index.categories[dt.index.codes].mid
    dt_lenbin = dt.index.categories[dt.index.codes].length / 2

    final_dt = SF.dropna()
    final_dt_midbin = final_dt.index.categories[final_dt.index.codes].mid
    final_dt_lenbin = final_dt.index.categories[final_dt.index.codes].length / 2
    target_intervals = final_dt.index

    maxerr_sf2 = maxerr_sf[maxerr_sf.index.isin(target_intervals)]
    minerr_sf2 = minerr_sf[maxerr_sf.index.isin(target_intervals)]

    if plot:
        # --- plotting (preserved behavior) ---
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

        if (cs_all is not None) and (not cs1.empty) and n > 1:
            try:
                ax[0].errorbar(cs['mjd'], cs[mag_column], yerr=cs['mag_err'], fmt='o', alpha=0.1, c='grey', label='cs')
            except Exception:
                pass

        ax[0].errorbar(ztf['mjd'], ztf[mag_column], yerr=ztf['mag_err'], fmt='o', alpha=0.7, c=color, label='source')

        if clip:
            block_boundaries = ztf.groupby('block_id')['mjd'].first()[1:]
            for i, boundary in enumerate(block_boundaries):
                label = 'epochs window' if i == 0 else None
                ax[0].axvline(boundary, color='xkcd:grey green', linestyle='--', linewidth=1, label=label)

        ax[0].invert_yaxis()
        ax[0].set_xlabel('MJD [days]', size =25)
        ax[0].set_ylabel('mag', size =25)
        ax[0].tick_params(labelsize=20)
        ax[0].legend()

        ax[1].errorbar(dt_midbin, sf, xerr=dt_lenbin, yerr=(minerr_corr, maxerr_corr),
                        label='source', fmt='o', c=color)

        if (cs_all is not None) and (not cs1.empty) and n > 1:
            ax[1].errorbar(dt_midbin_cs, sf_cs, xerr=dt_lenbin_cs, yerr=(maxerr_cs, minerr_cs),
                            label='calib stars', fmt='+', c='lightgrey')

            pos_mask = dif.loc[target_intervals] > 0
            ax[1].errorbar(final_dt_midbin[pos_mask], final_dt[pos_mask],
                            xerr=final_dt_lenbin[pos_mask],
                            yerr=(minerr_sf2[pos_mask], maxerr_sf2[pos_mask]),
                            fmt='o', c=color2, label='source-stars', capsize=3)

            neg_mask = dif.loc[target_intervals] < 0
            ax[1].errorbar(final_dt_midbin[neg_mask], final_dt[neg_mask],
                            xerr=final_dt_lenbin[neg_mask],
                            yerr=(minerr_sf2[neg_mask], minerr_sf2[neg_mask]),
                            fmt='o', linestyle='none', c=color2, capsize=3)

            if showallcs and (not df_cs.empty):
                binned_df_cs = temp_avg_sq_dmag_cs.unstack(level='ra')
                dt_cs = binned_df_cs.index.categories[binned_df_cs.index.codes].mid
                err_cs = grouped_cs.apply(lambda g: group_stats(g, f'dmag')).unstack(level='ra')
                for ra_col in binned_df_cs.columns:
                    ax[1].errorbar(dt_cs, np.sqrt(binned_df_cs[ra_col]),
                                    yerr=(err_cs['min_err'][ra_col], err_cs['max_err'][ra_col]),
                                    fmt='o', alpha=0.2)

        ax[1].axhline(y=0, color='xkcd:mocha', linestyle='-.')
        ax[1].set_xlabel('time difference [days]', size =25)
        ax[1].set_ylabel('SF [mag]', size =25)
        ax[1].tick_params(labelsize=20)
        ax[1].set_xscale('log')
        ax[1].set_xlim(4e-1,4e3)
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    if save_plt:
        plt.savefig(path + f'simulated_SF.png')
        print(f"Plot saved to {path} as f'simulated_SF.png'")

    # else:
    #     plt.close()

    # append results (keeps original dictionary structure)
    SF_dict.append({'SF': SF, 'SFmaxerr': maxerr_sf, 'SFminerr': minerr_sf})
                    #'#elements': ndmag, 'RA': ra, f'{phot}_mag': mztf, 'band': band})

    # periodic save
    if save:
        name_file = path + f'simulated_SF.pkl'
        with open(name_file, 'wb') as f:
            pickle.dump(SF_dict, f)
        print(f'pkl file sucessfully saved in {name_file}')
        print('\n')

    return SF_dict

def SF_basic(filenames, band, phot='PSF'):
    """Refactored from your original function. Behavior and signature preserved.

    Key changes:
    - helper functions moved outside
    - variables that may be referenced later are initialized early with safe defaults
    - flattened conditionals and fewer repeated computations inside loops
    - same return value (SF_dict)

    NOTES / ASSUMPTIONS:
    - `path`, `SFarray`, and `err_prop` must be available in the calling scope (same as original).
    - This refactor aims to preserve exact behavior; subtle numeric differences could come from
      reordering of operations but should be negligible.
    """

    tic = time.perf_counter()

    # --- constants and configuration ---
    pattern = r"(\d+\.\d+)_(-?\d+\.\d+)"


    mag_col = 'magtot_clr_corr'

    # bins for log scale
    min_value = 0.5
    max_value = 2500
    log_min = np.log10(min_value)  # -4.0
    log_max = np.log10(max_value)  # ~3.39794

    # Calculate number of bins needed for 0.1 dex spacing
    num_bins_required = int(np.ceil((log_max - log_min) / 0.1)) + 1
    #num_bins_required = np.ceil((np.log10(max_value) - np.log10(min_value)) / 0.5)
    num_bins = max(num_bins_required, 10)
    log_bins = np.logspace(log_min, log_max, num=num_bins)

    # Prepare output container
    SF_dict = []
    cs = pd.DataFrame()


    if band == 'g':
        color = 'mediumseagreen'
        #color2 = 'darkgreen'
        color2 = 'seagreen'
    elif band == 'r':
        color = 'firebrick'
        color2 = 'maroon'
    elif band =='i':
        color = 'gold'
        color2 = 'goldenrod'

    # loop over files (each file == target light curve)
    for file in filenames:
        print('target', file)
        ztf1 = pd.read_csv(file)
        ztf1 = ztf1[(ztf1['filter'] == f'ZTF_{band}')]

        # Build flux <-> mag column names
        modified_str = mag_col.replace('mag', 'flux_')
        flux_col = modified_str.replace('corr', 'uJy')

        # Decide mag_column and possibly convert flux->mag when mag column missing
        mag_column = None
        if mag_col in ztf1.columns:
            mag_column = mag_col if phot == 'PSF' else 'AP' + mag_col
        
        # If still no mag column, skip
        if mag_column is None or mag_column not in ztf1.columns:
            print('no mag column after attempts, skipping file')
            continue

        # quick stats
        if len(ztf1) == 0:
            print('empty LC')
            continue

        mztf = ztf1[mag_column].median()
        minztf = np.min(ztf1[mag_column])
        maxztf = np.max(ztf1[mag_column])
        mag_err = mag_column.replace('tot', 'unc')

        print(f'median {phot} mag = {mztf:.2f}, max = {maxztf:.2f} and min = {minztf:.2f}')

        match = re.search(pattern, file)
        if not match or math.isnan(mztf):
            print(match, mztf)
            print('no re match or nan values found')
            continue

        sf_corrLC = SFarray(ztf1.mjd.values, ztf1[mag_column].values, ztf1[mag_err].values)
        corr_sf = pd.DataFrame(sf_corrLC).transpose()
        corr_sf.columns = ['dt', f'dmag_{phot}', 'err']
        corr_sf[f'dmag_{phot}'] = np.abs(corr_sf[f'dmag_{phot}'])
        corr_sf['log_bin'] = pd.cut(corr_sf['dt'], bins=log_bins)
        grouped_corr = corr_sf.groupby('log_bin')
        print(grouped_corr.count())
        #plt.hist(grouped_corr.count())
    
    return grouped_corr


def check_memory_usage(threshold=0.9):
    """Check if memory usage is too high."""
    process = psutil.Process(os.getpid())
    memory_percent = process.memory_percent() / 100
    if memory_percent > threshold:
        print(f"Memory usage high: {memory_percent:.1%}, forcing cleanup...")
        gc.collect()
        return True
    return False

#for old_dict in new_dict:
def SF_linmix(old_dict,phot=False,verbose=False,amp_at=365,plot=False,save_plot=False,save_pkl=False):  
    
    interval_index = pd.IntervalIndex(old_dict['SF'].index)
    sf_mag=old_dict['SF'][(interval_index.left >= 1)&(interval_index.right <= 365)&(old_dict['SF']!=0)].dropna()
    dt_lenbin = sf_mag.index.categories[sf_mag.index.codes].length/2
    dt_data = sf_mag.index.categories[sf_mag.index.codes].mid
    log_dt = np.log10(dt_data)

    maxerr_sf2 = old_dict['SFmaxerr'][old_dict['SFmaxerr'].index.isin(sf_mag.index)]
    minerr_sf2 = old_dict['SFminerr'][old_dict['SFminerr'].index.isin(sf_mag.index)]

    sf_err=(maxerr_sf2+minerr_sf2)/2

    xerr= dt_lenbin/(dt_data*np.log(10))
    yerr = sf_err / (sf_mag * np.log(10))

    log_sf = np.log10(sf_mag)

    if phot:
        mag_key = list(old_dict.keys())[5]
        if mag_key.endswith('_mag'):
            phot = mag_key.split("_")[0]
        else:
            print(f'photometry is not defined')
            return
    else:
        phot=''

    # print(len(dt_data))
    if len(dt_data)<=3:
        print(f'too few data points to fit: {len(dt_data)} valid data')
        print("SKIPPING this dataset")
        # continue
        # pass
        return 

    # ===================================================================
    # DATA QUALITY CHECKS
    # ===================================================================
    # Check for NaNs or Infs
    if (np.any(np.isnan(log_dt)) or np.any(np.isnan(log_sf)) or 
        np.any(np.isinf(log_dt)) or np.any(np.isinf(log_sf))):
        print(f"SKIPPING: Data contains NaN or Inf values")
        # continue
        # pass
        return None
    
    if (np.any(np.isnan(xerr)) or np.any(np.isnan(yerr)) or
        np.any(np.isinf(xerr)) or np.any(np.isinf(yerr))):
        print(f"SKIPPING: Errors contain NaN or Inf values")
        # continue
        # pass
        return None
    
    # Check for zero or negative errors
    if np.any(xerr <= 0) or np.any(yerr <= 0):
        print(f"SKIPPING: Errors contain zero or negative values")
        # continue
        # pass
        return None
    
    # Check for extremely large errors (relative to data spread)
    #if np.mean(xerr) > 2*np.std(log_dt) or np.mean(yerr) > 2*np.std(log_sf):
    #    print(f"SKIPPING: Errors are too large relative to data spread")
    #    continue
    if verbose:
        print("=" * 70)
        print("LINMIX LOG-LOG FIT WITH ERRORS (SCALED DATA)")
        print("=" * 70)
        print(f"Data points: {len(dt_data)}")
        print(f"\nData ranges:")
        print(f"  log(dt): [{log_dt.min():.4f}, {log_dt.max():.4f}]")
        print(f"  log(sf): [{log_sf.min():.4f}, {log_sf.max():.4f}]")
        print(f"\nAssumed errors in linear space:")
        print(f"  sf_mag fractional error:  {np.mean(sf_err/sf_mag)*100:.1f}%")
        print(f"\nPropagated to log₁₀ space:")
        print(f"  log(dt) error: {np.mean(xerr):.4f} ± {np.std(xerr):.4f}")
        print(f"  log(sf) error: {np.mean(yerr):.4f} ± {np.std(yerr):.4f}")
    
    # ===================================================================
    # SCALE THE DATA
    # ===================================================================
    log_dt_mean = np.mean(log_dt)
    log_dt_std = np.std(log_dt)
    log_sf_mean = np.mean(log_sf)
    log_sf_std = np.std(log_sf)
    
    log_dt_scaled = (log_dt - log_dt_mean) / log_dt_std
    log_sf_scaled = (log_sf - log_sf_mean) / log_sf_std
    xerr_scaled = xerr / log_dt_std
    yerr_scaled = yerr / log_sf_std

    print(f"\nScaling parameters:")
    print(f"  log(dt): mean={log_dt_mean:.4f}, std={log_dt_std:.4f}")
    print(f"  log(sf): mean={log_sf_mean:.4f}, std={log_sf_std:.4f}")
    print("\nRunning LinMix MCMC on scaled data...")
    # ===================================================================

    # Run LinMix on SCALED data with timeout protection
    try:
        print(f'fitting {phot } photometry')
        lm = linmix.LinMix(log_dt_scaled, log_sf_scaled, xsig=xerr_scaled, ysig=yerr_scaled, nchains=2)
        lm.run_mcmc(miniter=1000, maxiter=3000, silent=True)
        print("LinMix complete!")
    except Exception as e:
        print(f"ERROR: LinMix failed with error: {e}")
        print("SKIPPING this dataset")
        # continue
        # pass
        return None

    # Extract results from SCALED fit
    chain_alpha_scaled = lm.chain['alpha']
    chain_beta_scaled = lm.chain['beta']
    
    # ===================================================================
    # TRANSFORM BACK TO ORIGINAL SCALE
    # ===================================================================
    # For a linear relationship: y_scaled = alpha_scaled + beta_scaled * x_scaled
    # where: y_scaled = (y - y_mean) / y_std
    #        x_scaled = (x - x_mean) / x_std
    # 
    # Solving for y:
    # y = y_std * y_scaled + y_mean
    # y = y_std * (alpha_scaled + beta_scaled * x_scaled) + y_mean
    # y = y_std * alpha_scaled + y_std * beta_scaled * (x - x_mean) / x_std + y_mean
    # y = [y_std * alpha_scaled + y_mean - beta_scaled * y_std * x_mean / x_std] + [beta_scaled * y_std / x_std] * x
    
    chain_beta = chain_beta_scaled * log_sf_std / log_dt_std
    chain_alpha = (chain_alpha_scaled * log_sf_std + log_sf_mean - 
                   chain_beta_scaled * log_sf_std * log_dt_mean / log_dt_std)
    # ===================================================================

    # Get median and uncertainties in ORIGINAL scale
    alpha_med = np.median(chain_alpha)
    alpha_err = np.percentile(chain_alpha, [16, 84])
    beta_med = np.median(chain_beta)
    beta_err = np.percentile(chain_beta, [16, 84])

    ## Convert to power law parameters
    A_at_1 = 10**alpha_med
    slope_med = beta_med

    # ===================================================================
    # NORMALIZATION: Choose your reference point
    # ===================================================================
    dt_ref = amp_at

    # Calculate amplitude at reference point
    A_ref_med = A_at_1 * dt_ref**beta_med

    # Also calculate uncertainties in A_ref using the posterior
    A_ref_samples = 10**chain_alpha * dt_ref**chain_beta
    A_ref_err = np.percentile(A_ref_samples, [16, 84])
    
    new_par = {f'{phot}_A_1':A_at_1,
               f'{phot}_A_{dt_ref}':A_ref_med,f'{phot}_A_maxerr':A_ref_err[1]-A_ref_med,
               f'{phot}_A_minerr': A_ref_med-A_ref_err[0],
                      f'{phot}_gamma': slope_med, f'{phot}_gamma_maxerr': beta_err[1]-beta_med,
               f'{phot}_gamma_minerr': beta_med-beta_err[0]}

    old_dict.update(new_par)
    
    if verbose:

        print("\n" + "=" * 70)
        print("RESULTS (TRANSFORMED BACK TO ORIGINAL SCALE)")
        print("=" * 70)
        print(f"Linear fit in log-log space:")
        print(f"  log₁₀(sf_mag) = α + β × log₁₀(dt_data)")
        print(f"\nFitted parameters (median with 68% credible interval):")
        print(f"  α (intercept): {alpha_med:.6f} +{alpha_err[1]-alpha_med:.6f} -{alpha_med-alpha_err[0]:.6f}")
        print(f"  β (slope):     {beta_med:.6f} +{beta_err[1]-beta_med:.6f} -{beta_med-beta_err[0]:.6f}")

        print(f"\nPower law (normalized at dt_data = 1):")
        print(f"  sf_mag = A × dt_data^β")
        print(f"  A(dt=1) = {A_at_1:.6e}")
        print(f"  β = {slope_med:.6f}")

        print(f"\nPower law (normalized at dt_data = {dt_ref}):")
        print(f"  sf_mag = A_ref × (dt_data / {dt_ref})^β")
        print(f"  A_ref = {A_ref_med:.6e} +{A_ref_err[1]-A_ref_med:.6e} -{A_ref_med-A_ref_err[0]:.6e}")
        print(f"  β = {slope_med:.6f} +{beta_err[1]-beta_med:.6f} -{beta_med-beta_err[0]:.6f}")
        print(f"  (This is sf_mag at dt_data = {dt_ref})")

        print(f"\nAlternative form:")
        print(f"  sf_mag = {A_ref_med:.6e} × (dt_data / {dt_ref})^{slope_med:.4f}")
        print("=" * 70)

    # Generate predictions with uncertainty
    dt_fit_range = np.logspace(np.log10(dt_data.min()), np.log10(dt_data.max()), 500)
    log_dt_range = np.log10(dt_fit_range)

    # Sample from posterior to get uncertainty band
    n_samples = 500
    sf_samples = np.zeros((n_samples, len(dt_fit_range)))
    for i in range(n_samples):
        idx = np.random.randint(len(chain_alpha))
        alpha_s = chain_alpha[idx]
        beta_s = chain_beta[idx]
        log_sf_s = alpha_s + beta_s * log_dt_range
        sf_samples[i] = 10**log_sf_s

    sf_fit_med = np.median(sf_samples, axis=0)
    sf_fit_lower = np.percentile(sf_samples, 16, axis=0)
    sf_fit_upper = np.percentile(sf_samples, 84, axis=0)
    if plot:
        # Create plots
        fig = plt.figure(figsize=(16, 10))

        # 1. Posterior distributions
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(chain_alpha, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(alpha_med, color='r', linestyle='--', linewidth=2, label=f'Median: {alpha_med:.4f}')
        ax1.set_xlabel('α (intercept)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Posterior: Intercept', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(chain_beta, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(beta_med, color='r', linestyle='--', linewidth=2, label=f'Median: {beta_med:.4f}')
        ax2.set_xlabel('β (slope)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Posterior: Slope', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(chain_alpha, chain_beta, alpha=0.3, s=1, c='black')
        ax3.axvline(alpha_med, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(beta_med, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('α (intercept)', fontsize=12)
        ax3.set_ylabel('β (slope)', fontsize=12)
        ax3.set_title('Parameter Correlation', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 2. Linear scale fit
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(dt_fit_range, sf_fit_med, 'r-', linewidth=2, label='Median fit')
        ax4.fill_between(dt_fit_range, sf_fit_lower, sf_fit_upper, alpha=0.3, 
                         color='red', label='68% credible interval')
        ax4.errorbar(dt_data, sf_mag, xerr=dt_lenbin, yerr=sf_err,
                     fmt='o', markersize=8, color='blue', ecolor='blue', capsize=4,
                     label='Data with errors', zorder=5)
        ax4.set_xlabel('dt_data', fontsize=12)
        ax4.set_ylabel('sf_mag', fontsize=12)
        ax4.set_title('Linear Scale', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 3. Log-log scale fit
        ax5 = plt.subplot(2, 3, 5)
        ax5.loglog(dt_fit_range, sf_fit_med, 'r-', linewidth=2, label='Median fit')
        ax5.fill_between(dt_fit_range, sf_fit_lower, sf_fit_upper, alpha=0.3,
                         color='red', label='68% credible interval')
        ax5.errorbar(dt_data, sf_mag, xerr=dt_lenbin, yerr=sf_err,
                     fmt='o', markersize=8, color='blue', ecolor='blue', capsize=4,
                     label='Data with errors', zorder=5)
        ax5.set_xlabel('dt_data (log)', fontsize=12)
        ax5.set_ylabel('sf_mag (log)', fontsize=12)
        ax5.set_title(f'Log-Log Scale (slope β = {beta_med:.3f})', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, which='both')

        # 4. Direct log-log with linear axes
        ax6 = plt.subplot(2, 3, 6)
        log_sf_fit = alpha_med + beta_med * log_dt_range
        ax6.plot(log_dt_range, log_sf_fit, 'r-', linewidth=2, 
                 label=f'y = {alpha_med:.3f} + {beta_med:.3f}×x')
        ax6.errorbar(log_dt, log_sf, xerr=xerr, yerr=yerr,
                     fmt='o', markersize=8, color='blue', ecolor='blue', capsize=4,
                     label='Data with errors', zorder=5)
        ax6.set_xlabel('log₁₀(dt_data)', fontsize=12)
        ax6.set_ylabel('log₁₀(sf_mag)', fontsize=12)
        ax6.set_title('Direct Log-Log Space', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    if 'RA' in old_dict.keys():
        ra = old_dict['RA']
    else:
        ra = 'simulatedLC'
    if save_plot:
    
        plt.savefig(f'PSFlinmix_loglog_fit_{ra}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    if verbose:
        print("\n" + "=" * 70)
        print("INTERPRETATION:")
        print("=" * 70)
        print(f"Power law: sf_mag ∝ dt_data^{beta_med:.3f}")
        print(f"The exponent β = {beta_med:.3f} ± {(beta_err[1]-beta_err[0])/2:.3f}")
        print(f"LinMix accounts for errors in BOTH x and y variables")
        print(f"Data was scaled during fitting for numerical stability")
        print("=" * 70)

    if save_pkl:
        # Save back to the same file
        #print(f'{phot}_{mag_column}_SF_clip{clip}.pkl')
        name_file = path+f'{phot}_SFdict_fit_linmix_{ra}.pkl'
        #name_file = path+'test.pkl'
        with open(name_file, 'wb') as f:
            pickle.dump(old_dict, f)
        print(f'{name_file} successfully saved')
    
    return old_dict

import emcee
#import corner
from scipy.optimize import curve_fit

# Define the broken power law model
def broken_power_law_flat(dt, A, gamma1, dt_break):
    """Broken power law with flat slope after break."""
    return np.where(
        dt <= dt_break,
        A * (dt ** gamma1),
        A * (dt_break ** gamma1)  # Flat after break
    )

# 2. Define log-likelihood (Gaussian)
def log_likelihood(theta, dt, sf, sf_err):
    """Gaussian log-likelihood"""
    A, gamma1, dt_break = theta
    
    # Model prediction
    model = broken_power_law_flat(dt, A, gamma1, dt_break)
    
    # Chi-squared term
    chi2 = np.sum(((sf - model) / sf_err) ** 2)
    
    # Normalization term
    norm = np.sum(np.log(2 * np.pi * sf_err ** 2))
    
    return -0.5 * (chi2 + norm)

# 3. Define log-prior (physically motivated)
def log_prior(theta, dt):
    """Log-prior probability"""
    A, gamma1, dt_break = theta
    
    # Physical constraints
    if A <= 0:  # Amplitude must be positive
        return -np.inf
    if gamma1 < 0 or gamma1 > 5:  # Reasonable power-law slope
        return -np.inf
    if dt_break < min(dt) or dt_break > max(dt):  # Break within data range
        return -np.inf
    
    # Uniform prior in log-space for A (scale-invariant)
    # This is equivalent to prior ∝ 1/A
    return -np.log(A)  # log of 1/A prior

# 4. Define log-posterior
def log_posterior(theta, dt, sf, sf_err):
    """Log-posterior probability"""
    lp = log_prior(theta, dt)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dt, sf, sf_err)

def bpl_mcmc(old_dict,initial_guess = [0.5, 0.5, 100],model_check=False, mcmc_check=False,plot=False):
    # initial_guess = [A, gamma, dt_break]
    # model check: plot model and data. checks it fit is right

    interval_index = pd.IntervalIndex(old_dict['SF'].index)
    sf_mag=old_dict['SF'][(interval_index.left >= 1)&(interval_index.left <= 365)&(old_dict['SF']!=0)].dropna()
    dt = sf_mag.index.categories[sf_mag.index.codes].mid
    dt_lenbin = sf_mag.index.categories[sf_mag.index.codes].length/2
    maxerr_sf2 = old_dict['SFmaxerr'][old_dict['SFmaxerr'].index.isin(sf_mag.index)]
    minerr_sf2 = old_dict['SFminerr'][old_dict['SFminerr'].index.isin(sf_mag.index)]

    err=(maxerr_sf2+minerr_sf2)/2

    if (len(sf_mag)>=3):
        # # 5. Get good starting values (use curve_fit)
        popt, pcov = curve_fit(broken_power_law_flat, dt, sf_mag, p0=initial_guess)
        A_fit, alpha_fit, t_break_fit = popt
        y_broken_pl = broken_power_law_flat(dt, A_fit, alpha_fit, t_break_fit)

        if model_check:

            plt.errorbar(dt, sf_mag, yerr=err, fmt='o', capsize=5, label='Data')
            plt.plot(dt,y_broken_pl, label='broken power law')
            plt.axvline(t_break_fit, color='black',linestyle=':')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()

        try:
            popt, pcov = curve_fit(broken_power_law_flat, dt, sf_mag, 
                                p0=initial_guess, sigma=err, absolute_sigma=True)
            A_guess, gamma_guess, dt_break_guess = popt
        except:
            # Fallback if curve_fit fails
            A_guess, gamma_guess, dt_break_guess = 0.5, 0.5, 100
            print("curve_fit failed, using initial guesses")

        print(f"Initial guesses: A={A_guess:.3f}, γ={gamma_guess:.3f}, dt_break={dt_break_guess:.1f}")

        # 6. Set up MCMC
        ndim = 3  # [A, gamma1, dt_break]
        nwalkers = 32
        nsteps = 1000
        burnin = 100

        # Initialize walkers around the good guess
        pos = np.array([A_guess, gamma_guess, dt_break_guess]) + \
            1e-4 * np.random.randn(nwalkers, ndim) * np.array([A_guess, gamma_guess, dt_break_guess])

        # 7. Run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                        args=(dt, sf_mag, err))

        print("Running MCMC...")
        sampler.run_mcmc(pos, nsteps, progress=True)

        # 8. Analyze results
        # Discard burn-in
        samples = sampler.get_chain(discard=burnin, flat=True)

        # Get best-fit values (median)
        A_mcmc, gamma_mcmc, dt_break_mcmc = np.median(samples, axis=0)

        # Get uncertainties (16th and 84th percentiles)
        A_err = np.percentile(samples[:, 0], [16, 84])
        gamma_err = np.percentile(samples[:, 1], [16, 84])
        dt_break_err = np.percentile(samples[:, 2], [16, 84])

        print("\nMCMC Results:")
        print(f"A = {A_mcmc:.3f} +{A_err[1]-A_mcmc:.3f} -{A_mcmc-A_err[0]:.3f}")
        print(f"γ = {gamma_mcmc:.3f} +{gamma_err[1]-gamma_mcmc:.3f} -{gamma_mcmc-gamma_err[0]:.3f}")
        print(f"dt_break = {dt_break_mcmc:.1f} +{dt_break_err[1]-dt_break_mcmc:.1f} -{dt_break_mcmc-dt_break_err[0]:.1f}")

        SF_dict = {'RA':old_dict['RA'],'A': A_mcmc,'A_uperr': A_err[1]-A_mcmc, 'A_loerr': A_mcmc-A_err[0],
                    'gamma': gamma_mcmc,'gamma_uperr': gamma_err[1]-gamma_mcmc, 
                                        'gamma_loerr': gamma_mcmc-gamma_err[0], 
                    'dt_break': dt_break_mcmc, 'dt_break_uperr':dt_break_err[1]-dt_break_mcmc,
                                        'dt_break_loerr':dt_break_mcmc-dt_break_err[0]}

        if mcmc_check:
            # 9. Plot results
            fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
            samples_chain = sampler.get_chain()
            labels = ["A", "γ", "dt_break"]

            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples_chain[:, :, i], "k", alpha=0.3)
                ax.set_ylabel(labels[i])
                ax.axvline(burnin, color='red', linestyle='--')
            axes[-1].set_xlabel("Step number")
            plt.tight_layout()
            plt.show()

            # 10. Corner plot
            fig = corner.corner(samples, labels=labels,
                            truths=[A_mcmc, gamma_mcmc, dt_break_mcmc],
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True)
            plt.show()

        if plot:
            # 11. Plot best fit with data
            dt_fit = np.logspace(np.log10(min(dt)), np.log10(max(dt)), 100)
            y_fit = broken_power_law_flat(dt_fit, A_mcmc, gamma_mcmc, dt_break_mcmc)

            # Also plot random draws from posterior to show uncertainty
            plt.figure(figsize=(10, 6))
            plt.errorbar(dt, sf_mag, yerr=err, fmt='o', capsize=3, 
                        alpha=0.7, label='Data', color='blue')

            # Plot 100 random posterior samples
            inds = np.random.randint(len(samples), size=100)
            for ind in inds:
                A_samp, gamma_samp, dt_break_samp = samples[ind]
                y_samp = broken_power_law_flat(dt_fit, A_samp, gamma_samp, dt_break_samp)
                plt.plot(dt_fit, y_samp, 'gray', alpha=0.1, linewidth=0.5)

            # Plot median fit
            plt.plot(dt_fit, y_fit, 'r-', linewidth=2, label='Median fit')
            plt.axvline(dt_break_mcmc, color='black', linestyle=':', 
                        label=f'Break at {dt_break_mcmc:.1f} days')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Time Delay (days)')
            plt.ylabel('Structure Function')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        # 12. Check convergence (autocorrelation time)
        try:
            tau = sampler.get_autocorr_time()
            print(f"\nAutocorrelation time: {tau}")
            print(f"Effective sample size: {len(samples) / tau}")
        except:
            print("Could not compute autocorrelation time")
        
        return SF_dict
    else:
        print(f'not enough data points ({len(sf_mag)})')
        return None 

def parse_to_dict(series_str):
    """
    Parse the string and return a dictionary with 'SF' as key.
    Returns: {'SF': parsed_series} or original value if parsing fails
    """
    if not isinstance(series_str, str):
        # If it's already a dictionary or something else
        if isinstance(series_str, dict) and 'SF' in series_str:
            return series_str
        # If it's already a Series, wrap it
        elif isinstance(series_str, pd.Series):
            return {'SF': series_str}
        else:
            return {'SF': series_str}  # Keep as is
    
    try:
        # Parse the string to create Series
        lines = [line.strip() for line in series_str.split('\n') 
                if line.strip() and 'dtype:' not in line]
        
        if not lines:
            return {'SF': None}
        
        # Skip header if present
        if lines[0] == 'log_bin' or '(' not in lines[0]:
            name = lines[0]
            lines = lines[1:]
        else:
            name = 'log_bin'
        
        intervals = []
        values = []
        
        for line in lines:
            if not line or ']' not in line:
                continue
            
            # Find interval end
            bracket_pos = line.find(']')
            interval_str = line[:bracket_pos + 1]
            value_str = line[bracket_pos + 1:].strip()
            
            # Parse interval
            clean_interval = interval_str.replace('(', '').replace('[', '').replace(']', '')
            left, right = map(float, clean_interval.split(','))
            intervals.append((left, right))
            
            # Parse value
            if not value_str or value_str.lower() in ['nan', 'null', 'na', 'none']:
                values.append(np.nan)
            else:
                try:
                    values.append(float(value_str))
                except:
                    values.append(np.nan)
        
        if intervals:
            interval_index = pd.IntervalIndex.from_tuples(intervals, closed='right')
            series = pd.Series(values, index=interval_index, name=name)
            # Convert the index
            series_categorical = series.copy()
            series_categorical.index = pd.CategoricalIndex(
            [idx for idx in series.index],
            categories=[idx for idx in series.index]
)
            return {'SF': series_categorical}
        else:
            return {'SF': None}
            
    except Exception as e:
        print(f"Error parsing: {e}")
        return {'SF': series_str}  # Return original as fallback


# Function to recreate old_dict format from parsed data
def format_SFdf(row, columns=['RA_x','DEC','SF','SFmaxerr',
                            'SFminerr', 'PSF_A_365', 'PSF_gamma', 
                                'PSF_A_minerr', 'PSF_A_maxerr']):
    """
    Create the old dictionary format from a DataFrame row.
    Assumes row has: SF_series, and other columns like SFmaxerr, SFminerr, etc.
    """
    old_dict = {}
    
    # Add SF Series
    #if 'SF_dict' in row and isinstance(row['SF_dict']['SF'], pd.Series):
        #print('yes')
    #    old_dict['SF'] = row['SF_dict']['SF']
    #print('oi')
    # Add other parameters if they exist
    for key in  columns:# ['RA_x','DEC','SF','SFmaxerr', 'SFminerr', 'PSF_A_365', 'PSF_gamma', 
                #'PSF_A_minerr', 'PSF_A_maxerr']:
        if key in row:
            #print(key)
            # Check if it's also a Series that needs parsing
            if isinstance(row[key], str):# and 'IntervalIndex' in row[key]:
                # Parse it
                parsed = parse_to_dict(row[key])
                old_dict[key] = parsed['SF'] if 'SF' in parsed else row[key]
            else:
                old_dict[key] = row[key]
    
    return old_dict


def plot_SF(old_dict,band, model,phot='PSF',use_all_points=True,label=True,color_data='blue'):
    if band == 'g':
        color = 'mediumseagreen'
        #color2 = 'darkgreen'
        color2 = 'seagreen'
        # color2 = 'blue'
    elif band == 'r':
        color = 'firebrick'
        # color2 = 'maroon'
        # color2 = 'red'
        color2 = '#E17701'
    elif band =='i':
        color = 'gold'
        color2 = 'goldenrod'

    #old_dict = SF_dict[2]
    interval_index = pd.IntervalIndex(old_dict['SF'].index)
    sf_mag_lim=old_dict['SF'][(interval_index.left >= 1)&(interval_index.right < 365)&(old_dict['SF']!=0)].dropna()
    
    if not use_all_points:
        print(sf_mag_lim[-1:],f'excluded from fit')
        sf_mag = sf_mag_lim[:-1]
    else:
        sf_mag = sf_mag_lim
    
    dt_data = sf_mag.index.categories[sf_mag.index.codes].mid
    dt_lenbin = sf_mag.index.categories[sf_mag.index.codes].length/2

    maxerr_sf2 = old_dict['SFmaxerr'][old_dict['SFmaxerr'].index.isin(sf_mag.index)]
    minerr_sf2 = old_dict['SFminerr'][old_dict['SFminerr'].index.isin(sf_mag.index)]
    #err = (maxerr_sf2+minerr_sf2)/2
    #ax2.scatter(dt_data, sf_mag, s=100, c='blue', marker='o', 
    #        label='Data', zorder=5)

    sf = old_dict['SF'].dropna()
    dt = sf.index.categories[sf.index.codes].mid
    
    # print(old_dict['SF'])
    # print(dt)
    if label:
        lab1 = 'Data'
        lab2 = 'fitted Data'
    else:
        lab1 = None
        lab2 = None
    
    plt.errorbar(dt,sf, yerr=(old_dict['SFminerr'].values,old_dict['SFmaxerr'].values),
                    fmt='o',color=color_data,capsize=3,label=lab1)
    
    plt.errorbar(dt_data,sf_mag, yerr=(minerr_sf2,maxerr_sf2),
                    xerr=dt_lenbin,fmt='o',color=color2,capsize=3,label=lab2)

    if model=='single':
        mod = '_spl'
        # mod = ''
        # xlin = np.linspace(1, 2000, 50)

        # CORRECTED: A_365 is amplitude at 365 days
        # y_lm = old_dict['PSF_A_365'] * (xlin/365)**old_dict['PSF_gamma']
        y_fit = old_dict[f'{phot}_A_365{mod}'] * (dt_data/365)**old_dict[f'{phot}_gamma{mod}']

        # plt.plot(dt_data, y_fit, label='single power law', linewidth=2, c='#CC4F1B')
        plt.plot(dt_data, y_fit, label='single power law', linewidth=2, c=color2)
        #plt.fill_between(xlin, y_lm/np.abs(old_dict['PSF_A_minerr']), y_lm*np.abs(old_dict['PSF_A_maxerr']),
        #            alpha=0.3, facecolor='#FF9848')
        # print(len(dt_data),len(sf_mag),len(minerr_sf2),len(maxerr_sf2))
        #print(minerr_sf2.max(),maxerr_sf2.max())
        #plt.fill_between(xlin, y_lm+minerr_sf2.max(), y_lm-maxerr_sf2.max(),
        #               alpha=0.3, facecolor='#FF9848')
        # plt.fill_between(dt_data, sf_mag-(minerr_sf2.max()), sf_mag+(maxerr_sf2.max()),
        # alpha=0.3, facecolor='#FF9848')

        #if len(dt_data) > 5:
        #    ax2.scatter(dt_data[5:], sf_mag[5:], s=100, c='gray', marker='x', 
        #                label='Data (not used)', zorder=5)
        #ax2.axvline(x_break, color='green', linestyle='--', alpha=0.5, 
        #            label=f'Break at x={x_break:.2f}')

    elif model=='broken':
        # xlog = np.logspace(np.log10(min(dt)), np.log10(max(dt)), 100)

        y_fit = broken_power_law_flat(dt_data, old_dict['PSF_A_bpl'], old_dict['PSF_gamma_bpl'], old_dict['dt_break_bpl'])
        # Plot median fit
        bk = old_dict['dt_break_bpl']
        plt.plot(dt_data, y_fit, c='#FF958F',linewidth=2, label='broken power law')
        plt.axvline(old_dict['dt_break_bpl'], color='black', linestyle=':', 
                    label=f'Break at {bk:.1f} days')

    plt.xlabel('Time diff (days)', fontsize=15)
    plt.ylabel('Structure Function (mag)', fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tick_params(axis='both', labelsize=12)
    plt.show()

    return None
    # return sf_mag,maxerr_sf2,minerr_sf2,y_fit

from scipy.stats import chi2

# CHI-SQUARE MODEL COMPARISON FUNCTION
def chi2_stats(y_data, y_pred, n_params, model_name,sigma=1):
    """Calculate chi-square statistics for a model."""
    n_points = len(y_data)
    
    # Chi-square calculation
    chi2_val = np.sum(((y_data - y_pred) / sigma) ** 2)

    # Degrees of freedom
    dof = n_points - n_params
    
    # Reduced chi-square
    reduced_chi2 = chi2_val / dof if dof > 0 else np.inf
    
    # p-value (probability that chi-square could be this large by chance)
    p_value = 1 - chi2.cdf(chi2_val, dof) if dof > 0 else 0
    
    print(f"\n--- {model_name} ---")
    print(f"Parameters: {n_params}")
    print(f"χ² = {chi2_val:.3f}")
    print(f"Degrees of freedom = {dof}")
    print(f"Reduced χ² = {reduced_chi2:.3f}")
    print(f"p-value = {p_value:.3f}")
    
    return chi2_val, reduced_chi2, p_value
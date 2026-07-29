import re
import numpy as np
#from astropy.coordinates import SkyCoord,EarthLocation, AltAz
#import astropy.units as u
#from astropy.time import Time
import time
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
#from collections import Counter
#import corner
#from scipy.stats import median_abs_deviation as mad
#from carma_utils import *

bands = ['g', 'r', 'i']

# def WhichFiles(filenames,RA_Row,DEC_Row):
#     typeI_AGN=[]
#     find_filename=False
#     for i in filenames:
#         match = re.search(r'lc_(\d+.\d+)_(-?\d+.\d+)', i)
#         if match:
#             regex_ra = match.group(1)
#             regex_ra = float(regex_ra)
#             regex_dec = match.group(2)
#             regex_dec = float(regex_dec)

#             for j,k in zip(RA_Row,DEC_Row):
#                 j = float(j)
#                 k = float(k)
#                 if np.isclose(j, regex_ra, atol=0.0005) and np.isclose(k, regex_dec, atol=0.0005):
#                 #if (np.abs(j == regex_ra) and np.abs(k == regex_dec)):
#                     typeI_AGN.append(i)
#                     find_filename=True
#             if find_filename
#     return(typeI_AGN)

def radec_filename(i):
    #path = os.environ['HOME']+r"/Nextcloud/Doutorado/Forced_Phot/"
    #calpath = path+r'calibration_sources/PS1/'
    
    match = re.search(r'(\d+.\d+)_([-+]?\d+.\d+)', i)
    if match:
        regex_ra = match.group(1)
        regex_ra = float(regex_ra)
        regex_dec = match.group(2)
        regex_dec = float(regex_dec)

        return regex_ra, regex_dec

def WhichFiles(filenames, RA_Row, DEC_Row):
    typeI_AGN = []

    for j, k in zip(RA_Row, DEC_Row):
        j = float(j)
        k = float(k)

        find_filename = False  # RESET HERE ✅

        for i in filenames:
            match = re.search(r'lc_(\d+.\d+)_(-?\d+.\d+)', i)
            if match:
                regex_ra = float(match.group(1))
                regex_dec = float(match.group(2))

                if np.isclose(j, regex_ra, atol=0.0005) and np.isclose(k, regex_dec, atol=0.0005):
                    typeI_AGN.append(i)
                    find_filename = True
                    break   # optional but recommended

        if not find_filename:
            print(f'no filename found for coords {j}, {k}')

    return typeI_AGN


#data is the table/dataframe with the catalogue information, i.e.:
#data = pd.read_csv(os.environ['HOME']+'/Nextcloud/Doutorado/Forced_Phot/DR2-105monthcatalog-105m_all.csv')
#data =data.rename(columns={"wise_ra":"ra","wise_dec":"dec"})
def LC2VarFeat(json_varfeat, data): 
    
    #RA = data.wise_ra.values
    RA = data.wise_ra.replace(',','.', regex=True).values
    DEC = data.wise_dec.values
    
    AGN = data.AGN.values
    
    bat_index = data.bat_index.values
    sources = data.ctpt_name.values
    
    for i in np.arange(len(json_varfeat)):
        for j in np.arange(len(RA)):
            #print(coord[i]['RA'], type(coord[i]['RA']))
            json_varfeat[i]['RA']=float(json_varfeat[i]['RA'])
            #if (RA[j]=='243,555'):
            #    print(j)
            RA[j] = float(RA[j])
            DEC[j] = float(DEC[j])
            if (np.abs(json_varfeat[i]['RA'] - RA[j]) <= 0.0001) and (~np.isnan(RA[j])) and ('name' not in json_varfeat[i].keys()): #and (np.abs(DEC[j] - json_varfeat[i]['DEC'])<=0.0001) 
                if (AGN[j] == 'TRUE'):
                    name_dic={'name':sources[j],'bat_index':bat_index[j],'DEC': DEC[j] }
                    json_varfeat[i].update(name_dic)
                elif (AGN[j] == 'FALSE'):
                    print(f"{sources[j]} is not an AGN!")
                    #name_dic={'name': 'not AGN'}
                #    json_varfeat[i].update(name_dic)
                    
    return(json_varfeat)

def LC2VarFeatdf(varfeat_df, data): 
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = varfeat_df.copy()
    
    # Process RA and DEC from the data DataFrame
    RA = data.wise_ra.replace(',','.', regex=True).values
    DEC = data.wise_dec.values
    AGN = data.AGN.values
    bat_index = data.bat_index.values
    sources = data.ctpt_name.values
    
    # Convert RA/DEC to float for comparison
    RA = [float(ra) if not pd.isna(ra) else np.nan for ra in RA]
    DEC = [float(dec) if not pd.isna(dec) else np.nan for dec in DEC]
    
    # Initialize new columns
    result_df['name'] = None
    result_df['bat_index'] = None
    result_df['DEC'] = None
    
    for i in range(len(result_df)):
        current_ra = float(result_df.iloc[i]['RA'])
        
        for j in range(len(RA)):
            if (np.abs(current_ra - RA[j]) <= 0.0001) and (~np.isnan(RA[j])):
                if AGN[j] == 'TRUE':
                    result_df.at[i, 'name'] = sources[j]
                    result_df.at[i, 'bat_index'] = bat_index[j]
                    result_df.at[i, 'DEC'] = DEC[j]
                elif AGN[j] == 'FALSE':
                    print(f"{sources[j]} is not an AGN!")
                    
    return result_df

#table_mbh=pd.read_excel(os.environ['HOME']+'/Nextcloud/Doutorado/Forced_Phot/mbh_all_aug2022.xlsx')
#table_mbh = table_mbh.rename(columns={'14-150 Lum':'lx'})#, table_mbh.iloc[0].index[25]:'MBH_errm',table_mbh.iloc[0].index[26]:'MBH_errP'})
#if you have a df, you can use df.to_dict('records) to transform it into a json format
#table_mbh=pd.read_csv(os.environ['HOME']+'/Nextcloud/Doutorado/Forced_Phot/DR2_DR3_Best MBH - DR2_final_use_this.csv')
#table_mbh = table_mbh.rename(columns={'14-150 Lum':'lx'})#, table_mbh.iloc[0].index[25]:'MBH_errm',table_mbh.iloc[0].index[26]:'MBH_errP'})
#table_mbh.dropna(subset=['Best_M_BH'], inplace=True)

def AGNProp (json_varfeat, table_mbh):

    table_mbh.drop(table_mbh.index[1194:],inplace=True)
    table_mbh.replace('#VALUE!', np.nan, inplace=True)
    #table_mbh.dropna(subset = ['Best_M_BH', 'Edd_rat'], inplace=True)
    log_mbh=table_mbh.Best_M_BH.astype(float).values
    zbest =table_mbh.zbest.values
    Lx = table_mbh.lx.values
    bat_id=table_mbh.BAT_ID.values
    Edd= table_mbh.Edd_rat.astype(float).values
    Lbol=table_mbh.L_bol.astype(float).values
    Type = table_mbh.Type_105.values
    Le = Lbol/Edd

    st='bat_index'
    for i in np.arange(len(json_varfeat)):
        #if st in json_varfeat[i]:
        #    json_varfeat[i]['bat_index']=int(json_varfeat[i]['bat_index'])

        for j in np.arange(len(log_mbh)):

            if (json_varfeat[i]['bat_index'] == bat_id[j]) and (~np.isnan(zbest[j])) and (~np.isnan(Lx[j])) and (~np.isnan(log_mbh[j])) and ('LogM_BH' not in json_varfeat[i].keys()):
                BHmass={'LogM_BH':log_mbh[j], 'Lx':Lx[j],'z':zbest[j],'Redd':Edd[j] ,'LogRedd':np.log10(Edd[j]),'Ledd':Le[j], 'Lbol':Lbol[j] ,'nH':table_mbh.NH.values[j], 'clasf':Type[j]}
                json_varfeat[i].update(BHmass)

    return(json_varfeat) 


def AGNPropdf(df_varfeat, table_mbh,overwrite=False):
    
    # Clean table_mbh
    #cleaned = table_mbh.dropna(subset=['Best_M_BH', 'Edd_rat']).copy()
    cleaned = table_mbh.replace('#VALUE!', np.nan).copy()

    # Create derived columns
    # cleaned['LogM_BH'] = cleaned['Best_M_BH'].astype(float)
    cleaned['LogM_BH'] = pd.to_numeric(cleaned['Best_M_BH'], errors='coerce')
    cleaned['Redd']    = cleaned['Edd_rat'].astype(float)
    cleaned['LogRedd'] = np.log10(cleaned['Redd'])
    cleaned['Lbol']    = cleaned['L_bol'].astype(float)
    cleaned['Ledd']    = cleaned['Lbol'] / cleaned['Redd']
    
    # Columns to merge
    cols_to_add = [
        'BAT_ID', 'LogM_BH', 'lx', 'zbest', 'Redd', 'LogRedd',
        'Ledd', 'Lbol', 'NH', 'Type_105'
    ]
    
    mbh_subset = cleaned[cols_to_add].rename(columns={
        'BAT_ID': 'bat_index',
        'lx': 'Lx',
        'zbest': 'z',
        'Type_105': 'clasf'
    })
    #print(mbh_subset[mbh_subset['bat_index']==413])

    renamed_cols = ['bat_index', 'LogM_BH','Lx','z','Redd', 'LogRedd',
        'Ledd', 'Lbol', 'NH', 'clasf']

    feat_col = set(df_varfeat.columns)
    diff_col= list(set(renamed_cols).difference(feat_col))
    
    if len(diff_col) !=0:
        print(f'adding {diff_col} columns')

        if 'bat_index' not in diff_col:
            diff_col.append('bat_index')

        df_merged = df_varfeat.merge(
        mbh_subset[diff_col],
        on='bat_index',
        how='left'
            )

    
    else:
        print('columns already exist')
        df_merged = df_varfeat

    return df_merged


def ra_dec_inf (path):
    with open(path,'r') as the_file:
        data2 = [line.strip() for line in the_file.readlines()]
        for line in data2:
            if '# Requested input R' in line:
                RA_info = line
                RA_info = RA_info.replace('# Requested input R.A. = ','') #, inplace = True) #'',' degrees','',
                RA_info = float(RA_info.replace(' degrees',''))
            if '# Requested input D' in line:
                Dec_info = line
                Dec_info = Dec_info.replace('# Requested input Dec. = ','') #, inplace = True) #'',' degrees','',
                Dec_info = float(Dec_info.replace(' degrees',''))
        return RA_info,Dec_info

# Obtain airmass (secz) for observations CAUTION - THIS MUST BE DONE FOR SDSS OBJECTS
def airmass(df, path):
    RA_info, Dec_info = ra_dec_inf(path)
    #SDSS_Apache_Point = EarthLocation.of_site('Apache Point Observatory') #change for ZTF
    Samuel_Oschin = EarthLocation(lat=(33 + (21 + 29 / 60.) / 60.) * u.deg,
                        lon=-(116 + (51 + 43 / 60.) / 60.) * u.deg,
                        height = 1712 * u.m)
    df['time'] = Time(df['jd'], format='jd')
    #df['secz'] = SkyCoord(RA_info, Dec_info, unit='deg').transform_to(AltAz(obstime=df.time, location=SDSS_Apache_Point)).secz
    df['secz'] = SkyCoord(RA_info, Dec_info, unit='deg').transform_to(AltAz(obstime=df.time, location=Samuel_Oschin)).secz
    return df
#EarthLocation.get_site_names()
#EarthLocation.of_site('Palomar')

#typeI_AGN = lista com filenames 
#Master_OE = df com CARMA parameter space best fit results
def LCfeatCARMA(typeI_AGN,Master_OE):
    dirname = os.path.dirname(typeI_AGN[0])
    feat=[]
    p=2
    for i in typeI_AGN:
        #if os.path.basename(dirname) == 'LC':
        #    lc_df = friendly_use(i)
        #    lc_df = calc_some(lc_df ,'forcediffimflux')
        #    lc_df = airmass(lc_df, i)
            
        #elif os.path.basename(dirname) == 'test':
        #    lc_df = pd.read_csv(i)
            
        #print(i)
        lc_df = pd.read_csv(i)
        if (lc_df.size) != 0:
            for fid in bands:
                a=Master_OE.loc[(Master_OE['fid'] == fid)]
                #print(a.index.values)

                mask = ((lc_df['filter'] == f'ZTF_{fid}')) #& (lc_df.infobitssci==0) & (lc_df.nearestrefmag < 17)& 
                        #(lc_df.mag_tot > 14) & (lc_df.ccdid == lc_df.ccdid.mode()[0]))

                npts = len(lc_df[mask].mjd)
                #print(npts)
                if (npts > 30):
                    #print(npts)
                    mjd_lc = lc_df[mask].mjd.values
                    delta_t=[]
                    #print(p)
                    for j in range(npts-1):
                        delta_t.append(np.abs(mjd_lc[j]-mjd_lc[j+1]))
                    
                        #print(delta_t)
                        
                    #print(a.index.values)
                    p75=np.percentile(delta_t, 75)
                    p90=np.percentile(delta_t, 90)
                    median_dt = np.median(delta_t)
                    rep_dt=Counter(delta_t).most_common(1)[0][0]
                    nrep = Counter(delta_t).most_common(1)[0][1]
                    y = np.mean(lc_df[mask].mag_tot.values)
                    am= np.mean(lc_df[mask].secz.values)
                    meandt= np.mean(delta_t)
                    
                    #print(p75,meandt)
                    #match = re.search(r'corrlc_(\d+.\d+)_(-?\d+.\d+)', i)
                    match = re.search(r'lc_(\d+.\d+)_(-?\d+.\d+)', i)
                    if match:
                        regex_ra = match.group(1)
                        regex_ra = float(regex_ra)
                        regex_dec = match.group(2)
                        regex_dec = float(regex_dec)
                        #print(a.index.values)
                        
                        
                        for idx in a.index.values:
                            #print(idx)
                            #print(regex_ra, regex_dec)
                            #print(idx, a.RA.loc[idx], a.DEC.loc[idx])
                            if (a.RA.loc[idx]==regex_ra) and (a.DEC.loc[idx]==regex_dec):
                                #print(a.RA.loc[idx],a.DEC.loc[idx])
                                #min_dt, max_dt = a[[f"min_dt", f"max_dt"]].loc[idx].values.astype(float)
                                feat.append(dict(RA=regex_ra,DEC=regex_dec, fid=fid, mean_mag =y,
                                                    delta_mag=max(lc_df[mask].mag_tot)-min(lc_df[mask].mag_tot),
                                                    min_dt = a.min_dt.loc[idx], max_dt = a.max_dt.loc[idx],
                                                    freq_dt=rep_dt, nrep=nrep,median_dt=median_dt,p75=p75,
                                                    p90=p90,lcn=a.lcN.loc[idx], 
                                                    MCMCtau_perturb = a.mcmc_tau_perturb.loc[idx],
                                                    MCMCsigma_e = a.mcmc_sigma_e.loc[idx],
                                                    MCMCtau_decay = a.mcmc_tau_decay.loc[idx],
                                                    MCMCsigma_dho = a.mcmc_sigma_dho.loc[idx],
                                                    burned_MCMCtau_perturb = a.burned_mcmc_tau_perturb.loc[idx],
                                                    burned_MCMCsigma_e = a.burned_mcmc_sigma_e.loc[idx],
                                                    burned_MCMCtau_decay = a.burned_mcmc_tau_decay.loc[idx],
                                                    burned_MCMCsigma_dho = a.burned_mcmc_sigma_dho.loc[idx],
                                                    tau_perturb=a.perturb.loc[idx],sigma_e=a.sigma_e.loc[idx],
                                                    tau_decay=a.decay.loc[idx],sigma_dho=a.sigma_dho.loc[idx],
                                                    airmass=am))
                            #else:
                                #print(f'no df a, RA={a.RA.loc[idx]}, regex={regex_ra}, DEC={a.DEC.loc[idx]} e regex {regex_dec}')
                        
                        
                    else:
                        print(f'{i} found {match} matches')
    return(feat)

#gives a list with names of sources (with path included) whose the max errors of the parameters
#are in between x and y (closed intervale i.e., including those values)
def namelist(OE_files,x,y,parameter_names=np.array(['log_a1','log_a2','log_b0','log_b1'])):
    error1to2=[]
    for name in OE_files:
        ex = np.load(name)
        if ex['flatchain'].size != 0:
            par = []
            for i,parameter in enumerate(parameter_names): # must be done once per variable
                q_16, q_50, q_84 = corner.quantile(ex['flatchain'][:,i], [0.16, 0.5, 0.84]) # your x is q_50
                dx_down, dx_up = q_50-q_16, q_84-q_50

                par.extend([dx_down, dx_up])
                if parameter == 'log_b1':
                    if (max(par)>=x) & (max(par)<=y):
                        #print(i, name,"%.2f" %dx_up,"%.2f" % dx_down)
                        error1to2.append(name)
    error1to2 = set(error1to2)
    error1to2=list(error1to2)
    return(error1to2)

#OE_files = .npz corner plot output files, feat = list that will be returned with errors from the 
#parameter space corner plot
def CornerErrors(OE_files,feat,parameter_names=np.array(['log_a1','log_a2','log_b0','log_b1'])):
    for name in OE_files:
        #print(outdir+"noprescriptionerrors/OE)
        #for name in OE_files[:3]:
            #print(i,name)
        ex = np.load(name)
        if ex['flatchain'].size == 0:
            print(f'{name} is an empty file')
        else: 

            #match = re.search(r'O?E?\d+?_(\d+.\d+)_(-?\d+.\d+)_(\w)', name)
            match = re.search(r'O?E?\d+?_(\d+.\d+)_(-?\d+.\d+)_(\w)', name)

            if match:
                regex_ra = match.group(1)
                regex_ra = float(regex_ra)
                regex_dec = match.group(2)
                regex_dec = float(regex_dec)
                regex_fid = match.group(3) 
                #regex_fid = (f"'{regex_fid}'")

                #for i in range(len(feat)):
                ra=[d['RA'] for d in feat]
                #fid = [d['RA'] for d in feat]
                #print(regex_ra,ra.index(regex_ra))
                #print(regex_fid)
                #print(feat[ra.index(regex_ra)]['fid'])
                if (regex_ra in ra) and (feat[ra.index(regex_ra)]['fid'] == regex_fid):
                    #if feat[ra.index(regex_ra)]['fid'] == regex_fid:

                    for j,parameter in enumerate(parameter_names): # must be done once per variable
                        q_16, q_50, q_84 = corner.quantile(ex['flatchain'][:,j], [0.16, 0.5, 0.84]) # your x is q_50
                        dx_down, dx_up = q_50-q_16, q_84-q_50
                        

                        new_feat = {f'err+_{parameter}' : dx_up, f'err-_{parameter}':dx_down}
                        #print(regex_ra,feat[ra.index(regex_ra)], ra.index(regex_ra),regex_fid,new_feat)
                        feat[ra.index(regex_ra)].update(new_feat)
                else:
                    #print(f'regex is {regex_fid} and fid is',feat[ra.index(regex_ra)]['fid'])
                    continue    
            else:
                print(f'{name} found {match} matches')
    return(feat)

#concatenar no df original (onde estao os valores do best fit)
#os valors medianos da amplitude e timescale do mcmc
def mcmcfeatures(outdir,typeI_df,cp):
    #cp = glob.glob(outdir+"corner/batch/*.npz")

    for file in cp:
        print(file)
        ex = np.load(file)
        match = re.search(r'\d+?_(\d+.\d+)_(-?\d+.\d+)_(\w)', file)

        if match:
            regex_ra = match.group(1)
            regex_ra = float(regex_ra)
            regex_dec = match.group(2)
            regex_dec = float(regex_dec)
            regex_fid = match.group(3) 

            #print(regex_ra,regex_dec,regex_fid)
            #typeI_df.loc[25]['DEC']-
            #print(fid, regex_fid)
            #print(typeI_df[typeI_df['fid']==f'{regex_fid}'])
            mask = (np.abs(typeI_df['RA']-regex_ra)<=0.00001) & \
                   (np.abs(typeI_df['DEC']-regex_dec)<=0.00001) & \
                    (typeI_df['fid']==f'{regex_fid}')#.index[0]
            
            #print(len(typeI_df[mask]))
            if (len(typeI_df[mask]) > 0):
                idx = typeI_df[mask].index[0]

                #print(idx) 
                #print(np.median(ex['ampflatchain']))
                #not in log
                #cols of flat chain = ['a1','a2','b0','b1']
                #cols of ts flatchain = ['xi', 'tau_decay', 
                #     'tau_rise_dqpo', 'tau_perturb', 'tau_decorr', 'omega_0']
                # 
                typeI_df.loc[idx, ['mcmc_sigma_dho']] = np.median(ex['ampflatchain'])
                typeI_df.loc[idx, ['mcmc_sigma_e']] = np.median(ex['flatchain'][:,2])
                typeI_df.loc[idx, ['mcmc_tau_perturb']] = np.median(ex['tsflatchain'][:, 3])
                typeI_df.loc[idx, ['mcmc_tau_decay']] = np.median(ex['tsflatchain'][:, 1])
                
                #in log
                #cols of final sample = ['a1','a2','b0','b1','xi', 'tau_decay', 
                #     'tau_rise_dqpo', 'tau_perturb', 'tau_decorr', 'omega_0', 'sigma_dho']
                typeI_df.loc[idx, ['burned_mcmc_sigma_dho']] = np.median(ex['final_samples'][:,10])
                typeI_df.loc[idx, ['burned_mcmc_sigma_e']] = np.median(ex['final_samples'][:,2])
                typeI_df.loc[idx, ['burned_mcmc_tau_perturb']] = np.median(ex['final_samples'][:, 7])
                typeI_df.loc[idx, ['burned_mcmc_tau_decay']] = np.median(ex['final_samples'][:, 5])
                #typeI_df.loc[idx, ['mcmc_tau_perturb']] = np.median(ex['tsflatchain'])
                #typeI_df.insert(idx, 'mcmc_sigma', np.median(ex['ampflatchain']))
                #typeI_df['mcmc_sigma'].loc[idx]= np.median(ex['ampflatchain'])
                #typeI_df['mcmc_tau'].loc[idx]= np.median(ex['tsflatchain'])
            
    return(typeI_df)

        
def zmad_metric(file, path, band='g', flux_column='flux_tot_clr_uJy_corr'):
    """Calculate MAD and ZMAD metrics for AGN variability detection."""
    # Constants
    if band=='g':
        COLOR = 'mediumseagreen'
    elif band=='r':
        COLOR = 'firebrick'

    FX_COL = flux_column.replace('_corr', '')
    columns_to_check = ['ra', 'dec', 'flux_tot_uJy_corr', 'magtot_clr_corr']
    
    # Load and prepare data
    ztf = pd.read_csv(file)

    ra, dec = radec_filename(file)
    cs_all = pd.read_parquet(f'{path}/calibration_sources/PS1/{ra}_{dec}/calstars_{FX_COL}.parquet')
    
    # Remove duplicates
    cs_clean = cs_all.drop_duplicates(subset=columns_to_check)
    
    # Filter by band and CCD quadrant
    ztf_band = ztf[(ztf['filter'] == f'ZTF_{band}')].dropna(subset=[flux_column])
    
    if (ztf_band.size == 0):
        print(f'RA: {ra},DEC:{dec}, empty {band}-band LC')
        #return None, None, None, None
        return None, None, None
        #raise ValueError(f'RA: {ra},DEC:{dec}, empty {band}-band LC')
        
    ccd_mode = ztf_band['CCDquadID'].mode()[0]
    cs_band = cs_clean[(cs_clean['filter'] == f'ZTF_{band}') & 
                       (cs_clean['mjd'].isin(ztf['mjd'])) & 
                      (cs_clean['CCDquadID'] == ccd_mode)].copy()
    
    # Create magnitude masks
    def create_flux_mask(star_df, flux_col, target_df):
        median_flux = star_df.groupby('ra')[flux_col].median()
        min_flux = np.min(target_df[flux_col]) - 100
        max_flux = np.max(target_df[flux_col]) + 100
        return (median_flux > min_flux) & (median_flux < max_flux)
    
    mag_mask = create_flux_mask(cs_band, flux_column, ztf_band)
    APmag_mask = create_flux_mask(cs_band, f'AP{flux_column}', ztf_band)
    
    # Apply masks
    cs_mask = cs_band[cs_band['ra'].isin(mag_mask[mag_mask].index)].copy()
    APcs_mask = cs_band[cs_band['ra'].isin(APmag_mask[APmag_mask].index)].copy()
    
    # Calculate statistics
    def calculate_star_stats(df, flux_col, prefix=''):
        """Calculate median flux, residuals, MAD and ZMAD for stars without warnings."""
        # Make an explicit copy to avoid warnings
        result = df.copy()
        
        # Per-star calculations
        grouped_star = result.groupby(['ra', 'filter', 'CCDquadID'])
        result.loc[:, f'{prefix}median_flux'] = grouped_star[flux_col].transform('median')
        result.loc[:, f'{prefix}dflux'] = np.abs(result[flux_col] - result[f'{prefix}median_flux'])
        
        # Per-epoch calculations
        grouped_epoch = result.groupby(['filter', 'CCDquadID', 'mjd'])
        result.loc[:, f'{prefix}median_per_epoch'] = grouped_epoch[f'{prefix}dflux'].transform('median')
        result.loc[:, f'{prefix}residual'] = np.abs(result[f'{prefix}dflux'] - result[f'{prefix}median_per_epoch'])
        result.loc[:, f'{prefix}MAD'] = grouped_epoch[f'{prefix}residual'].transform('median')
        
        # ZMAD calculation (with proper scaling factor 0.6745)
        result.loc[:, f'{prefix}ZMAD'] = (np.abs(result[f'{prefix}residual']) / result[f'{prefix}MAD'])
    
        return result
    
    cs_mask = calculate_star_stats(cs_mask, flux_column)
    APcs_mask = calculate_star_stats(APcs_mask, f'AP{flux_column}', 'AP')
    #print(cs_mask.columns)

    if len(APcs_mask)==0:
        cs_mask[['APmedian_flux', 'APdflux','APmedian_per_epoch','APresidual','APMAD', 'APZMAD']] = np.zeros((len(cs_mask), 6), dtype=float)
        cs_df = cs_mask

    elif len(APcs_mask) > len(cs_mask):
        #print('yes')
        cs_df = APcs_mask.merge(cs_mask[['filter', 'CCDquadID', 'mjd', 'ra', 'dec',
                'median_flux', 'dflux', 'median_per_epoch', 
                'residual', 'MAD', 'ZMAD']],  # ← ADD MERGE KEYS HERE!
        on=['filter', 'CCDquadID', 'mjd', 'ra', 'dec'],
        how='left'
        )
    elif len(cs_mask)==0:
        APcs_mask[['median_flux', 'dflux','median_per_epoch','residual','MAD', 'ZMAD']] = np.zeros((len(APcs_mask), 6), dtype=float)
        cs_df = APcs_mask

    else:
        cs_df = cs_mask.merge(APcs_mask[['filter', 'CCDquadID', 'mjd', 'ra', 'dec',
                'APmedian_flux', 'APdflux', 'APmedian_per_epoch', 
                'APresidual', 'APMAD', 'APZMAD']],  # ← ADD MERGE KEYS HERE!
        on=['filter', 'CCDquadID', 'mjd', 'ra', 'dec'],
        how='left'
        )


    # Process AGN data
    ztf_clean = ztf.dropna(subset=[flux_column]).copy()
    ztf_clean = ztf_clean.merge(
        cs_df[['filter', 'CCDquadID', 'mjd','ra', 'dec' ,'MAD', 'APMAD', 'ZMAD', 'APZMAD', 
                       'median_flux', 'APmedian_flux', 'median_per_epoch', 'APmedian_per_epoch']],
        on=['filter', 'CCDquadID', 'mjd'],
        how='left'
        )
    
    # Calculate AGN statistics
    group_ztf = ztf_clean.groupby(['filter','CCDquadID','ra_x','dec_x'])
    ztf_clean['median_flux_agn'] = group_ztf[flux_column].transform('median')
    ztf_clean['APmedian_flux_agn'] = group_ztf[f'AP{flux_column}'].transform('median')
    
    ztf_clean['dflux_agn'] = np.abs(ztf_clean[flux_column] - ztf_clean['median_flux_agn'])
    ztf_clean['APdflux_agn'] = np.abs(ztf_clean[f'AP{flux_column}'] - ztf_clean['APmedian_flux_agn'])
    
    ztf_clean['residual_agn'] = np.abs(ztf_clean['dflux_agn'] - ztf_clean['median_per_epoch'])
    ztf_clean['APresidual_agn'] = np.abs(ztf_clean['APdflux_agn'] - ztf_clean['APmedian_per_epoch'])
    
    ztf_clean['MAD_agn'] = group_ztf['residual_agn'].transform('median')
    ztf_clean['APMAD_agn'] = group_ztf['APresidual_agn'].transform('median')
    
    ztf_clean['ZMAD_agn'] = ztf_clean['residual_agn'] / ztf_clean['MAD']
    ztf_clean['APZMAD_agn'] = ztf_clean['APresidual_agn'] / ztf_clean['APMAD']
    
    # Calculate percentiles
    def calculate_percentile(target_df, zmad_col):
        target_df.dropna(subset=[zmad_col,f'{zmad_col}_agn'], inplace=True)
        star_sums = target_df.groupby(['filter', 'CCDquadID','ra_y', 'dec_y'])[zmad_col].sum()
        target_sum = target_df.groupby(['filter', 'CCDquadID','ra_x','dec_x'])[f'{zmad_col}_agn'].apply(lambda x: x.unique().sum())
        # target_sum = sum(target_zmad)
        #print(star_sums,target_sum)
        # return np.sum(star_sums < target_sum) / len(star_sums) * 100
        return star_sums, target_sum
    
    # szmad,tzmad = calculate_percentile(ztf_clean, 'ZMAD')
    # sap_zmad,tsap_zmad = calculate_percentile(ztf_clean,'APZMAD')
    # print(len(ztf_clean))
    if (len(ztf_clean)!=0):
        zmad = calculate_percentile(ztf_clean, 'ZMAD')
        ap_zmad = calculate_percentile(ztf_clean,'APZMAD')
        
        s = zmad[0]
        # print(s)
        if len(s)>5:
            clipped = sigma_clip(s.values, sigma=3)
            # print(clipped)
            good_points = ~clipped.mask
            clipped_s = s[good_points]
        else:
            clipped_s = s 

        s_ap = ap_zmad[0]
        if len(s_ap)>5:
            clipped = sigma_clip(s_ap.values, sigma=3)
            ap_good_points = ~clipped.mask
            clipped_aps = s_ap[ap_good_points]
        else:
            clipped_aps = s_ap
        # print(len(zmad[0]),len(zmad[1]))
        # if (len(zmad[0])!=0) and (len(zmad[1])!=0):
        #print(zmad[0])
        #print(clipped_s)
        #print(zmad[1])
        PSF_mean = clipped_s.groupby(['filter', 'CCDquadID']).mean()
        #print(PSF_mean)
        PSF_std = clipped_s.groupby(['filter', 'CCDquadID']).std()
        #print('clipped std', PSF_std)
        psf_std = zmad[0].groupby(['filter', 'CCDquadID']).std()
        #print('unclipped std', psf_std)
        PSF_sigma = (zmad[1] - PSF_mean)/PSF_std
        #print(PSF_sigma)

        # print(zmad[0].values,zmad[1].values)
        ntrue = (clipped_s.values < zmad[1].values).sum()
        pct = ntrue*100/len(clipped_s.values)

        #if (len(ap_zmad[0])!=0) and (len(ap_zmad[1])!=0):

        AP_mean = clipped_aps.groupby(['filter', 'CCDquadID']).mean()
        AP_std = clipped_aps.groupby(['filter', 'CCDquadID']).std()
        AP_sigma = (ap_zmad[1] - AP_mean)/AP_std

        ap_ntrue = (clipped_aps.values < ap_zmad[1].values).sum()
        ap_pct = ap_ntrue*100/len(clipped_aps.values)
        
        #else:
        #    return None,None,None

        zmad_dict = {'RA':ra, 'DEC': dec,'PSF_dist':(clipped_s,zmad[1]),'AP_dist':(clipped_aps,ap_zmad[1]),
                    'PSF_csmean': PSF_mean,
                    'PSF_csmedian': zmad[0].groupby(['filter', 'CCDquadID']).median(),
                    'PSF_csstd': PSF_std,
                    'PSF_sigma': PSF_sigma,
                    'PSF_csmax': zmad[0].groupby(['filter', 'CCDquadID']).max(),
                    'PSF_cscount': zmad[0].groupby(['filter', 'CCDquadID']).size(),
                    'PSF_perc':pct,
                    'AP_csmean': AP_mean,
                    'AP_csmedian': ap_zmad[0].groupby(['filter', 'CCDquadID']).median(),
                    'AP_csstd': AP_std,
                    'AP_sigma':AP_sigma,
                    'AP_csmax': ap_zmad[0].groupby(['filter', 'CCDquadID']).max(),
                    'AP_cscount': ap_zmad[0].groupby(['filter', 'CCDquadID']).size(),
                    'AP_perc':ap_pct}
        
        #return ztf_clean, cs_df, zmad, ap_zmad
        # return szmad,tzmad, sap_zmad,tsap_zmad
        return ztf_clean, cs_df, zmad_dict
    else:
        return None,None,None

def zmad_hist(df_stars, df_agn, prefix=''):
    col=f'{prefix}ZMAD'
    agn_col=f'{col}_agn'
    """
    Plot histograms of star ZMAD values with AGN ZMAD as vertical lines.
    
    Parameters:
    - df_stars: DataFrame with star ZMAD values
    - df_agn: DataFrame with AGN ZMAD values
    - stars_col: Column name for star ZMAD values
    - agn_col: Column name for AGN ZMAD values
    """
    # Get unique filter-CCD groups
    star_groups = df_stars.groupby(['filter', 'CCDquadID'])
    agn_groups = df_agn.groupby(['filter', 'CCDquadID'])
    
    # Create subplots for each group
    n_groups = len(star_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(5, 4*n_groups), squeeze=False)
    axes = axes.flatten()
    
    for idx, ((filter_val, ccd_id), star_group) in enumerate(star_groups):
        ax = axes[idx]
        
        # Plot star histogram
#         star_zmad = star_group[stars_col]
        star_zmad = star_group.values
        ax.hist(star_zmad, bins=20, alpha=0.7, color='#9F9F9F', 
                edgecolor='black', label=f'Stars (N={len(star_zmad)})')
        
        # Add AGN vertical line if exists
        agn_key = (filter_val, ccd_id)
        if agn_key in agn_groups.groups:
            agn_group = agn_groups.get_group(agn_key)
            agn_zmad = agn_group.values[0]  # Assuming one AGN per group
            
            ax.axvline(agn_zmad, color='green', linestyle='--', linewidth=2,
                      label=f'AGN: {agn_zmad:.2f}')
            
            # Add text annotation
            ax.text(agn_zmad-20, ax.get_ylim()[1]*0.9, f'AGN\n{agn_zmad:.1f}', 
                   ha='center', va='top', color='green', fontweight='bold')

            agn_sigma = (agn_zmad-star_zmad.mean())/star_zmad.std()
            ax.text(agn_zmad-20, ax.get_ylim()[1]*0.5, f'std \n{agn_sigma:.1f}', 
                   ha='center', va='top', color='green', fontweight='bold')

            ax.axvline(star_zmad.mean(), color='black', linestyle='-.', linewidth=2)#,
                      #label=f'Stars: {star_zmad.mean():.2f}')
            ax.axvline(star_zmad.mean()+star_zmad.std(), color='black', linestyle='dotted', linewidth=2)
            ax.axvline(star_zmad.mean()+3*star_zmad.std(), color='red', linestyle='dotted', linewidth=2)
            #ax.axvline(star_zmad.mean()+4*star_zmad.std(), color='red', linestyle='dotted', linewidth=2)
            ax.axvline(star_zmad.mean()-star_zmad.std(), color='black', linestyle='dotted', linewidth=2)#,
                      #label=f'Stars: {star_zmad.mean():.2f}')
        
        # Format plot
        ax.set_xlabel('ZMAD', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Filter: {filter_val}, CCD: {ccd_id}', fontsize=14, fontweight='bold')
        #ax.legend()
        ax.grid(True, alpha=0.3)
        
        # # Add statistics
        # print(f'AGN\n{agn_zmad:.1f}')
        # print(f'Mean: {star_zmad.mean():.2f}')
        # print(f'Std: {star_zmad.std():.2f}')
        # print(f'sigma: {(agn_zmad-star_zmad.mean())/star_zmad.std()}')
        ax.text(0.4, 0.98, f'Mean: {star_zmad.mean():.2f}\nStd: {star_zmad.std():.2f}\nMax: {star_zmad.max():.2f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, axes

def zmad_plots(df_stars, df_agn,cs_mask, ztf, prefix=''):
    fig, axes = plt.subplots(1, 2, figsize=(10,6))
    axes = axes.flatten()

    # flux_column = f'{prefix}flux_tot_clr_uJy_corr'
    flux_column = f'{prefix}magtot_clr_corr'
    # flux_unc = flux_column.split('_', 1)[0] + 'unc_' + flux_column.split('_', 1)[1] + '_alt2'
    flux_unc = f'{prefix}magunc_clr_corr'

    # Get unique filter-CCD groups
    star_groups = df_stars.groupby(['filter', 'CCDquadID'])
    agn_groups = df_agn.groupby(['filter', 'CCDquadID'])
    
    for idx, ((filter_val, ccd_id), star_group) in enumerate(star_groups):
        if filter_val == 'ZTF_g':
            color = 'mediumseagreen'
        elif filter_val == 'ZTF_r':
            color = 'firebrick'

        # Plot star histogram
#         star_zmad = star_group[stars_col]
        star_zmad = star_group.values
        ztf_band = ztf[(ztf['filter'] == filter_val)&(ztf['CCDquadID'] == ccd_id)]
        cs_band = cs_mask[(cs_mask['filter'] == filter_val)&(cs_mask['CCDquadID'] == ccd_id)]

        # Plot 1: Aperture Photometry Flux
        axes[0].errorbar(
            cs_band['mjd'], 
            cs_band[f'{flux_column}'],
            yerr=cs_band[f'{flux_unc}'], 
            fmt='o', c='#9F9F9F', alpha=0.1, label='Stars'
        )
        axes[0].errorbar(
            ztf_band['mjd'], 
            ztf_band[f'{flux_column}'], 
            yerr=ztf_band[f'{flux_unc}'], 
            fmt='o', color=color, label=f'AGN {filter_val}'
        )
        axes[1].hist(star_zmad, bins=20, alpha=0.7, color='#9F9F9F', 
                    edgecolor='black', label=f'Stars (N={len(star_zmad)})')
        
        # Add AGN vertical line if exists
        agn_key = (filter_val, ccd_id)
        if agn_key in agn_groups.groups:
            agn_group = agn_groups.get_group(agn_key)
            agn_zmad = agn_group.values[0]  # Assuming one AGN per group

            axes[1].axvline(agn_zmad, color=color, linestyle='--', linewidth=2,
                    label=f'AGN: {agn_zmad:.2f}')

            # Add text annotation
            axes[1].text(agn_zmad, axes[1].get_ylim()[1]*0.9, f'AGN\n{agn_zmad:.1f}', 
                ha='center', va='top', color='red', fontweight='bold')

        # Format plot
        axes[1].set_xlabel('ZMAD', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Filter: {filter_val}, CCD: {ccd_id}', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[0].set_xlabel('time [days]', fontsize=12)
        axes[0].set_ylabel(flux_column, fontsize=12)
        axes[0].invert_yaxis()
        # axes[0].set_title(f'Filter: {filter_val}, CCD: {ccd_id}', fontsize=14, fontweight='bold')
        # axes[0].legend()

        # Add statistics
        #print(f"Group ({filter_val}, {ccd_id}):")
        #print(f"  Group type: {type(star_group)}")
        #print(f"  Group values: {star_group.values[:5]}")  # First 5 values
        #print(f"  Group.std(): {star_group.std():.2f}")
        #print(f"  group.values.std(): {star_group.values.std():.2f}")
        #print(f"  group.values.std(ddof=1): {star_group.values.std(ddof=1):.2f}")

        #print(star_zmad)
        #print(f'AGN: {agn_zmad:.1f}')
        #print(f'Mean: {star_zmad.mean():.2f}')
        #print(f'Std: {star_zmad.std():.2f}')
        #print(f'sigma: {(agn_zmad-star_zmad.mean())/star_zmad.std()}')
        axes[1].text(0.4, 0.95, f'Mean: {star_zmad.mean():.2f}\nStd: {star_zmad.std():.2f}\nMax: {star_zmad.max():.2f}', 
            transform=axes[1].transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, axes

def plot_results(ztf, cs_mask, APcs_mask, percentile,APpercentile, band='g', color='mediumseagreen'):
    """
    Plot light curves and ZMAD distributions for stars and AGN.
    
    Parameters:
    - ztf: DataFrame with AGN data and calculated metrics
    - cs_mask: DataFrame with PSF star data
    - APcs_mask: DataFrame with aperture star data
    - percentile: Variability percentile score
    - band: Filter band (default 'g')
    - color: Color for AGN plots (default 'mediumseagreen')
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Filter data
    band_filter = f'ZTF_{band}'
    ccd_id = ztf['CCDquadID'].mode()[0]  # Use mode instead of unique()[0] for safety
    flux_column = 'flux_tot_clr_uJy_corr'
    flux_unc = flux_column.split('_', 1)[0] + 'unc_' + flux_column.split('_', 1)[1] + '_alt2'
    
    # Get data for plotting
    masked_cs = cs_mask[(cs_mask['CCDquadID'] == ccd_id) & 
                       (cs_mask['filter'] == band_filter)].copy()
    APmasked_cs = APcs_mask[(APcs_mask['CCDquadID'] == ccd_id) & 
                           (APcs_mask['filter'] == band_filter)].copy()
    ztf_band = ztf[ztf['filter'] == band_filter].copy()
    
    # Plot 1: Aperture Photometry Flux
    axes[0].errorbar(
        APmasked_cs['mjd'], 
        APmasked_cs[f'AP{flux_column}'],
        yerr=APmasked_cs[f'AP{flux_unc}'], 
        fmt='o', c='red', alpha=0.5, label='Stars'
    )
    axes[0].errorbar(
        ztf_band['mjd'], 
        ztf_band[f'AP{flux_column}'], 
        yerr=ztf_band[f'AP{flux_unc}'], 
        fmt='o', color=color, label='AGN'
    )
    axes[0].set(xlabel='MJD', ylabel='AP Flux [μJy]', title='Aperture Photometry')
    axes[0].legend()
    
    # Plot 2: Aperture ZMAD Distribution
    star_ap_zmad = APmasked_cs.groupby(['ra', 'dec'])['APZMAD'].sum()
    agn_ap_zmad = ztf_band['APZMAD_agn'].sum()
    
    axes[1].hist(
        star_ap_zmad.values,
        bins=30,
        color='red', 
        alpha=0.7,
        label=f'Stars (N={len(star_ap_zmad)})'
    )
    axes[1].axvline(
        agn_ap_zmad, 
        color=color, 
        linestyle='--', 
        linewidth=2,
        label=f'AGN (percentile: {percentile:.1f}%)'
    )
    axes[1].set(xlabel='AP ZMAD (Sum across epochs)', ylabel='Number of sources')
    axes[1].legend()
    
    # Plot 3: PSF Photometry Flux
    axes[2].errorbar(
        masked_cs['mjd'], 
        masked_cs[flux_column],
        yerr=masked_cs[flux_unc], 
        fmt='o', c='red', alpha=0.5, label='Stars'
    )
    axes[2].errorbar(
        ztf_band['mjd'], 
        ztf_band[flux_column], 
        yerr=ztf_band[flux_unc], 
        fmt='o', color=color, label='AGN'
    )
    axes[2].set(xlabel='MJD', ylabel='PSF Flux [μJy]', title='PSF Photometry')
    axes[2].legend()
    
    # Plot 4: PSF ZMAD Distribution
    star_psf_zmad = masked_cs.groupby(['ra', 'dec'])['ZMAD'].sum()
    agn_psf_zmad = ztf_band['ZMAD_agn'].sum()
    
    axes[3].hist(
        star_psf_zmad.values,
        bins=30,
        color='red', 
        alpha=0.7,
        label=f'Stars (N={len(star_psf_zmad)})'
    )
    axes[3].axvline(
        agn_psf_zmad, 
        color=color, 
        linestyle='--', 
        linewidth=2,
        label=f'AGN (AP percentile: {APpercentile:.1f}%)'
    )
    axes[3].set(xlabel='PSF ZMAD (Sum across epochs)', ylabel='Number of sources')
    axes[3].legend()
    
    plt.tight_layout()
    plt.suptitle(f'AGN Variability Analysis (ZTF {band}-band)', y=1.02)
    return fig

def plot_1point(band,grouped_per_epoch):
    ztf_band = ztf[ztf['filter']==f'ZTF_{band}']
    for group in grouped_per_epoch:
        #print(group[1][['ra','dec','mjd','AP'+flux_column,'APmedian_flux','APdflux','APmedian_per_epoch','APresidual','APMAD','APZMAD']])#[['ra','dec','mjd','APresidual','APMAD','APZMAD']])

        #ztf[ztf['filter']=='ZTF_g'][['mjd','APZMAD_agn']]
        #print(ztf_band[ztf_band['mjd']==group[0][2]]['APZMAD_agn'].values)
        ztf_window = ztf_band[ztf_band['mjd']==group[0][2]]
        if (len(ztf_window['APZMAD_agn'])>0): #and (group[0][2]>59700) and (group[0][2]<60000):
    #         print(group[0])
    #         print(group[1][['ra','dec','APflux_tot_clr_uJy_corr','APresidual','APMAD']])
    # #         print(group[1]['APresidual'].values)
    #         #print(group[1][['mjd','APresidual','APMAD','APZMAD']])
    #         #print(group[1]['APflux_tot_clr_uJy_corr'].values)
    #         print(np.median(group[1]['APflux_tot_clr_uJy_corr']))
    #         print(np.median(np.abs(group[1]['APflux_tot_clr_uJy_corr']-np.median(group[1]['APflux_tot_clr_uJy_corr']))))


            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes = axes.flatten()
            axes[0].plot(group[1]['mjd'],group[1]['APflux_tot_uJy_corr'],'o',label='stars')
        #         axes[0].plot(group[1]['mjd'],group[1]['APresidual'],'o',label='residual stars',color='black')
            axes[0].plot(ztf_window['mjd'],ztf_window['APflux_tot_uJy_corr'],'o',color='red',label='source')
            axes[1].hist(group[1]['APZMAD'])
            axes[0].axhline(np.median(ztf_window['APmedian_flux_agn']),ls=':',label='AGN median', color='red')
            axes[1].axvline(ztf_window['APZMAD_agn'].values,color='red')
            #axes[1].set_title(f'epoch {group[0][2]}')
            for n,y_val in enumerate(group[1]['APmedian_flux']):
                label = 'star median' if n==0 else None
                axes[0].axhline(y=y_val, color='black', linestyle='--', linewidth=1,label=label)
            axes[0].legend()
            plt.show()
            
def plot_epochs(band,cs_mask):
    ztf_band = ztf[ztf['filter']==f'ZTF_{band}']
    for window in cs_mask.groupby(['CCDquadID','block_id']):
        #print(window[1][['mjd','CCDquadID','block_id','APresidual','APMAD','APZMAD']])
        ztf_window = ztf_band[(ztf_band['block_id']==window[0][1])
                              &(~ztf_band['APZMAD_agn'].isna())]#['APZMAD_agn'].values
        if all(~window[1]['APZMAD'].isna()):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes = axes.flatten()
            group = window[1]
            block = ztf_window[ztf_window['block_id']==window[0][1]]
            #print(block[])


            axes[1].hist(group['APZMAD'], label='stars')
            axes[1].hist(block['APZMAD_agn'], alpha=0.5, label = 'source', color='red')
            #axes[1].axvline(ztf_window[ztf_window['block_id']==window[0][1]]['APZMAD_agn'].values,color='red')
            axes[1].legend()

            star_sums = window[1].groupby(['ra', 'dec'])['APZMAD'].sum()
            #print(star_sums)

            axes[1].set_xlabel('ZMAD per epoch')
            max_dt = window[1]['mjd'].max()
            min_dt = window[1]['mjd'].min()
            axes[1].set_title(f'CCDquadID {window[0][0]} Epoch from {min_dt:.2f}-{max_dt:.2f}')
            axes[1].set_xscale('log')

            axes[0].plot(group['mjd'],group['APflux_tot_uJy_corr'],'o',label='stars')
            axes[0].plot(ztf_window['mjd'],ztf_window['APflux_tot_uJy_corr'],'o',color='red',label='source')
            #axes[0].axhline(np.median(group['APflux_tot_uJy_corr']),ls='-.',color='black',label='median stars')
            axes[0].axhline(np.median(block['APmedian_flux_agn']),ls=':',label='AGN full LC median', color='red')
            axes[0].axhline(np.median(block['APflux_tot_uJy_corr']),ls='-.',label='AGN block median', color='red')
            axes[2].hist(star_sums, alpha=0.5, label = 'source')
            axes[2].axvline(block['APZMAD_agn'].sum(),color='red')
            axes[2].set_xlabel('sum ZMAD per source')
            axes[2].set_xscale('log')

            for n,y_val in enumerate(group['APmedian_flux']):
                label = 'star median' if n==0 else None
                axes[0].axhline(y=y_val, color='black', linestyle='--', linewidth=1,label=label)


            plt.show()
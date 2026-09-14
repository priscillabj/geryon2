mask_nSy1=(feat['RA']>=0)&(feat['DEC']>=0)&(feat['clasf']=='Sy1+Sy1.2')&(feat['SF_ML_amplitude_1']<0.1)
mask_nSy15=(feat['RA']>=0)&(feat['DEC']>=0)&(feat['clasf']=='Sy1.5')&(feat['SF_ML_amplitude_1']<0.1)
mask_sSy1=(feat['RA']>=0)&(feat['DEC']<0)&(feat['clasf']=='Sy1+Sy1.2')&(feat['SF_ML_amplitude_1']<0.1)
mask_sSy15=(feat['RA']>=0)&(feat['DEC']<0)&(feat['clasf']=='Sy1.5')&(feat['SF_ML_amplitude_1']<0.1)
mask_nSy2=(feat['RA']>=0)&(feat['DEC']>=0)&(feat['clasf']=='Sy2')&(feat['SF_ML_amplitude_1']>=0.1)
mask_nSy19=(feat['RA']>=0)&(feat['DEC']>=0)&(feat['clasf']=='Sy1.8+Sy1.9')&(feat['SF_ML_amplitude_1']>=0.1)
mask_sSy2=(feat['RA']>=0)&(feat['DEC']<0)&(feat['clasf']=='Sy2')&(feat['SF_ML_amplitude_1']>=0.1)
mask_sSy19=(feat['RA']>=0)&(feat['DEC']<0)&(feat['clasf']=='Sy1.8+Sy1.9')&(feat['SF_ML_amplitude_1']>=0.1)

agn = feat[mask_nSy1|mask_sSy15|mask_nSy15| mask_sSy15|mask_nSy19|mask_sSy19|mask_nSy2|mask_sSy2] 
candidates = agn[['bat_index','RA','DEC','name','z','clasf','nH','SF_ML_gamma_1','SF_ML_amplitude_1']]
#candidates
cti = feat[(feat['clasf'] == 'Sy1.5') & (feat['SF_ML_amplitude_1'] < 0.05)][['name','clasf','bat_index', 'RA', 'DEC', 'z']]
ct1 = feat.loc[(feat['clasf'] == 'Sy1+Sy1.2') & (feat['SF_ML_amplitude_1'] < 0.05)][['name','clasf','bat_index', 'RA', 'DEC', 'z']]
cti2 = feat[(feat['clasf'] == 'Sy1.8+Sy1.9') & (feat['SF_ML_amplitude_1'] > 0.05)][['name','clasf','bat_index', 'RA', 'DEC', 'z']]
ct2= feat[(feat['clasf'] == 'Sy2') & (feat['SF_ML_amplitude_1'] > 0.05)][['name','clasf','bat_index', 'RA', 'DEC', 'z']]
# ct2.head()

tII=candidates[(candidates['clasf']=='Sy2')|(candidates['clasf']=='Sy1.8+Sy1.9')]
tI=candidates[(candidates['clasf']=='Sy1+Sy1.2')|(candidates['clasf']=='Sy1.5')]
plt.style.use('seaborn-white')

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="k")

plt.hist(feat.nH, ls='dashed',fill=False,histtype='step',label='BASS',density=True)
plt.hist(tII.nH, **kwargs, label='CS-typeII candidates')
plt.hist(tI.nH, **kwargs, label='CS-typeI candidates')
plt.xlabel('log nH')
plt.ylabel('Frequency')
plt.legend()
#plt.title('Sy2 above 0.1 z distribution')

import seaborn as sns
h = sns.jointplot(x=feat['SF_ML_gamma_1'].values ,y=feat['SF_ML_amplitude_1'].values,height=10,xlim=(-0.3, 1), ylim=(-0.1, 0.8), hue=feat['clasf'].values)
h.set_axis_labels(xlabel='SF_ML_gamma_1', ylabel='SF_ML_amplitude_1', fontsize = 16)

MT = [72, 184, 280, 349, 757, 981, 1037, 1070, 106, 116, 471, 530, 1327] #bat index for Matthew Temple CL AGNs [Temple+22]
CL = [73,216,557,656,595, 994, 1188, 1189, 1194] #(Tohline & Osterbrock 1976; Penston & Perez 1984; 
#Kollatschny & Fricke 1985; Wamsteker et al. 1985; Malkov et al. 1997; Aretxaga et al. 1999; 
#Kollatschny et al. 2000; Shapovalova et al. 2008; Parker et al. 2016; Oknyansky et al. 2019; 
#Kollatschny et al. 2020; Jiang et al. 2021; Liu et al. 2022;
#14 sources
#feat.loc[(feat['bat_index'].isin(MT))]
CLf=feat.loc[(feat['bat_index'].isin(CL))| (feat['bat_index'].isin(MT)) ]

#h = sns.jointplot(x=feat['SF_ML_gamma_1'].values ,y=feat['SF_ML_amplitude_1'].values,height=10,xlim=(-0.3, 1), ylim=(-0.1, 0.8), hue=feat['clasf'].values)
#h.set_axis_labels(xlabel='SF_ML_gamma_1', ylabel='SF_ML_amplitude_1', fontsize = 16)
graph = sns.jointplot(x=feat['SF_ML_gamma_1'].values, y=feat['SF_ML_amplitude_1'].values,color='r',height=10,xlim=(-0.3, 1), ylim=(-0.1, 0.8))

graph.x = CLf['SF_ML_gamma_1'].values
graph.y =  CLf['SF_ML_amplitude_1'].values
#graph.hue = 
graph.plot_joint(plt.scatter, marker='X', c='b', s=70)

#def EVS (PathAndFileNameList):
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
candidates = []
for i in filenames:
    df_lc = friendly_use(i)
    df_lc = calc_some(df_lc,'forcediffimflux')

    
    if (df_lc.size) != 0:
        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        for fid, color in [("ZTF_g","mediumseagreen"),("ZTF_r","firebrick")]:
            mask = ((df_lc['filter'] == fid) & (df_lc.infobitssci==0) & (df_lc.nearestrefmag < 17)& 
                    (df_lc.mag_tot > 14) & (df_lc.ccdid == df_lc.ccdid.mode()[0]))

            npts = len(df_lc[mask].mjd)

            if (npts > 30):
                mjd = df_lc[mask].mjd.values
                mag = df_lc[mask].mag_tot.values
                magerr = np.sqrt(0.01**2+df_lc[mask].magerr.values**2)
                
                t_max = mjd.max()
                t = mjd.min()
                #n=0
                #print(mjd[n])
                #n += 3
                #print(mjd[n])

                while t<= t_max:

                    t_i=closest(mjd, t)
                    #print(f'o t_0 mais proximo é {t_i} com indice {np.where(mjd==t_i)[0]}')
                    idx_t1=np.where(mjd==t_i)[0]
                    #print(mag[idx_t1])
                    #print(mjd[idx_t1])
                    t+=180
                    #idx_f= bisect(mjd, t)
                    t_f=closest(mjd, t)
                    #print(np.where(mjd==t_f))
                    #print(f'o t_f mais proximo é {t_f} com indice {np.where(mjd==t_f)[0]}')
                    idx_t2=np.where(mjd==t_f)[0]


                    if idx_t2-idx_t1 >= 2 and np.abs(mag[idx_t2]-mag[idx_t1])>=0.4 :

                        #window=np.arange(idx_t1[0],idx_t2[0])
                        window_mag=mag[idx_t1[0]:idx_t2[0]]
                        #print(window_mag)
                        window_t = mjd[idx_t1[0]:idx_t2[0]]
                        #print(window_t)
                        #max_mag=np.where(window_mag == np.amax(window_mag))[0]
                        #min_mag=np.where(window_mag == np.amin(window_mag))[0]
                        #print(max_mag[0],window_mag[max_mag], window_mag[max_mag-1],window_mag[max_mag-2] )
                        
                        #case = {'filename': i, 'fid':fid ,'delta_mag': np.abs(mag[idx_t2]-mag[idx_t1]), 'window_mag':window_mag, 'window_t':window_t }
                        #candidate.append(case)
                        
                        match = re.search(r'lc_(\d+.\d+)_(-?\d+.\d+)_', i)
                        if match:
                            regex_ra = match.group(1)
                            regex_ra = float(regex_ra)
                            regex_dec = match.group(2)
                            regex_dec = float(regex_dec)

                            case = {'filename': i, 'RA': regex_ra, 'DEC': regex_dec,'fid':fid ,'delta_mag': np.abs(mag[idx_t2]-mag[idx_t1]), 'window_mag':window_mag, 'window_t':window_t }
                            candidates.append(case)

                        #for index, row in pd.DataFrame(candidates).iterrows():
                        #    if i == row['filename'] and row['bat_index'] != 'nan':
                        #        
                        #        ax.errorbar(mjd,mag,yerr=magerr,c=color,label=f'data_{fid}',fmt='.')
                        #        ax.axvspan(mjd[idx_t1[0]], mjd[idx_t2[0]], color='red', alpha=0.5)
                        #        ax.set_xlabel("MJD")
                        #        ax.set_ylabel("Apparent Magnitude")
                        #        ax.set_ylim([14.5,16.5])
                        #        ax.invert_yaxis()
                        #        ax.legend()
                        #        ax.set_title(row['bat_index'])
                        
    #type(candidates)                    
    #output_dir = os.path.join(path+'EVS')
    #with open(output_dir, 'w') as fout:
    #    json.dump(candidates, fout)

for index, row in pd.DataFrame(candidate).iterrows():
        #data = ascii.read(i)
        #df_lc = pd.DataFrame()
        #print(i)
        df_lc = friendly_use(row['filename'])
        df_lc = calc_some(df_lc,'forcediffimflux')
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        for fid, color in [("ZTF_g","mediumseagreen"),("ZTF_r","firebrick")]:
            mask = ((df_lc['filter'] == fid) & (df_lc.infobitssci==0) & (df_lc.nearestrefmag < 17)& 
                    (df_lc.mag_tot > 14) & (df_lc.ccdid == df_lc.ccdid.mode()[0]))

            

            tIn = df_lc[mask].mjd.values
            y = df_lc[mask].mag_tot.values
            yerr = np.sqrt(0.01**2+df_lc[mask].magerr.values**2)

            if fid == row['fid']:
                ax.errorbar(tIn,y,yerr=yerr,c=color,label=f'data_{fid}',fmt='.')
                ax.axvspan(candidate[index]['window_t'].min(),candidate[index]['window_t'].max(), color='red', alpha=0.5)
                ax.set_xlabel("MJD")
                ax.set_ylabel("Apparent Magnitude")
                ax.legend()
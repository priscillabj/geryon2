python -c "

import json, os, pandas as pd

ok = pd.read_parquet(os.path.expanduser('~/results/bat_master_zmadflux2_outliers.parquet'))
cand = ok[(ok.max_z_rob > 4) & (ok.dgamma > 0.3) & (ok.n_sf >= 5)]
cand = cand.drop_duplicates('lc_file').sort_values('dgamma', ascending=False)

print(cand[['lc_file','n_sf','ts_gamma','max_z_rob','dt_max_z','sign_max_z','dgamma']]
      .to_string(index=False))

# paths = [os.path.expanduser(f'~/results/partials/{r.source}/{r.lc_file}_10cs_linmix.pkl')
#          for r in cand.itertuples()]
paths = cand['lc_file'].dropna().unique().tolist()
missing = [p for p in paths if not os.path.exists(p)]
print(f'{len(paths)} paths, {len(missing)} missing')

json.dump(paths, open(os.path.expanduser('~/results/bad_outliers.json'), 'w'), indent=1)
"
#  ----------------------------------------------------

python -c "
import json, pandas as pd
df = pd.read_parquet('~/results/bat_master_zmadflux2.parquet').drop_duplicates()
g = df[df['band']=='g']
fit_model = 'spl'
col = f'gamma_{fit_model}'
sub = g[g[col]<0]['pkl_spl']
json.dump(sub.dropna().unique().tolist(), open('/home/pjorge/results/neg_SFgamma_g.json', 'w'), indent=1)
"
#  ----------------------------------------------------

python -c "
import pandas as pd, os
df = pd.read_parquet('~/results/bat_master_zmadflux2.parquet').drop_duplicates()
fit_model = 'spl'
# col = f'gamma_{fit_model}'
col = f'A_365_{fit_model}'

lo = df[col] < 0.1
hi = df[col] >= 0.1
m = (lo & df['clasf'].isin(['Sy1','Sy1.2'])) | (hi & df['clasf'].isin(['Sy1.8','Sy1.9','Sy2','Sy1.5']))
CL_cand = df[m]
CL_cand.to_parquet(f'{os.path.expanduser('~')}/results/{col}_CL_candidates.parquet')
"


python -c "
import pandas as pd
df = pd.read_parquet('~/results/A_365_spl_CL_candidates.parquet')
# g = df[df['band']=='g']
# print(len(g))
print(len(df))"


python -c "
import os, pandas as pd
R  = os.path.expanduser('~/results')
cl = pd.read_parquet(f'{R}/A_365_spl_CL_candidates.parquet')
sl = pd.read_parquet(f'{R}/A_365_spl_CL_shortlist.parquet')

cl['object'] = cl.lc_file.str.split('_').str[:2].str.join('_')
assert cl.groupby('object').bat_index.nunique().max() <= 1, 'bat_index differs across bands'
meta = cl.drop_duplicates('object').set_index('object')[['bat_index', 'clasf']]
g    = cl[cl.lc_file.str.contains('_zg_')].groupby('object').lc_file.agg(list)

# sl = sl.join(meta, on='object').assign(g_lc_file=sl.object.map(g))
sl = (sl.drop(columns=['bat_index', 'clasf', 'g_lc_file'], errors='ignore')
        .join(meta, on='object'))
sl['g_lc_file'] = sl.object.map(g)
# print(sl[['tier', 'object', 'bat_index', 'clasf', 'ref_band', 'snr', 'd', 'g_lc_file']]
#       .to_string(index=False))
# sl.to_parquet(f'{os.path.expanduser('~')}/results/A_365_spl_CL_shortlist.parquet')
ov = cl[cl.object.isin(sl.object)].merge(
        sl.drop(columns=['bat_index', 'clasf']), on='object')   # already in cl
ov.drop(columns='object').to_parquet(f'{R}/A_365_spl_CL_overlay.parquet')

# print(sl[['tier', 'object', 'bat_index', 'clasf', 'ref_band', 'snr', 'd', 'g_lc_file']]
#       .to_string(index=False))
# print()
print(ov.groupby(['tier', 'band']).size())
"

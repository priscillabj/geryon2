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
g = df[df['band']=='g']
print(len(g))"

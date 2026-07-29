import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

PKL_PATH = '/home/pjorge/results/partials/'

'''
check histogram or scatter plot of slopes difference of
two different cadences mock LCs
'''
# filename = sys.argv[1]
# key = sys.argv[2]

# PATH = PKL_PATH+filename
# if os.path.exists(PATH):
#     sf = pd.read_pickle(PATH)
    
#     if key not in sf[0]:
#         raise ValueError(f"Invalid key '{key}'. Choose from: {list(sf[0].keys())}")
    
#     result = sf[0][key]
#     bins = np.arange(min(result), max(result), 0.05)
#     plt.hist(result, bins=bins)
    
#     save_name = filename.split('_')[0] + f'{key}_hist.png'
#     plt.savefig(PKL_PATH + save_name, dpi=300, bbox_inches='tight')
#     plt.close()
# else:
#     raise ValueError(f"{PATH} does not exist.")

filename = sys.argv[1]
#  ex filename: '1200simfit_results.pkl'
# key1     = sys.argv[2]
# key2     = sys.argv[3]

PATH = PKL_PATH + filename

if not os.path.exists(PATH):
    raise ValueError(f"{PATH} does not exist.")

sf = pd.read_pickle(PATH)

key1, key2 = '1day_gamma', 'orgcad_gamma'
src = [d for d in sf if key1 in d and key2 in d and 'ref_band' in d
       and len(d[key1]) and len(d[key2])]

vals1 = [d[key1] for d in src]
vals2 = [d[key2] for d in src]
bands = np.array([d['ref_band'] for d in src])

median1 = np.array([np.median(v) for v in vals1])
median2 = np.array([np.median(v) for v in vals2])

p16_1, p84_1 = zip(*(np.percentile(v, [16, 84]) if len(v) else [np.nan, np.nan] for v in vals1))
p16_2, p84_2 = zip(*(np.percentile(v, [16, 84]) if len(v) else [np.nan, np.nan] for v in vals2))
p16_1, p84_1, p16_2, p84_2 = map(np.array, (p16_1, p84_1, p16_2, p84_2))

fig, ax = plt.subplots(figsize=(12, 5))

colors = {'g': 'mediumseagreen', 'r': 'firebrick', 'i': 'gold'}
for b in np.unique(bands):
    m = bands == b
    ax.errorbar(median1[m], median2[m],
                xerr=[median1[m] - p16_1[m], p84_1[m] - median1[m]],
                yerr=[median2[m] - p16_2[m], p84_2[m] - median2[m]],
                fmt='o', ms=4, alpha=0.6, elinewidth=0.5, capsize=0,
                color=colors.get(b, 'gray'), label=b)

ax.axline((0, 0), slope=1, lw=1, alpha=0.7, label='1:1')
ax.set_xlabel(key1); ax.set_ylabel(key2)
ax.legend(); ax.set_title('Median')

save_name = filename.split('_')[0] + f'_{key1}_vs_{key2}.png'
print(PKL_PATH + save_name)
fig.savefig(PKL_PATH + save_name, dpi=300, bbox_inches='tight')
plt.close()
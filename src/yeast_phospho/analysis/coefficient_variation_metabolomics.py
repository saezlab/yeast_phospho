import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv

# -- Import coefficient variation
dynamic_met_cv = read_csv('%s/tables/dynamic_metabolomics_cv.csv' % wd, index_col=0)
dynamic_met_cv = dynamic_met_cv.ix[np.sort(dynamic_met_cv.index)]

dynamic_comb_met_cv = read_csv('%s/tables/dynamic_combination_metabolomics_cv.csv' % wd, index_col=0)

# -- Plot
plt.figure(figsize=(10, 35))
sns.set(style='ticks', context='paper')
sns.boxplot(dynamic_met_cv.T, orient='h', color='#95a5a6')
plt.axvline(dynamic_met_cv.T.median().median(), ls='--', lw=1.5, c='#e74c3c')
sns.despine()
plt.ylabel('ions')
plt.title('Dynamic Metabolomics (TOR perturbations)\nCV (median): %.2f' % dynamic_met_cv.T.median().median())
plt.xlabel('coefficient of variation')
plt.savefig('%s/reports/dynamic_metabolomics_cv.pdf' % wd, bbox_inches='tight')
plt.close('all')

plt.figure(figsize=(10, 35))
sns.set(style='ticks', context='paper')
sns.boxplot(dynamic_comb_met_cv.T, orient='h', color='#95a5a6')
plt.axvline(dynamic_comb_met_cv.T.median().median(), ls='--', lw=1.5, c='#e74c3c')
sns.despine()
plt.ylabel('ions')
plt.xlabel('coefficient of variation')
plt.title('Dynamic Combination Metabolomics (NaCl/alpha)\nCV (median): %.2f' % dynamic_comb_met_cv.T.median().median())
plt.savefig('%s/reports/dynamic_combination_metabolomics_cv.pdf' % wd, bbox_inches='tight')
plt.close('all')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv

# Import
dyn_trans = read_csv('%s/tables/transcriptomics_dynamic.tab' % wd, sep='\t', index_col=0)

# Clustermap
cmap = sns.diverging_palette(220, 10, n=9, as_cmap=True)
sns.set(context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3})

g = sns.clustermap(dyn_trans.T.corr(), figsize=(14, 14), linewidth=.5, cmap=cmap, metric='correlation', xticklabels=False, yticklabels=False)
plt.title('Nitrogen metabolism\n(pearson)')
plt.savefig('%s/reports/transcriptomics_clustermap_nitrogen.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Clustermap done'

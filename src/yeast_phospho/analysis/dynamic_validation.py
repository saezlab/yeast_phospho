import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas.stats.misc import zscore
from sklearn.linear_model import ElasticNet
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import pearson
from yeast_phospho.utilities import get_metabolites_name


# -- Import IDs maps
met_name = get_metabolites_name()
met_name = {'%.2f' % float(k): met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# -- Import data-sets
# Dynamic without growth
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn_ng = metabolomics_dyn_ng[metabolomics_dyn_ng.std(1) > .4]
metabolomics_dyn_ng.index = ['%.2f' % i for i in metabolomics_dyn_ng.index]

k_activity_dyn_ng = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng = k_activity_dyn_ng[(k_activity_dyn_ng.count(1) / k_activity_dyn_ng.shape[1]) > .75].replace(np.NaN, 0.0)

# Dynamic combination
k_activity_dyn_comb_ng = read_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[(k_activity_dyn_comb_ng.count(1) / k_activity_dyn_comb_ng.shape[1]) > .75].replace(np.NaN, 0.0)

metabolomics_dyn_comb = read_csv('%s/tables/dynamic_combination_metabolomics.csv' % wd, index_col=0)[k_activity_dyn_comb_ng.columns]
metabolomics_dyn_comb = metabolomics_dyn_comb[metabolomics_dyn_comb.std(1) > .4]
metabolomics_dyn_comb.index = ['%.2f' % i for i in metabolomics_dyn_comb.index]

# Variables
ions = list(set(metabolomics_dyn_ng.index).intersection(metabolomics_dyn_comb.index))
kinases = list(set(k_activity_dyn_ng.index).intersection(k_activity_dyn_comb_ng.index))

train, test = list(metabolomics_dyn_ng), list(set(metabolomics_dyn_comb).intersection(k_activity_dyn_comb_ng))


# -- Linear regressions
df = []
for ion in ions:
    lm = ElasticNet(alpha=.01).fit(k_activity_dyn_ng.ix[kinases, train].T, metabolomics_dyn_ng.ix[ion, train])

    pred, meas = Series(lm.predict(k_activity_dyn_comb_ng.ix[kinases, test].T), index=test), metabolomics_dyn_comb.ix[ion, test]
    # pred, meas = zscore(pred), zscore(meas)

    cor, pval, nmeas = pearson(pred, meas)

    for c in test:
        df.append((ion, met_name[ion], cor, pval, nmeas, pred.ix[c], metabolomics_dyn_comb.ix[ion, c]))

    print '%s: %.2f, %.2e' % (met_name[ion], cor, pval)

df = DataFrame(df, columns=['ion', 'name', 'cor', 'pval', 'nmeas', 'pred', 'meas']).sort('cor', ascending=False)
print '[INFO] Linear regressions done!'


# -- Plot
sns.set(style='ticks')
g = sns.FacetGrid(df, col='ion', col_wrap=5, sharey=False, sharex=False)
g.map(sns.regplot, 'meas', 'pred', color='#34495e')
g.map(plt.axhline, y=0, lw=.3, ls='--', c='gray', alpha=.6)
g.map(plt.axvline, x=0, lw=.3, ls='--', c='gray', alpha=.6)
g.set_axis_labels('Measured', 'Estimated')
plt.savefig('%s/reports/Figure_5.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Figure 5 exported'

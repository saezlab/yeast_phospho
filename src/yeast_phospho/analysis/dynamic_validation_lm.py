import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas.stats.misc import zscore
from sklearn.linear_model import ElasticNet
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import pearson, spearman
from yeast_phospho.utilities import get_metabolites_name


# -- Import IDs maps
met_name = get_metabolites_name()
met_name = {'%.2f' % float(k): met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}


# -- Import data-sets
# Dynamic without growth
x_train = read_csv('%s/tables/kinase_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
x_train = x_train[(x_train.count(1) / x_train.shape[1]) > .75].replace(np.NaN, 0.0)

y_train = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
y_train = y_train[y_train.std(1) > .4]
y_train.index = ['%.2f' % i for i in y_train.index]

# Dynamic combination
x_test = read_csv('%s/tables/kinase_activity_dynamic_combination.tab' % wd, sep='\t', index_col=0)
x_test = x_test[[c for c in x_test if not c.startswith('NaCl+alpha_')]]
x_test = x_test[(x_test.count(1) / x_test.shape[1]) > .75].replace(np.NaN, 0.0)

y_test = read_csv('%s/tables/metabolomics_dynamic_combination_cor_samples.csv' % wd, index_col=0)[x_test.columns]
y_test.index = ['%.2f' % i for i in y_test.index]

# Variables
ions = list(set(y_train.index).intersection(y_test.index))
kinases = list(set(x_test.index).intersection(x_train.index))

train, test = list(x_train), list(y_test)


# -- Linear regressions
df = []
for ion in ions:
    lm = ElasticNet(alpha=.01).fit(x_train.ix[kinases, train].T, y_train.ix[ion, train])

    pred, meas = Series(lm.predict(x_test.ix[kinases, test].T), index=test), y_test.ix[ion, test]
    pred, meas = zscore(pred), zscore(meas)

    cor, pval, nmeas = spearman(pred, meas)

    title = '%s\n%.2f, %.2e' % (met_name[ion][:11], cor, pval)

    for c in test:
        df.append((ion, met_name[ion], cor, pval, nmeas, pred.ix[c], meas.ix[c], title))

    print '%s: %.2f, %.2e' % (met_name[ion], cor, pval)

df = DataFrame(df, columns=['ion', 'name', 'cor', 'pval', 'nmeas', 'pred', 'meas', 'title']).sort('cor', ascending=False)
print '[INFO] Linear regressions done!'


# -- Plot
sns.set(style='ticks')
g = sns.FacetGrid(df, col='title', col_wrap=5, sharey=False, sharex=False)
g.map(sns.regplot, 'meas', 'pred', color='#34495e')
g.map(plt.axhline, y=0, lw=.3, ls='--', c='gray', alpha=.6)
g.map(plt.axvline, x=0, lw=.3, ls='--', c='gray', alpha=.6)
g.set_axis_labels('Measured', 'Estimated')
plt.savefig('%s/reports/dynamic_validation_lm.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Figure 5 exported'

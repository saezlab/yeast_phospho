import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import LeaveOneOut
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import pearson


# -- Import
# Steady-state with growth
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = ['%.4f' % i for i in metabolomics.index]

k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0)
tf_activity = tf_activity[tf_activity.std(1) > .4]


# Steady-state without growth
metabolomics_ng = read_csv('%s/tables/metabolomics_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_ng = metabolomics_ng[metabolomics_ng.std(1) > .4]
metabolomics_ng.index = ['%.4f' % i for i in metabolomics_ng.index]

k_activity_ng = read_csv('%s/tables/kinase_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_ng = k_activity_ng[(k_activity_ng.count(1) / k_activity_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_ng = read_csv('%s/tables/tf_activity_steady_state_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_ng = tf_activity_ng[tf_activity_ng.std(1) > .4]


# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn = metabolomics_dyn[metabolomics_dyn.std(1) > .4]
metabolomics_dyn.index = ['%.4f' % i for i in metabolomics_dyn.index]

k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity_dyn = k_activity_dyn[(k_activity_dyn.count(1) / k_activity_dyn.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn = tf_activity_dyn[tf_activity_dyn.std(1) > .4]


# Dynamic without growth
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn_ng = metabolomics_dyn_ng[metabolomics_dyn_ng.std(1) > .4]
metabolomics_dyn_ng.index = ['%.4f' % i for i in metabolomics_dyn_ng.index]

k_activity_dyn_ng = read_csv('%s/tables/kinase_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng = k_activity_dyn_ng[(k_activity_dyn_ng.count(1) / k_activity_dyn_ng.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn_ng = read_csv('%s/tables/tf_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_ng = tf_activity_dyn_ng[tf_activity_dyn_ng.std(1) > .4]

# Dynamic combination
k_activity_dyn_comb_ng = read_csv('%s/tables/kinase_activity_dynamic_combination.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_ng = k_activity_dyn_comb_ng[(k_activity_dyn_comb_ng.count(1) / k_activity_dyn_comb_ng.shape[1]) > .75].replace(np.NaN, 0.0)

metabolomics_dyn_comb = read_csv('%s/tables/dynamic_combination_metabolomics.csv' % wd, index_col=0)[k_activity_dyn_comb_ng.columns]
metabolomics_dyn_comb = metabolomics_dyn_comb[metabolomics_dyn_comb.std(1) > .4]
metabolomics_dyn_comb.index = ['%.4f' % i for i in metabolomics_dyn_comb.index]


# -- Build linear regression models
comparisons = [
    (k_activity, metabolomics, 'Kinases', 'Steady-state', 'with'),
    (tf_activity, metabolomics, 'TFs', 'Steady-state', 'with'),

    (k_activity_ng, metabolomics_ng, 'Kinases', 'Steady-state', 'without'),
    (tf_activity_ng, metabolomics_ng, 'TFs', 'Steady-state', 'without'),

    (k_activity_dyn, metabolomics_dyn, 'Kinases', 'Dynamic', 'with'),
    (tf_activity_dyn, metabolomics_dyn, 'TFs', 'Dynamic', 'with'),

    (k_activity_dyn_ng, metabolomics_dyn_ng, 'Kinases', 'Dynamic', 'without'),
    (tf_activity_dyn_ng, metabolomics_dyn_ng, 'TFs', 'Dynamic', 'without'),

    (k_activity_dyn_comb_ng, metabolomics_dyn_comb, 'Kinases', 'Combination', 'with'),
]


def loo_regressions(xs, ys, ft, dt, mt):
    print '[INFO]', ft, dt

    # Align matricies
    x = xs.loc[:, ys.columns].dropna(axis=1).T
    y = ys[x.index].T

    # Define cross-validation
    cv = LeaveOneOut(len(y))

    # Run regressions
    y_pred, y_betas = {}, {}
    for m in y:
        y_pred[m] = {}

        betas = []
        for train, test in cv:
            lm = ElasticNet(alpha=0.01).fit(x.ix[train], y.ix[train, m])
            y_pred[m][x.index[test][0]] = lm.predict(x.ix[test])[0]

            betas.append(dict(zip(*(x.columns, lm.coef_))))

        y_betas[m] = DataFrame(betas).median().to_dict()

    y_pred = DataFrame(y_pred).ix[y.index, y.columns]
    print '[INFO] Regression done: ', ft, dt

    # Perform correlation with predicted values
    metabolites_corr = [(ft, dt, f, mt, 'metabolites', pearson(y[f], y_pred[f])[0]) for f in y_pred]
    conditions_corr = [(ft, dt, s, mt, 'conditions', pearson(y.ix[s], y_pred.ix[s])[0]) for s in y_pred.index]

    return (metabolites_corr + conditions_corr), (ft, dt, mt, y_betas)

lm_res = [loo_regressions(xs, ys, ft, dt, mt) for xs, ys, ft, dt, mt in comparisons]

lm_cor = [(ft, dt, f, mt, ct, c) for c in lm_res for ft, dt, f, mt, ct, c in c[0]]
lm_cor = DataFrame(lm_cor, columns=['feature', 'dataset', 'variable', 'growth', 'corr_type', 'cor'])

lm_betas = [c[1] for c in lm_res]
print '[INFO] Regressions done'


# -- Plot linear regression predictions correlation
palette = {'TFs': '#34495e', 'Kinases': '#3498db'}

sns.set(style='ticks')
g = sns.FacetGrid(lm_cor, row='dataset', col='growth', legend_out=True)
g.map(sns.boxplot, 'corr_type', 'cor', 'feature', palette=palette, sym='')
g.map(sns.stripplot, 'corr_type', 'cor', 'feature', palette=palette, jitter=True, size=5)
g.map(plt.axhline, y=0, ls='--', c='.5')
plt.ylim([-1, 1])
g.add_legend()
sns.despine(trim=True)
plt.savefig('%s/reports/linear_regressions_lm.pdf' % wd, bbox_inches='tight')
plt.close('all')

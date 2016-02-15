import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import LeaveOneOut
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import pearson

# -- Imports
# GSEA kinases activities
k_activity_dyn_comb_gsea = read_csv('%s/tables/kinase_activity_dynamic_combination_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_gsea = k_activity_dyn_comb_gsea[[c for c in k_activity_dyn_comb_gsea if not c.startswith('NaCl+alpha_')]]
k_activity_dyn_comb_gsea = k_activity_dyn_comb_gsea[(k_activity_dyn_comb_gsea.count(1) / k_activity_dyn_comb_gsea.shape[1]) > .75].replace(np.NaN, 0.0)

# LM kinases activites
k_activity_dyn_comb_lm = read_csv('%s/tables/kinase_activity_dynamic_combination.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_comb_lm = k_activity_dyn_comb_lm[[c for c in k_activity_dyn_comb_lm if not c.startswith('NaCl+alpha_')]]
k_activity_dyn_comb_lm = k_activity_dyn_comb_lm[(k_activity_dyn_comb_lm.count(1) / k_activity_dyn_comb_lm.shape[1]) > .75].replace(np.NaN, 0.0)

# Metabolomics
metabolomics_dyn_comb = read_csv('%s/tables/metabolomics_dynamic_combination_cor_samples.csv' % wd, index_col=0)[k_activity_dyn_comb_gsea.columns]
metabolomics_dyn_comb = metabolomics_dyn_comb[metabolomics_dyn_comb.std(1) > .4]
metabolomics_dyn_comb.index = ['%.4f' % i for i in metabolomics_dyn_comb.index]

kinases, ions, conditions = set(k_activity_dyn_comb_lm.index), set(metabolomics_dyn_comb.index), set(metabolomics_dyn_comb)

# -- Define metabolic to analyse
m = '606.0736'
k_activities = [('gsea', k_activity_dyn_comb_gsea), ('lm', k_activity_dyn_comb_lm)]

# -- Kinase activities correlation
m_cor = [(k, pearson(k_activity_dyn_comb_lm.ix[k, conditions], metabolomics_dyn_comb.ix[m, conditions])[0], pearson(k_activity_dyn_comb_gsea.ix[k, conditions], metabolomics_dyn_comb.ix[m, conditions])[0]) for k in kinases]
m_cor = DataFrame(m_cor, columns=['kinase', 'lm_cor', 'gsea_cor']).set_index('kinase')

lm_top_features = list(m_cor['lm_cor'].abs().sort(inplace=False, ascending=False).head(5).index)
gsea_top_features = list(m_cor['gsea_cor'].abs().sort(inplace=False, ascending=False).head(5).index)
top_features = list(set(lm_top_features).union(gsea_top_features))

plot_df = [(m, k, c, method, metabolomics_dyn_comb.ix[m, c], df.ix[k, c]) for k in top_features for c in conditions for method, df in k_activities]
plot_df = DataFrame(plot_df, columns=['ion', 'kinase', 'condition', 'method', 'fc', 'activity'])

sns.set(style='ticks')
g = sns.FacetGrid(plot_df, row='method', col='kinase', legend_out=True, sharey=False)
g.map(sns.regplot, 'fc', 'activity')
g.map(plt.axhline, y=0, ls='-', c='gray', lw=.3)
g.map(plt.axvline, x=0, ls='-', c='gray', lw=.3)
g.add_legend()
sns.despine(trim=True)
plt.savefig('%s/reports/regression_test_scatter.pdf' % wd, bbox_inches='tight')
plt.close('all')

# -- Linear regressions
pred = {}
for df_type, df in k_activities:
    # df_type, df = [('gsea', k_activity_dyn_comb_gsea), ('lm', k_activity_dyn_comb_lm)][0]

    ys = metabolomics_dyn_comb.ix[m]
    xs = df[ys.index].T

    pred[df_type] = Series({xs.ix[test].index[0]: ElasticNet(alpha=0.01).fit(xs.ix[train], ys.ix[train]).predict(xs.ix[test])[0] for train, test in LeaveOneOut(len(ys))})

pred = DataFrame(pred)
pred['measured'] = metabolomics_dyn_comb.ix[m, pred.index]

sns.set(style='ticks')
sns.pairplot(pred, kind='reg')
plt.savefig('%s/reports/regression_test.pdf' % wd, bbox_inches='tight')
plt.close('all')

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from sklearn.linear_model import ElasticNet
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics.regression import mean_squared_error
from pandas import DataFrame, read_csv, Series
from yeast_phospho.utilities import pearson


# -- Imports
# Metabolomics
metabolomics_dyn_ng = read_csv('%s/tables/metabolomics_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
metabolomics_dyn_ng = metabolomics_dyn_ng[metabolomics_dyn_ng.std(1) > .4]
metabolomics_dyn_ng.index = ['%.4f' % i for i in metabolomics_dyn_ng.index]

# GSEA
k_activity_dyn_ng_gsea = read_csv('%s/tables/kinase_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng_gsea = k_activity_dyn_ng_gsea[(k_activity_dyn_ng_gsea.count(1) / k_activity_dyn_ng_gsea.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn_gsea = read_csv('%s/tables/tf_activity_dynamic_gsea_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_gsea = tf_activity_dyn_gsea[tf_activity_dyn_gsea.std(1) > .4]

# LM
k_activity_dyn_ng_lm = read_csv('%s/tables/kinase_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
k_activity_dyn_ng_lm = k_activity_dyn_ng_lm[(k_activity_dyn_ng_lm.count(1) / k_activity_dyn_ng_lm.shape[1]) > .75].replace(np.NaN, 0.0)

tf_activity_dyn_ng_lm = read_csv('%s/tables/tf_activity_dynamic_no_growth.tab' % wd, sep='\t', index_col=0)
tf_activity_dyn_ng_lm = tf_activity_dyn_ng_lm[tf_activity_dyn_ng_lm.std(1) > .4]

# Conditions
conditions = {'_'.join(c.split('_')[:-1]) for c in metabolomics_dyn_ng}

comparisons = [
    (k_activity_dyn_ng_gsea, metabolomics_dyn_ng, 'Kinases', 'Gsea'),
    (tf_activity_dyn_gsea, metabolomics_dyn_ng, 'TFs', 'Gsea'),

    (k_activity_dyn_ng_lm, metabolomics_dyn_ng, 'Kinases', 'Lm'),
    (tf_activity_dyn_ng_lm, metabolomics_dyn_ng, 'TFs', 'Lm'),
]

# -- Linear regressions
lm_res, rmse_res = [], []
for (x, y, feature_type, method_type) in comparisons:
    for m in metabolomics_dyn_ng.index:
        for c in conditions:
            ys = y.ix[m, [i for i in y if not i.startswith(c)]]
            xs = x[ys.index].T

            yss = y.ix[m, [i for i in y if i.startswith(c)]]
            xss = x[yss.index].T

            lm = ElasticNet(alpha=0.01).fit(xs, ys)
            pred = Series(lm.predict(xss), index=xss.index)

            train_error = mean_squared_error(ys, lm.predict(xs))
            test_error = mean_squared_error(yss, pred)

            lm_res.append((feature_type, method_type, m, c, pearson(yss, pred)[0]))

            rmse_res.append((feature_type, method_type, m, c, 'train', train_error))
            rmse_res.append((feature_type, method_type, m, c, 'test', test_error))
lm_res = DataFrame(lm_res, columns=['feature', 'method', 'ion', 'condition', 'pearson'])
rmse_res = DataFrame(rmse_res, columns=['feature', 'method', 'ion', 'condition', 'dataset', 'rmse'])
print lm_res.head()

# -- Plot
palette = {'TFs': '#34495e', 'Kinases': '#3498db'}

# Cor
sns.set(style='ticks')
g = sns.FacetGrid(lm_res, row='method', legend_out=True)
g.map(sns.boxplot, 'condition', 'pearson', 'feature', palette=palette, sym='')
g.map(sns.stripplot, 'condition', 'pearson', 'feature', palette=palette, jitter=True, size=5, split=True, edgecolor='white', linewidth=.75)
g.map(plt.axhline, y=0, ls='-', c='gray', lw=.3)
plt.ylim([-1, 1])
g.add_legend()
sns.despine(trim=True)
plt.savefig('%s/reports/dynamic_overfitting.pdf' % wd, bbox_inches='tight')
plt.close('all')

# RMSE
sns.set(style='ticks')
g = sns.FacetGrid(rmse_res, row='method', col='feature', legend_out=True, sharey=False)
g.map(sns.boxplot, 'condition', 'rmse', 'dataset', sym='')
g.map(sns.stripplot, 'condition', 'rmse', 'dataset', jitter=True, size=5, split=True, edgecolor='white', linewidth=.75)
g.map(plt.axhline, y=0, ls='-', c='gray', lw=.3)
g.add_legend()
sns.despine(trim=True)
plt.savefig('%s/reports/dynamic_overfitting_rmse.pdf' % wd, bbox_inches='tight')
plt.close('all')

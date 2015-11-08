import numpy as np
import seaborn as sns
import itertools as it
import statsmodels.api as sm
import statsmodels.tools as st
import matplotlib.pyplot as plt
from yeast_phospho import wd, data
from sklearn.cross_validation import KFold, ShuffleSplit
from statsmodels.tools.eval_measures import rmse
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.linear_model import Lasso, RandomizedLasso
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from pandas import DataFrame, Series, read_csv, concat, melt, pivot_table
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name


# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# Import data-set
tf_activity = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)

metabolomics = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)[tf_activity.columns]
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]
metabolomics = metabolomics[[i in met_name for i in metabolomics.index]]


# Import TF KO data-set metabolomics
tf_ko_map = read_csv('%s/yeast_tf/tf_map.txt' % data, sep='\t', index_col=0)['orf'].to_dict()


def import_tf_ko(file_path):
    df = read_csv(file_path, sep='\t', index_col=0)
    df.index = ['%.2f' % i for i in df.index]

    counts = {mz: counts for mz, counts in zip(*(np.unique(df.index, return_counts=True)))}
    df = df[[counts[i] == 1 for i in df.index]]

    df.columns = ['_'.join(c.split('_')[:-1]) for c in df]
    df = df[list(set(df).intersection(tf_ko_map))]
    df.columns = [c.split('_')[0] + '_' + tf_ko_map[c] for c in df.columns]

    return df

tf_logfc = import_tf_ko('%s/yeast_tf/yeast_tf_ko_logfc.txt' % data)
tf_pvalue = import_tf_ko('%s/yeast_tf/yeast_tf_ko_pvalues.txt' % data)


# Linear regression
y = metabolomics.T
x = tf_activity[y.index].T

x_coefs = {m: {f: sm.OLS(y[m], st.add_constant(x[f])).fit() for f in x} for m in y}
print '[INFO] Regressions done: ', len(x_coefs)

df = DataFrame([(m, f, x_coefs[m][f].params[f], x_coefs[m][f].pvalues[f]) for f in x for m in y], columns=['variable', 'feature', 'coef', 'pvalue']).sort('pvalue')
df['cor'] = [pearson(y[m], x[f])[0] for m, f in df[['variable', 'feature']].values]
df['ci_lb'] = [x_coefs[m][f].conf_int().ix[f, 0] for m, f in df[['variable', 'feature']].values]
df['ci_ub'] = [x_coefs[m][f].conf_int().ix[f, 1] for m, f in df[['variable', 'feature']].values]
df['var_name'] = [met_name[i] for i in df['variable']]
df['feature_name'] = [acc_name[i] for i in df['feature']]
df.to_csv('%s/tables/tf_coefficients.txt' % wd, sep='\t', index=False)
print df.head(15)


plot_df = [(m, f, coef, tf_logfc.ix[m, c], pvalue) for m, f, coef, pvalue in df[['variable', 'feature', 'coef', 'pvalue']].values if pvalue < 0.01 and m in tf_logfc.index for c in tf_logfc if f in c]
plot_df = DataFrame(plot_df, columns=['variable', 'feature', 'coef', 'fold-change', 'pvalue'])
plot_df['log10_pvalue'] = [-np.log10(p) if c > 0 else np.log10(p) for p, c in plot_df[['pvalue', 'coef']].values]

sns.set(style='ticks')
margin = 1.20
xlim, ylim = (plot_df['coef'].min() * margin, plot_df['coef'].max() * margin), (plot_df['fold-change'].min() * margin, plot_df['fold-change'].max() * margin)
sns.jointplot('log10_pvalue', 'fold-change', data=plot_df, kind='reg', xlim=xlim, ylim=ylim, marginal_kws={'hist': False}, stat_func=pearsonr)
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('TF (coefficient)')
plt.ylabel('TF knockout (log fold-change)')
plt.savefig('%s/reports/TF_KO_validation.pdf' % wd, bbox_inches='tight')
plt.close('all')

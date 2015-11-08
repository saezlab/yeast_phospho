import numpy as np
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools as st
from yeast_phospho import wd, data
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.cross_validation import KFold, ShuffleSplit
from statsmodels.tools.eval_measures import rmse
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from pandas import DataFrame, Series, read_csv, concat, melt, pivot_table
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name


# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# Import data-set
metabolomics = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]

k_activity = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

#
metabolomics_ss = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics_ss.index = [str(i) for i in metabolomics_ss.index]

# Linear regression
y = metabolomics.T
x = k_activity[y.index].T

x_coefs = {m: {f: sm.OLS(y[m], st.add_constant(x[f])).fit() for f in x} for m in y}
print '[INFO] Regressions done: ', len(x_coefs)

df = DataFrame([(m, f, x_coefs[m][f].params[f], x_coefs[m][f].pvalues[f]) for f in x for m in y], columns=['variable', 'feature', 'coef', 'pvalue']).sort('pvalue')
df['cor'] = [pearson(y[m], x[f])[0] for m, f in df[['variable', 'feature']].values]
df['ci_lb'] = [x_coefs[m][f].conf_int().ix[f, 0] for m, f in df[['variable', 'feature']].values]
df['ci_ub'] = [x_coefs[m][f].conf_int().ix[f, 1] for m, f in df[['variable', 'feature']].values]
df['var_name'] = [met_name[i] if i in met_name else i for i in df['variable']]
df['feature_name'] = [acc_name[i] for i in df['feature']]
df.to_csv('%s/tables/tf_coefficients.txt' % wd, sep='\t', index=False)
print df.head(15)


plot_df = [(m, f, coef, metabolomics_ss.ix[m, c]) for m, f, coef, pvalue in df[['variable', 'feature', 'coef', 'pvalue']].values if pvalue < 0.01 and m in metabolomics_ss.index for c in metabolomics_ss if f in c]
plot_df = DataFrame(plot_df, columns=['variable', 'feature', 'coef', 'fold-change'])

sns.set(style='ticks')
margin = 1.20
xlim, ylim = (plot_df['coef'].min() * margin, plot_df['coef'].max() * margin), (plot_df['fold-change'].min() * margin, plot_df['fold-change'].max() * margin)
sns.jointplot('coef', 'fold-change', data=plot_df, kind='reg', xlim=xlim, ylim=ylim, marginal_kws={'hist': False}, stat_func=pearsonr)
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('Kinase (coefficient)')
plt.ylabel('Kinase knockout (log fold-change)')
plt.savefig('%s/reports/Kinase_KO_validation.pdf' % wd, bbox_inches='tight')
plt.close('all')

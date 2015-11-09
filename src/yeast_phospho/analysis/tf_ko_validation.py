import numpy as np
import seaborn as sns
import itertools as it
import statsmodels.api as sm
import statsmodels.tools as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from yeast_phospho import wd, data
from sklearn.cross_validation import KFold, ShuffleSplit
from statsmodels.tools.eval_measures import rmse
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.metrics.scorer import make_scorer
from sklearn.decomposition.pca import PCA
from sklearn.feature_selection.rfe import RFECV, RFE
from sklearn.preprocessing.data import StandardScaler
from sklearn.linear_model import Lasso, RandomizedLasso, LinearRegression, LassoCV, ElasticNet
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from pandas import DataFrame, Series, read_csv, concat, melt, pivot_table
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name


# Import annotations
acc_name = get_proteins_name()
met_name = get_metabolites_name()


# Import data-set
tf_activity = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)

k_activity = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

metabolomics = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)[tf_activity.columns]
metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]
metabolomics = metabolomics[[i in met_name for i in metabolomics.index]]

# PCA
n_components = 10

sns.set(style='ticks')
fig, gs, pos = plt.figure(figsize=(10, 15)), GridSpec(3, 2, hspace=.3), 0
for df, df_type in [(metabolomics, 'Metabolomics'), (tf_activity, 'TF activity'), (k_activity, 'Kinases activity')]:
    pca = PCA(n_components=n_components).fit(df)
    pcs = DataFrame(pca.transform(df), columns=['PC%d' % i for i in range(1, n_components + 1)], index=df.index)
    pcs['condition'] = ['_'.join(i.split('_')[:-1]) for i in pcs.index]

    condition_palette = {'N_downshift': '#3498db', 'N_upshift': '#95a5a6', 'Rapamycin': '#e74c3c'}

    ax = plt.subplot(gs[pos])
    for condition in set(pcs['condition']):
        ax.scatter(pcs.ix[pcs['condition'] == condition, 'PC1'], pcs.ix[pcs['condition'] == condition, 'PC2'], label=condition, lw=0, marker='o', c=condition_palette[condition])
    ax.legend()
    ax.set_title(df_type)
    ax.set_xlabel('PC1 (%.1f%%)' % (pca.explained_variance_ratio_[0] * 100))
    ax.set_ylabel('PC2 (%.1f%%)' % (pca.explained_variance_ratio_[1] * 100))
    sns.despine(trim=True, ax=ax)

    ax = plt.subplot(gs[pos + 1])
    plot_df = DataFrame(zip(['PC%d' % i for i in range(1, n_components + 1)], pca.explained_variance_ratio_), columns=['PC', 'var'])
    plot_df['var'] *= 100
    sns.barplot('var', 'PC', data=plot_df, color='gray', linewidth=0, ax=ax)
    ax.set_title(df_type)
    ax.set_xlabel('Explained variance ratio')
    ax.set_ylabel('Principal component')
    sns.despine(trim=True, ax=ax)
    ax.figure.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

    pos += 2

    print '[INFO] PCA analysis done: %s' % df_type

plt.savefig('%s/reports/dynamic_pca.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] PCA done'


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
sc_x, sc_y = StandardScaler(), StandardScaler()

x = tf_activity.T
x = DataFrame(sc_x.fit_transform(x), index=x.index, columns=x.columns)

y = metabolomics[x.index].T
y = DataFrame(sc_y.fit_transform(y), index=y.index, columns=y.columns)

rmse_scorer = make_scorer(rmse, greater_is_better=False)

x_coefs = {}
for m in y.columns:
    lm = LinearRegression()
    cv = ShuffleSplit(len(y[m]), test_size=.3, n_iter=30)

    rfe = RFE(lm, step=1, n_features_to_select=1).fit(x, y[m])

    for train, test in cv:
        lm_m = lm.fit(x.ix[train, rfe.support_], y.ix[train, m])
        train_rmse = rmse(lm_m.predict(x.ix[train, rfe.support_]), y.ix[train, m])
        test_rmse = rmse(lm_m.predict(x.ix[test, rfe.support_]), y.ix[test, m])
        print train_rmse, test_rmse

    # lm = lm.fit(x.loc[:, rfe.support_], y[m])

    lm = sm.OLS(y[m], st.add_constant(x.loc[:, rfe.support_])).fit()
    cor, pvalue, _ = pearson(lm.predict(st.add_constant(x.loc[:, rfe.support_])), y[m].values)

    if lm.f_pvalue < .05:
        # x_coefs[m] = dict(zip(*(x.ix[:, rfe.support_], lm.coef_)))
        x_coefs[m] = {x.loc[:, rfe.support_]: cor}
print '[INFO] Regressions done: ', len(x_coefs)

df = DataFrame([(m, f, x_coefs[m][f]) for m in x_coefs for f in x_coefs[m] if f != 'const'], columns=['variable', 'feature', 'coef'])
df['cor'] = [pearson(y[m], x[f])[0] for m, f in df[['variable', 'feature']].values]
# df['ci_lb'] = [x_coefs[m][f].conf_int().ix[f, 0] for m, f in df[['variable', 'feature']].values]
# df['ci_ub'] = [x_coefs[m][f].conf_int().ix[f, 1] for m, f in df[['variable', 'feature']].values]
df['var_name'] = [met_name[i] for i in df['variable']]
df['feature_name'] = [acc_name[i] for i in df['feature']]
df.to_csv('%s/tables/tf_coefficients.txt' % wd, sep='\t', index=False)
print df.head(15)


plot_df = [(m, f, coef, tf_logfc.ix[m, c]) for m, f, coef in df[['variable', 'feature', 'coef']].values if m in tf_logfc.index for c in tf_logfc if f in c]
plot_df = DataFrame(plot_df, columns=['variable', 'feature', 'coef', 'fold-change'])
plot_df = plot_df[plot_df['coef'] != 0]

sns.set(style='ticks')
margin = 1.20
xlim, ylim = (plot_df['coef'].min() * margin, plot_df['coef'].max() * margin), (plot_df['fold-change'].min() * margin, plot_df['fold-change'].max() * margin)
sns.jointplot('coef', 'fold-change', data=plot_df, kind='reg', xlim=xlim, ylim=ylim, marginal_kws={'hist': False}, stat_func=pearsonr)
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('TF (coefficient)')
plt.ylabel('TF knockout (log fold-change)')
plt.savefig('%s/reports/TF_KO_validation.pdf' % wd, bbox_inches='tight')
plt.close('all')

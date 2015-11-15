import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools as st
from yeast_phospho import wd, data
from pandas import DataFrame, Series, read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.classification import jaccard_similarity_score
from sklearn.decomposition import FactorAnalysis, PCA
from yeast_phospho.utilities import pearson, get_proteins_name, get_metabolites_name, regress_out, jaccard, get_tfs_targets


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


# Import annotations
acc_name = get_proteins_name()

met_name = get_metabolites_name()
met_name = {k: met_name[k] for k in met_name if len(met_name[k].split('; ')) == 1}

tf_targets = get_tfs_targets()

# Import data-set
metabolomics = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0)
# metabolomics = metabolomics[metabolomics.std(1) > .4]
metabolomics.index = [str(i) for i in metabolomics.index]

k_activity = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity = k_activity[(k_activity.count(1) / k_activity.shape[1]) > .75].replace(np.NaN, 0.0)

k_activity_gsea = read_csv('%s/tables/kinase_activity_dynamic_gsea.tab' % wd, sep='\t', index_col=0)
k_activity_gsea = k_activity_gsea[(k_activity_gsea.count(1) / k_activity_gsea.shape[1]) > .75].replace(np.NaN, 0.0)


tf_activity = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0)
tf_activity = tf_activity[tf_activity.std(1) > .4]

#
x = tf_activity.T
y = metabolomics.T

ssx = StandardScaler()
x = DataFrame(ssx.fit_transform(x), index=x.index, columns=x.columns)

ssy = StandardScaler()
y = DataFrame(ssy.fit_transform(y), index=y.index, columns=y.columns)

cs, features, variables = list(set(y.index).intersection(x.index)), list(x), list(y)

lms = {m: {f: sm.OLS(y.ix[cs, m], st.add_constant(x.ix[cs, f])).fit() for f in features} for m in variables}

df = [(m, f, lms[m][f].rsquared, lms[m][f].llf, lms[m][f].params[f]) for m in variables for f in features]
df = [(m, f, cor, llf, coef, tf_logfc.ix[m, c], tf_pvalue.ix[m, c]) for m, f, cor, llf, coef in df if m in tf_logfc.index for c in tf_logfc if f in c]
df = DataFrame(df, columns=['var', 'fea', 'cor', 'llf', 'coef', 'logfc', 'pvalfc']).sort('cor', ascending=False)
df = df[[i in met_name for i in df['var']]]
df['var_name'] = [met_name[i] for i in df['var']]
df['fea_name'] = [acc_name[i] for i in df['fea']]
print df.sort('cor', ascending=False).head(15)

sns.set(style='ticks')
sns.jointplot('coef', 'logfc', data=df, kind='reg', marginal_kws={'hist': False})
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('coefficient')
plt.ylabel('KO log-FC')
plt.savefig('%s/reports/single_feature_regression.pdf' % wd, bbox_inches='tight')
plt.close('all')

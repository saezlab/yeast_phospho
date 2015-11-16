import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd, data
from pandas import DataFrame, read_csv, melt
from yeast_phospho.utilities import get_proteins_name, get_metabolites_name


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


# Linear regression results
with open('%s/tables/linear_regressions.pickle' % wd, 'rb') as handle:
    lm_res = pickle.load(handle)


df = DataFrame([i[1][3] for i in lm_res if i[1][0] == 'TFs' and i[1][1] == 'Dynamic' and i[1][2] == 'without'][0])
df['feature'] = df.index
df = melt(df, id_vars='feature', var_name='variable', value_name='coef')
df = df[df['coef'] != 0.0]

df = [(m, f, coef, tf_logfc.ix[m, c], tf_pvalue.ix[m, c]) for f, m, coef in df.values if m in tf_logfc.index for c in tf_logfc if f in c]
df = DataFrame(df, columns=['variable', 'feature', 'coef', 'logfc', 'pvalfc'])
df = df[[i in met_name for i in df['variable']]]
df['var_name'] = [met_name[i] for i in df['variable']]
df['fea_name'] = [acc_name[i] for i in df['feature']]
df = df[df['coef'] != 0]
print df.sort('logfc', ascending=False).tail(15)


sns.set(style='ticks')
sns.jointplot('coef', 'logfc', data=df, kind='reg', marginal_kws={'hist': False})
plt.axhline(y=0, ls='--', c='.5', lw=.3)
plt.axvline(x=0, ls='--', c='.5', lw=.3)
plt.xlabel('coefficient')
plt.ylabel('KO log-FC')
plt.savefig('%s/reports/single_feature_regression.pdf' % wd, bbox_inches='tight')
plt.close('all')

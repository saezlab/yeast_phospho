import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas import DataFrame, Series, read_csv, pivot_table
from statsmodels.stats.multitest import multipletests


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, np.sum(mask)

# Version
sns.set_style('white')
version = 'v1'
print '[INFO] Version: %s' % version

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import conversion table
name2id = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=0).to_dict()['name']

# Import gene-expression data-set
gexp = read_csv(wd + 'data/gene_expresion/tf_ko_gene_expression.tab', sep='\t', header=False)
gexp = gexp[gexp['study'] == 'Kemmeren_2014']
gexp['tf'] = [name2id[i] if i in name2id else id2name[i] for i in gexp['tf']]
gexp = pivot_table(gexp, values='value', index='target', columns='tf')
print '[INFO] Gene-expression imported!'

# Import TF enrichment
tf_df = read_csv(wd + 'tables/tf_enrichment_df.tab', sep='\t', index_col=0)

# Import kinase enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)

# Overlapping strains
strains = set(tf_df.columns).intersection(kinase_df.columns)
tf_df, kinase_df, gexp = tf_df.loc[:, strains], kinase_df.loc[:, strains], gexp.loc[:, strains]

# Significantly chaning TF
tfs = gexp.ix[tf_df.index].dropna()
tfs = list(tfs.index[tfs.std(1) > 1])

plot_df = [pearson(tf_df.ix[tf, strains], gexp.ix[tf, strains])[0] for tf in tfs if tf in gexp.index]
sns.boxplot(plot_df, widths=0.3)
sns.despine(bottom=True)
plt.title('TF correlation with gene expression (%d TFs)\naverage correlation: %.2f' % (len(plot_df), np.mean(plot_df)))
plt.ylabel('pearson r')
plt.savefig(wd + 'reports/%s_TF_correlation_gexp.pdf' % version, bbox_inches='tight')
plt.close('all')
print '[INFO] TF correlation with gene expression (%d): %.2f' % (len(plot_df), np.mean(plot_df))
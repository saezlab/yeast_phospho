import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.decomposition.pca import PCA
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv, pivot_table, Index

# ---- Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth'] / 100


# ---- Steady-state: gene-expression data-set
# Import conversion table
name2id = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=0).to_dict()['name']

# TF targets
tf_targets = read_csv('%s/files/tf_gene_network_chip_only.tab' % wd, sep='\t')
tf_targets['tf'] = [name2id[i] if i in name2id else id2name[i] for i in tf_targets['tf']]
tf_targets['interaction'] = 1
tf_targets = pivot_table(tf_targets, values='interaction', index='target', columns='tf', fill_value=0)
print '[INFO] TF targets calculated!'

# Import data-set
gexp = read_csv('%s/data/Kemmeren_2014_zscores_parsed_filtered.tab' % wd, sep='\t', header=False)
gexp['tf'] = [name2id[i] if i in name2id else id2name[i] for i in gexp['tf']]
gexp = pivot_table(gexp, values='value', index='target', columns='tf')

# Overlap conditions with growth measurements
strains = list(set(gexp.columns).intersection(growth.index))

# Filter gene-expression to overlapping conditions
gexp = gexp[strains]
print '[INFO] Gene expression data imported'


# ---- Steady-state: metabolomics data-set
metabolomics = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0)
metabolomics.index = Index(metabolomics.index, dtype=str)
print '[INFO] Metabolomics data imported'


# ---- Run PCA
n_components = 10

pca_metabolomics = PCA(n_components=n_components).fit(metabolomics.T)
pca_metabolomics_pc = DataFrame(pca_metabolomics.transform(metabolomics.T), columns=['PC%d' % i for i in range(1, n_components + 1)], index=metabolomics.columns)

pca_gexp = PCA(n_components=n_components).fit(gexp.T)
pca_gexp_pc = DataFrame(pca_gexp.transform(gexp.T), columns=['PC%d' % i for i in range(1, n_components + 1)], index=strains)


sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
g = sns.jointplot(growth[pca_metabolomics_pc.index], pca_metabolomics_pc['PC1'], kind='reg', color='gray')
plt.xlabel('Relative growth')
plt.ylabel('PC1 (%.1f%%)' % (pca_metabolomics.explained_variance_ratio_[0] * 100))
plt.savefig('%s/reports/pca_growth_metabolomics.pdf' % wd, bbox_inches='tight')
plt.close('all')


sns.set(style='ticks', palette='pastel', color_codes=True, context='paper')
g = sns.jointplot(growth[pca_gexp_pc.index], pca_gexp_pc['PC1'], kind='reg', color='gray')
plt.xlabel('Relative growth')
plt.ylabel('PC1 (%.1f%%)' % (pca_gexp.explained_variance_ratio_[0] * 100))
plt.savefig('%s/reports/pca_growth_gene_expression.pdf' % wd, bbox_inches='tight')
plt.close('all')


plot_df = DataFrame(zip(['PC%d' % i for i in range(1, n_components + 1)], pca_metabolomics.explained_variance_ratio_), columns=['PC', 'var'])
plot_df['var'] *= 100
sns.barplot('var', 'PC', data=plot_df, color='gray', linewidth=0)
plt.xlabel('Explained variance ratio')
plt.ylabel('Principal component')
sns.despine(trim=True)
plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
plt.savefig('%s/reports/pca_growth_metabolomics_pcs.pdf' % wd, bbox_inches='tight')
plt.close('all')


plot_df = DataFrame(zip(['PC%d' % i for i in range(1, n_components + 1)], pca_gexp.explained_variance_ratio_), columns=['PC', 'var'])
plot_df['var'] *= 100
sns.barplot('var', 'PC', data=plot_df, color='gray', linewidth=0)
plt.xlabel('Explained variance ratio')
plt.ylabel('Principal component')
sns.despine(trim=True)
plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
plt.savefig('%s/reports/pca_growth_gene_expression_pcs.pdf' % wd, bbox_inches='tight')
plt.close('all')

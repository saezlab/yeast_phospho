import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from pandas.stats.misc import zscore
from sklearn.decomposition.pca import PCA
from matplotlib.gridspec import GridSpec
from yeast_phospho.utilities import pearson, regress_out


# Regress-out Factor correlated with growth rate
datasets = [
    # Linear regression
    ('metabolomics_steady_state', 'Metabolomics', 'strain_relative_growth_rate.txt', 'Genetic perturbations', 'PC1'),
    ('kinase_activity_steady_state', 'Kinases/Phosphatases', 'strain_relative_growth_rate.txt', 'Genetic perturbations', 'PC1'),
    ('tf_activity_steady_state', 'TFs', 'strain_relative_growth_rate.txt', 'Genetic perturbations', 'PC2'),

    ('metabolomics_dynamic', 'Metabolomics', 'dynamic_growth.txt', 'Nitrogen metabolism', 'PC2'),
    ('kinase_activity_dynamic', 'Kinases/Phosphatases', 'dynamic_growth.txt', 'Nitrogen metabolism', 'PC2'),
    ('tf_activity_dynamic', 'TFs', 'dynamic_growth.txt', 'Nitrogen metabolism', 'PC3')
]

n_components = 10
sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3}, font_scale=0.75)
fig, gs, pos = plt.figure(figsize=(7, 4 * len(datasets))), GridSpec(1 * len(datasets), 2, hspace=.425, wspace=.3), 0

for df_file, df_type, growth_file, dataset_type, pc in datasets:
    print df_file

    # Import growth rates
    growth = zscore(read_csv('%s/files/%s' % (wd, growth_file), sep='\t', index_col=0)['relative_growth'])

    # Import data-set
    df = read_csv('%s/tables/%s.tab' % (wd, df_file), sep='\t', index_col=0)

    if df_type == 'Kinases/Phosphatases':
        df = df[(df.count(1) / df.shape[1]) > .75]

    # Conditions overlap
    conditions = list(set(growth.index).intersection(df))

    # PCA analysis
    pca = PCA(n_components=n_components).fit(df.T.replace(np.nan, 0))
    pca_pc = DataFrame(pca.transform(df.T.replace(np.nan, 0)), columns=['PC%d' % i for i in range(1, n_components + 1)], index=df.columns)

    # Plot correlation with PCA
    ax = plt.subplot(gs[pos])
    cor, pvalue, nmeas = pearson(growth[pca_pc.index], pca_pc[pc])
    sns.regplot(growth[pca_pc.index], pca_pc[pc], ax=ax, color='#4c4c4c')
    ax.axhline(0, ls='-', lw=0.1, c='black', alpha=.3)
    ax.axvline(0, ls='-', lw=0.1, c='black', alpha=.3)
    ax.set_title('%s - %s\n(Pearson: %.2f, p-value: %.1e)' % (dataset_type, df_type, cor, pvalue))
    ax.set_xlabel('Relative growth (centered)')
    ax.set_ylabel('PC%d (%.1f%%)' % (int(pc[-1:]), pca.explained_variance_ratio_[int(pc[-1:]) - 1] * 100))
    sns.despine(trim=True, ax=ax)

    ax = plt.subplot(gs[pos + 1])
    plot_df = DataFrame(zip(['PC%d' % i for i in range(1, n_components + 1)], pca.explained_variance_ratio_), columns=['PC', 'var'])
    plot_df['var'] *= 100
    sns.barplot('var', 'PC', data=plot_df, color='#4c4c4c', linewidth=0, ax=ax)
    ax.set_xlabel('Explained variance ratio')
    ax.set_ylabel('Principal component')
    sns.despine(trim=True, ax=ax)
    ax.figure.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

    pos += 2

    # Regress-out factor
    df = DataFrame({m: regress_out(growth[conditions], df.ix[m, conditions]) for m in df.index}).T

    # Export regressed-out data-set
    df.to_csv('%s/tables/%s_no_growth.tab' % (wd, df_file), sep='\t')
    print '[INFO] Growth regressed-out: ', 'tables/%s_no_growth.tab' % df_file

plt.savefig('%s/reports/PCA_growth_correlation.pdf' % wd, bbox_inches='tight')
plt.close('all')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from yeast_phospho import wd
from yeast_phospho.utilities import pearson
from matplotlib.gridspec import GridSpec
from sklearn.decomposition.pca import PCA
from pandas import DataFrame, read_csv


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)


# Import data-sets
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)[ko_strains]

trans = read_csv('%s/tables/transcriptomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)

phospho_df = read_csv('%s/tables/pproteomics_steady_state.tab' % wd, sep='\t', index_col=0).loc[:, ko_strains].dropna(how='all', axis=1)
phospho_df = phospho_df[(phospho_df.count(1) / phospho_df.shape[1]) > .25].replace(np.NaN, 0.0)


# PCA analysis
n_components = 10

sns.set(style='ticks')
fig, gs, pos = plt.figure(figsize=(10, 15)), GridSpec(3, 2, hspace=.3), 0
for df, df_type in [(trans.T, 'Transcriptomics'), (phospho_df.T, 'Phospho-proteomics'), (metabolomics.T, 'Metabolomics')]:
    pca = PCA(n_components=n_components).fit(df)
    pca_pc = DataFrame(pca.transform(df), columns=['PC%d' % i for i in range(1, n_components + 1)], index=df.index)

    ax = plt.subplot(gs[pos])
    cor, pvalue, nmeas = pearson(growth[pca_pc.index], pca_pc['PC1'])
    sns.regplot(growth[pca_pc.index], pca_pc['PC1'], ax=ax)
    ax.set_title('%s (pearson: %.2f, p-value: %.2e)' % (df_type, cor, pvalue))
    ax.set_xlabel('Relative growth')
    ax.set_ylabel('PC1 (%.1f%%)' % (pca.explained_variance_ratio_[0] * 100))
    sns.despine(trim=True, ax=ax)

    ax = plt.subplot(gs[pos + 1])
    plot_df = DataFrame(zip(['PC%d' % i for i in range(1, n_components + 1)], pca.explained_variance_ratio_), columns=['PC', 'var'])
    plot_df['var'] *= 100
    sns.barplot('var', 'PC', data=plot_df, color='gray', linewidth=0, ax=ax)
    ax.set_xlabel('Explained variance ratio')
    ax.set_ylabel('Principal component')
    sns.despine(trim=True, ax=ax)
    ax.figure.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

    pos += 2

    print '[INFO] PCA analysis done: %s' % df_type

plt.savefig('%s/reports/Figure_Supp_1.pdf' % wd, bbox_inches='tight')
plt.close('all')

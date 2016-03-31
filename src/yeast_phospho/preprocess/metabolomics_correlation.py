import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as it
from yeast_phospho import wd
from pandas.stats.misc import zscore
from scipy.stats.stats import spearmanr
from pandas import DataFrame, Series, read_csv, concat


# -- Imports
# Annotation
annot = read_csv('%s/files/2015_12_10_fia_experiments_imm904_annotation.txt' % wd, sep='\t', index_col=0)
annot['mz'] = ['%.4f' % i for i in annot['mz']]
annot = annot[annot['mod'] == '-H(+)']
annot = annot.groupby('mz')['id'].agg(lambda x: set(x)).to_dict()

m_map = read_csv('%s/files/james_yeast.txt' % wd, sep='\t', index_col=0).dropna()['id'].to_dict()

# Data-sets
m_targeted = read_csv('%s/tables/metabolomics_dynamic_combination_targeted.csv' % wd, index_col=0)
m_targeted = m_targeted[[i in m_map for i in m_targeted.index]]
m_targeted.index = [m_map[i] for i in m_targeted.index]

m_untargeted = read_csv('%s/tables/metabolomics_dynamic_combination.csv' % wd, index_col=0)
m_untargeted.index = ['; '.join(annot['%.4f' % i]) for i in m_untargeted.index]

# -- Overlap data-sets
metabolites = list(set(m_targeted.index).intersection(m_untargeted.index))

samples = list(set(m_targeted).intersection(m_untargeted))
samples = [c for c in samples if (c.split('_')[1] not in ['0', '25']) and (not c.startswith('NaCl+alpha_'))]

m_targeted, m_untargeted = m_targeted.ix[metabolites, samples], m_untargeted.ix[metabolites, samples]

# -- Plot
plot_df = [(m, c, m_targeted.ix[m, c], m_untargeted.ix[m, c]) for m, c in it.product(metabolites, samples)]
plot_df = DataFrame(plot_df, columns=['metabolite', 'condition', 'LC-MS', 'QTOF'])

color = '#808080'

sns.set(style='ticks')
g = sns.jointplot(
    'LC-MS', 'QTOF', plot_df, 'reg', color=color, joint_kws={'scatter_kws': {'s': 40, 'edgecolor': 'w', 'linewidth': .5}},
    marginal_kws={'hist': False, 'rug': True}, annot_kws={'template': 'Spearman: {val:.2g}, p-value: {p:.1e}'}, space=0,
    stat_func=spearmanr, xlim=(-1.5, 4), ylim=(-1, 2.5)
)
plt.axhline(0, ls='-', lw=0.3, c=color, alpha=.5)
plt.axvline(0, ls='-', lw=0.3, c=color, alpha=.5)
g.plot_marginals(sns.kdeplot, shade=True, color=color)
g.set_axis_labels('LC-MS (log2 fold-change)', 'QTOF (log2 fold-change)')
plt.savefig('%s/reports/metabolomics_correlation.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Corr plotted!'

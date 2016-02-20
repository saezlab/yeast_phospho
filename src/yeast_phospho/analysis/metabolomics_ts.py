import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from numpy.lib.nanfunctions import nanmedian
from pandas import DataFrame, Series, read_csv

conditions = ['NaCl', 'alpha']
palette = {'NaCl': '#CC2229', 'pheromone': '#6FB353'}

# -- Import annotation
annot = read_csv('%s/files/2015_12_10_fia_experiments_imm904_annotation.txt' % wd, sep='\t', index_col=0)
annot['mz'] = ['%.4f' % i for i in annot['mz']]
annot = annot[annot['mod'] == '-H(+)']
annot = annot.groupby('mz')['name'].agg(lambda x: '; '.join(set(x))).to_dict()
annot['341.1089'] = 'Trehalose'
annot['338.9885'] = 'D-Fructose 1,6-bisphosphate'
annot['160.0615'] = 'O-Acetyl-L-homoserine'
annot['421.0750'] = 'Trehalose 6-phosphate'
annot['418.9559'] = '1D-myo-Inositol\n1,4,5-trisphosphate'


# -- Import data-sets
m_untargeted_std = read_csv('%s/tables/metabolomics_dynamic_combination_std.csv' % wd, index_col=0)
m_untargeted_std = m_untargeted_std[[c for c in m_untargeted_std if c.split('_')[0] in conditions]]
m_untargeted_std.index = ['%.4f' % i for i in m_untargeted_std.index]


# -- tsplot: untargeted matebolomics
m_to_show = ['Glycerol 3-phosphate', 'Trehalose', 'L-Malate', 'N-Acetyl-L-glutamate']

plot_df = m_untargeted_std.unstack().reset_index()
plot_df.columns = ['sample', 'metabolite', 'fc']
plot_df['condition'] = [i.split('_')[0] for i in plot_df['sample']]
plot_df['time'] = [int(i.split('_')[1].split('.')[0]) / 60 for i in plot_df['sample']]
plot_df['replicate'] = [0 if len(i.split('_')[1].split('.')) == 1 else i.split('_')[1].split('.')[1] for i in plot_df['sample']]
plot_df['metabolite'] = [annot[m] for m in plot_df['metabolite']]
plot_df = plot_df[[i in m_to_show for i in plot_df['metabolite']]]
plot_df['condition'] = ['pheromone' if i == 'alpha' else i for i in plot_df['condition']]

sns.set(style='ticks')
g = sns.FacetGrid(plot_df, col='metabolite', col_wrap=len(m_to_show), sharey=False)
g.map_dataframe(sns.tsplot, time='time', unit='replicate', condition='condition', value='fc', estimator=nanmedian, color=palette, err_kws={'lw': 0})
g.map(plt.axhline, ls='-', lw=.3, c='gray')
g.set_titles('{col_name}')
g.set_axis_labels('Time (minutes)', 'Metabolite\n(log2 fold-change)')
g.add_legend()
plt.savefig('%s/reports/metabolomics_facetgrid_ts_untargeted.pdf' % wd, bbox_inches='tight')
plt.close('all')
print '[INFO] Done'
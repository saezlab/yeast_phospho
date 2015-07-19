import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from yeast_phospho import wd
from yeast_phospho.utils import spearman
from pandas import DataFrame, read_csv, Index
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import LinearRegression

sns.set_style('ticks')

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

# Import metabolites map
m_map = read_csv('%s/files/metabolomics/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = Index([str(i) for i in m_map['mz']], dtype=str)
m_map = m_map.groupby('mz')['name'].apply(lambda x: '; '.join(x)).to_dict()

# Import kinase activity
k_activity = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)

# Import metabolomics
metabolomics = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0)
metabolomics.index = Index([str(i) for i in metabolomics.index], dtype=str)

# Overlapping kinases/phosphatases knockout
strains = list(set(k_activity.columns).intersection(set(metabolomics.columns)))
k_activity, metabolomics = k_activity[strains], metabolomics[strains]


# ---- Steady-state: predict metabolites FC with kinases
x, y = k_activity[strains].replace(np.NaN, 0.0).T, metabolomics[strains].T
m_predicted = DataFrame({strains[test]: dict(zip(*(y.columns, LinearRegression().fit(x.ix[train], y.ix[train]).predict(x.ix[test])[0]))) for train, test in LeaveOneOut(len(strains))})

# Plot predicted prediction scores
m_score = [(m, spearman(metabolomics.ix[m, strains], m_predicted.ix[m, strains])) for m in m_predicted.index]
m_score = DataFrame([(m, c, p) for m, (c, p) in m_score], columns=['metabolite', 'correlation', 'pvalue'])
m_score = m_score.set_index('metabolite')
m_score['adjpvalue'] = multipletests(m_score['pvalue'], method='fdr_bh')[1]
m_score['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in m_score['adjpvalue']]
m_score['name'] = [m_map[m] if m in m_map else m for m in m_score.index]
m_score = m_score.sort('correlation', ascending=False)
print 'Mean correlation metabolites: ', m_score['correlation'].mean()

s_score = [(s, spearman(metabolomics.ix[metabolomics.index, s], m_predicted.ix[metabolomics.index, s])) for s in strains]
s_score = DataFrame([(m, c, p) for m, (c, p) in s_score], columns=['strain', 'correlation', 'pvalue'])
s_score = s_score.set_index('strain')
s_score['adjpvalue'] = multipletests(s_score['pvalue'], method='fdr_bh')[1]
s_score['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in s_score['adjpvalue']]
s_score['name'] = [acc_name[s] for s in s_score.index]
s_score = s_score.sort('correlation', ascending=False)
print 'Mean correlation samples: ', s_score['correlation'].mean()

plot_df = m_score[m_score['adjpvalue'] < 0.1].index
plot_df = DataFrame([(m_score.ix[m, 'name'], m_predicted.ix[m, s], metabolomics.ix[m, s], m_score.ix[m, 'signif']) for m in plot_df for s in strains])
plot_df.columns = ['metabolite', 'predicted', 'measured', 'signif']

colour_pallete = list(reversed(sns.color_palette('Paired')[:2]))
g = sns.lmplot(x='measured', y='predicted', col='metabolite', hue='signif', data=plot_df, col_wrap=12, palette=colour_pallete, sharex=False, sharey=False, scatter_kws={'s': 80})
plt.savefig('%s/reports/lm_metabolites_steadystate_corr.png' % wd, bbox_inches='tight')
g.set_axis_labels('Measured', 'Predicted').fig.subplots_adjust(wspace=.02)
plt.close('all')
print '[INFO] Plot done!'

plot_df = s_score[s_score['adjpvalue'] < 0.1].index
plot_df = DataFrame([(acc_name[s], m_predicted.ix[m, s], metabolomics.ix[m, s], s_score.ix[s, 'signif']) for s in plot_df for m in m_predicted.index])
plot_df.columns = ['strain', 'predicted', 'measured', 'signif']

colour_pallete = list(reversed(sns.color_palette('Paired')[:2]))
g = sns.lmplot(x='measured', y='predicted', col='strain', hue='signif', data=plot_df, col_wrap=12, palette=colour_pallete, sharex=False, sharey=False, scatter_kws={'s': 80})
plt.savefig('%s/reports/lm_samples_steadystate_corr.png' % wd, bbox_inches='tight')
g.set_axis_labels('Measured', 'Predicted').fig.subplots_adjust(wspace=.02)
plt.close('all')
print '[INFO] Plot done!'


# ---- Dynamic: predict metabolites FC with kinases
# Import kinase activity
k_activity_dyn = read_csv(wd + 'tables/kinase_activity_dynamic.tab', sep='\t', index_col=0)

# Import metabolomics
metabolomics_dyn = read_csv(wd + 'tables/dynamic_metabolomics.tab', sep='\t', index_col=0)
metabolomics_dyn.index = Index([str(i) for i in metabolomics_dyn.index], dtype=str)

metabolomics.index = ['%.2f' % float(i) for i in metabolomics.index]

kinases_ov = list(set(k_activity.index).intersection(k_activity_dyn.index))
metabol_ov = list(set(metabolomics.index).intersection(metabolomics_dyn.index))
conditions = list(metabolomics_dyn.columns)

# Import metabolites map
m_map_dyn = read_csv('%s/files/metabolomics/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map_dyn['mz'] = Index(['%.2f' % i for i in m_map_dyn['mz']], dtype=str)
m_map_dyn = m_map_dyn.groupby('mz')['name'].apply(lambda x: '; '.join(x)).to_dict()

# Fit linear regression model
x_train, y_train = k_activity.ix[kinases_ov, strains].replace(np.NaN, 0.0).T, metabolomics.ix[metabol_ov, strains].T
x_test, y_test = k_activity_dyn.ix[kinases_ov, conditions].replace(np.NaN, 0.0).T, metabolomics_dyn.ix[metabol_ov, conditions].T

m_dyn_predicted = DataFrame(LinearRegression().fit(x_train, y_train).predict(x_test), index=conditions, columns=metabol_ov).T

# Plot predicted prediction scores
m_score_dyn = [(m, spearman(metabolomics_dyn.ix[m, conditions], m_dyn_predicted.ix[m, conditions])) for m in m_dyn_predicted.index]
m_score_dyn = DataFrame([(m, c, p) for m, (c, p) in m_score_dyn], columns=['metabolite', 'correlation', 'pvalue'])
m_score_dyn = m_score_dyn.set_index('metabolite')
m_score_dyn['adjpvalue'] = multipletests(m_score_dyn['pvalue'], method='fdr_bh')[1]
m_score_dyn['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in m_score_dyn['adjpvalue']]
m_score_dyn['name'] = [m for m in m_score_dyn.index]
m_score_dyn = m_score_dyn.sort('correlation', ascending=False)
print 'Mean correlation metabolites: ', m_score_dyn['correlation'].mean()

s_score_dyn = [(s, spearman(metabolomics_dyn.ix[metabolomics.index, s], m_dyn_predicted.ix[metabolomics.index, s])) for s in conditions]
s_score_dyn = DataFrame([(m, c, p) for m, (c, p) in s_score_dyn], columns=['condition', 'correlation', 'pvalue'])
s_score_dyn = s_score_dyn.set_index('condition')
s_score_dyn['adjpvalue'] = multipletests(s_score_dyn['pvalue'], method='fdr_bh')[1]
s_score_dyn['signif'] = ['FDR < 5%' if x < 0.05 else ' FDR >= 5%' for x in s_score_dyn['adjpvalue']]
s_score_dyn['name'] = [i for s in s_score_dyn.index]
s_score_dyn = s_score_dyn.sort('correlation', ascending=False)
print 'Mean correlation samples: ', s_score_dyn['correlation'].mean()

plot_df = DataFrame([(m_score_dyn.ix[m, 'name'], m_dyn_predicted.ix[m, s], metabolomics_dyn.ix[m, s], m_score_dyn.ix[m, 'signif']) for m in m_score_dyn.index for s in conditions])
plot_df.columns = ['metabolite', 'predicted', 'measured', 'signif']

colour_pallete = list(reversed(sns.color_palette('Paired')[:2]))
g = sns.lmplot(x='measured', y='predicted', col='metabolite', hue='signif', data=plot_df, col_wrap=6, palette=colour_pallete, sharex=False, sharey=False, scatter_kws={'s': 80})
plt.savefig('%s/reports/lm_metabolites_dynamic_corr.pdf' % wd, bbox_inches='tight')
g.set_axis_labels('Measured', 'Predicted').fig.subplots_adjust(wspace=.02)
plt.close('all')
print '[INFO] Plot done!'

plot_df = DataFrame([(s, m_dyn_predicted.ix[m, s], metabolomics_dyn.ix[m, s], s_score_dyn.ix[s, 'signif']) for s in conditions for m in m_dyn_predicted.index])
plot_df.columns = ['strain', 'predicted', 'measured', 'signif']

colour_pallete = list(reversed(sns.color_palette('Paired')[:2]))
g = sns.lmplot(x='measured', y='predicted', col='strain', hue='signif', data=plot_df, col_wrap=6, palette=colour_pallete, sharex=False, sharey=False, scatter_kws={'s': 80})
plt.savefig('%s/reports/lm_samples_dynamic_corr.pdf' % wd, bbox_inches='tight')
g.set_axis_labels('Measured', 'Predicted').fig.subplots_adjust(wspace=.02)
plt.close('all')
print '[INFO] Plot done!'

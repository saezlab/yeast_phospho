import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from yeast_phospho import wd
from pandas import DataFrame, read_csv, Index, concat, melt, pivot_table
from scipy.interpolate.interpolate import interp1d


# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)


# --  Process steady-state metabolomics
metabol_df = read_csv(wd + 'data/steady_state_metabolomics.tab', sep='\t').dropna()
metabol_df['m/z'] = ['%.4f' % i for i in metabol_df['m/z']]

counts = {mz: counts for mz, counts in zip(*(np.unique(metabol_df['m/z'], return_counts=True)))}
metabol_df = metabol_df[[counts[i] == 1 for i in metabol_df['m/z']]]
metabol_df = metabol_df.set_index('m/z')

metabol_df = metabol_df[ko_strains]

metabol_df.to_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t')


# --  Process dynamic metabolomics
dyn_metabol = read_csv(wd + 'data/metabol_intensities.tab', sep='\t').dropna()
dyn_metabol['m/z'] = ['%.4f' % i for i in dyn_metabol['m/z']]

counts = {mz: counts for mz, counts in zip(*(np.unique(dyn_metabol['m/z'], return_counts=True)))}
dyn_metabol = dyn_metabol[[counts[i] == 1 for i in dyn_metabol['m/z']]]
dyn_metabol = dyn_metabol.set_index('m/z')

# Import samplesheet
ss = read_csv(wd + 'data/metabol_samplesheet.tab', sep='\t', index_col=0)
ss['time_value'] = [float(i.replace('min', '')) for i in ss['time']]

conditions, p_timepoints, dyn_metabol_df, cv = ['N_downshift', 'N_upshift', 'Rapamycin'], [-10, 5, 9, 15, 25, 44, 79], DataFrame(), DataFrame()
for condition in conditions:
    ss_cond = ss[ss['condition'] == condition]

    # Coefficient of variation
    cv_df = DataFrame({m: {t: dyn_metabol.ix[m, ss_cond[ss_cond['time'] == t].index].std() / dyn_metabol.ix[m, ss_cond[ss_cond['time'] == t].index].mean() for t in set(ss_cond['time'])} for m in dyn_metabol.index})
    cv = concat([cv, cv_df])

    # Average metabolite replicates
    m_df_cond = DataFrame({m: {t: dyn_metabol.ix[m, ss_cond[ss_cond['time'] == t].index].mean() for t in set(ss_cond['time'])} for m in dyn_metabol.index})
    m_df_cond.index = Index([float(i.replace('min', '')) for i in m_df_cond.index])
    m_df_cond = m_df_cond.ix[np.sort(m_df_cond.index)].T

    # Interpolate phospho time-points
    m_df_cond = DataFrame({m: interp1d(m_df_cond.ix[m].index, m_df_cond.ix[m].values)(p_timepoints) for m in dyn_metabol.index}, index=['%s_%dmin' % (condition, i) for i in p_timepoints]).T

    # Calculate log2 fold-change
    tx, t0 = ['%s_%dmin' % (condition, i) for i in p_timepoints if i != -10], '%s_-10min' % condition
    m_df_cond = np.log2(m_df_cond[tx].div(m_df_cond[t0], axis=0))

    # Append to existing data-set
    dyn_metabol_df = dyn_metabol_df.join(m_df_cond, how='outer')

print '[INFO] Done'

# Export processed data-set
dyn_metabol_df.to_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t')
cv.T.to_csv('%s/tables/dynamic_metabolomics_cv.csv' % wd)
print '[INFO] Metabolomics preprocessing done'

# Replicates correlation
dyn_metabol = dyn_metabol.drop(['metabolite', 'metabolite_charge'], axis=1)

condition = 'N_upshift'

tec_rep = {
    'BY003_02': 'BY003_01', 'BY003_03': 'BY003_01', 'BY003_04': 'BY003_01',
    'BY008_02': 'BY008_01', 'BY008_03': 'BY008_01', 'BY008_04': 'BY008_01',
    'BY012_02': 'BY012_01', 'BY012_03': 'BY012_01', 'BY012_04': 'BY012_01',

    'BY009_03': 'BY009_01', 'BY009_04': 'BY009_01',
    'BY010_02': 'BY010_01', 'BY010_03': 'BY010_01', 'BY010_04': 'BY010_01',
    'BY013_02': 'BY013_01', 'BY013_03': 'BY013_01', 'BY013_04': 'BY013_01',

    'BY005_02': 'BY005_01', 'BY005_03': 'BY005_01', 'BY005_04': 'BY005_01',
    'BY006_02': 'BY006_01', 'BY006_03': 'BY006_01', 'BY006_04': 'BY006_01',
    'BY007_02': 'BY007_01', 'BY007_03': 'BY007_01', 'BY007_04': 'BY007_01'
}


dyn_metabol.columns = [tec_rep[c] if c in tec_rep else c for c in dyn_metabol]
dyn_metabol = DataFrame({c: dyn_metabol[c].mean(1) if len(dyn_metabol[c].shape) > 1 else dyn_metabol[c] for c in dyn_metabol})

for time in [.1, .25, .5, .75, 1., 2., 7., 10., 20., 45., 60., 120.]:
    t0 = set(ss[(ss['condition'] == condition) & (ss['time_value'] < 0)].index).intersection(dyn_metabol)
    tx = set(ss[(ss['condition'] == condition) & (ss['time_value'] == time)].index).intersection(dyn_metabol)

    logfc_m = DataFrame({'%svs%s' % (c0, cx): np.log2(dyn_metabol[cx]).subtract(np.log2(dyn_metabol[c0])) for c0, cx in it.product(t0, tx)})

    plot_df = logfc_m.corr()
    plot_df = [(c[0], c[1], plot_df.ix[c[0], c[1]]) for c in it.product(plot_df, plot_df) if len(set(c[0].split('vs')).intersection(c[1].split('vs'))) == 0]
    plot_df = DataFrame(plot_df, columns=['cond1', 'cond2', 'spearman'])
    plot_df = pivot_table(plot_df, index='cond1', columns='cond2', values='spearman', fill_value=np.nan)

    plt.figure(figsize=(30, 30))
    sns.clustermap(plot_df.replace(np.nan, 0), annot=True, fmt='.2f', linewidths=.5, cmap='YlGnBu', square=True)
    plt.title('%s, %.2f min (median spearman: %.2f)' % (condition, time, plot_df.median().median()))
    plt.savefig('%s/reports/replicates_correlation_%s_%.2fmin.pdf' % (wd, condition, time), bbox_inches='tight')
    plt.close('all')
print '[INFO] Done'

import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv, Index
from yeast_phospho.utilities import regress_out
from scipy.interpolate.interpolate import interp1d


# ---- Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)['relative_growth']


# ----  Process steady-state metabolomics
metabol_df = read_csv(wd + 'data/steady_state_metabolomics.tab', sep='\t')
metabol_df['m/z'] = ['%.2f' % i for i in metabol_df['m/z']]
print '[INFO] [METABOLOMICS]: ', metabol_df.shape

counts = {mz: counts for mz, counts in zip(*(np.unique(metabol_df['m/z'], return_counts=True)))}
metabol_df = metabol_df[[counts[i] == 1 for i in metabol_df['m/z']]]
metabol_df = metabol_df.set_index('m/z')
print '[INFO] [METABOLOMICS] remove duplicated m/z: ', metabol_df.shape

metabol_df = metabol_df.dropna()
print '[INFO] [METABOLOMICS] drop NaN: ', metabol_df.shape

metabol_df_file = wd + 'tables/metabolomics_steady_state_growth_rate.tab'
metabol_df.to_csv(metabol_df_file, sep='\t')
print '[INFO] [METABOLOMICS] Exported to: %s' % metabol_df_file

metabol_df = DataFrame({i: regress_out(growth, metabol_df.ix[i, growth.index]) for i in metabol_df.index}).T
print '[INFO] Growth regressed out from the metabolites: ', metabol_df.shape

metabol_df_file = wd + 'tables/metabolomics_steady_state.tab'
metabol_df.to_csv(metabol_df_file, sep='\t')
print '[INFO] [METABOLOMICS] Exported to: %s' % metabol_df_file


# ----  Process dynamic metabolomics
dyn_metabol = read_csv(wd + 'data/metabol_intensities.tab', sep='\t').dropna()
dyn_metabol['m/z'] = ['%.2f' % i for i in dyn_metabol['m/z']]
print '[INFO] [METABOLOMICS]: ', dyn_metabol.shape

counts = {mz: counts for mz, counts in zip(*(np.unique(dyn_metabol['m/z'], return_counts=True)))}
dyn_metabol = dyn_metabol[[counts[i] == 1 for i in dyn_metabol['m/z']]]
dyn_metabol = dyn_metabol.set_index('m/z')
print '[INFO] [METABOLOMICS] remove duplicated m/z: ', dyn_metabol.shape

# Import samplesheet
ss = read_csv(wd + 'data/metabol_samplesheet.tab', sep='\t', index_col=0)
ss['time_value'] = [float(i.replace('min', '')) for i in ss['time']]

conditions, p_timepoints, dyn_metabol_df = ['N_downshift', 'N_upshift', 'Rapamycin'], [-10, 5, 9, 15, 25, 44, 79], DataFrame()
for condition in conditions:
    ss_cond = ss[ss['condition'] == condition]

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

# Export processed data-set
dyn_metabol_df_file = wd + 'tables/metabolomics_dynamic.tab'
dyn_metabol_df.to_csv(dyn_metabol_df_file, sep='\t')
print '[INFO] [METABOLOMICS] Exported to: %s' % dyn_metabol_df_file

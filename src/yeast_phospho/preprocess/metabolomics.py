import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv, Index, concat
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

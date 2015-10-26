import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv, pivot_table
from scipy.interpolate.interpolate import interp1d


# ---- Process: Steady-state gene-expression
# Import conversion table
name2id = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv('%s/files/orf_name_dataframe.tab' % wd, sep='\t', index_col=0).to_dict()['name']

transcriptomics = read_csv('%s/data/Kemmeren_2014_zscores_parsed_filtered.tab' % wd, sep='\t', header=False)
transcriptomics['tf'] = [name2id[i] if i in name2id else id2name[i] for i in transcriptomics['tf']]
transcriptomics = pivot_table(transcriptomics, values='value', index='target', columns='tf')
print '[INFO] Gene-expression imported!'

# Export processed data-set
transcriptomics_file = wd + 'tables/transcriptomics_steady_state.tab'
transcriptomics.to_csv(transcriptomics_file, sep='\t')
print '[INFO] [TRANSCRIPTOMICS] Exported to: %s' % transcriptomics_file


# ---- Process dynamic transcriptomics
dyn_transcriptomics = read_csv('%s/data/dynamic_transcriptomics.tab' % wd, sep='\t')
dyn_transcriptomics = dyn_transcriptomics[[(dyn_transcriptomics['YORF'] == i).sum() == 1 for i in dyn_transcriptomics['YORF']]]
dyn_transcriptomics = dyn_transcriptomics.groupby('YORF').first()

# Import samplesheet
ss = read_csv('%s/data/dynamic_transcriptomics_samplesheet.tab' % wd, sep='\t', index_col=0)
ss['time_value'] = [float(i.replace('min', '')) for i in ss['time']]

conditions, p_timepoints, dyn_trans_df = ['N_downshift', 'N_upshift', 'Rapamycin'], [-10, 5, 9, 15, 25, 44, 79], DataFrame()
for condition in conditions:
    ss_cond = ss[ss['condition'] == condition]

    # Sub-set data-set
    t_df_cond = dyn_transcriptomics[ss_cond.index]
    t_df_cond.columns = [ss_cond.ix[c, 'time_value'] for c in t_df_cond]

    # Interpolate phospho time-points
    t_df_cond = DataFrame({t: interp1d(t_df_cond.ix[t].index, t_df_cond.ix[t].values)(p_timepoints) for t in t_df_cond.index}, index=['%s_%dmin' % (condition, i) for i in p_timepoints]).T

    # Calculate log2 fold-change
    tx, t0 = ['%s_%dmin' % (condition, i) for i in p_timepoints if i != -10], '%s_-10min' % condition
    t_df_cond = np.log2(t_df_cond[tx].div(t_df_cond[t0], axis=0))

    # Append to existing data-set
    dyn_trans_df = dyn_trans_df.join(t_df_cond, how='outer')

# Export processed data-set
dyn_trans_df_file = wd + 'tables/transcriptomics_dynamic.tab'
dyn_trans_df.to_csv(dyn_trans_df_file, sep='\t')
print '[INFO] [TRANSCRIPTOMICS] Exported to: %s' % dyn_trans_df_file

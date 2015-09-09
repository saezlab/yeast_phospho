import re
import numpy as np
from yeast_phospho import wd
from sklearn.linear_model import LinearRegression
from pandas import DataFrame, read_csv, Index, pivot_table
from scipy.interpolate.interpolate import interp1d


def get_site(protein, peptide):
    pep_start = protein.find(re.sub('\[[0-9]*\.?[0-9]*\]', '', peptide))
    pep_site_strat = peptide.find('[')
    site_pos = pep_start + pep_site_strat
    return protein[site_pos - 1] + str(site_pos)


def get_multiple_site(protein, peptide):
    n_sites = len(re.findall('\[[0-9]*\.?[0-9]*\]', peptide))
    return [get_site(protein, peptide if i == 0 else re.sub('\[[0-9]*\.?[0-9]*\]', '', peptide, i)) for i in xrange(n_sites)]


# Import Phosphogrid network
network = read_csv(wd + 'files/PhosphoGrid.txt', sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SEQUENCE']]
protein_seq = network.groupby('ORF_NAME').first()['SEQUENCE'].to_dict()
print '[INFO] [PHOSPHOGRID] ', network.shape

# Import growth rates
growth = read_csv(wd + 'files/strain_relative_growth_rate.txt', sep='\t', index_col=0)['relative_growth']


# ----  Process steady-state phosphoproteomics
phospho_df = read_csv(wd + 'data/steady_state_phosphoproteomics.tab', sep='\t')
phospho_df = phospho_df.pivot_table(values='logFC', index=['peptide', 'target'], columns='regulator', aggfunc=np.median)
print '[INFO] [PHOSPHOPROTEOMICS] : ', phospho_df.shape

# Define lists of strains
strains = list(set(phospho_df.columns).intersection(growth.index))

# Filter ambigous peptides
phospho_df = phospho_df[[len(i[0].split(',')) == 1 for i in phospho_df.index]]

# Remove K and R aminoacids from the peptide head
phospho_df.index = phospho_df.index.set_levels([re.split('^[K|R]\.', x)[1] for x in phospho_df.index.levels[0]], 'peptide')

# Match peptide sequences to protein sequence and calculate peptide phosphorylation site
pep_match = {peptide: set(network.loc[network['ORF_NAME'] == target, 'SEQUENCE']) for peptide, target in phospho_df.index}
pep_site = {peptide: target + '_' + '_'.join(get_multiple_site(list(pep_match[peptide])[0].upper(), peptide)) for peptide, target in phospho_df.index if len(pep_match[peptide]) == 1}

# Merge phosphosites with median
phospho_df['site'] = [pep_site[peptide] if peptide in pep_site else np.NaN for peptide, target in phospho_df.index]
phospho_df = phospho_df.groupby('site').median()
print '[INFO] [PHOSPHOPROTEOMICS] (merge phosphosites, i.e median): ', phospho_df.shape

# Export processed data-set
phospho_df_file = wd + 'tables/pproteomics_steady_state.tab'
phospho_df.to_csv(phospho_df_file, sep='\t')
print '[INFO] [PHOSPHOPROTEOMICS] Exported to: %s' % phospho_df_file


# ----  Process steady-state metabolomics
metabol_df = read_csv(wd + 'data/steady_state_metabolomics.tab', sep='\t')
metabol_df['m/z'] = ['%.2f' % i for i in metabol_df['m/z']]
metabol_df = metabol_df.groupby('m/z').median()
print '[INFO] [METABOLOMICS]: ', metabol_df.shape

metabol_df = metabol_df.dropna()[strains]
print '[INFO] [METABOLOMICS] drop NaN: ', metabol_df.shape

metabol_df_file = wd + 'tables/metabolomics_steady_state_growth_rate.tab'
metabol_df.to_csv(metabol_df_file, sep='\t')
print '[INFO] [METABOLOMICS] Exported to: %s' % metabol_df_file


def regress_out_growth_metabolite(metabolite):
    x, y = growth.ix[strains].values, metabol_df.ix[metabolite, strains].values

    lm = LinearRegression().fit(np.mat(x).T, y)

    y_ = y - lm.coef_[0] * x - lm.intercept_

    return dict(zip(np.array(strains), y_))

metabol_df = DataFrame({metabolite: regress_out_growth_metabolite(metabolite) for metabolite in metabol_df.index}).T.dropna(axis=0, how='all')
print '[INFO] Growth regressed out from the metabolites: ', metabol_df.shape

# Export processed data-set
metabol_df_file = wd + 'tables/metabolomics_steady_state.tab'
metabol_df.to_csv(metabol_df_file, sep='\t')
print '[INFO] [METABOLOMICS] Exported to: %s' % metabol_df_file


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


# ---- Process dynamic phosphoproteomics
phospho_dyn_df = read_csv(wd + 'data/dynamic_phosphoproteomics.tab', sep='\t')
phospho_dyn_map = read_csv(wd + 'data/dynamic_peptides_map.tab', sep='\t')

dyn_phospho_df, conditions, timepoints = DataFrame(), ['N_downshift', 'N_upshift', 'Rapamycin'], ['5min', '9min', '15min', '25min', '44min', '79min']

for condition in conditions:
    phospho_dyn_df_cond = phospho_dyn_df[phospho_dyn_df['condition'] == condition]
    phospho_dyn_map_cond = phospho_dyn_map[phospho_dyn_map['condition'] == condition]

    phospho_dyn_df_cond = phospho_dyn_df_cond[[1 == len(phospho_dyn_map_cond.loc[phospho_dyn_map_cond['peptide'] == i, 'phosphopeptide']) for i in phospho_dyn_df_cond['peptide']]]
    phospho_dyn_df_cond = phospho_dyn_df_cond[[phospho_dyn_map_cond.loc[phospho_dyn_map_cond['peptide'] == i, ['site_%d' % p for p in [1, 2, 3, 4]]].count(1).values[0] == 1 for i in phospho_dyn_df_cond['peptide']]]
    phospho_dyn_df_cond = phospho_dyn_df_cond[[p in network['ORF_NAME'].values for p in phospho_dyn_df_cond['protein']]]
    phospho_dyn_df_cond = phospho_dyn_df_cond[[len(p.split('/')) == 1 for p in phospho_dyn_df_cond['protein']]]
    phospho_dyn_df_cond = phospho_dyn_df_cond.set_index('peptide')

    protein_pos = [tuple(phospho_dyn_map_cond.loc[phospho_dyn_map_cond['peptide'] == i, ['protein', 'site_1']].values[0]) for i in phospho_dyn_df_cond.index]
    phospho_dyn_df_cond['site'] = ['%s_%s%d' % (protein, protein_seq[protein][int(pos - 1)].upper(), int(pos)) for protein, pos in protein_pos]

    phospho_dyn_df_cond = phospho_dyn_df_cond.groupby('site').mean()[timepoints]

    phospho_dyn_df_cond.columns = ['%s_%s' % (condition, c) for c in phospho_dyn_df_cond.columns]

    dyn_phospho_df = phospho_dyn_df_cond.join(dyn_phospho_df, how='outer')

    print '[INFO] %s' % condition

# Export processed data-set
dyn_phospho_df_file = '%s/tables/pproteomics_dynamic.tab' % wd
dyn_phospho_df.to_csv(dyn_phospho_df_file, sep='\t')
print '[INFO] [PHOSPHOPROTEOMICS] Exported to: %s' % dyn_phospho_df_file


# ----  Process dynamic metabolomics
dyn_metabol = read_csv(wd + 'data/metabol_intensities.tab', sep='\t')
dyn_metabol.index = Index(dyn_metabol['m/z'], dtype=np.str)
dyn_metabol = dyn_metabol.groupby(level=0).first()
print '[INFO] [METABOLOMICS]: ', dyn_metabol.shape

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

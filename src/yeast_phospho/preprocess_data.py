import re
import numpy as np
from pandas import read_csv, Index, DataFrame, concat


def get_site(protein, peptide):
    pep_start = protein.find(re.sub('\[.+\]', '', peptide))
    pep_site_strat = peptide.find('[')
    site_pos = pep_start + pep_site_strat
    return protein[site_pos - 1] + str(site_pos)

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import Phosphogrid network
network = read_csv(wd + 'files/phosphosites.txt', sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SEQUENCE']]
protein_seq = network.groupby('ORF_NAME').first()['SEQUENCE'].to_dict()
print '[INFO] [PHOSPHOGRID] ', network.shape


####  Process steady-state phosphoproteomics
phospho_df = read_csv(wd + 'data/steady_state_phosphoproteomics.tab', sep='\t')
phospho_df = phospho_df.pivot_table(values='logFC', index=['peptide', 'target'], columns='regulator', aggfunc=np.median)
print '[INFO] [PHOSPHOPROTEOMICS] merge repeated phosphopeptides, i.e. median : ', phospho_df.shape

# Consider phoshopeptides with only one phosphosite
phospho_df = phospho_df.loc[[len(re.findall('\[[0-9]*\.?[0-9]*\]', peptide)) == 1 for peptide in phospho_df.index.levels[0]]]
print '[INFO] [PHOSPHOPROTEOMICS] (filtered phosphopetides with multiple phosphosites): ', phospho_df.shape

# Remove K and R aminoacids from the peptide head
phospho_df.index = phospho_df.index.set_levels([re.split('^[K|R]\.', x)[1] for x in phospho_df.index.levels[0]], 'peptide')

# Match peptide sequences to protein sequence and calculate peptide phosphorylation site
pep_match = {peptide: set(network.loc[network['ORF_NAME'] == target, 'SEQUENCE']) for (peptide, target), r in phospho_df.iterrows()}
pep_site = {peptide: target + '_' + get_site(list(pep_match[peptide])[0].upper(), peptide) for (peptide, target), r in phospho_df.iterrows() if len(pep_match[peptide]) == 1}

# Merge phosphosites with median
phospho_df['site'] = [pep_site[peptide] if peptide in pep_site else np.NaN for (peptide, target), r in phospho_df.iterrows()]
phospho_df = phospho_df.groupby('site').median()
print '[INFO] [PHOSPHOPROTEOMICS] (merge phosphosites, i.e median): ', phospho_df.shape

# Export processed data-set
phospho_df_file = wd + 'tables/steady_state_phosphoproteomics.tab'
phospho_df.to_csv(phospho_df_file, sep='\t')
print '[INFO] [PHOSPHOPROTEOMICS] Exported to: %s' % phospho_df_file


####  Process steady-state metabolomics
metabol_df = read_csv(wd + 'data/steady_state_metabolomics.tab', sep='\t')
metabol_df.index = Index(metabol_df['m/z'], dtype=np.str)
print '[INFO] [METABOLOMICS]: ', metabol_df.shape

metabol_df = metabol_df.drop('m/z', 1).dropna()
print '[INFO] [METABOLOMICS] drop NaN: ', metabol_df.shape

fc_thres, n_fc_thres = 1.0, 1
metabol_df = metabol_df[(metabol_df.abs() > fc_thres).sum(1) > n_fc_thres]
print '[INFO] [METABOLOMICS] drop metabolites with less than %d abs FC higher than %.2f : ' % (n_fc_thres, fc_thres), metabol_df.shape

# Export processed data-set
metabol_df_file = wd + 'tables/steady_state_metabolomics.tab'
metabol_df.to_csv(metabol_df_file, sep='\t')
print '[INFO] [METABOLOMICS] Exported to: %s' % metabol_df_file


####  Process dynamic phosphoproteomics
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
dyn_phospho_df_file = wd + 'tables/dynamic_phosphoproteomics.tab'
dyn_phospho_df.to_csv(dyn_phospho_df_file, sep='\t')
print '[INFO] [PHOSPHOPROTEOMICS] Exported to: %s' % dyn_phospho_df_file


####  Process dynamic metabolomics
dyn_metabol = read_csv(wd + 'data/metabol_intensities.tab', sep='\t')
dyn_metabol.index = Index(dyn_metabol['m/z'], dtype=np.str)
print '[INFO] [METABOLOMICS]: ', dyn_metabol.shape

ss = read_csv(wd + 'data/metabol_samplesheet.tab', sep='\t', index_col=0)

timepoints = ['5min', '15min']

dyn_metabol_df = [np.log2(dyn_metabol[ss.query('time == "%s" & condition == "%s"' % (t, c)).index].mean(1) / dyn_metabol[ss.query('time == "%s" & condition == "%s"' % ('-10min', c)).index].mean(1)) for c in conditions for t in timepoints]
dyn_metabol_df = concat(dyn_metabol_df, axis=1)
dyn_metabol_df.columns = Index(['%s_%s' % (c, t) for c in conditions for t in timepoints], dtype=str, name='m/z')

# Export processed data-set
dyn_metabol_df_file = wd + 'tables/dynamic_metabolomics.tab'
dyn_metabol_df.to_csv(dyn_metabol_df_file, sep='\t', )
print '[INFO] [METABOLOMICS] Exported to: %s' % dyn_metabol_df_file
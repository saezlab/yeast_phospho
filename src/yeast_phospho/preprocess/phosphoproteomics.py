import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from pandas import DataFrame, Series, read_csv
from yeast_phospho.utilities import get_protein_sequence, get_multiple_site
from scipy.interpolate.interpolate import interp1d


# ---- Get protein sequence
protein_seq = get_protein_sequence()
protein_seq = {k: protein_seq[k] for k in protein_seq if len(protein_seq[k]) > 1}


# ----  Process steady-state phosphoproteomics
phospho_df = read_csv(wd + 'data/steady_state_phosphoproteomics.tab', sep='\t')
phospho_df = phospho_df.pivot_table(values='logFC', index=['peptide', 'target'], columns='regulator', aggfunc=np.median)
print '[INFO] [PHOSPHOPROTEOMICS] : ', phospho_df.shape

# Filter ambigous peptides
phospho_df = phospho_df[[len(i[0].split(',')) == 1 for i in phospho_df.index]]
print '[INFO] [PHOSPHOPROTEOMICS] Filter ambigous peptides: ', phospho_df.shape

# Remove K and R aminoacids from the peptide head
phospho_df.index = phospho_df.index.set_levels([re.split('^[K|R]\.', x)[1] for x in phospho_df.index.levels[0]], 'peptide')

# Match peptide sequences to protein sequence and calculate peptide phosphorylation site
pep_site = {peptide: target + '_' + '_'.join(get_multiple_site(protein_seq[target].upper(), peptide)) for peptide, target in phospho_df.index if target in protein_seq}
print '[INFO] [PHOSPHOPROTEOMICS] Peptides mapped to proteins'

# Merge phosphosites with median
phospho_df['site'] = [pep_site[peptide] if peptide in pep_site else np.NaN for peptide, target in phospho_df.index]
phospho_df = phospho_df.groupby('site').median()
print '[INFO] [PHOSPHOPROTEOMICS] (merge phosphosites, i.e median): ', phospho_df.shape

# Export processed data-set
phospho_df_file = wd + 'tables/pproteomics_steady_state.tab'
phospho_df.to_csv(phospho_df_file, sep='\t')
print '[INFO] [PHOSPHOPROTEOMICS] Exported to: %s' % phospho_df_file


# ---- Process dynamic phosphoproteomics
phospho_dyn_df = read_csv(wd + 'data/dynamic_phosphoproteomics.tab', sep='\t')
phospho_dyn_map = read_csv(wd + 'data/dynamic_peptides_map.tab', sep='\t')

dyn_phospho_df, conditions, timepoints = DataFrame(), ['N_downshift', 'N_upshift', 'Rapamycin'], ['5min', '9min', '15min', '25min', '44min', '79min']

for condition in conditions:
    phospho_dyn_df_cond = phospho_dyn_df[phospho_dyn_df['condition'] == condition]
    phospho_dyn_map_cond = phospho_dyn_map[phospho_dyn_map['condition'] == condition]

    phospho_dyn_df_cond = phospho_dyn_df_cond[[1 == len(phospho_dyn_map_cond.loc[phospho_dyn_map_cond['peptide'] == i, 'phosphopeptide']) for i in phospho_dyn_df_cond['peptide']]]
    phospho_dyn_df_cond = phospho_dyn_df_cond[[phospho_dyn_map_cond.loc[phospho_dyn_map_cond['peptide'] == i, ['site_%d' % p for p in [1, 2, 3, 4]]].count(1).values[0] == 1 for i in phospho_dyn_df_cond['peptide']]]
    phospho_dyn_df_cond = phospho_dyn_df_cond[[p in protein_seq for p in phospho_dyn_df_cond['protein']]]
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

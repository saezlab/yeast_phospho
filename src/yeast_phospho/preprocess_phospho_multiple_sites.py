import re
import numpy as np
from yeast_phospho import wd
from pandas import read_csv, Index, DataFrame, concat
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
network = read_csv(wd + 'files/phosphosites.txt', sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SEQUENCE']]
protein_seq = network.groupby('ORF_NAME').first()['SEQUENCE'].to_dict()
print '[INFO] [PHOSPHOGRID] ', network.shape


# ----  Process steady-state phosphoproteomics
phospho_df = read_csv(wd + 'data/steady_state_phosphoproteomics.tab', sep='\t')
phospho_df = phospho_df.pivot_table(values='logFC', index=['peptide', 'target'], columns='regulator', aggfunc=np.median)
print '[INFO] [PHOSPHOPROTEOMICS] : ', phospho_df.shape

# Filter ambigous peptides
phospho_df = phospho_df[[len(i[0].split(',')) == 1 for i in phospho_df.index]]

# Remove K and R aminoacids from the peptide head
phospho_df.index = phospho_df.index.set_levels([re.split('^[K|R]\.', x)[1] for x in phospho_df.index.levels[0]], 'peptide')

# Match peptide sequences to protein sequence and calculate peptide phosphorylation site
pep_match = {peptide: set(network.loc[network['ORF_NAME'] == target, 'SEQUENCE']) for (peptide, target), r in phospho_df.iterrows()}
pep_site = {peptide: target + '_' + '_'.join(get_multiple_site(list(pep_match[peptide])[0].upper(), peptide)) for (peptide, target), r in phospho_df.iterrows() if len(pep_match[peptide]) == 1}

# Merge phosphosites with median
phospho_df['site'] = [pep_site[peptide] if peptide in pep_site else np.NaN for (peptide, target), r in phospho_df.iterrows()]
phospho_df = phospho_df.groupby('site').median()
print '[INFO] [PHOSPHOPROTEOMICS] (merge phosphosites, i.e median): ', phospho_df.shape

# Export processed data-set
phospho_df_file = wd + 'tables/steady_state_phosphoproteomics_multiple_psites.tab'
phospho_df.to_csv(phospho_df_file, sep='\t')
print '[INFO] [PHOSPHOPROTEOMICS] Exported to: %s' % phospho_df_file

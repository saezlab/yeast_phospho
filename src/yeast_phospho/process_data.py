import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from yeast_phospho import wd
from yeast_phospho.utils import pearson
from sklearn.linear_model import LinearRegression
from pandas import DataFrame, read_csv, Index


def get_site(protein, peptide):
    pep_start = protein.find(re.sub('\[[0-9]*\.?[0-9]*\]', '', peptide))
    pep_site_strat = peptide.find('[')
    site_pos = pep_start + pep_site_strat
    return protein[site_pos - 1] + str(site_pos)


def get_multiple_site(protein, peptide):
    n_sites = len(re.findall('\[[0-9]*\.?[0-9]*\]', peptide))
    return [get_site(protein, peptide if i == 0 else re.sub('\[[0-9]*\.?[0-9]*\]', '', peptide, i)) for i in xrange(n_sites)]


sns.set_style('ticks')

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
pep_match = {peptide: set(network.loc[network['ORF_NAME'] == target, 'SEQUENCE']) for (peptide, target), r in phospho_df.iterrows()}
pep_site = {peptide: target + '_' + '_'.join(get_multiple_site(list(pep_match[peptide])[0].upper(), peptide)) for (peptide, target), r in phospho_df.iterrows() if len(pep_match[peptide]) == 1}

# Merge phosphosites with median
phospho_df['site'] = [pep_site[peptide] if peptide in pep_site else np.NaN for (peptide, target), r in phospho_df.iterrows()]
phospho_df = phospho_df.groupby('site').median()
print '[INFO] [PHOSPHOPROTEOMICS] (merge phosphosites, i.e median): ', phospho_df.shape

# Regress out growth
psites_growth_cor = [pearson(phospho_df.ix[i, strains].values, growth.ix[strains].values)[0] for i in phospho_df.index if phospho_df.ix[i, strains].count() > 3]
plt.hist(psites_growth_cor, lw=0, bins=30)
sns.despine(offset=10, trim=True)
plt.title('pearson(p-site, growth)')
plt.xlabel('pearson')
plt.ylabel('counts')
plt.savefig(wd + 'reports/p-sites_growth_correlation_hist.pdf', bbox_inches='tight')
plt.close('all')


def regress_out_growth(site):
    x, y = growth.ix[strains].values, phospho_df.ix[site, strains].values

    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))

    if sum(mask) > 3:
        x, y = x[mask], y[mask]

        effect_size = LinearRegression().fit(np.mat(x).T, y).coef_[0]

        y_ = y - effect_size * x

        return dict(zip(np.array(strains)[mask], y_))

    else:
        return {}

phospho_df_ = DataFrame({site: regress_out_growth(site) for site in phospho_df.index}).T.dropna(axis=0, how='all')

psites_growth_cor = [pearson(phospho_df_.ix[i, strains].values, growth.ix[strains].values)[0] for i in phospho_df_.index if phospho_df_.ix[i, strains].count() > 3]
plt.hist(psites_growth_cor, lw=0, bins=30)
sns.despine(offset=10, trim=True)
plt.title('pearson(p-site, growth)')
plt.xlabel('pearson')
plt.ylabel('counts')
plt.savefig(wd + 'reports/p-sites_growth_correlation_growth_out_hist.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Growth regressed out from the p-sites: ', phospho_df_.shape

# Round values
phospho_df_ = np.round(phospho_df_, decimals=6)

# Export processed data-set
phospho_df_file = wd + 'tables/pproteomics_steady_state.tab'
phospho_df_.to_csv(phospho_df_file, sep='\t')
print '[INFO] [PHOSPHOPROTEOMICS] Exported to: %s' % phospho_df_file


# ----  Process steady-state metabolomics
metabol_df = read_csv(wd + 'data/steady_state_metabolomics.tab', sep='\t')
metabol_df.index = Index(metabol_df['m/z'], dtype=np.str)
print '[INFO] [METABOLOMICS]: ', metabol_df.shape

metabol_df = metabol_df.drop('m/z', 1).dropna()[strains]
print '[INFO] [METABOLOMICS] drop NaN: ', metabol_df.shape

fc_thres, n_fc_thres = 1.0, 1
metabol_df = metabol_df[(metabol_df.abs() > fc_thres).sum(1) > n_fc_thres]
print '[INFO] [METABOLOMICS] drop metabolites with less than %d abs FC higher than %.2f : ' % (n_fc_thres, fc_thres), metabol_df.shape

metabolites_growth_cor = [pearson(metabol_df.ix[i, strains].values, growth.ix[strains].values)[0] for i in metabol_df.index]
plt.hist(metabolites_growth_cor, lw=0, bins=30)
sns.despine(offset=10, trim=True)
plt.title('pearson(metabolite, growth)')
plt.xlabel('pearson')
plt.ylabel('counts')
plt.savefig(wd + 'reports/metabolites_growth_correlation_hist.pdf', bbox_inches='tight')
plt.close('all')


def regress_out_growth_metabolite(metabolite):
    x, y = growth.ix[strains].values, metabol_df.ix[metabolite, strains].values

    effect_size = LinearRegression().fit(np.mat(x).T, y).coef_[0]

    y_ = y - effect_size * x

    return dict(zip(np.array(strains), y_))

metabol_df_ = DataFrame({metabolite: regress_out_growth_metabolite(metabolite) for metabolite in metabol_df.index}).T.dropna(axis=0, how='all')

metabolites_growth_cor = [pearson(metabol_df_.ix[i, strains].values, growth.ix[strains].values)[0] for i in metabol_df_.index if metabol_df_.ix[i, strains].count() > 3]
plt.hist(metabolites_growth_cor, lw=0, bins=30)
sns.despine(offset=10, trim=True)
plt.title('pearson(metabolite, growth)')
plt.xlabel('pearson')
plt.ylabel('counts')
plt.savefig(wd + 'reports/metabolites_growth_correlation_growth_out_hist.pdf', bbox_inches='tight')
plt.close('all')
print '[INFO] Growth regressed out from the metabolites: ', metabol_df_.shape

# Export processed data-set
metabol_df_file = wd + 'tables/metabolomics_steady_state.tab'
metabol_df_.to_csv(metabol_df_file, sep='\t')
print '[INFO] [METABOLOMICS] Exported to: %s' % metabol_df_file
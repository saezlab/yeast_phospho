import numpy as np
from scipy.stats import pearsonr
from pandas import DataFrame, Series, read_csv


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, x[mask], y[mask], np.sum(mask)


wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import phospho log2 FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')

# Import kinases enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment.tab', sep='\t', index_col=0)
kinase_ztest_df = read_csv(wd + 'tables/kinase_enrichment.tab', sep='\t', index_col=0)

# Regulatory-sites
reg_motif, kinases, strains = ['Activates The Protein Function', 'Inhibits The Protein Function'], list(kinase_ztest_df.index), list(kinase_ztest_df.columns)
regulatory_sites = read_csv(wd + 'files/phosphosites.txt', sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'SITE_FUNCTIONS']]
regulatory_sites = regulatory_sites[regulatory_sites['SITE_FUNCTIONS'] != '-']
regulatory_sites = regulatory_sites[[k in kinases for k in regulatory_sites['ORF_NAME']]]
regulatory_sites['SITE'] = regulatory_sites['ORF_NAME'] + '_' + regulatory_sites['PHOSPHO_SITE']
regulatory_sites = regulatory_sites[[k in phospho_df.index for k in regulatory_sites['SITE']]]
regulatory_sites = regulatory_sites.set_index('SITE')

gsea_reg_sites_cor = [(k, pearson(phospho_df.ix[k, strains].values, kinase_df.ix[k.split('_')[0], strains].values)) for k in regulatory_sites.index]
gsea_reg_sites_cor = [(k, c, p, x_values[i], y_values[i], count) for (k, (c, p, x_values, y_values, count)) in gsea_reg_sites_cor if np.isfinite(c) for i in range(len(x_values))]
gsea_reg_sites_cor = DataFrame(gsea_reg_sites_cor, columns=['kinase', 'correlation', 'pvalue', 'phospho', 'enrichment', 'count'])
gsea_reg_sites_cor['method'] = 'GSEA'
gsea_reg_sites_cor['function'] = [regulatory_sites.ix[k, 'SITE_FUNCTIONS'] for k in gsea_reg_sites_cor['kinase']]

ztest_reg_sites_cor = [(k, pearson(phospho_df.ix[k, strains].values, kinase_ztest_df.ix[k.split('_')[0], strains].values)) for k in regulatory_sites.index]
ztest_reg_sites_cor = [(k, c, p, x_values[i], y_values[i], count) for (k, (c, p, x_values, y_values, count)) in ztest_reg_sites_cor if np.isfinite(c) for i in range(len(x_values))]
ztest_reg_sites_cor = DataFrame(ztest_reg_sites_cor, columns=['kinase', 'correlation', 'pvalue', 'phospho', 'enrichment', 'count'])
ztest_reg_sites_cor['method'] = 'z_test'
ztest_reg_sites_cor['function'] = [regulatory_sites.ix[k, 'SITE_FUNCTIONS'] for k in ztest_reg_sites_cor['kinase']]

regulatory_sites_cor = gsea_reg_sites_cor.append(ztest_reg_sites_cor)
regulatory_sites_cor.to_csv(wd + 'tables/regulatory_sites_cor.tab', sep='\t', index=False)
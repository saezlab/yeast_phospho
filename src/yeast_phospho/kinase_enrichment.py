import time
import numpy as np
from pandas import DataFrame, read_csv, pivot_table
from pymist.enrichment.gsea import gsea

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import phospho FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')
strains = set(phospho_df.columns)

# Import dynamic phospho FC
dyn_phospho_fc = read_csv(wd + 'tables/dynamic_phosphoproteomics.tab', sep='\t', index_col='site')

# Import Phosphogrid network
network = read_csv(wd + 'files/phosphosites.txt', sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SEQUENCE']]
print '[INFO] [PHOSPHOGRID] ', network.shape

# Filter interactions without kinases/phosphatases
network = network.loc[np.bitwise_or(network['KINASES_ORFS'] != '-', network['PHOSPHATASES_ORFS'] != '-')]
print '[INFO] [PHOSPHOGRID] (filter non-kinase/phosphatase interactions): ', network.shape

# Split into multiple interactions into different lines and remove self-phosphorylation events
network['SOURCE'] = network['KINASES_ORFS'] + '|' + network['PHOSPHATASES_ORFS']
network = [(k, r['ORF_NAME'] + '_' + r['PHOSPHO_SITE']) for i, r in network.iterrows() for k in r['SOURCE'].split('|') if k != '-' and r['ORF_NAME'] != k]
network = DataFrame(network, columns=['SOURCE', 'TARGET'])
network = network[network['SOURCE'] != '']
network.to_csv(wd + 'tables/kinases_phosphatases_targets.tab', sep='\t', index=False)
print '[INFO] [PHOSPHOGRID] (split into multiple interactions into different lines and remove self-phosphorylation events): ', network.shape

# Set kinases targets dictionary
kinases = set(network['SOURCE'])
kinases_targets = {k: set(network.loc[network['SOURCE'] == k, 'TARGET']) for k in kinases}


####  Steady-state phosphoproteomics kinases enrichment
start_time = time.time()
kinase_df = [(k, ko, gsea(phospho_df[ko], targets, True, 10000)[:2]) for k, targets in kinases_targets.items() for ko in strains]
kinase_df = [(k, ko, -np.log10(pvalue) if es < 0 else np.log10(pvalue)) for k, ko, (es, pvalue) in kinase_df]
kinase_df = DataFrame(kinase_df, columns=['kinase', 'strain', 'score']).dropna()
kinase_df = pivot_table(kinase_df, values='score', index='kinase', columns='strain')
print '[INFO] GSEA for kinase enrichment done (ellapsed time %.2fmin).' % ((time.time() - start_time) / 60), kinase_df.shape

# Export matrix
kinase_df_file = wd + 'tables/kinase_enrichment_df.tab'
kinase_df.to_csv(kinase_df_file, sep='\t')
print '[INFO] Kinase enrichment matrix exported to: %s' % kinase_df_file


####  Dynamic phosphoproteomics kinases enrichment
start_time = time.time()
dyn_kinase_df = [(k, c, gsea(dyn_phospho_fc[c], targets, True, 10000)[:2]) for k, targets in kinases_targets.items() for c in dyn_phospho_fc.columns]
dyn_kinase_df = [(k, c, -np.log10(pvalue) if es < 0 else np.log10(pvalue)) for k, c, (es, pvalue) in dyn_kinase_df]
dyn_kinase_df = DataFrame(dyn_kinase_df, columns=['kinase', 'strain', 'score']).dropna()
dyn_kinase_df = pivot_table(dyn_kinase_df, values='score', index='kinase', columns='strain')
print '[INFO] GSEA for kinase enrichment done (ellapsed time %.2fmin).' % ((time.time() - start_time) / 60), dyn_kinase_df.shape

# Export matrix
dyn_kinase_df_file = wd + 'tables/kinase_enrichment_dynamic_df.tab'
dyn_kinase_df.to_csv(dyn_kinase_df_file, sep='\t')
print '[INFO] Kinase enrichment matrix exported to: %s' % dyn_kinase_df_file
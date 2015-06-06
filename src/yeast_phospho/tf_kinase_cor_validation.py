import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, pivot_table
from pymist.enrichment.gsea import gsea
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, np.sum(mask)

# Version
sns.set_style('white')
version = 'v1'
print '[INFO] Version: %s' % version

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import gene names
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()

# Import phospho FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')

# Import Phosphogrid network
network = read_csv(wd + 'files/phosphosites.txt', sep='\t').loc[:, ['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SEQUENCE']]
print '[INFO] [PHOSPHOGRID] ', network.shape

# Import TF enrichment
tf_df = read_csv(wd + 'tables/tf_enrichment_df.tab', sep='\t', index_col=0)

# Import conversion table
name2id = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=0).to_dict()['name']

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

# Import TF network
tf_network = read_csv(wd + 'data/tf_network/tf_gene_network_chip_only.tab', sep='\t')
tf_network['tf'] = [name2id[i] if i in name2id else id2name[i] for i in tf_network['tf']]
tfs = set(tf_network['tf'])
tfs_targets = {tf: set(tf_network.loc[tf_network['tf'] == tf, 'target']) for tf in tfs}
print '[INFO] TF targets calculated!'

# ---- Remove regulatory TF sites
tf_reg_sites = read_csv(wd + 'files/phosphosites.txt', sep='\t')[['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SITE_FUNCTIONS']]
tf_reg_sites = tf_reg_sites[[p in tfs_targets for p in tf_reg_sites['ORF_NAME']]]
tf_reg_sites = tf_reg_sites[['Inhibits The Protein Function' in f or 'Activates The Protein Function' in f for f in tf_reg_sites['SITE_FUNCTIONS']]]
tf_reg_sites.index = tf_reg_sites['ORF_NAME'] + '_' + tf_reg_sites['PHOSPHO_SITE']
tf_reg_sites = tf_reg_sites['SITE_FUNCTIONS'].to_dict()

test_cases = [(k, s, tf_reg_sites[s]) for k in kinases_targets for s in kinases_targets[k] if s in tf_reg_sites]

kinases_targets = {k: {s for s in kinases_targets[k] if s not in tf_reg_sites} for k in kinases_targets}

# ---- Kinase Enrichment without TF p-sites
strains = list(set(tf_df.columns).intersection(phospho_df.columns))

conditions = {ko: phospho_df[ko].dropna().to_dict() for ko in strains}

start_time = time.time()
kinase_df = [(k, ko, gsea(conditions[ko], kinases_targets[k], 1000)) for k in kinases_targets for ko in strains]
kinase_df = [(k, ko, -np.log10(pvalue) if es < 0 else np.log10(pvalue)) for k, ko, (es, pvalue) in kinase_df]
kinase_df = DataFrame(kinase_df, columns=['kinase', 'strain', 'score']).dropna()
kinase_df = pivot_table(kinase_df, values='score', index='kinase', columns='strain')
print '[INFO] GSEA for kinase enrichment done (ellapsed time %.2fmin).' % ((time.time() - start_time) / 60), kinase_df.shape

# Export matrix
kinase_df_file = wd + 'tables/kinase_enrichment_df_without_TF.tab'
kinase_df.to_csv(kinase_df_file, sep='\t')
print '[INFO] Kinase enrichment matrix exported to: %s' % kinase_df_file

# ---- Correlation Kinases vs TF
test_cases_simp = {(k, s.split('_')[0]) for k, s, _ in test_cases}

k_tf_cor = [(k, tf, pearson(tf_df.ix[tf, strains], kinase_df.ix[k, strains]), int((k, tf) in test_cases_simp)) for tf in tf_df.index for k in kinase_df.index]
k_tf_cor = DataFrame([(k, tf, c, p, n, tp) for k, tf, (c, p, n), tp in k_tf_cor], columns=['kinase', 'TF', 'cor', 'pvalue', 'n', 'TP']).dropna()
k_tf_cor = k_tf_cor[k_tf_cor['n'] > 5]
k_tf_cor['log10_pvalue'] = -np.log10(k_tf_cor['pvalue'])
k_tf_cor['kinase_name'] = [acc_name[p].split(';')[0] for p in k_tf_cor['kinase']]
k_tf_cor['TF_name'] = [acc_name[p].split(';')[0] for p in k_tf_cor['TF']]
print '[INFO] Kinase vs TF Correlation: ', k_tf_cor.shape

curve_fpr, curve_tpr, _ = roc_curve(k_tf_cor['TP'], k_tf_cor['log10_pvalue'])
curve_auc = auc(curve_fpr, curve_tpr)

plt.plot(curve_fpr, curve_tpr, label='area = %0.2f' % curve_auc)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.savefig(wd + 'reports/%s_TF_Kinase_validation.pdf' % version, bbox_inches='tight')
plt.close('all')
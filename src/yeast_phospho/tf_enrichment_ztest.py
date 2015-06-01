import time
import numpy as np
from pandas import DataFrame, read_csv, pivot_table, Index
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW


def ztest(dataset, targets):
    targets_values = DescrStatsW([dataset[t] for t in targets if t in dataset])
    non_targets_values = DescrStatsW([dataset[t] for t in dataset if t not in targets])

    if len(targets_values.data) > 1:
        t_stat, p_value = CompareMeans(targets_values, non_targets_values).ztest_ind(usevar='unequal')
        mean_diff = targets_values.data.mean() - non_targets_values.data.mean()

        return np.log10(p_value) if mean_diff < 0 else -np.log10(p_value)

    else:
        return np.NaN

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import metabol log2 FC
metabol_df = read_csv(wd + 'tables/steady_state_metabolomics.tab', sep='\t', index_col=0)
metabol_df.index = Index(metabol_df.index, dtype=str)

# Import conversion table
name2id = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=1).to_dict()['orf']
id2name = read_csv(wd + 'tables/orf_name_dataframe.tab', sep='\t', index_col=0).to_dict()['name']

# Import gene-expression data-set
gexp = read_csv(wd + 'data/gene_expresion/tf_ko_gene_expression.tab', sep='\t', header=False)
gexp = gexp[gexp['study'] == 'Kemmeren_2014']
gexp['tf'] = [name2id[i] if i in name2id else id2name[i] for i in gexp['tf']]
gexp = pivot_table(gexp, values='value', index='target', columns='tf')
print '[INFO] Gene-expression imported!'

# Import TF network
tf_network = read_csv(wd + 'data/tf_network/tf_gene_network_chip_only.tab', sep='\t')
tf_network['tf'] = [name2id[i] if i in name2id else id2name[i] for i in tf_network['tf']]

# Overlap conditions with metabolomcis
strains = list(set(gexp.columns).intersection(metabol_df.columns))

# Filter gene-expression to overlapping conditions
gexp = gexp[strains]

# TF targets
tfs = set(tf_network['tf'])
tfs_targets = {tf: set(tf_network.loc[tf_network['tf'] == tf, 'target']) for tf in tfs}
print '[INFO] TF targets calculated!'

# TF enrichment
start_time = time.time()

conditions = {ko: gexp[ko].dropna().to_dict() for ko in strains}

tf_df = [(k, ko, ztest(conditions[ko], tfs_targets[k])) for k in tfs_targets for ko in strains]
tf_df = DataFrame(tf_df, columns=['tf', 'strain', 'score']).dropna()
tf_df = pivot_table(tf_df, values='score', index='tf', columns='strain')
print '[INFO] GSEA for TF enrichment done (ellapsed time %.2fmin).' % ((time.time() - start_time) / 60)

# Export matrix
tf_df_file = wd + 'tables/tf_enrichment_df_ztest.tab'
tf_df.to_csv(tf_df_file, sep='\t')
print '[INFO] Kinase enrichment matrix exported to: %s' % tf_df_file
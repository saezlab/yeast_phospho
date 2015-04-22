import numpy as np
from pandas import DataFrame, read_csv, pivot_table
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW


def ztest(dataset, targets):
    targets_values = DescrStatsW(dataset[targets].dropna().values)
    non_targets_values = DescrStatsW(dataset.drop(set(dataset.index).intersection(targets)).dropna().values)

    if len(targets_values.data) > 1:
        t_stat, p_value = CompareMeans(targets_values, non_targets_values).ztest_ind(usevar='unequal')
        mean_diff = targets_values.data.mean() - non_targets_values.data.mean()

        return np.log10(p_value) if mean_diff < 0 else -np.log10(p_value)

    else:
        return np.NaN

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import phospho FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')
strains = set(phospho_df.columns)

# Import kinase targets
network = read_csv(wd + 'tables/kinases_phosphatases_targets.tab', sep='\t')
kinases = set(network['SOURCE'])
kinases_targets = {k: set(network.loc[network['SOURCE'] == k, 'TARGET']) for k in kinases}

# Kinase enrichment with z-test
kinases_ztest_df = [(k, ko, ztest(phospho_df[ko], targets)) for k, targets in kinases_targets.items() for ko in strains]
print '[INFO] GSEA for kinase enrichment done z-test. ', len(kinases_ztest_df)

kinases_ztest_df = DataFrame(kinases_ztest_df, columns=['kinase', 'strain', 'score']).dropna()
kinases_ztest_df = pivot_table(kinases_ztest_df, values='score', index='kinase', columns='strain')
kinases_ztest_df.to_csv(wd + 'tables/kinase_enrichment_df_ztest.tab', sep='\t')
print '[INFO] Kinase enrichment matrix exported to: ', wd + 'kinase_enrichment_df_ztest.tab'
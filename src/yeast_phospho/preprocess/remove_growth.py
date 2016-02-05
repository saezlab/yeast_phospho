import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import regress_out, pearson
from sklearn.decomposition import PCA


# Regress-out Factor correlated with growth rate
datasets = [
    ('metabolomics_steady_state', 'Metabolomics', 'PC1', 'strain_relative_growth_rate.txt'),
    ('kinase_activity_steady_state_gsea', 'Kinases', 'PC1', 'strain_relative_growth_rate.txt'),
    ('tf_activity_steady_state_gsea', 'TFs', 'PC1', 'strain_relative_growth_rate.txt'),

    ('metabolomics_dynamic', 'Metabolomics', 'PC2', 'dynamic_growth.txt'),
    ('kinase_activity_dynamic_gsea', 'Kinases', 'PC3', 'dynamic_growth.txt'),
    ('tf_activity_dynamic_gsea', 'TFs', 'PC2', 'dynamic_growth.txt'),

    ('kinase_activity_steady_state', 'Kinases', 'PC1', 'strain_relative_growth_rate.txt'),
    ('tf_activity_steady_state', 'TFs', 'PC2', 'strain_relative_growth_rate.txt'),

    ('kinase_activity_dynamic', 'Kinases', 'PC2', 'dynamic_growth.txt'),
    ('tf_activity_dynamic', 'TFs', 'PC3', 'dynamic_growth.txt'),
]

n_components = 3
for df_file, df_type, selected_pc, growth_file in datasets:
    print df_file

    # Import growth rates
    growth = read_csv('%s/files/%s' % (wd, growth_file), sep='\t', index_col=0)['relative_growth']
    conditions = list(growth.index)

    # Import data-set
    df = read_csv('%s/tables/%s.tab' % (wd, df_file), sep='\t', index_col=0).T

    if df_type == 'Kinases':
        df = df.loc[:, (df.count() / df.shape[0]) > .75]

    # Factor Analysis
    fa = PCA(n_components=n_components).fit(df.replace(np.nan, 0.0))
    pc = DataFrame(fa.transform(df.replace(np.nan, 0.0)), index=df.index, columns=['PC%d' % (i+1) for i in range(n_components)])

    print [c + ' (c: %.2f, p: %.2e, n: %d' % pearson(growth, pc.ix[growth.index, c]) + ', v: %.2f' % (fa.explained_variance_ratio_[int(c[-1:]) - 1] * 100) for c in pc]

    # Regress-out factor
    df = DataFrame({m: regress_out(pc.ix[conditions, selected_pc], df.ix[conditions, m]) for m in df}).T

    # # Export regressed-out data-set
    # df.to_csv('%s/tables/%s_no_growth.tab' % (wd, df_file), sep='\t')
    # print '[INFO] Growth regressed-out: ', 'tables/%s_no_growth.tab' % df_file

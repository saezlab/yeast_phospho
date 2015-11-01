import numpy as np
from yeast_phospho import wd
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import regress_out, pearson
from sklearn.decomposition.factor_analysis import FactorAnalysis


# Import growth rates
growth = read_csv('%s/files/strain_relative_growth_rate.txt' % wd, sep='\t', index_col=0)['relative_growth']
ko_strains = list(growth.index)


# Regress-out Factor correlated with growth rate
datasets = [
    ('metabolomics_steady_state', 'Metabolomics', 'PC1'),
    ('kinase_activity_steady_state', 'Kinases', 'PC1'),
    ('tf_activity_steady_state', 'TFs', 'PC1')
]

n_components = 10
for df_file, df_type, selected_pc in datasets:
    # Import data-set
    df = read_csv('%s/tables/%s.tab' % (wd, df_file), sep='\t', index_col=0).T

    if df_type == 'Kinases':
        df = df[(df.count(1) / df.shape[1]) > .25].replace(np.NaN, 0)

    # Factor Analysis
    fa = FactorAnalysis(n_components=n_components).fit(df)
    pc = DataFrame(fa.transform(df), index=df.index, columns=['PC%d' % (i+1) for i in range(n_components)])
    print [(c, pearson(growth, pc.ix[growth.index, c])) for c in pc]

    # Regress-out factor
    df = DataFrame({m: regress_out(pc.ix[ko_strains, selected_pc], df.ix[ko_strains, m]) for m in df}).T

    # Export regressed-out data-set
    df.to_csv('%s/tables/%s_no_growth.tab' % (wd, df_file), sep='\t')
    print '[INFO] Growth regressed-out: ', 'tables/%s_no_growth.tab' % df_file

from yeast_phospho import wd
from pandas import DataFrame, read_csv
from yeast_phospho.utilities import regress_out


# Regress-out Factor correlated with growth rate
datasets = [
    ('metabolomics_steady_state', 'Metabolomics', 'strain_relative_growth_rate.txt'),
    ('kinase_activity_steady_state_gsea', 'Kinases', 'strain_relative_growth_rate.txt'),
    ('tf_activity_steady_state_gsea', 'TFs', 'strain_relative_growth_rate.txt'),

    ('metabolomics_dynamic', 'Metabolomics', 'dynamic_growth.txt'),
    ('kinase_activity_dynamic_gsea', 'Kinases', 'dynamic_growth.txt'),
    ('tf_activity_dynamic_gsea', 'TFs', 'dynamic_growth.txt'),

    ('kinase_activity_steady_state', 'Kinases', 'strain_relative_growth_rate.txt'),
    ('tf_activity_steady_state', 'TFs', 'strain_relative_growth_rate.txt'),

    ('kinase_activity_dynamic', 'Kinases', 'dynamic_growth.txt'),
    ('tf_activity_dynamic', 'TFs', 'dynamic_growth.txt'),
]

for df_file, df_type, growth_file in datasets:
    print df_file

    # Import growth rates
    growth = read_csv('%s/files/%s' % (wd, growth_file), sep='\t', index_col=0)['relative_growth']

    # Import data-set
    df = read_csv('%s/tables/%s.tab' % (wd, df_file), sep='\t', index_col=0)

    if df_type == 'Kinases':
        df = df[(df.count(1) / df.shape[1]) > .75]

    # Conditions overlap
    conditions = list(set(growth.index).intersection(df))

    # Regress-out factor
    df = DataFrame({m: regress_out(growth[conditions], df.ix[m, conditions]) for m in df.index}).T

    # Export regressed-out data-set
    df.to_csv('%s/tables/%s_no_growth.tab' % (wd, df_file), sep='\t')
    print '[INFO] Growth regressed-out: ', 'tables/%s_no_growth.tab' % df_file

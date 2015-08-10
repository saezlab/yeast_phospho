import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from yeast_phospho import wd
from bioservices import KEGG, KEGGParser, QuickGO
from pymist.enrichment.gsea import gsea
from sklearn.linear_model import Lasso
from pandas import DataFrame, read_csv
from scipy.stats.distributions import hypergeom


# ---- Set-up KEGG bioservice
kegg, kegg_parser = KEGG(cache=True), KEGGParser()

kegg.organism = 'sce'
print '[INFO] KEGG service configured'

kegg_pathways = {p: kegg.parse_kgml_pathway(p) for p in kegg.pathwayIds}
kegg_pathways = {p: {x.split(':')[1] for i in kegg_pathways[p]['entries'] if i['type'] == 'gene' for x in i['name'].split(' ')} for p in kegg_pathways}
print '[INFO] KEGG pathways extracted: ', len(kegg_pathways)


# ---- Set-up QuickGO bioservice
quickgo = QuickGO(cache=True)

go_terms = read_csv('%s/files/gene_association.goa_ref_yeast' % wd, sep='\t', skiprows=12, header=None)[[4, 10]]
go_terms_set = set(go_terms[4])
go_terms = {go: set(go_terms.loc[go_terms[4] == go, 10]) for go in go_terms_set}
go_terms = {go: {p for k in go_terms[go] for p in k.split('|')} for go in go_terms}

go_terms = {go: go_terms[go] for go in go_terms if '<namespace>biological_process</namespace>' in quickgo.Term(go)}
print '[INFO] Consider only biological process GO Terms: ', len(go_terms)

# ---- Import metabolites map
m_map = read_csv('%s/files/metabolite_mz_map_kegg.txt' % wd, sep='\t')
m_map['mz'] = [float('%.2f' % i) for i in m_map['mz']]
m_map = m_map.drop_duplicates('mz').drop_duplicates('formula')
m_map = m_map.groupby('mz')['name'].apply(lambda i: '; '.join(i)).to_dict()


# ---- Import YORF names
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)['gene'].to_dict()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}


# ---- Import data-sets
# Steady-state
metabolomics_std = read_csv('%s/tables/metabolomics_steady_state.tab' % wd, sep='\t', index_col=0).dropna()
metabolomics_std = metabolomics_std[metabolomics_std.std(1) > .2]

k_activity_std = read_csv('%s/tables/kinase_activity_steady_state.tab' % wd, sep='\t', index_col=0)
k_activity_std = k_activity_std[(k_activity_std.count(1) / k_activity_std.shape[1]) > .6].replace(np.NaN, 0.0)

tf_activity_std = read_csv('%s/tables/tf_activity_steady_state.tab' % wd, sep='\t', index_col=0).dropna()

# Steady-state with growth
metabolomics_std_g = read_csv('%s/tables/metabolomics_steady_state_growth_rate.tab' % wd, sep='\t', index_col=0).dropna()
metabolomics_std_g = metabolomics_std_g[metabolomics_std_g.std(1) > .2]

k_activity_std_g = read_csv('%s/tables/kinase_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0)
k_activity_std_g = k_activity_std_g[(k_activity_std_g.count(1) / k_activity_std_g.shape[1]) > .6].replace(np.NaN, 0.0)

tf_activity_std_g = read_csv('%s/tables/tf_activity_steady_state_with_growth.tab' % wd, sep='\t', index_col=0).dropna()

# Dynamic
metabolomics_dyn = read_csv('%s/tables/metabolomics_dynamic.tab' % wd, sep='\t', index_col=0).dropna()
metabolomics_dyn = metabolomics_dyn[metabolomics_dyn.std(1) > .2]

k_activity_dyn = read_csv('%s/tables/kinase_activity_dynamic.tab' % wd, sep='\t', index_col=0)
k_activity_dyn = k_activity_dyn[(k_activity_dyn.count(1) / k_activity_dyn.shape[1]) > .6].replace(np.NaN, 0.0)

tf_activity_dyn = read_csv('%s/tables/tf_activity_dynamic.tab' % wd, sep='\t', index_col=0).dropna()

# ---- Overlap
strains = list(set(metabolomics_std.columns).intersection(k_activity_std.columns).intersection(tf_activity_std.columns))
conditions = list(set(metabolomics_dyn.columns).intersection(k_activity_dyn.columns).intersection(tf_activity_dyn.columns))
metabolites = list(set(metabolomics_std.index).intersection(metabolomics_dyn.index))
kinases = list(set(k_activity_std.index).intersection(k_activity_dyn.index))
tfs = list(set(tf_activity_std.index).intersection(tf_activity_dyn.index))

metabolomics_std, k_activity, tf_activity = metabolomics_std.ix[metabolites, strains], k_activity_std.ix[kinases, strains], tf_activity_std.ix[tfs, strains]
metabolomics_std_g, k_activity_g, tf_activity_g = metabolomics_std_g.ix[metabolites, strains], k_activity_std_g.ix[kinases, strains], tf_activity_std_g.ix[tfs, strains]
metabolomics_dyn, k_activity_dyn, tf_activity_dyn = metabolomics_dyn.ix[metabolites, conditions], k_activity_dyn.ix[kinases, conditions], tf_activity_dyn.ix[tfs, conditions]

k_tf_activity = k_activity.append(tf_activity)
k_tf_activity_g = k_activity_g.append(tf_activity_g)
k_tf_activity_dyn = k_activity_dyn.append(tf_activity_dyn)

# Comparisons
datasets_files = [
    (k_tf_activity, metabolomics_std, 'Steady-state'),
    (k_tf_activity_g, metabolomics_std_g, 'Steady-state (with growth)'),
    (k_tf_activity_dyn, metabolomics_dyn, 'Dynamic')
]

for xs, ys, condition in datasets_files:
    # Lasso regression
    lm = Lasso(alpha=.01).fit(xs.T, ys.T)

    features = DataFrame(lm.coef_, index=ys.index, columns=xs.index)
    features = features.sum().to_dict()

    # Kegg enrichment
    kegg_pathways_hyper = [(p, gsea(features, kegg_pathways[p], 10000)) for p in kegg_pathways]
    kegg_pathways_hyper = DataFrame([(p, es, pval) for p, (es, pval) in kegg_pathways_hyper], columns=['pathway', 'es', 'pvalue']).dropna()
    kegg_pathways_hyper['name'] = [re.findall('NAME\s*(.*)Saccharomyces cerevisiae \(budding yeast\)\n?', kegg.get(p))[0].split(' - ')[0] for p in kegg_pathways_hyper['pathway']]
    kegg_pathways_hyper['intersection'] = [len(kegg_pathways[p].intersection(features)) for p in kegg_pathways_hyper['pathway']]
    kegg_pathways_hyper = kegg_pathways_hyper[kegg_pathways_hyper['pvalue'] < .05].sort('pvalue')
    kegg_pathways_hyper['source'] = 'kegg'

    # GO term enrichment
    go_terms_hyper = [(p, gsea(features, go_terms[p], 10000)) for p in go_terms]
    go_terms_hyper = DataFrame([(p, es, pval) for p, (es, pval) in go_terms_hyper], columns=['pathway', 'es', 'pvalue']).dropna()
    go_terms_hyper['name'] = [re.findall('name: (.*)\n?', quickgo.Term(go, frmt='obo'))[0] for go in go_terms_hyper['pathway']]
    go_terms_hyper['intersection'] = [len(go_terms[p].intersection(features)) for p in go_terms_hyper['pathway']]
    go_terms_hyper = go_terms_hyper[go_terms_hyper['pvalue'] < .05].sort('pvalue')
    go_terms_hyper['source'] = 'go'

    # Plot GO terms enrichment values barplot
    plot_df = go_terms_hyper.append(kegg_pathways_hyper).sort('pvalue')
    plot_df['pvalue_log10'] = -np.log10(plot_df['pvalue'])
    plot_df['short_name'] = [n[:50] for n in plot_df['name']]

    sns.set(style='ticks', palette='pastel', color_codes=True)
    g = sns.FacetGrid(data=plot_df, size=8, aspect=1)
    g.map(plt.axvline, x=-np.log10(.05), ls=':', c='.5')
    g.map(sns.barplot, 'pvalue_log10', 'short_name', color='#95a5a6', lw=0, orient='h', ci=None)
    plt.savefig('%s/reports/kinase_enzyme_enrichment_gsea_%s.pdf' % (wd, condition), bbox_inches='tight')
    plt.close('all')

    print '[INFO] Enrichment plotted'

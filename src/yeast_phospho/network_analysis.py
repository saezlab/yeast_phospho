import re
import pydot
import igraph
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyparsing import col
from pandas.stats.misc import zscore
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC, SVC
from scipy.stats.distributions import hypergeom
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
from scipy.stats import pearsonr
from sklearn.cross_validation import KFold
from statsmodels.distributions import ECDF
from pandas import DataFrame, Series, read_csv, pivot_table
from sklearn.metrics import roc_curve, auc, jaccard_similarity_score, f1_score
from pymist.reader.sbml_reader import read_sbml_model


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, np.sum(mask)

sns.set_style('white')

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

# Import metabolic model
model = read_sbml_model('/Users/emanuel/Projects/resources/metabolic_models/1752-0509-4-145-s1/yeast_4.04.xml')
enzymes = model.get_genes()

# Import acc map to name form uniprot
acc_name = read_csv('/Users/emanuel/Projects/resources/yeast/yeast_uniprot.txt', sep='\t', index_col=1)
acc_name = acc_name['gene'].to_dict()
acc_name = {k: acc_name[k].split(';')[0] for k in acc_name}

# Import phospho FC
phospho_df = read_csv(wd + 'tables/steady_state_phosphoproteomics.tab', sep='\t', index_col='site')

# Import multiple phospho p-sites
phospho_df_ms = read_csv(wd + 'tables/steady_state_phosphoproteomics_multiple_psites.tab', sep='\t', index_col='site')

# Import kinase enrichment
kinase_df = read_csv(wd + 'tables/kinase_enrichment_df.tab', sep='\t', index_col=0)

# Import Phosphogrid network
network = read_csv(wd + 'files/phosphosites.txt', sep='\t')[['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'SEQUENCE']]
network = network.loc[network['KINASES_ORFS'] != '-']
network['SOURCE'] = network['KINASES_ORFS']
network = [(k, r['ORF_NAME'] + '_' + r['PHOSPHO_SITE']) for i, r in network.iterrows() for k in r['SOURCE'].split('|') if k != '-' and r['ORF_NAME'] != k]
network = DataFrame(network, columns=['SOURCE', 'TARGET'])
network = network[network['SOURCE'] != '']

# Set kinases targets dictionary
kinases = set(network['SOURCE'])
kinases_targets = {k: set(network.loc[network['SOURCE'] == k, 'TARGET']) for k in kinases}


# Improt regulatory sites
def is_regulatory(x):
    regulatory_functions = ['Required For Protein Function', 'Activates The Protein Function', 'Inhibits The Protein Function']
    return sum([i in x for i in regulatory_functions]) > 0

reg_sites = read_csv(wd + 'files/phosphosites.txt', sep='\t')[['ORF_NAME', 'PHOSPHO_SITE', 'SITE_FUNCTIONS', 'KINASES_ORFS', 'PHOSPHATASES_ORFS', 'SITE_CONDITIONS']]
reg_sites = reg_sites[reg_sites['SITE_FUNCTIONS'] != '-']
reg_sites['SITE'] = reg_sites['ORF_NAME'] + '_' + reg_sites['PHOSPHO_SITE']
reg_sites = reg_sites.set_index('SITE')
reg_sites = reg_sites[[is_regulatory(x) for x in reg_sites['SITE_FUNCTIONS']]]
reg_sites_proteins = {s.split('_')[0] for s in reg_sites.index}

# ---- Define condition
strains = kinase_df.columns

# ---- Weight network
plot_networks = False
condition_networks, condition_networks_nweighted = {}, {}
for condition in strains:
    print '[INFO] Condition: %s' % condition

    c_kinase = kinase_df[condition].dropna().abs()
    c_phosph = phospho_df[condition].dropna().abs()

    # ---- Scale kinase enrichment
    c_kinase_ecdf = ECDF(c_kinase.values)
    c_kinase_weights = {k: c_kinase_ecdf(c_kinase.ix[k]) for k in c_kinase.index}

    # plot_df = zip(*[(c_kinase_weights[k], c_kinase.ix[k]) for k in c_kinase_weights])
    # plt.scatter(plot_df[0], plot_df[1])
    # plt.close('all')

    # ---- Scale p-sites fold-change
    c_phosph_ecdf = ECDF(c_phosph.values)
    c_phosph_weights = {s: c_phosph_ecdf(c_phosph.ix[s]) for s in c_phosph.index}

    # plot_df = zip(*[(c_phosph_weights[k], c_phosph.ix[k]) for k in c_phosph_weights])
    # plt.scatter(plot_df[0], plot_df[1])
    # plt.close('all')

    # ---- Create network
    network_i = igraph.Graph(directed=True)

    vertices = list(set(network['SOURCE']).union(network['TARGET']).union([s.split('_')[0] for s in network['TARGET']]))
    network_i.add_vertices(vertices)

    edges, edges_names, edges_weights = [], [], []
    for i in network.index:
        source, site, substrate = network.ix[i, 'SOURCE'], network.ix[i, 'TARGET'], network.ix[i, 'TARGET'].split('_')[0]

        edges.append((source, site))
        edges_names.append('%s -> %s' % (source, site))
        edges_weights.append(1.0 - c_kinase_weights[source] if source in c_kinase_weights else 1.0)

        edges.append((site, substrate))
        edges_names.append('%s -> %s' % (site, substrate))
        edges_weights.append(1.0 - c_phosph_weights[site] if site in c_phosph_weights else 1.0)

    network_i.add_edges(edges)

    network_i.es['name'] = edges_names
    network_i.es['weight'] = edges_weights

    network_i.simplify(True, False, 'first')

    # ---- Sub-set network to differentially phosphorylated sites
    sub_network = network_i.subgraph({x for i in c_phosph[c_phosph > .8].index if i in vertices for x in network_i.neighborhood(i, order=5, mode='IN')})
    print '[INFO] Sub-network created: ', sub_network.summary()

    condition_networks_nweighted[condition] = sub_network.copy()
    print '[INFO] Unweighted sub-network created: ', condition_networks_nweighted[condition].summary()

    condition_networks[condition] = sub_network.spanning_tree('weight')
    print '[INFO] Weighted sub-network created: ', condition_networks[condition].summary()

    # ---- Plot consensus network
    if plot_networks:
        sub_network = condition_networks[condition]

        graph = pydot.Dot(graph_type='digraph', rankdir='LR')

        graph.set_node_defaults(fontcolor='white', penwidth='3')
        graph.set_edge_defaults(color='gray', arrowhead='vee')

        freq_ecdf = ECDF(sub_network.es.get_attribute_values('weight'))

        for edge_index in sub_network.es.indices:
            edge = sub_network.es[edge_index]

            source_id, target_id = sub_network.vs[edge.source].attributes()['name'], sub_network.vs[edge.target].attributes()['name']

            source = pydot.Node(source_id, style='filled', shape='box', penwidth='0')
            target = pydot.Node(target_id, style='filled')

            for node in [source, target]:
                node_name = node.get_name()

                # Set node colour
                if node_name.split('_')[0] in enzymes:
                    node.set_fillcolor('#8EC127')

                elif node_name in c_phosph.index:
                    node.set_fillcolor('#3498db')

                elif node_name in c_kinase:
                    node.set_fillcolor('#BB3011')

                # Set node name
                if len(node_name.split('_')) == 2:
                    node_name, res = node_name.split('_')
                    node.set_name((acc_name[node_name] if node_name in acc_name else node_name) + '_' + res)

                else:
                    node.set_name(acc_name[node_name] if node_name in acc_name else node_name)

                graph.add_node(node)

            # Set edge width
            edge_width = str((1 - freq_ecdf(sub_network.es[edge_index].attributes()['weight'])) * 5 + 1)

            edge = pydot.Edge(source, target, penwidth=edge_width)
            graph.add_edge(edge)

        graph.write_pdf(wd + 'reports/networks/consensus_network_%s.pdf' % condition)
        print '[INFO] Network PDF saved!\n'

print '[INFO] Sub-networks calculation done'

# ---- Regulatory sites validation
site_edges = [(condition_networks[condition].vs[e.source]['name'], e['weight']) for condition in strains for e in condition_networks[condition].es]
site_edges = {n for n, w in site_edges if len(n.split('_')) > 1}

site_edges_nweighted = [(condition_networks_nweighted[condition].vs[e.source]['name'], e['weight']) for condition in strains for e in condition_networks_nweighted[condition].es]
site_edges_nweighted = {n for n, w in site_edges_nweighted if len(n.split('_')) > 1}

M = site_edges_nweighted
n = M.intersection(reg_sites.index)
N = site_edges
x = site_edges.intersection(n)

p_value = hypergeom.sf(len(x), len(M), len(n), len(N))
print '[INFO] x: %d, M: %d, n: %d, N: %d' % (len(x), len(M), len(n), len(N))
print '[INFO] network_regulatory_sites done: %.2e' % p_value


# ---- Predict multiple phosphorylated peptides
phospho_df_ms = phospho_df_ms[phospho_df_ms.count(1) > 50]

site_edges = DataFrame({condition: {condition_networks[condition].vs[e.source]['name']: e['weight'] for e in condition_networks[condition].es} for condition in strains})
site_edges_nweighted = DataFrame({condition: {condition_networks_nweighted[condition].vs[e.source]['name']: 1 for e in condition_networks_nweighted[condition].es} for condition in strains}).replace(np.NaN, 0.0)

predictions, site_edges_predictions, site_edges_nweighted_predictions = {}, {}, {}
for p in phospho_df_ms.index:
    y = phospho_df_ms.ix[p].dropna()

    X = phospho_df[y.index].T
    X = X.loc[:, X.count() > 50].replace(np.NaN, 0.0)

    # Only data
    x = X
    predictions[p] = np.mean([pearson(LinearRegression().fit(x.ix[train], y[train]).predict(x.ix[test]), y[test].values)[0] for train, test in KFold(len(y))])

    # Data + edges weights
    x = (1 - site_edges.T).replace(np.NaN, 0.0)
    x = X.join(x, lsuffix='_prediction')
    site_edges_predictions[p] = np.mean([pearson(LinearRegression().fit(x.ix[train], y[train]).predict(x.ix[test]), y[test].values)[0] for train, test in KFold(len(y))])

    # Data + edges topology
    x = X.join(site_edges_nweighted.T, lsuffix='_prediction')
    site_edges_nweighted_predictions[p] = np.mean([pearson(LinearRegression().fit(x.ix[train], y[train]).predict(x.ix[test]), y[test].values)[0] for train, test in KFold(len(y))])
print '[INFO] Peptides with multiple p-sites prediction complete: %.3f vs %.3f vs %.3f' % (np.mean(predictions.values()), np.mean(site_edges_predictions.values()), np.mean(site_edges_nweighted_predictions.values()))

sns.boxplot([predictions.values(), site_edges_predictions.values(), site_edges_nweighted_predictions.values()], notch=True, names=['Data', 'Data + weights', 'Data + topology'])
sns.despine()
plt.ylabel('Mean pearson (3 fold cross-validation)')
plt.title('Fold-change prediction of multiple-phosphorylated peptides')
plt.savefig(wd + 'reports/multiple_phospho_peptides_prediction_boxplot.pdf')
plt.close('all')
print '[INFO] Figure saved!'

# ---- Calculate weighted shortest-paths between differential phospho sites and all the kinases
c_sites = set(c_phosph.index).intersection(vertices)
c_kinases = set(c_kinase.index).intersection(vertices)

shortest_paths = [(k, s, network_i.get_all_shortest_paths(vertices[k], to=vertices[s], weights='weight')) for k in c_kinases for s in c_sites]
print '[INFO] Weighted shortest paths calculated: ', len(shortest_paths)

shortest_paths_all = [(k, s, network_i.get_all_shortest_paths(vertices[k], to=vertices[s])) for k in c_kinases for s in c_sites]
print '[INFO] Non-weighted shortest paths calculated: ', len(shortest_paths_all)

shortest_paths_len = [(k, s, network_i.shortest_paths(vertices[k], vertices[s], 'weight')) for k in c_kinases for s in c_sites]
print '[INFO] Lenght of shortest paths calculated: ', len(shortest_paths_len)


# ---- Assemble all shortest-paths
# Weighted shortest-paths
shortest_paths_edges = [p for k, s, paths in shortest_paths for path in paths for p in zip(path, path[1:])]
shortest_paths_edges = [network_i.get_eid(s, t) for s, t in shortest_paths_edges]
shortest_paths_edges_freq = np.unique(shortest_paths_edges, return_counts=True)
shortest_paths_edges_freq = dict(zip(shortest_paths_edges_freq[0], np.log2(shortest_paths_edges_freq[1])))

# Unweighted shortest-paths
shortest_paths_edges_all = [p for k, s, paths in shortest_paths_all for path in paths for p in zip(path, path[1:])]
shortest_paths_edges_all = [network_i.get_eid(s, t) for s, t in shortest_paths_edges_all]
shortest_paths_edges_all_freq = np.unique(shortest_paths_edges_all, return_counts=True)
shortest_paths_edges_all_freq = dict(zip(shortest_paths_edges_all_freq[0], np.log2(shortest_paths_edges_all_freq[1])))

# Save edges frequency as attribute
network_i.es.set_attribute_values('freq', [(1.0 * shortest_paths_edges_freq[index]) if index in shortest_paths_edges_freq else np.NaN for index in network_i.es.indices])

# ---- Calculate consensus network
cutoff = 5.0
sub_network = network_i.subgraph_edges([k for k, v in shortest_paths_edges_freq.items() if v >= cutoff])
print '[INFO] cutoff: %d, network: %s' % (cutoff, sub_network.summary())

sub_network = network_i.spanning_tree('weight')
sub_network = sub_network.subgraph([i for i in vertices if len(i.split('_')) == 1 or i in c_phosph.index and c_phosph.ix[i] > 1.0])
print sub_network.summary()

sub_network = network_i.subgraph({x for i in c_phosph[c_phosph > 1.0].index if i in vertices for x in network_i.neighborhood(i, order=5, mode='IN')})
sub_network = sub_network.spanning_tree('weight')
print sub_network.summary()
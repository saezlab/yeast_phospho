import pydot
import igraph
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyparsing import col
from scipy.stats import pearsonr
from statsmodels.distributions import ECDF
from pandas import DataFrame, Series, read_csv, pivot_table
from pymist.reader.sbml_reader import read_sbml_model


def pearson(x, y):
    mask = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    cor, pvalue = pearsonr(x[mask], y[mask]) if np.sum(mask) > 1 else (np.NaN, np.NaN)
    return cor, pvalue, np.sum(mask)

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

regulatory_sites = read_csv(wd + 'files/phosphosites.txt', sep='\t')[['ORF_NAME', 'PHOSPHO_SITE', 'KINASES_ORFS', 'SITE_FUNCTIONS']]
regulatory_sites['site'] = regulatory_sites['ORF_NAME'] + '_' + regulatory_sites['PHOSPHO_SITE']
regulatory_sites = regulatory_sites[[x != '-' for x in regulatory_sites['KINASES_ORFS']]]
regulatory_sites = regulatory_sites[[is_regulatory(x) for x in regulatory_sites['SITE_FUNCTIONS']]]
regulatory_sites = regulatory_sites[[x in phospho_df.index for x in regulatory_sites['site']]]
regulatory_sites = regulatory_sites[['KINASES_ORFS', 'site', 'SITE_FUNCTIONS']]

# ---- Define condition
strains = kinase_df.columns
condition = 'YJL187C'

c_kinase = kinase_df[condition].dropna().abs()
c_phosph = phospho_df[condition].dropna().abs()

# ---- Scale kinase enrichment
c_kinase_ecdf = ECDF(c_kinase.values)
c_kinase_weights = {k: c_kinase_ecdf(c_kinase.ix[k]) for k in c_kinase.index}

plot_df = zip(*[(c_kinase_weights[k], c_kinase.ix[k]) for k in c_kinase_weights])
plt.scatter(plot_df[0], plot_df[1])
plt.close('all')

# ---- Scale p-sites fold-change
c_phosph_ecdf = ECDF(c_phosph.values)
c_phosph_weights = {s: c_phosph_ecdf(c_phosph.ix[s]) for s in c_phosph.index}

plot_df = zip(*[(c_phosph_weights[k], c_phosph.ix[k]) for k in c_phosph_weights])
plt.scatter(plot_df[0], plot_df[1])
plt.close('all')

# ---- Create network
network_i = igraph.Graph(directed=True)

vertices = list(set(network['SOURCE']).union(network['TARGET']).union([s.split('_')[0] for s in network['TARGET']]))
network_i.add_vertices(vertices)

edges, edges_weights = [], []
for i in network.index:
    source, site, substrate = network.ix[i, 'SOURCE'], network.ix[i, 'TARGET'], network.ix[i, 'TARGET'].split('_')[0]

    edges.append((source, site))
    edges_weights.append(1.0 - c_kinase_weights[source] if source in c_kinase_weights else 1.0)

    edges.append((site, substrate))
    edges_weights.append(1.0 - c_phosph_weights[site] if site in c_phosph_weights else 1.0)

network_i.add_edges(edges)

network_i.es['weight'] = edges_weights
network_i.es['inv_weight'] = [1 - i['weight'] for i in network_i.es]

network_i.simplify(True, False, 'first')
print '[INFO] Network created: ', network_i.summary()

# ---- Calculate weighted shortest-paths between differential phospho sites and all the kinases
c_sites = set(c_phosph.index).intersection(vertices)
c_kinases = set(c_kinase.index).intersection(vertices)

s = c_phosph[set(c_phosph.index).intersection(vertices)].argmax()
k = s.split('_')[0]

k_s_paths = [(k, network_i.vs[x]['name'], network_i.vs[p]['name'], network_i.shortest_paths(p[0], k, 'weight')) for x in network_i.neighborhood(k, 5, 'IN') for p in network_i.get_all_shortest_paths(x, k) if len(p) == 5]

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
# sub_network = sub_network.spanning_tree('weight')
print sub_network.summary()

# ---- Plot consensus network
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

        elif node_name in c_sites:
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

graph.write_pdf(wd + 'reports/consensus_network.pdf')
print '[INFO] Network PDF saved!'
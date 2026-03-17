import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

from collections import defaultdict

def build_nx_graph(graph_nodes, graph_relations, extr=1):

    G = nx.Graph()

    #Adds nodes
    for node in graph_nodes:
        G.add_node(node.id,name=node.name,label=getattr(node, "label", None),)

    if extr:
        #Adds edges(of type Relation)
        for rel in graph_relations:
            G.add_edge(rel.source_id,rel.target_id,label=rel.label)
    else:
        #Adds edges(of type tuple)
        for subj, rel, obj, desc in graph_relations:
            if subj not in G or obj not in G:
                continue
            G.add_edge(subj,obj,key=rel,relation=rel,description=desc)

    return G

def graph_statistics(G, top_k=10, plot=True):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    degrees = dict(G.degree())
    degree_values = list(degrees.values())

    avg_degree = sum(degree_values) / num_nodes if num_nodes > 0 else 0

    isolated_nodes = [n for n, d in degrees.items() if d == 0]
    isolated_ratio = len(isolated_nodes) / num_nodes if num_nodes > 0 else 0

    components = list(nx.connected_components(G))
    largest_component = max(components, key=len) if components else set()
    largest_component_ratio = len(largest_component) / num_nodes if num_nodes > 0 else 0

    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)

    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Isolated nodes: {len(isolated_nodes)} ({isolated_ratio:.2%})")
    print(f"Connected components: {len(components)}")
    print(f"Largest component size: {len(largest_component)}")
    print(f"({largest_component_ratio:.2%})")

    print("\nTop nodes by degree:")
    for node_id, deg in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        print(f"  {node_id} — degree {deg}")

    #Degree distribution
    if plot:
        degree_counts = Counter(degree_values)
        x = sorted(degree_counts.keys())
        y = [degree_counts[k] for k in x]

        plt.figure()
        plt.bar(x, y)
        plt.xlabel("Degree")
        plt.ylabel("Number of nodes")
        plt.title("Degree Distribution")
        plt.show()

    return


def debug_graph(graph_nodes, graph_relations, extr=1):
    G = build_nx_graph(graph_nodes, graph_relations, extr)
    return graph_statistics(G)
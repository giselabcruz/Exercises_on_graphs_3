from BFS import bfs_unweighted
from Dijkstra import dijkstra_weighted

import random
import networkx as nx

def generate_random_graph_unweighted_adjlist(n_nodes, p=0.2, seed=42):
    G = nx.fast_gnp_random_graph(n=n_nodes, p=p, seed=seed)
    graph = {u: [] for u in G.nodes()}
    for u, v in G.edges():
        graph[u].append(v)
        graph[v].append(u)
    return graph


def generate_random_graph_weighted_dict(n_nodes, p=0.2, low=1.0, high=10.0, seed=42):
    rng = random.Random(seed)
    G = nx.fast_gnp_random_graph(n=n_nodes, p=p, seed=seed)
    graph = {u: {} for u in G.nodes()}
    for u, v in G.edges():
        w = rng.uniform(low, high)
        graph[u][v] = w
        graph[v][u] = w
    return graph


if __name__ == "__main__":
    g_unweighted = generate_random_graph_unweighted_adjlist(10, 0.3, seed=42)
    g_weighted = generate_random_graph_weighted_dict(10, 0.3, low=1.0, high=10.0, seed=42)

    src, tgt = random.sample(list(g_unweighted.keys()), 2)

    print(f"BFS path from {src} to {tgt}:")
    print(bfs_unweighted(g_unweighted, src, tgt))

    print(f"\nDijkstra path from {src} to {tgt}:")
    path, dist = dijkstra_weighted(g_weighted, src, tgt)
    if path:
        print(f"Path: {path} | Distance: {dist:.2f}")
    else:
        print("No path found between the selected nodes (graph is disconnected).")

import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import os

import networkx as nx
import matplotlib.pyplot as plt
import statistics as stats

from algorithms.BFS import bfs_unweighted
from algorithms.Dijkstra import dijkstra_weighted


def gen_adjlist(n: int, p: float, seed: int = 42) -> Dict[int, List[int]]:
    G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
    adj = {u: [] for u in G.nodes()}
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return adj

def to_weighted(adj: Dict[int, List[int]], low=1.0, high=10.0, seed: int = 42) -> Dict[int, Dict[int, float]]:
    rng = random.Random(seed)
    gw = {u: {} for u in adj}
    for u, nbrs in adj.items():
        for v in nbrs:
            if v not in gw[u]:
                w = rng.uniform(low, high)
                gw[u][v] = w
                gw[v][u] = w
    return gw

@dataclass
class Ops:
    edge_checks: int = 0
    queue_push: int = 0
    pq_push: int = 0
    pq_pop: int = 0
    relax: int = 0
    path_len: int = 0
    distance: float = float("inf")

def bfs_ops(adj: Dict[int, List[int]], s: int, t: int) -> Tuple[List[int], Ops]:
    parents = {s: None}
    q = deque([s])
    ops = Ops(queue_push=1)
    found = False
    while q:
        u = q.popleft()
        for v in adj[u]:
            ops.edge_checks += 1
            if v not in parents:
                parents[v] = u
                q.append(v)
                ops.queue_push += 1
                if v == t:
                    found = True
                    q.clear()
                    break
        if found:
            break
    path = []
    if t in parents:
        x = t
        while x is not None:
            path.insert(0, x)
            x = parents[x]
    ops.path_len = len(path)
    ops.distance = len(path) - 1 if path else float("inf")
    return path, ops

def dijkstra_ops(gw: Dict[int, Dict[int, float]], s: int, t: int) -> Tuple[List[int], Ops]:
    import heapq
    dist = {u: float("inf") for u in gw}
    prev = {u: None for u in gw}
    dist[s] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, s)]
    seen = set()
    ops = Ops(pq_push=1)
    while pq:
        du, u = heapq.heappop(pq); ops.pq_pop += 1
        if u in seen:
            continue
        seen.add(u)
        if u == t:
            break
        for v, w in gw[u].items():
            ops.edge_checks += 1
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v)); ops.pq_push += 1; ops.relax += 1
    path = []
    if dist[t] != float("inf"):
        x = t
        while x is not None:
            path.insert(0, x)
            x = prev[x]
    ops.path_len = len(path)
    ops.distance = dist[t]
    return path, ops


def time_ms(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


COLLECT_OPS = True

def run_once(adj, gw, rng, trials, collect_ops):
    nodes = list(adj.keys())
    bfs_times, dij_times = [], []
    ops_rows = []

    for t in range(trials):
        if len(nodes) < 2:
            break
        s, tgt = rng.sample(nodes, 2)

        (_, t_bfs) = time_ms(bfs_unweighted, adj, s, tgt)
        ((_, _), t_dij) = time_ms(dijkstra_weighted, gw, s, tgt)
        bfs_times.append(t_bfs); dij_times.append(t_dij)

        if collect_ops:
            _, bfsop = bfs_ops(adj, s, tgt)
            _, dijop = dijkstra_ops(gw, s, tgt)
            ops_rows.append(("BFS", bfsop.edge_checks, bfsop.queue_push, 0, 0, bfsop.path_len))
            ops_rows.append(("Dijkstra", dijop.edge_checks, 0, dijop.pq_push, dijop.pq_pop, dijop.path_len))
    return bfs_times, dij_times, ops_rows

def main():
    sizes   = [200, 500, 1000, 2000, 5000]
    p       = 0.01
    trials  = 300
    seed    = 8
    rng     = random.Random(seed)

    os.makedirs("plots", exist_ok=True)

    avg_bfs, avg_dij = [], []
    ops_rows_all = []

    for n in sizes:
        adj = gen_adjlist(n, p, seed=seed)
        gw  = to_weighted(adj, seed=seed)

        bfs_times, dij_times, ops_rows = run_once(adj, gw, rng, trials, COLLECT_OPS)
        if bfs_times and dij_times:
            avg_bfs.append(sum(bfs_times)/len(bfs_times))
            avg_dij.append(sum(dij_times)/len(dij_times))
        else:
            avg_bfs.append(float("nan")); avg_dij.append(float("nan"))

        if COLLECT_OPS:
            for algo, ec, qb, pqb, pqp, plen in ops_rows:
                ops_rows_all.append((algo, n, ec, qb, pqb, pqp, plen))

    plt.figure()
    plt.plot(sizes, avg_bfs, marker="o", label="BFS (unweighted)")
    plt.plot(sizes, avg_dij, marker="s", label="Dijkstra (weighted)")
    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Average runtime (ms)")
    plt.title(f"BFS vs Dijkstra (p={p}, trials={trials})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("plots/runtime_bfs_vs_dijkstra.png", dpi=300); plt.close()

    if COLLECT_OPS:
        def mean_ops(algo, n, idx):
            vals = [row[idx] for row in ops_rows_all if row[0]==algo and row[1]==n]
            return stats.mean(vals) if vals else 0.0

        bfs_edge = [mean_ops("BFS", n, 2) for n in sizes]
        dij_edge = [mean_ops("Dijkstra", n, 2) for n in sizes]
        bfs_q    = [mean_ops("BFS", n, 3) for n in sizes]
        dij_qp   = [mean_ops("Dijkstra", n, 4) for n in sizes]
        dij_qo   = [mean_ops("Dijkstra", n, 5) for n in sizes]

        plt.figure()
        plt.plot(sizes, bfs_edge, marker="o", label="BFS edge checks")
        plt.plot(sizes, dij_edge, marker="s", label="Dijkstra edge checks")
        plt.xlabel("Number of nodes (n)"); plt.ylabel("Average operations")
        plt.title(f"Edge examinations (p={p}, trials={trials})")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig("plots/ops_edge_exams.png", dpi=300); plt.close()

        plt.figure()
        plt.plot(sizes, bfs_q, marker="o", label="BFS queue pushes")
        plt.plot(sizes, dij_qp, marker="s", label="Dijkstra PQ pushes")
        plt.plot(sizes, dij_qo, marker="^", label="Dijkstra PQ pops")
        plt.xlabel("Number of nodes (n)"); plt.ylabel("Average operations")
        plt.title(f"Queue/PQ operations (p={p}, trials={trials})")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig("plots/ops_queue_vs_pq.png", dpi=300); plt.close()

    ratios = [d/b if (b and b>0) else float("inf") for b, d in zip(avg_bfs, avg_dij)]
    crossover = next((n for n, r in zip(sizes, ratios) if r > 1.0 and r < float("inf")), None)

    print("Saved: plots/runtime_bfs_vs_dijkstra.png")
    if COLLECT_OPS:
        print("Saved: plots/ops_edge_exams.png, plots/ops_queue_vs_pq.png")
    print(f"Crossover n where Dijkstra > BFS (avg): {crossover} (ratios: {['%.2f'%r for r in ratios]})")

if __name__ == "__main__":
    main()

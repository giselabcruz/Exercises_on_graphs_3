import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import math

import networkx as nx
import matplotlib.pyplot as plt
import statistics as stats

@dataclass
class Ops:
    edge_checks: int = 0
    queue_push: int = 0
    pq_push: int = 0
    pq_pop: int = 0
    relax: int = 0
    path_len: int = 0
    distance: float = float("inf")

def bfs_unweighted_ops(adj: Dict[int, List[int]], s: int, t: int) -> Tuple[List[int], Ops]:
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
                q.append(v); ops.queue_push += 1
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
            path.append(x)
            x = parents[x]
        path.reverse()
    ops.path_len = len(path)
    ops.distance = len(path) - 1 if path else float("inf")
    return path, ops

def dijkstra_weighted_ops(gw: Dict[int, Dict[int, float]], s: int, t: int) -> Tuple[List[int], Ops]:
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
            path.append(x)
            x = prev[x]
        path.reverse()
    ops.path_len = len(path)
    ops.distance = dist[t]
    return path, ops

def gen_adjlist_constant_degree(n: int, c: float, seed: int = 42) -> Dict[int, List[int]]:
    p = min(1.0, max(0.0, c / n))
    G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        comps = sorted(comps, key=len, reverse=True)
        main = comps[0]
        H = G.copy()
        rng = random.Random(seed + 12345)
        for C in comps[1:]:
            u = rng.choice(tuple(main))
            v = rng.choice(tuple(C))
            H.add_edge(u, v)
            main = main.union(C)
        G = H
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

def pick_pair_same_component(adj: Dict[int, List[int]], rng: random.Random) -> Tuple[int, int]:
    G = nx.Graph()
    for u, nbrs in adj.items():
        for v in nbrs:
            G.add_edge(u, v)
    comp = rng.choice(list(nx.connected_components(G)))
    if len(comp) < 2:
        nodes = list(adj.keys())
        s, t = rng.sample(nodes, 2)
        return s, t
    nodes = list(comp)
    s, t = rng.sample(nodes, 2)
    return s, t

def time_ms(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0

def mean_ci(values, alpha=0.05):
    if not values:
        return float("nan"), float("nan")
    m = stats.mean(values)
    sd = stats.pstdev(values) if len(values) == 1 else stats.stdev(values)
    half = 1.96 * (sd / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return m, half

def run():
    sizes   = [2, 10, 30, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    c_deg   = 10
    trials  = 1000
    seed    = 8
    rng     = random.Random(seed)

    os.makedirs("plots", exist_ok=True)
    os.makedirs("csv", exist_ok=True)

    avg_bfs, avg_dij = [], []
    err_bfs, err_dij = [], []
    ratios = []

    ops_rows = []

    with open("csv/summary_per_size.csv", "w") as fsum:
        fsum.write("n,avg_bfs_ms,ci_bfs_ms,avg_dij_ms,ci_dij_ms,ratio_dij_over_bfs\n")

    with open("csv/trials.csv", "w") as ftr:
        ftr.write("n,algo,time_ms,edge_checks,queue_push,pq_push,pq_pop,relax,path_len,distance\n")

        for n in sizes:
            adj = gen_adjlist_constant_degree(n, c_deg, seed=seed)
            gw  = to_weighted(adj, seed=seed)

            bfs_times, dij_times = [], []
            for _ in range(trials):
                s, t = pick_pair_same_component(adj, rng)

                (path_b, ops_b), tb = time_ms(bfs_unweighted_ops, adj, s, t)
                bfs_times.append(tb)
                ftr.write(f"{n},BFS,{tb:.6f},{ops_b.edge_checks},{ops_b.queue_push},0,0,{ops_b.relax},{ops_b.path_len},{ops_b.distance}\n")
                ops_rows.append(("BFS", n, ops_b.edge_checks, ops_b.queue_push, 0, 0, ops_b.relax, ops_b.path_len))

                (path_d, ops_d), td = time_ms(dijkstra_weighted_ops, gw, s, t)
                dij_times.append(td)
                ftr.write(f"{n},Dijkstra,{td:.6f},{ops_d.edge_checks},0,{ops_d.pq_push},{ops_d.pq_pop},{ops_d.relax},{ops_d.path_len},{ops_d.distance}\n")
                ops_rows.append(("Dijkstra", n, ops_d.edge_checks, 0, ops_d.pq_push, ops_d.pq_pop, ops_d.relax, ops_d.path_len))

            mb, hb = mean_ci(bfs_times)
            md, hd = mean_ci(dij_times)
            avg_bfs.append(mb); err_bfs.append(hb)
            avg_dij.append(md); err_dij.append(hd)
            ratio = (md / mb) if mb > 0 else float("inf")
            ratios.append(ratio)

            with open("csv/summary_per_size.csv", "a") as fsum:
                fsum.write(f"{n},{mb:.6f},{hb:.6f},{md:.6f},{hd:.6f},{ratio:.6f}\n")

    crossover_n = None
    xs = sizes
    diffs = [d - b for b, d in zip(avg_bfs, avg_dij)]
    for i in range(1, len(xs)):
        if diffs[i-1] <= 0 < diffs[i]:
            x0, y0 = xs[i-1], diffs[i-1]
            x1, y1 = xs[i], diffs[i]
            if (y1 - y0) != 0:
                crossover_n = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
            else:
                crossover_n = xs[i]
            break

    plt.figure()
    plt.errorbar(sizes, avg_bfs, yerr=err_bfs, marker="o", label="BFS (unweighted)")
    plt.errorbar(sizes, avg_dij, yerr=err_dij, marker="s", label="Dijkstra (weighted)")
    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Runtime (ms) — mean ± 95% CI")
    plt.title(f"BFS vs Dijkstra (constant avg degree ≈ {c_deg}, trials={trials})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("plots/runtime_bfs_vs_dijkstra.png", dpi=300); plt.close()

    plt.figure()
    plt.plot(sizes, ratios, marker="^", label="Dijkstra / BFS runtime")
    plt.axhline(1.0, linestyle="--")
    if crossover_n is not None:
        plt.axvline(crossover_n, linestyle="--", label=f"Crossover ≈ n={crossover_n:.0f}")
    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Runtime ratio")
    plt.title("When does Dijkstra’s overhead dominate?")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("plots/ratio_dijkstra_over_bfs.png", dpi=300); plt.close()

    def mean_for(algo, n, idx):
        vals = [row[idx] for row in ops_rows if row[0]==algo and row[1]==n]
        return stats.mean(vals) if vals else 0.0

    with open("csv/ops_summary_per_size.csv", "w") as fops:
        fops.write("n,algo,edge_checks,queue_push,pq_push,pq_pop,relax,path_len\n")
        for n in sizes:
            bfs_edge = mean_for("BFS", n, 2); bfs_q = mean_for("BFS", n, 3)
            dij_edge = mean_for("Dijkstra", n, 2); dij_qp = mean_for("Dijkstra", n, 4); dij_qo = mean_for("Dijkstra", n, 5)
            bfs_rel = mean_for("BFS", n, 6); dij_rel = mean_for("Dijkstra", n, 6)
            bfs_pl  = mean_for("BFS", n, 7); dij_pl  = mean_for("Dijkstra", n, 7)
            fops.write(f"{n},BFS,{bfs_edge},{bfs_q},0,0,{bfs_rel},{bfs_pl}\n")
            fops.write(f"{n},Dijkstra,{dij_edge},0,{dij_qp},{dij_qo},{dij_rel},{dij_pl}\n")

    bfs_edge = [mean_for("BFS", n, 2) for n in sizes]
    dij_edge = [mean_for("Dijkstra", n, 2) for n in sizes]
    plt.figure()
    plt.plot(sizes, bfs_edge, marker="o", label="BFS edge checks")
    plt.plot(sizes, dij_edge, marker="s", label="Dijkstra edge checks")
    plt.xlabel("Number of nodes (n)"); plt.ylabel("Average operations")
    plt.title(f"Edge examinations (c≈{c_deg}, trials={trials})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("plots/ops_edge_exams.png", dpi=300); plt.close()

    bfs_q   = [mean_for("BFS", n, 3) for n in sizes]
    dij_qp  = [mean_for("Dijkstra", n, 4) for n in sizes]
    dij_qo  = [mean_for("Dijkstra", n, 5) for n in sizes]
    plt.figure()
    plt.plot(sizes, bfs_q, marker="o", label="BFS queue pushes")
    plt.plot(sizes, dij_qp, marker="s", label="Dijkstra PQ pushes")
    plt.plot(sizes, dij_qo, marker="^", label="Dijkstra PQ pops")
    plt.xlabel("Number of nodes (n)"); plt.ylabel("Average operations")
    plt.title(f"Queue/PQ operations (c≈{c_deg}, trials={trials})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("plots/ops_queue_vs_pq.png", dpi=300); plt.close()

    ratio_str = ", ".join(f"{r:.2f}" for r in ratios)
    print("Saved figures in plots/ and CSVs in csv/")
    print(f"Runtime ratios (Dijkstra/BFS) by n: [{ratio_str}]")
    print(f"Crossover n (interp) where Dijkstra > BFS: {crossover_n}")

if __name__ == "__main__":
    run()

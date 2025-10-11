from collections import deque
import math

INFINITY = math.inf

def dijkstra_weighted(graph, start, end):
    """
    Implementación de Dijkstra para grafos ponderados.
    'graph' debe tener el formato:
        {
            'A': {'B': 5, 'C': 2},
            'B': {'A': 5, 'C': 1, 'D': 3},
            ...
        }
    Devuelve (path, distance)
    """
    if start not in graph or end not in graph:
        raise AttributeError(f"El nodo '{start}' o '{end}' no existe en el grafo")

    unvisited = set(graph.keys())
    dist = {node: INFINITY for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0

    while unvisited:
        current = min(unvisited, key=lambda node: dist[node])
        unvisited.remove(current)

        if dist[current] == INFINITY:
            break

        for neighbor, weight in graph[current].items():
            new_dist = dist[current] + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current

        if current == end:
            break

    path = deque()
    node = end
    while node is not None:
        path.appendleft(node)
        node = prev[node]

    return list(path), dist[end]

if __name__ == "__main__":
    test_graph = {
        "A": {"B": 4, "C": 2},
        "B": {"A": 4, "C": 5, "D": 10},
        "C": {"A": 2, "B": 5, "D": 3},
        "D": {"B": 10, "C": 3},
    }

    path, dist = dijkstra_weighted(test_graph, "A", "D")
    print(f"Camino más corto: {path}")
    print(f"Distancia total: {dist}")

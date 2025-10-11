from collections import deque

def bfs_unweighted(graph, src, tgt):
    """Return the shortest path from the source (src) to the target (tgt) in the graph"""
    if src not in graph:
        raise AttributeError(f"The source '{src}' is not in the graph")
    if tgt not in graph:
        raise AttributeError(f"The target '{tgt}' is not in the graph")

    parents = {src: None}
    queue = deque([src])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
                if neighbor == tgt:
                    queue.clear()
                    break

    if tgt not in parents:
        return []

    path = [tgt]
    while parents[tgt] is not None:
        path.insert(0, parents[tgt])
        tgt = parents[tgt]
    return path

if __name__ == "__main__":
    test = {
        "a": ["b", "f"],
        "b": ["a", "c", "g"],
        "c": ["b", "d", "g", "l"],
        "d": ["c", "e", "k"],
        "e": ["d", "f"],
        "f": ["a", "e"],
        "g": ["b", "c", "h", "l"],
        "h": ["g", "i"],
        "i": ["h", "j", "k"],
        "j": ["i", "k"],
        "k": ["d", "i", "j", "l"],
        "l": ["c", "g", "k"],
    }

    assert bfs_unweighted(test, "a", "e") == ['a', 'f', 'e']
    assert bfs_unweighted(test, "a", "i") == ['a', 'b', 'g', 'h', 'i']
    assert bfs_unweighted(test, "a", "l") == ['a', 'b', 'c', 'l']
    assert bfs_unweighted(test, "b", "k") == ['b', 'c', 'd', 'k']
    print("All tests passed.")


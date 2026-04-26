"""Smoke test: run all the non-API code from the notebook to verify nothing is broken
before someone pays API tokens to find out it doesn't work.
"""
import random
import time
import json
import re
import networkx as nx
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)


def build_graph(n_nodes, avg_degree=3, weight_range=(1, 20), seed=RNG_SEED):
    rng = random.Random(seed)
    g = nx.Graph()
    nodes = [f'N{i}' for i in range(n_nodes)]
    g.add_nodes_from(nodes)
    shuffled = nodes[:]
    rng.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        u = shuffled[i]
        v = rng.choice(shuffled[:i])
        w = rng.randint(*weight_range)
        g.add_edge(u, v, weight=w)
    target_edges = int(n_nodes * avg_degree / 2)
    while g.number_of_edges() < target_edges:
        u, v = rng.sample(nodes, 2)
        if not g.has_edge(u, v):
            w = rng.randint(*weight_range)
            g.add_edge(u, v, weight=w)
    return g


def graph_to_text(g):
    lines = []
    for u, v, data in sorted(g.edges(data=True)):
        lines.append(f'{u} -- {v}  (weight {data["weight"]})')
    return '\n'.join(lines)


def dijkstra_path(g, source, target):
    path = nx.shortest_path(g, source=source, target=target, weight='weight')
    weight = nx.path_weight(g, path, weight='weight')
    return path, weight


def time_dijkstra(g, source, target, runs=1000):
    start = time.perf_counter()
    for _ in range(runs):
        nx.shortest_path(g, source=source, target=target, weight='weight')
    elapsed = time.perf_counter() - start
    return elapsed / runs


def extract_answer(response_text):
    matches = re.findall(r'\{[^{}]*"path"[^{}]*\}', response_text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


# Test extract_answer with realistic LLM outputs
test_responses = [
    '{"path": ["N0", "N3", "N4"], "total_weight": 12}',
    'Let me think...\nThe shortest path is N0 -> N1 -> N4.\n\n{"path": ["N0", "N1", "N4"], "total_weight": 8}',
    'No JSON here',
    '{"path": ["N0", "N4"], "total_weight": 5}\n\nActually wait, let me reconsider.\n\n{"path": ["N0", "N2", "N4"], "total_weight": 4}',
]
for i, r in enumerate(test_responses):
    print(f'Test {i}: {extract_answer(r)}')

print()

# Test all three graph sizes
for n in [5, 20, 50]:
    g = build_graph(n_nodes=n, avg_degree=3 if n > 5 else 2, seed=RNG_SEED)
    source, target = 'N0', f'N{n-1}'
    path, weight = dijkstra_path(g, source, target)
    elapsed = time_dijkstra(g, source, target, runs=100)
    text = graph_to_text(g)
    n_edges = len(text.splitlines())
    print(f'{n}-node graph: {n_edges} edges, path = {" -> ".join(path)}, weight = {weight}, dijkstra = {elapsed*1e6:.1f} us')

print()
print('All non-API code paths working.')

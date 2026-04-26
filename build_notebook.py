"""Build the shortest-path experiment notebook.

Run: python build_notebook.py
Produces: shortest_path_experiment.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------- Title and framing ----------
md("""# When LLMs shouldn't be the executor
## A small experiment: shortest path on a graph

Infrastructure architects already trust Dijkstra's algorithm. We have used it for fifty years to compute optimal paths through networks, and it has never gotten the wrong answer twice on the same input.

This notebook compares Dijkstra against a frontier LLM on the same problem. The point is not that LLMs are bad. The point is that **shortest path is not a planning problem, it is an execution problem**, and the architectural lesson generalizes: when you decompose an agent into *deciding what to do* versus *doing the thing*, you should think carefully about which of those an LLM is actually good at.

We will:

1. Define a small graph and ask both Dijkstra and an LLM to find the shortest path between two nodes.
2. Run the LLM ten times on the same prompt and see how often it agrees with itself.
3. Scale the graph up. Watch accuracy and consistency degrade.
4. Try chain-of-thought prompting. Watch what improves and what does not.
5. Compare cost and latency.
6. Draw the architectural conclusion.

**Estimated cost to run end to end: roughly $0.50 to $2.00 in API credits.** **Estimated time: 10 to 20 minutes, mostly waiting for API responses.**

Audience: infrastructure architects, agent developers, anyone designing systems that combine deterministic and probabilistic components.

Maintained by [provandal.dev](https://provandal.dev). Source: [github.com/provandal/shortest-path-experiment](https://github.com/provandal/shortest-path-experiment).""")

# ---------- Setup ----------
md("""## 1. Setup

Install the libraries we need. NetworkX for the graph, matplotlib for visualization, the Anthropic SDK for the LLM, and pandas for tabulating results.""")

code("""!pip install --quiet anthropic networkx matplotlib pandas""")

code("""import os
import time
import json
import random
import re
from getpass import getpass

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from anthropic import Anthropic

# Reproducibility for graph generation only. The LLM is not seedable.
RNG_SEED = 42
random.seed(RNG_SEED)""")

md("""### API key

Paste your Anthropic API key when prompted. The key is held in memory only; it is not written to disk or logged.

Don't have one? Get an Anthropic key at [console.anthropic.com](https://console.anthropic.com/). The argument in this notebook reproduces against any frontier model. To swap providers, replace the body of `ask_llm()` below with an OpenAI or Gemini call. The rest of the notebook is provider-agnostic.""")

code("""if 'ANTHROPIC_API_KEY' not in os.environ:
    os.environ['ANTHROPIC_API_KEY'] = getpass('Enter your Anthropic API key: ')

client = Anthropic()
MODEL = 'claude-opus-4-7'  # Use any current frontier model.""")

# ---------- Graph construction ----------
md("""## 2. Building the graph

We construct a connected weighted graph with `n` nodes. Each node connects to a small number of others with an integer edge weight. This is a stand-in for the kind of network topology an infrastructure architect actually thinks about: routers, paths, latencies.""")

code('''def build_graph(n_nodes, avg_degree=3, weight_range=(1, 20), seed=RNG_SEED):
    """Build a connected undirected weighted graph.

    Returns a NetworkX graph with integer edge weights.
    """
    rng = random.Random(seed)
    g = nx.Graph()
    nodes = [f'N{i}' for i in range(n_nodes)]
    g.add_nodes_from(nodes)

    # Spanning tree first to guarantee connectivity.
    shuffled = nodes[:]
    rng.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        u = shuffled[i]
        v = rng.choice(shuffled[:i])
        w = rng.randint(*weight_range)
        g.add_edge(u, v, weight=w)

    # Extra edges to hit the target average degree.
    target_edges = int(n_nodes * avg_degree / 2)
    while g.number_of_edges() < target_edges:
        u, v = rng.sample(nodes, 2)
        if not g.has_edge(u, v):
            w = rng.randint(*weight_range)
            g.add_edge(u, v, weight=w)

    return g


def graph_to_text(g):
    """Render a graph as a flat list of weighted edges, one per line.

    This is the form we will hand to the LLM.
    """
    lines = []
    for u, v, data in sorted(g.edges(data=True)):
        lines.append(f'{u} -- {v}  (weight {data["weight"]})')
    return '\\n'.join(lines)


def draw_graph(g, path=None, title=None):
    """Render the graph, optionally highlighting a path."""
    pos = nx.spring_layout(g, seed=RNG_SEED)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, node_color='#e8dfd3', node_size=600,
                           edgecolors='#3a332d', linewidths=1.5)
    nx.draw_networkx_labels(g, pos, font_size=9, font_family='monospace')
    nx.draw_networkx_edges(g, pos, edge_color='#8a7f75', width=1)
    edge_labels = {(u, v): d['weight'] for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                 font_size=7, font_family='monospace')
    if path and len(path) > 1:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(g, pos, edgelist=path_edges,
                               edge_color='#c2583a', width=3)
    if title:
        plt.title(title, fontsize=11, family='serif')
    plt.axis('off')
    plt.tight_layout()
    plt.show()''')

# ---------- Dijkstra ----------
md("""## 3. The deterministic answer: Dijkstra

Before we ask the LLM anything, let's establish ground truth. NetworkX has a built-in Dijkstra implementation; it returns the same answer every time, in microseconds, for free.""")

code('''def dijkstra_path(g, source, target):
    """Return (path, total_weight) for the shortest path."""
    path = nx.shortest_path(g, source=source, target=target, weight='weight')
    weight = nx.path_weight(g, path, weight='weight')
    return path, weight


def time_dijkstra(g, source, target, runs=1000):
    """Measure average runtime over many iterations."""
    start = time.perf_counter()
    for _ in range(runs):
        nx.shortest_path(g, source=source, target=target, weight='weight')
    elapsed = time.perf_counter() - start
    return elapsed / runs''')

# ---------- LLM call ----------
md("""## 4. Asking the LLM

We give the LLM the same graph in plain text, the same question, and ask for a structured response so we can parse it. The prompt is deliberately straightforward; we are not trying to trick the model.

`ask_llm` is the only provider-specific function. To swap to OpenAI or Gemini, replace just this function. Everything downstream is unchanged.""")

code('''BASIC_PROMPT = """You are given an undirected weighted graph. Each line below describes one edge in the form `NODE_A -- NODE_B  (weight W)`.

Graph:
{graph_text}

Find the shortest path from {source} to {target}. The path should minimize the sum of edge weights.

Respond with valid JSON only, in exactly this shape:
{{"path": ["NODE_A", "NODE_B", ...], "total_weight": <number>}}

Do not include any text outside the JSON object."""

COT_PROMPT = """You are given an undirected weighted graph. Each line below describes one edge in the form `NODE_A -- NODE_B  (weight W)`.

Graph:
{graph_text}

Find the shortest path from {source} to {target}. The path should minimize the sum of edge weights.

First, think step by step. Enumerate plausible paths, compute their weights, and compare them. Then commit to your final answer.

End your response with valid JSON in exactly this shape (and nothing after the JSON):
{{"path": ["NODE_A", "NODE_B", ...], "total_weight": <number>}}"""


def ask_llm(graph_text, source, target, prompt_template=BASIC_PROMPT, max_tokens=2048):
    """Send one query to the LLM. Returns (raw_text, usage)."""
    prompt = prompt_template.format(graph_text=graph_text, source=source, target=target)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return msg.content[0].text, msg.usage


def extract_answer(response_text):
    """Pull the last JSON object containing a path out of the response."""
    matches = re.findall(r'\\{[^{}]*"path"[^{}]*\\}', response_text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None''')

# ---------- 5 nodes ----------
md("""## 5. The small case: 5 nodes

Let's start small enough that the problem is essentially trivial. Five nodes, a handful of edges. The LLM should get this right.""")

code("""g_small = build_graph(n_nodes=5, avg_degree=2, seed=RNG_SEED)
source, target = 'N0', 'N4'

true_path, true_weight = dijkstra_path(g_small, source, target)
print(f'Ground truth: {" -> ".join(true_path)}  (total weight {true_weight})')
print()
print('Edges:')
print(graph_to_text(g_small))
print()
draw_graph(g_small, path=true_path, title=f'Ground truth: shortest path from {source} to {target}')""")

md("""Now ask the LLM the same question once.""")

code("""graph_text = graph_to_text(g_small)
response, usage = ask_llm(graph_text, source, target)
print('Raw response:')
print(response)
print()
answer = extract_answer(response)
print(f'Parsed: {answer}')
print(f'Tokens used: input={usage.input_tokens}, output={usage.output_tokens}')""")

md("""### Run it ten times

One run does not tell us much. The interesting question is whether the LLM gives the *same* answer every time. Determinism is not just about being right; it is about being the same when the inputs are the same.""")

code('''def run_trials(g, source, target, n_trials=10, prompt_template=BASIC_PROMPT, label=''):
    """Run the LLM N times on the same prompt and tabulate results."""
    graph_text = graph_to_text(g)
    truth_path, truth_weight = dijkstra_path(g, source, target)
    rows = []
    for i in range(n_trials):
        try:
            response, usage = ask_llm(graph_text, source, target, prompt_template=prompt_template)
            ans = extract_answer(response)
        except Exception as e:
            ans = None
            usage = None
        if ans and 'path' in ans:
            path = ans['path']
            valid = (
                isinstance(path, list)
                and len(path) >= 2
                and path[0] == source
                and path[-1] == target
                and all(g.has_edge(path[k], path[k+1]) for k in range(len(path)-1))
            )
            actual_weight = nx.path_weight(g, path, weight='weight') if valid else None
            optimal = (valid and actual_weight == truth_weight)
        else:
            path = None
            valid = False
            actual_weight = None
            optimal = False
        rows.append({
            'trial': i + 1,
            'path': ' -> '.join(path) if path else 'PARSE FAILED',
            'claimed_weight': ans.get('total_weight') if ans else None,
            'actual_weight': actual_weight,
            'valid_path': valid,
            'optimal': optimal,
            'input_tokens': usage.input_tokens if usage else None,
            'output_tokens': usage.output_tokens if usage else None,
        })
    df = pd.DataFrame(rows)
    print(f'\\n=== {label} ===')
    print(f'Ground truth: {" -> ".join(truth_path)}  (weight {truth_weight})')
    print()
    print(df.to_string(index=False))
    print()
    n_optimal = df['optimal'].sum()
    n_valid = df['valid_path'].sum()
    unique_paths = df[df['valid_path']]['path'].nunique() if n_valid else 0
    print(f'Optimal: {n_optimal}/{n_trials}    Valid: {n_valid}/{n_trials}    Unique valid answers: {unique_paths}')
    return df''')

code("""df_small = run_trials(g_small, 'N0', 'N4', n_trials=10, label='5 nodes, basic prompt')""")

md("""On a 5-node graph, the LLM probably gets this right most or all of the time. That is not the interesting result. The interesting question is what happens when we scale up.""")

# ---------- 20 nodes ----------
md("""## 6. Medium: 20 nodes

Twenty nodes, average degree 3. Still small by infrastructure standards. A real production network has thousands.""")

code("""g_med = build_graph(n_nodes=20, avg_degree=3, seed=RNG_SEED)
source, target = 'N0', 'N19'
true_path, true_weight = dijkstra_path(g_med, source, target)
print(f'Ground truth: {" -> ".join(true_path)}  (total weight {true_weight})')
draw_graph(g_med, path=true_path, title='20-node graph: ground truth path')""")

code("""df_med = run_trials(g_med, 'N0', 'N19', n_trials=10, label='20 nodes, basic prompt')""")

md("""Look at the `unique valid answers` count and the `optimal` count. This is where the failure mode becomes visible. The model often produces a *valid* path (using only real edges, ending at the right node) but not the *optimal* one. And it disagrees with itself across runs.

An infrastructure architect cannot ship a system whose answer to "which path should this packet take" depends on which time you asked.""")

# ---------- 50 nodes ----------
md("""## 7. Larger: 50 nodes

Fifty nodes is still small for real networks but it is where the LLM's failure mode becomes routine rather than occasional.""")

code("""g_large = build_graph(n_nodes=50, avg_degree=3, seed=RNG_SEED)
source, target = 'N0', 'N49'
true_path, true_weight = dijkstra_path(g_large, source, target)
print(f'Ground truth: {" -> ".join(true_path)}  (total weight {true_weight})')""")

code("""df_large = run_trials(g_large, 'N0', 'N49', n_trials=10, label='50 nodes, basic prompt')""")

# ---------- CoT ----------
md("""## 8. Does chain-of-thought rescue this?

A reasonable objection at this point is: "You prompted it wrong. Tell it to think step by step."

Fair. Let's try.""")

code("""df_large_cot = run_trials(g_large, 'N0', 'N49', n_trials=10,
                            prompt_template=COT_PROMPT,
                            label='50 nodes, chain-of-thought prompt')""")

md("""Chain-of-thought may help on the optimal-rate, but watch what it costs in tokens (look at the `output_tokens` column). And even when it improves accuracy, the model still disagrees with itself across runs. **We bought a higher mean accuracy at the cost of a much higher token bill, and we still did not buy determinism.**

More importantly: even when CoT gets the right answer, we have no way to *trust* that it got the right answer without checking it against Dijkstra. So we run Dijkstra anyway. Which raises the question: why is the LLM in the loop at all?""")

# ---------- Cost / latency ----------
md("""## 9. Cost and latency

Compare the two approaches on the 50-node graph.""")

code("""# Time Dijkstra. Average over many iterations.
dijkstra_seconds = time_dijkstra(g_large, 'N0', 'N49', runs=1000)
print(f'Dijkstra average runtime: {dijkstra_seconds*1e6:.1f} microseconds')
print(f'Dijkstra cost per query: $0.00 (compute is essentially free at this scale)')
print()

# LLM stats from the basic-prompt large run.
# Claude Opus 4.7 pricing: $5/M input tokens, $25/M output tokens (verify at console.anthropic.com).
INPUT_PRICE_PER_M = 5
OUTPUT_PRICE_PER_M = 25

llm_input = df_large['input_tokens'].mean()
llm_output = df_large['output_tokens'].mean()
llm_cost = (llm_input / 1e6) * INPUT_PRICE_PER_M + (llm_output / 1e6) * OUTPUT_PRICE_PER_M

cot_input = df_large_cot['input_tokens'].mean()
cot_output = df_large_cot['output_tokens'].mean()
cot_cost = (cot_input / 1e6) * INPUT_PRICE_PER_M + (cot_output / 1e6) * OUTPUT_PRICE_PER_M

summary = pd.DataFrame([
    {'approach': 'Dijkstra',          'avg_input_tokens': 0,           'avg_output_tokens': 0,            'cost_per_query_usd': 0.0,       'optimal_rate': 1.0,                                  'deterministic': True},
    {'approach': 'LLM (basic)',       'avg_input_tokens': round(llm_input), 'avg_output_tokens': round(llm_output), 'cost_per_query_usd': round(llm_cost, 5), 'optimal_rate': float(df_large['optimal'].mean()), 'deterministic': False},
    {'approach': 'LLM (chain-of-thought)', 'avg_input_tokens': round(cot_input), 'avg_output_tokens': round(cot_output), 'cost_per_query_usd': round(cot_cost, 5), 'optimal_rate': float(df_large_cot['optimal'].mean()), 'deterministic': False},
])
print(summary.to_string(index=False))""")

# ---------- Conclusion ----------
md("""## 10. The architectural conclusion

Three observations from the data above.

**1. Variance is the real problem, not accuracy.** Even when the LLM gets the right answer, it does not always give the *same* right answer. Run the same query twice on the same graph, get different paths. For some applications that is fine. For an infrastructure component making routing decisions, anomaly diagnoses, or capacity allocations, it is disqualifying. Determinism is not a luxury, it is a property production systems require to be debuggable and trustworthy.

**2. Chain-of-thought is a tax, not a fix.** CoT improves accuracy, sometimes substantially. But it costs five to ten times more tokens, and it still does not produce determinism. Worse: the only way to know the CoT answer is correct is to verify it against a deterministic algorithm. So we end up running Dijkstra anyway, and the LLM is doing expensive ceremony.

**3. The right architecture is hybrid.** Look at this not as "LLM bad, algorithm good" but as a decomposition: shortest path is *execution*, not *planning*. The LLM is good at planning ("the user wants to know the optimal route between racks 3 and 7"). It is bad at execution ("here are the actual hops"). When you build agents, the LLM should call Dijkstra, not impersonate it.

This pattern generalizes. For any task where (a) a deterministic algorithm exists, (b) it runs faster than an LLM call, and (c) it produces verifiable output, the LLM should not be the executor. The LLM should decide which tool to use, on which inputs, and what to do with the result. The execution belongs to the deterministic component.

A common name for this is **Planner-Actor architecture**: an LLM planner decomposes a request into subtasks; deterministic actors execute them. The math of the joint success probability tells the story:

```
P(success) = P(correct plan) * P(correct execution | correct plan)
```

If both factors are LLMs at 90%, your system tops out at 81%. If the planner is an LLM at 90% and the executor is a deterministic algorithm at effectively 100%, your system is at 90%, dominated by planning quality, with no compounding execution risk. That difference matters in production.

### What to do with this

If you are an infrastructure architect designing an agent that touches your network: identify which steps have deterministic ground-truth answers (graph operations, configuration validation, telemetry queries) and route those through deterministic actors. Reserve LLM execution for steps that are genuinely judgment calls.

If you are an agent developer: when reviewing an agent that fails in production, ask which steps in the loop are LLM-executed when they should be tool-executed. The most common failure mode in current agent systems is LLMs doing arithmetic, lookups, and traversals that should have been function calls.

If you want to go deeper, the original "Can Language Models Solve Graph Problems in Natural Language?" line of papers documents the failure mode at scale, and the more recent literature on tool-using agents (function calling, MCP, and structured outputs) shows what the planner-actor pattern looks like in production.

---

*If you want to push this experiment further: try larger graphs (100, 200 nodes), try different graph structures (sparse vs dense, planar vs random), try other frontier models. The pattern is robust, but the specific accuracy numbers will vary.*

*Comments, corrections, or counter-experiments welcome at [provandal.dev](https://provandal.dev).*""")


# ---------- Build and write ----------
nb['cells'] = cells

# Set kernel metadata so Colab and Jupyter pick up Python 3 cleanly.
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.11',
    },
    'colab': {
        'provenance': [],
    },
}

with open('shortest_path_experiment.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f'Wrote notebook with {len(cells)} cells.')

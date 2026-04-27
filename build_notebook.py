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
2. Run the LLM repeatedly on the same prompt and see how often it agrees with itself.
3. Scale the graph up to 100 nodes. Watch what holds up and what does not.
4. Try chain-of-thought prompting. Watch what improves and what does not.
5. Show the right architecture: have the LLM call Dijkstra as a tool.
6. Compare cost and latency.
7. Draw the architectural conclusion.

**Estimated cost to run end to end: roughly $3 to $6 in API credits.** **Estimated time: 20 to 40 minutes, mostly waiting for API responses.**

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
from anthropic import Anthropic, AuthenticationError

# Reproducibility for graph generation only. The LLM is not seedable.
RNG_SEED = 42
random.seed(RNG_SEED)""")

md("""### API key

Paste your Anthropic API key when prompted. The key is held in memory only; it is not written to disk or logged.

Don't have one? Get an Anthropic key at [console.anthropic.com](https://console.anthropic.com/). The argument in this notebook reproduces against any frontier model. To swap providers, replace the body of `ask_llm()` below with an OpenAI or Gemini call. The rest of the notebook is provider-agnostic.""")

code("""MODEL = 'claude-opus-4-7'  # Use any current frontier model.


def _key_works(key):
    \"\"\"One-token probe to verify the key. Costs a fraction of a cent.\"\"\"
    try:
        Anthropic(api_key=key).messages.create(
            model=MODEL,
            max_tokens=1,
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        return True
    except AuthenticationError:
        return False


client = None
for attempt in range(3):
    key = (os.environ.get('ANTHROPIC_API_KEY') or '').strip()
    if not key:
        key = getpass('Enter your Anthropic API key: ').strip()
    if _key_works(key):
        os.environ['ANTHROPIC_API_KEY'] = key
        client = Anthropic()
        print('Key validated.')
        break
    print('Key rejected by the API. Verify it at https://console.anthropic.com/settings/keys')
    os.environ.pop('ANTHROPIC_API_KEY', None)

if client is None:
    raise RuntimeError('Could not authenticate with Anthropic after 3 attempts.')""")

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

md("""Let's see what these produce. We'll build a small 5-node graph, print its edge list (the same format we'll later hand to the LLM), and draw it.""")

code("""demo_g = build_graph(n_nodes=5, avg_degree=2, seed=RNG_SEED)
print(f'Generated a graph with {demo_g.number_of_nodes()} nodes and {demo_g.number_of_edges()} edges.')
print()
print('Edges:')
print(graph_to_text(demo_g))
print()
draw_graph(demo_g, title='Demo: 5-node weighted graph')""")

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

md("""Run Dijkstra on the demo graph from above and visualize the result. The red edges are the shortest path; the timing tells us what we're competing against.""")

code("""demo_path, demo_weight = dijkstra_path(demo_g, 'N0', 'N4')
print(f'Shortest path from N0 to N4: {" -> ".join(demo_path)}')
print(f'Total weight: {demo_weight}')
print()

avg_runtime = time_dijkstra(demo_g, 'N0', 'N4', runs=1000)
print(f'Average Dijkstra runtime: {avg_runtime*1e6:.1f} microseconds per query')

draw_graph(demo_g, path=demo_path, title='Dijkstra shortest path: N0 to N4')""")

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


def ask_llm(graph_text, source, target, prompt_template=BASIC_PROMPT, max_tokens=4096):
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

md("""At 20 nodes a frontier model is still solid — typically 100% optimal, with most runs returning the same path. The architectural problem has not yet shown up. But notice the input_tokens column: it has roughly tripled from the 5-node case. The cost per query is climbing fast even when accuracy is not falling.""")

# ---------- 50 nodes ----------
md("""## 7. Larger: 50 nodes

Fifty nodes is still small for real networks. The prompt is now around 1,300 input tokens and the model has many more candidate paths to evaluate. A frontier model still tends to handle this — let's confirm.""")

code("""g_large = build_graph(n_nodes=50, avg_degree=3, seed=RNG_SEED)
source, target = 'N0', 'N49'
true_path, true_weight = dijkstra_path(g_large, source, target)
print(f'Ground truth: {" -> ".join(true_path)}  (total weight {true_weight})')""")

code("""df_large = run_trials(g_large, 'N0', 'N49', n_trials=10, label='50 nodes, basic prompt')""")

# ---------- CoT ----------
md("""## 8. Does chain-of-thought help?

A reasonable objection at this point is: "You prompted it wrong. Tell the model to think step by step and it will do even better."

Let's run the same 50-node graph with a chain-of-thought prompt and see what changes.""")

code("""df_large_cot = run_trials(g_large, 'N0', 'N49', n_trials=10,
                            prompt_template=COT_PROMPT,
                            label='50 nodes, chain-of-thought prompt')""")

md("""At 50 nodes the basic prompt was already at 100% optimal, so CoT has nothing to rescue here. But notice the `output_tokens` column: CoT spent meaningfully more output tokens for no accuracy gain. We will see in section 9 whether that pattern holds, or even reverses, at scale.

The deeper point — independent of any single run's numbers — is that we have no way to *verify* a CoT answer is correct without checking it against Dijkstra. So we end up running Dijkstra anyway, and the LLM is doing expensive ceremony.""")

# ---------- 100 nodes ----------
md("""## 9. The largest case: 100 nodes

Real production networks have thousands of nodes. We are nowhere near that. But 100 nodes is the point where the prompt starts to feel large, where the basic-prompt LLM finally drops below 100% optimal, and where the chain-of-thought tax is most visible. We will run both prompts here, with twenty trials each instead of ten, so the comparison is statistically meaningful.""")

code("""g_xl = build_graph(n_nodes=100, avg_degree=3, seed=RNG_SEED)
source, target = 'N0', 'N99'
true_path, true_weight = dijkstra_path(g_xl, source, target)
print(f'Ground truth: {" -> ".join(true_path)}  (total weight {true_weight})')
print(f'Edges in the prompt: {g_xl.number_of_edges()}')""")

md("""### Basic prompt on 100 nodes""")

code("""df_xl = run_trials(g_xl, 'N0', 'N99', n_trials=20, label='100 nodes, basic prompt')""")

md("""### Chain-of-thought on 100 nodes

Same graph, same question, but with the step-by-step prompt. Watch the `output_tokens` column — this is the most expensive single block in the notebook.""")

code("""df_xl_cot = run_trials(g_xl, 'N0', 'N99', n_trials=20,
                          prompt_template=COT_PROMPT,
                          label='100 nodes, chain-of-thought prompt')""")

md("""This is where the story gets interesting. The basic prompt holds up well — most of the 20 trials are optimal. But **chain-of-thought lagged the basic prompt by roughly 20 percentage points on our run**, and the same direction held in our smaller n=10 pilot. Whatever "thinking step by step" does for the model here, it is not buying us a better shortest path.

Both prompts still occasionally produce the wrong answer, with no signal at runtime to tell which runs are wrong. Meanwhile, Dijkstra has been producing the same correct answer in microseconds, for free, since section 3.""")

# ---------- Tool use ----------
md("""## 10. The right architecture: let the LLM call Dijkstra

Everything so far has been the LLM trying to *be* Dijkstra. That is the wrong job.

Modern LLM APIs let us split the work cleanly: expose Dijkstra as a tool, hand the model only the tool's *description* (not the graph itself), and ask the question. The model decides whether and how to call the tool. Our code runs Dijkstra and returns the result. The model formats the answer.

This is the planner-actor pattern with the wires visible. In production, the same structure is what an **MCP server** gives you — a standardized way to expose tools to LLM agents. The architectural point is the same whether the wiring is MCP, OpenAI function calling, or the Anthropic tool-use API used here.""")

code('''SHORTEST_PATH_TOOL = {
    'name': 'shortest_path',
    'description': "Compute the shortest path between two nodes in the network using Dijkstra's algorithm. Returns the optimal path and its total edge weight.",
    'input_schema': {
        'type': 'object',
        'properties': {
            'source': {'type': 'string', 'description': 'Starting node identifier, e.g. "N0"'},
            'target': {'type': 'string', 'description': 'Destination node identifier, e.g. "N99"'},
        },
        'required': ['source', 'target'],
    },
}


def ask_llm_with_tool(g, source, target, max_iters=5):
    """Multi-turn loop: the LLM may request tool calls; we execute and reply.

    The graph is NOT in the prompt. It lives in our process, behind the
    tool. The model only learns about the network through tool calls.
    """
    messages = [{
        'role': 'user',
        'content': f'On our test network, what is the shortest path from {source} to {target}? Use any available tools.',
    }]
    total_input = total_output = 0
    tool_called = False
    claimed_path = None
    claimed_weight = None

    for _ in range(max_iters):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=[SHORTEST_PATH_TOOL],
            messages=messages,
        )
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        if response.stop_reason == 'tool_use':
            tool_called = True
            messages.append({'role': 'assistant', 'content': response.content})
            tool_results = []
            for block in response.content:
                if block.type == 'tool_use' and block.name == 'shortest_path':
                    path, weight = dijkstra_path(g, block.input['source'], block.input['target'])
                    claimed_path = path
                    claimed_weight = weight
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.id,
                        'content': json.dumps({'path': path, 'total_weight': weight}),
                    })
            messages.append({'role': 'user', 'content': tool_results})
            continue

        text = ''.join(b.text for b in response.content if b.type == 'text')
        return text, total_input, total_output, claimed_path, claimed_weight, tool_called

    return None, total_input, total_output, claimed_path, claimed_weight, tool_called''')

md("""### One run: see the LLM call the tool

Run it once on the 100-node graph and look at what comes back.""")

code("""text, tin, tout, path, weight, called = ask_llm_with_tool(g_xl, 'N0', 'N99')
print('Final answer from the model:')
print(text)
print()
print(f'Tool called: {called}')
print(f'Path returned by tool: {" -> ".join(path) if path else None}')
print(f'Total weight: {weight}')
print(f'Tokens used: input={tin}, output={tout}')""")

md("""### Twenty trials: how does this compare to the LLM doing it itself?

Same setup as before — 20 runs on the 100-node graph — but now the LLM has the tool available.""")

code("""truth_path, truth_weight = dijkstra_path(g_xl, 'N0', 'N99')
rows = []
for i in range(20):
    text, tin, tout, path, weight, called = ask_llm_with_tool(g_xl, 'N0', 'N99')
    optimal = (path is not None and weight == truth_weight)
    rows.append({
        'trial': i + 1,
        'tool_called': called,
        'path': ' -> '.join(path) if path else 'NO TOOL CALL',
        'weight': weight,
        'optimal': optimal,
        'input_tokens': tin,
        'output_tokens': tout,
    })
df_xl_tool = pd.DataFrame(rows)
print(df_xl_tool.to_string(index=False))
print()
print(f'Optimal: {df_xl_tool[\"optimal\"].sum()}/{len(df_xl_tool)}    Tool called: {df_xl_tool[\"tool_called\"].sum()}/{len(df_xl_tool)}')""")

md("""Compare this to the basic and chain-of-thought runs on the same graph: same problem, but the model now gets the verifiably correct answer every time it calls the tool, at a fraction of the token cost. The LLM is doing the part it is good at — *deciding what to compute and framing the answer* — and Dijkstra is doing the part Dijkstra is good at.""")

# ---------- Cost / latency ----------
md("""## 11. Cost and latency

Now compare every approach we have run, side by side.""")

code("""# Claude Opus 4.7 pricing: $5/M input tokens, $25/M output tokens (verify at console.anthropic.com).
INPUT_PRICE_PER_M = 5
OUTPUT_PRICE_PER_M = 25


def _row(approach, df):
    inp = df['input_tokens'].mean()
    out = df['output_tokens'].mean()
    cost = (inp / 1e6) * INPUT_PRICE_PER_M + (out / 1e6) * OUTPUT_PRICE_PER_M
    return {
        'approach': approach,
        'avg_input_tokens': round(inp),
        'avg_output_tokens': round(out),
        'cost_per_query_usd': round(cost, 5),
        'optimal_rate': round(float(df['optimal'].mean()), 2),
        'deterministic': False,
    }


# Time Dijkstra at the largest scale. Still microseconds.
dijkstra_seconds = time_dijkstra(g_xl, 'N0', 'N99', runs=1000)
print(f'Dijkstra average runtime at 100 nodes: {dijkstra_seconds*1e6:.1f} microseconds per query')
print(f'Dijkstra cost per query: $0.00 (compute is essentially free at this scale)')
print()

summary = pd.DataFrame([
    {'approach': 'Dijkstra (any scale)', 'avg_input_tokens': 0, 'avg_output_tokens': 0, 'cost_per_query_usd': 0.0, 'optimal_rate': 1.0, 'deterministic': True},
    _row('LLM basic, 5 nodes',     df_small),
    _row('LLM basic, 20 nodes',    df_med),
    _row('LLM basic, 50 nodes',    df_large),
    _row('LLM CoT,   50 nodes',    df_large_cot),
    _row('LLM basic, 100 nodes',   df_xl),
    _row('LLM CoT,   100 nodes',   df_xl_cot),
    _row('LLM + tool, 100 nodes',  df_xl_tool),
])
print(summary.to_string(index=False))""")

# ---------- Conclusion ----------
md("""## 12. The architectural conclusion

Three observations from the data above.

**1. The LLM is competent here — but not verifiable.** A frontier model gets this right 75 to 100 percent of the time, even at 100 nodes. The problem is not capability. The problem is that on the runs where the model is wrong, you have no signal at runtime to tell which run is which. To know, you would have to verify against Dijkstra — at which point, why was the LLM in the loop at all? "Mostly correct" is not a property production systems can ship on when a deterministic alternative exists.

**2. Chain-of-thought is not a free upgrade.** CoT cost more tokens at every scale and produced equal-or-worse accuracy at 100 nodes than the basic prompt at the same scale. "Tell it to think step by step" is a tax that may or may not buy you anything, with no way to predict in advance which. It is also not a path to determinism: the model still disagrees with itself across runs, and you still cannot trust any single answer without verification.

**3. The right architecture is hybrid — and section 10 shows it concretely.** Expose Dijkstra as a tool and let the LLM *call* it instead of *impersonate* it. The model gets the verifiably correct answer at a fraction of the token cost, and uses its strengths — choosing what to compute, framing the result — instead of fighting them. This is a decomposition: shortest path is *execution*, not *planning*. The LLM is good at planning ("the user wants the optimal route between racks 3 and 7"). It is bad at execution ("here are the actual hops"). When you build agents, the LLM should call Dijkstra, not impersonate it.

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

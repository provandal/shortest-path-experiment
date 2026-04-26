# Shortest path: when LLMs shouldn't be the executor

A small experiment comparing Dijkstra's algorithm against a frontier LLM on shortest-path queries over weighted graphs. Built for infrastructure architects and agent developers thinking about when to use deterministic vs. probabilistic components in a system.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/provandal/shortest-path-experiment/blob/main/shortest_path_experiment.ipynb)

## What's in here

`shortest_path_experiment.ipynb` walks through:

1. Building weighted graphs at four scales (5, 20, 50, 100 nodes).
2. Running NetworkX's Dijkstra to establish ground truth.
3. Asking a frontier LLM the same question, in plain text.
4. Repeating each query ten times to measure consistency, not just accuracy.
5. Trying chain-of-thought prompting to see what it improves and what it doesn't.
6. Comparing cost and latency.
7. Drawing the architectural conclusion: shortest path is execution, not planning.

Read the [companion post on provandal.dev](https://provandal.dev/) for the longer version of the argument.

## Running

Click the Colab badge above. You will need an Anthropic API key from [console.anthropic.com](https://console.anthropic.com/). Estimated cost: $2 to $5 in API credits to run the whole notebook.

To use a different provider (OpenAI, Gemini, etc.), replace just the body of the `ask_llm()` function. The rest of the notebook is provider-agnostic.

## Why this matters

When you build agent systems, the LLM should plan and the deterministic algorithm should execute. The math:

```
P(success) = P(correct plan) * P(correct execution | correct plan)
```

If both factors are LLMs at 90%, the system tops out at 81%. If the planner is at 90% and the executor is deterministic at ~100%, the system is at 90%, dominated by planning quality. Compounding execution risk goes away. The notebook makes this concrete on a problem infrastructure architects already trust the answer to.

## License

MIT. Use, fork, extend, write a counter-experiment. Linkbacks appreciated but not required.

---

Maintained at [provandal.dev](https://provandal.dev). Comments, corrections, and counter-experiments welcome.

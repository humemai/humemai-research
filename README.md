# humemai

[![DOI](https://zenodo.org/badge/614376180.svg)](https://zenodo.org/doi/10.5281/zenodo.10876440)
[![PyPI
version](https://badge.fury.io/py/humemai.svg)](https://badge.fury.io/py/humemai)

<div align="center">
    <img src="./figures/humemai-with-text-below.png" alt="Image" style="width: 50%; max-width: 600px;">
</div>

- Built on a cognitive architecture
  - Functions as the brain 🧠 of your own agent
  - It has human-like short-term and long-term memory
- The memory is represented as a knowledge graph
  - A graph database (JanusGraph + Cassandra) is used for persistence and fastgraph
    traversal
  - The user does not have to know graph query languages, e.g., Gremlin, since HumemAI
    handles read from / write to the database
- The interface of HumemAI is natural language, just like a chatbot.
  - This requires the Text2Graph and Graph2Text modules, which are part of HumemAI
- Everything is open-sourced, including the database

## Installation

The `humemai` python package can already be found in [the PyPI server](https://pypi.org/project/humemai/)

```sh
pip install humemai
```

or

```sh
pip install 'humemai[dev]'
```

for the development

Supports python>=3.10

## Text2Graph and Graph2Text

These two modules are critical in HumemAI. At the moment, they are achieved with [LLM
prompting](./humemai/prompt/), which is not ideal. They'll be replaced with Transformer
and GNN based neural networks.

## Example

- [`example-janus-agent.ipynb`](./examples/janus-graph-parse-text/example-janus-agent.ipynb):
  This Jupyter Notebook reads the Harry Potter book paragraph by paragraph and turns it
  into a knowledge graph. Text2Graph and Graph2Text are achieved with LLM prompting.
- More to come ...

## Visualizaing Graph

Use [`JanusGraph-Visualizer`](https://github.com/JanusGraph/janusgraph-visualizer) to
visualize the graph.

Run below:

```sh
docker run --rm -d -p 3000:3000 -p 3001:3001 --name=janusgraph-visualizer --network=host janusgraph/janusgraph-visualizer:latest
```

And open `http://localhost:3001/` on your web browser

## Work in progress

Currently this is a one-man job. [Click here to see the current
progress](https://github.com/orgs/humemai/projects/2/).

<!-- ## List of academic papers that use HumemAI

- ["A Machine With Human-Like Memory Systems"](https://arxiv.org/abs/2204.01611)
- ["A Machine with Short-Term, Episodic, and Semantic Memory
  Systems"](https://arxiv.org/abs/2212.02098)

## List of applications that use HumemAI -->

## pdoc documentation

Click on [this link](https://humemai.github.io/humemai) to see the HTML rendered
docstrings

## Contributing

Contributions are what make the open source community such an amazing place to be learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure
   code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Authors

- [Taewoon Kim](https://taewoon.kim/)

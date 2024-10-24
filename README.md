# humemai

[![DOI](https://zenodo.org/badge/614376180.svg)](https://zenodo.org/doi/10.5281/zenodo.10876440)
[![PyPI
version](https://badge.fury.io/py/humemai.svg)](https://badge.fury.io/py/humemai)

![](./figures/humemai-diagram.png)

This repo hosts a package `humemai`, a human-like memory system, modeled with knowledge
knoweldge graphs (KGs).

## Installation

The `humemai` python package can already be found in [the PyPI server](https://pypi.org/project/humemai/)

```sh
pip install humemai
```

Supports python>=3.10

## Updates

### 24-Oct-2024

- Finished implementing RDFLib-based HumemAI
  - [`humemai/rdflib/memory.py`](./humemai/rdflib/memory.py)
- Made a very sipmle example
  - [`examples/example-rdflib-agent.ipynb`](./examples/example-rdflib-agent.ipynb)
- Released v2

### 11-Aug-2024

- Used RL to learn policies, e.g., memory management, maze navigation, question answering.
- The HumemAI memories are nothing but python objects, i.e., list of quadruples
- Released v1.1.2

## TODOs

- Implement JanusGraph + Cassandra DB
  - RDF-based graph as pros and cons. Let's try out a property graph.
- Implement a very simple text2graph and graph2text based on [GraphRAG](https://github.com/microsoft/graphrag)
- Implement a sipmle image2text2graph, e.g., face recognition.

## List of academic papers that use HumemAI

- ["A Machine With Human-Like Memory Systems"](https://arxiv.org/abs/2204.01611)
- ["A Machine with Short-Term, Episodic, and Semantic Memory
  Systems"](https://arxiv.org/abs/2212.02098)
- ["Leveraging Knowledge Graph-Based Human-Like Memory Systems to Solve Partially Observable Markov Decision Processes"](https://arxiv.org/abs/2408.05861)

## List of applications that use HumemAI

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

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors

- [Taewoon Kim](https://taewoon.kim/)

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

### Python Package

The `humemai` Python package is available on the [PyPI server](https://pypi.org/project/humemai/). You can install it using `pip`:

```bash
pip install humemai
```

For development purposes, use:

```bash
pip install 'humemai[dev]'
```

**Supported Python Versions:** Python >= 3.10

### Docker Compose

To set up Docker Compose, follow these steps:

#### Update package lists

```bash
sudo apt-get update
```

#### Install Docker Compose

```bash
sudo apt-get install -y docker-compose
```

## Text2Graph and Graph2Text

These two modules are critical in HumemAI. At the moment, they are achieved with [LLM
prompting](./humemai/prompt/), which is not ideal. They'll be replaced with Transformer
and GNN based neural networks.

## Example

- [`example-janus-agent.ipynb`](./examples/harry-potter/harry-potter-agent.ipynb):
  This Jupyter Notebook reads the Harry Potter book paragraph by paragraph and turns it
  into a knowledge graph. Text2Graph and Graph2Text are achieved with LLM prompting.
- More to come ...

## Docker Compose for JanusGraph with Cassandra and Elasticsearch

This project uses a `docker-compose-cql-es.yml` file to set up a JanusGraph instance with Cassandra, Elasticsearch, and other supporting services.

### Key Points

1. Unless you instantiate the `Humemai` class with a specified `compose_file_path`, it will always use the default `docker-compose-cql-es.yml` provided in the repository.
2. Port numbers, container names, and other configurations are currently fixed. Future updates will make these configurable for running multiple instances on the same machine.
3. The Docker Compose file starts four Docker containers:
   - `janusgraph`: The main JanusGraph instance.
   - `cassandra`: The backend storage for JanusGraph.
   - `elasticsearch`: The index/search backend for JanusGraph.
   - `janusgraph-visualizer`: A visualization tool for JanusGraph.
4. Not all Docker images are the latest versions. Future work includes updating to the latest compatible versions.

### Instructions

#### Start the Containers

Run the following command in the same directory as the `docker-compose-cql-es.yml` file:

```bash
docker-compose -f docker-compose-cql-es.yml up -d
```

#### Check the Status of Containers

Verify the running containers:

```
docker ps
```

You should see containers named:

- `jce-janusgraph`
- `jce-cassandra`
- `jce-elastic`
- `jce-visualizer`

#### Access the Services

- **JanusGraph Gremlin Server:** Accessible at `localhost:8182`.
- **Elasticsearch:** Accessible at `localhost:9200`.
- **Visualizer:** Accessible at `http://localhost:3000`.

#### Stop the Containers

To stop the services:

```bash
docker-compose -f docker-compose-cql-es.yml down
```

#### Clean Up

If you want to remove the containers and associated volumes:

```bash
docker-compose -f docker-compose-cql-es.yml down --volumes
```

### Future Work

- Make port numbers, container names, and other configurations customizable.
- Update Docker images to the latest compatible versions.
- Add support for running multiple instances on the same machine.

## Visualizaing Graph

HumemAI runs four docker containers with docker-compose and one of them is visualizer.
Open your browser and type `http://localhost:3001/`. Rename "host" to `jce-janusgraph`
and query the graph.

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

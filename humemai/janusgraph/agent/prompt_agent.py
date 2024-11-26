"""PromptAgent class using JanusGraph-Cassandra as backend."""

import re
import json
from datetime import datetime
from gremlin_python.structure.graph import Vertex, Edge
from humemai.janusgraph import Humemai
from humemai.utils import disable_logger
from humemai.prompt import (
    get_hf_pipeline,
    text2graph,
    graph2text,
)


class PromptAgent:
    """Agent class using JanusGraph-Cassandra as backend.

    This is the simplest agent that uses JanusGraph-Cassandra as the backend. It
    does not use properties of the vertices and edges. It takes one text at a time,
    converts it into a knowledge graph, and saves it as short-term memory. It then
    moves the short-term memory to long-term memory. There is no forgetting mechanism
    implemented in this agent. You also can't talk to the agent. It only processes
    the input text and updates the memory.

    """

    def __init__(
        self,
        warmup_seconds: int = 10,
        remove_data_on_start: bool = True,
        num_hops_for_working_memory: int = 4,
        turn_on_logger: bool = True,
        llm_config: dict = {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "device": "cuda",
            "quantization": "16bit",
            "max_new_tokens": 1024,
        },
        text2graph_template: str = "text2graph_without_properties",
    ):
        """Initialize the agent.

        Args:
            warmup_seconds (int): The number of seconds to wait for the containers to
                warm up. Defaults to 10.
            remove_data_on_start (bool): Whether to remove all data from the database
                on start. Defaults to True.
            num_hops_for_working_memory (int): The number of hops to consider for the
                working memory. Defaults to 2.
            turn_on_logger (bool): Whether to turn on the logger. Defaults to True.
            llm_config (dict): The configuration for the Hugging Face pipeline.
                Defaults to {
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "device": "cuda",
                    "quantization": "16bit",
                }. The model, device, and quantization can be changed.
            text2graph_template (str): The template to use for text2graph. Defaults to
                "text2graph_without_properties".
        """
        self.warmup_seconds = warmup_seconds
        self.remove_data_on_start = remove_data_on_start
        self.num_hops_for_working_memory = num_hops_for_working_memory
        self.turn_on_logger = turn_on_logger
        self.llm_config = llm_config
        self.text2graph_template = text2graph_template

        self.humemai = Humemai()
        self.humemai.start_containers(warmup_seconds=self.warmup_seconds)
        self.humemai.connect()
        self.pipeline = get_hf_pipeline(
            model=self.llm_config["model"],
            device=self.llm_config["device"],
            quantization=self.llm_config["quantization"],
        )

        if self.remove_data_on_start:
            self.humemai.remove_all_data()
        if not turn_on_logger:
            disable_logger()

        # HumemAI reserved keys
        self.reserved_keys = [
            "num_recalled",
            "current_time",
            "event_time",
            "known_since",
        ]

        self.reset_working_memory()

    def reset_working_memory(self) -> None:
        """Reset the working memory of the agent."""

        self.working_memory = {
            "long_term_entities": [],
            "long_term_relations": [],
            "short_term_entities": [],
            "short_term_relations": [],
        }

    def update_working_memory(self) -> None:
        """Update the working memory of the agent."""

        if len(self.humemai.get_all_short_term_vertices()) > 0:
            (
                short_term_vertices,
                long_term_vertices,
                short_term_edges,
                long_term_edges,
            ) = self.humemai.get_working_vertices_and_edges(
                short_term_vertices=self.humemai.get_all_short_term_vertices(),
                short_term_edges=self.humemai.get_all_short_term_edges(),
                include_all_long_term=False,
                hops=self.num_hops_for_working_memory,
            )

            self.working_memory = {
                "long_term_entities": long_term_vertices,
                "long_term_relations": long_term_edges,
                "short_term_entities": short_term_vertices,
                "short_term_relations": short_term_edges,
            }

    def return_working_memory_as_dict(self) -> dict:
        """Return the working memory as a dictionary.

        This is necessary for generating the prompt with an LLM. It merges short-term
        and (partial) long-term memories.

        Returns:
            dict: The working memory as a dictionary.
        """
        entities = []
        for vertex in self.working_memory["short_term_entities"]:
            entities.append({"label": vertex.label})
        for vertex in self.working_memory["long_term_entities"]:
            entities.append({"label": vertex.label})

        relations = []
        for edge in self.working_memory["short_term_relations"]:
            relations.append(
                {
                    "source": edge.outV.label,
                    "relation": edge.label,
                    "target": edge.inV.label,
                }
            )
        for edge in self.working_memory["long_term_relations"]:
            relations.append(
                {
                    "source": edge.outV.label,
                    "relation": edge.label,
                    "target": edge.inV.label,
                }
            )

        return {"entities": entities, "relations": relations}

    def generate_graph(self, text: str) -> None:
        """Process the input text and convert it into a knowledge graph.

        Args:
            text (str): The input text to process.

        """
        prompt = text2graph(
            memory=self.return_working_memory_as_dict(),
            next_text=text,
            template=self.text2graph_template,
        )

        outputs = self.pipeline(
            prompt, max_new_tokens=self.llm_config["max_new_tokens"]
        )
        text_content = outputs[0]["generated_text"][-1]["content"]
        json_match = re.search(r"```json\n(.*?)\n```", text_content, re.DOTALL)
        try:
            json_text = json_match.group(1)  # Extract JSON content
        except AttributeError:
            return [], []

        try:
            extracted_dict = json.loads(json_text)
        except json.JSONDecodeError:
            return [], []

        if "entities" not in extracted_dict or "relations" not in extracted_dict:
            return [], []

        return extracted_dict["entities"], extracted_dict["relations"]

    def save_as_short_term_memory(self, entities: dict, relations: dict) -> None:
        """Save the entities and relations as short-term memory.

        Args:
            entities (dict): The entities to save as short-term memory.
            relations (dict): The relations to save as short-term memory.

        """
        current_time = datetime.now().isoformat(timespec="seconds")
        for entity in entities:
            label = entity.get("label")
            self.humemai.write_short_term_vertex(
                label=label, properties={"current_time": current_time}
            )

        for relation in relations:
            head_label = relation.get("source")
            # assumes that there is only one head vertex
            head_vertex = self.humemai.find_vertex_by_label(head_label)
            if len(head_vertex) == 0:
                continue
            head_vertex = head_vertex[0]

            edge_label = relation.get("relation")

            tail_label = relation.get("target")
            # assumes that there is only one tail vertex
            tail_vertex = self.humemai.find_vertex_by_label(tail_label)
            if len(tail_vertex) == 0:
                continue
            tail_vertex = tail_vertex[0]

            edge = self.humemai.write_short_term_edge(
                head_vertex=head_vertex,
                edge_label=edge_label,
                tail_vertex=tail_vertex,
                properties={"current_time": current_time},
            )

    def save_as_long_term_memory(self) -> None:
        """Move the short-term memories to the long-term memory.

        At the moment, they are all moved as "episodic". In the future, we may want to
        move some of them as "semantic" memories, or even forget some of them.

        """
        for vertex in self.working_memory["short_term_entities"]:
            self.humemai.move_short_term_vertex(vertex, "episodic")

        for edge in self.working_memory["short_term_relations"]:
            self.humemai.move_short_term_edge(edge, "episodic")

        self.humemai.remove_all_short_term()

    def step(self, text: str) -> str:
        """Process the input (text), convert it into a knowledge graph, and save it as
        short-term memory.

        Args:
            text (str): The input text to process.

        Returns:
            str: HumemAI's generated text from the working memory.

        """
        # Step 1: Process the input text and convert it into a knowledge graph.
        entities, relations = self.generate_graph(text)

        # Step 2: Save the extracted entities and relations as short-term memory.
        self.save_as_short_term_memory(entities, relations)

        # Step 3: Update the working memory.
        self.update_working_memory()

        # Step 4: move the short-term memories to the long-term memory.
        self.save_as_long_term_memory()

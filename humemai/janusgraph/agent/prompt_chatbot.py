"""Prompt Chatbot"""

import re
import json
from humemai.prompt import graph2text
from .prompt_agent import PromptAgent


class PromptChatbot(PromptAgent):
    """A chatbot that uses a language model to generate responses to user inputs."""

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
        graph2text_template: str = "graph2text_without_properties",
    ):
        """Initialize the chatbot.

        Args:
            warmup_seconds (int): The number of seconds to wait for the containers to
                warm up. Defaults to 10.
            remove_data_on_start (bool): Whether to remove all data from the database
                on start. Defaults to True.
            num_hops_for_working_memory (int): The number of hops to consider for the
                working memory. Defaults to 4.
            turn_on_logger (bool): Whether to turn on the logger. Defaults to True.
            llm_config (dict): The configuration for the Hugging Face pipeline.
                Defaults to {
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "device": "cuda",
                    "quantization": "16bit",
                }. The model, device, and quantization can be changed.
            text2graph_template (str): The template to use for text2graph. Defaults to
                "text2graph_without_properties".
            graph2text_template (str): The template to use for graph2text. Defaults to
                "graph2text_without_properties".
        """
        super().__init__(
            warmup_seconds=warmup_seconds,
            remove_data_on_start=remove_data_on_start,
            num_hops_for_working_memory=num_hops_for_working_memory,
            turn_on_logger=turn_on_logger,
            llm_config=llm_config,
            text2graph_template=text2graph_template,
        )
        self.graph2text_template = graph2text_template
        self.introduced = False

    def generate_text(self) -> str:
        """Generate text from the working memory.

        Returns:
            str: The generated text
        """
        prompt = graph2text(
            memory=self.return_working_memory_as_dict(),
            template=self.graph2text_template,
        )
        outputs = self.pipeline(
            prompt, max_new_tokens=self.llm_config["max_new_tokens"]
        )
        text_content = outputs[0]["generated_text"][-1]["content"]
        json_match = re.search(r"```json\n(.*?)\n```", text_content, re.DOTALL)
        try:
            json_text = json_match.group(1)  # Extract JSON content
        except AttributeError:
            return ""

        try:
            extracted_text = json.loads(json_text)
        except json.JSONDecodeError:
            return ""

        return extracted_text["text"]

    def introduce(self):
        """Introduce the chatbot to the user."""
        if self.introduced:
            return

        intro_text = ["Hello! I am a chatbot. What's your name?", "Nice to meet you, "]
        print(intro_text[0])
        name = input("You: ")
        self.name = name
        intro_text[1] += self.name + "!"
        print(intro_text[1])
        self.introduced = True

        intro_text = " ".join(intro_text)

        # Step 1: Process the input text and convert it into a knowledge graph.
        entities, relations = self.generate_graph(intro_text)

        # Step 2: Save the extracted entities and relations as short-term memory.
        self.save_as_short_term_memory(entities, relations)

        # Step 3: Update the working memory.
        self.update_working_memory()

    def step(self):
        """Get user input and generate a response."""
        if not self.introduced:
            self.introduce()

        # Step 1: Process the input text and convert it into a knowledge graph.
        user_input = self.name + ": " + input()

        # entities, relations = self.generate_graph(user_input)

        # # Step 2: Save the extracted entities and relations as short-term memory.
        # self.save_as_short_term_memory(entities, relations)

        # # Step 3: Update the working memory.
        # self.update_working_memory()

        # # Step 4: Generate a response using the language model.
        # response = self.generate_text()

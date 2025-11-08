"""Propmt specific functions and classes."""

import torch
import transformers
from .templates import (
    text2graph_without_properties,
    text2graph_with_properties,
    graph2text_without_properties,
    graph2text_with_properties,
)


def get_hf_pipeline(
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    device: str = "cpu",
    quantization: str = "16bit",
) -> transformers.Pipeline:
    """Get a text generation pipeline with the specified device and quantization.

    Args:
        model (str): The model to use for text generation.
            Defaults to "meta-llama/Llama-3.2-1B-Instruct".
            meta-llama/Llama-3.2-3B-Instruct,
            meta-llama/Llama-3.1-8B-Instruct
            ...

            are also available.
        device (str): The device to run the pipeline on. Should be either "cuda" or
            "cpu".Defaults to "cpu".
        quantization (str): The quantization to apply to the model. Defaults to "16bit".

    Returns:
        transformers.Pipeline: The text generation pipeline.

    """

    if quantization == "16bit":
        quantization_config = None
    elif quantization == "8bit":
        quantization_config = {"load_in_8bit": True}
    elif quantization == "4bit":
        quantization_config = {"load_in_4bit": True}
    else:
        raise ValueError(
            f"Invalid quantization value: {quantization}. Must be '16bit', '8bit', "
            f"or '4bit'."
        )

    return transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": quantization_config,
        },
        device_map=device,
    )


def text2graph(memory: dict, next_text: str, template: str) -> list[dict]:
    """
    Generate the prompt for the AI assistant to convert text to a simplified knowledge graph.

    Args:
        memory (dict): The memory of the knowledge graph (history) extracted so far.
        next_text (str): The new text to convert into a knowledge graph.
        template (str): The template for the text-to-graph conversion. Currently it can
            be either "text2graph_without_properties" or "text2graph_with_properties".


    Returns:
        list[dict]: A structured prompt for the AI assistant to build a knowledge graph.
    """
    if template == "text2graph_without_properties":
        text2graph_template = text2graph_without_properties
    else:
        text2graph_template = text2graph_with_properties

    prompt = [
        {
            "role": "system",
            "content": text2graph_template,
        },
        {
            "role": "user",
            "content": f"Here is the knowledge graph extracted (memory) so far: "
            f"{memory}. The new text to process: {next_text}",
        },
    ]

    return prompt


def graph2text(memory: dict, template: str) -> list[dict]:
    """
    Generate the prompt for the AI assistant to convert a knowledge graph into text.

    Args:
        memory (dict): The knowledge graph to convert into text.
        template (str): The template for the graph-to-text conversion. Currently it can
            be either "graph2text_without_properties" or "graph2text_with_properties

    Returns:
        list[dict]: A structured prompt for the AI assistant to generate natural
        language text in JSON format.
    """
    if template == "graph2text_without_properties":
        graph2text_template = graph2text_without_properties
    else:
        graph2text_template = graph2text_with_properties

    prompt = [
        {
            "role": "system",
            "content": graph2text_template,
        },
        {
            "role": "user",
            "content": f"Here is the knowledge graph to convert into text: {memory}",
        },
    ]

    return prompt

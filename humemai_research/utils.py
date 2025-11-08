"""Utility functions."""

import csv
import json
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml


def remove_timestamp(entry: List[str]) -> List[str]:
    """Remove the timestamp from a given observation/episodic memory.

    Args:
        entry: An observation / episodic memory in a quadruple format
            (i.e., (head, relation, tail, timestamp))

    Returns:
        entry_without_timestamp: i.e., (head, relation, tail)
    """
    assert len(entry) == 4, "Entry must be a quadruple."
    entry_without_timestamp = entry[:-1]

    return entry_without_timestamp


def seed_everything(seed: int) -> None:
    """Seed every source of randomness for reproducibility.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_json(fname: str) -> Dict[str, Any]:
    """Read a JSON file and return its content.

    Args:
        fname: Path to the JSON file.

    Returns:
        The content of the JSON file as a dictionary.
    """
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: Dict[str, Any], fname: str) -> None:
    """Write a dictionary to a JSON file.

    Args:
        content: The content to write to the JSON file.
        fname: Path to the JSON file.
    """
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> Dict[str, Any]:
    """Read a YAML file and return its content.

    Args:
        fname: Path to the YAML file.

    Returns:
        The content of the YAML file as a dictionary.
    """
    with open(fname, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: Dict[str, Any], fname: str) -> None:
    """Write a dictionary to a YAML file.

    Args:
        content: The content to write to the YAML file.
        fname: Path to the YAML file.
    """
    with open(fname, "w") as stream:
        yaml.dump(content, stream, indent=2, sort_keys=False)


def write_pickle(to_pickle: Any, fname: str) -> None:
    """Serialize an object and write it to a pickle file.

    Args:
        to_pickle: The object to serialize.
        fname: Path to the pickle file.
    """
    with open(fname, "wb") as stream:
        pickle.dump(to_pickle, stream)


def read_pickle(fname: str) -> Any:
    """Deserialize an object from a pickle file.

    Args:
        fname: Path to the pickle file.

    Returns:
        The deserialized object.
    """
    with open(fname, "rb") as stream:
        return pickle.load(stream)


def write_csv(content: List[List[Any]], fname: str) -> None:
    """Write a list of lists to a CSV file.

    Args:
        content: The content to write to the CSV file.
        fname: Path to the CSV file.
    """
    with open(fname, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(content)


def read_data(data_path: str) -> Dict[str, List[Any]]:
    """Read train, validation, and test splits from a JSON file.

    Args:
        data_path: Path to the data JSON file.

    Returns:
        A dictionary containing 'train', 'val', and 'test' splits.
    """
    data = read_json(data_path)
    return data


def load_questions(path: str) -> Dict[str, Any]:
    """Load pre-made questions from a JSON file.

    Args:
        path: Path to the questions JSON file.

    Returns:
        A dictionary containing the loaded questions.
    """
    questions = read_json(path)
    return questions


def argmax(iterable: List[float]) -> int:
    """Return the index of the maximum value in the iterable.

    Args:
        iterable: A list of numerical values.

    Returns:
        The index of the maximum value.
    """
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def get_duplicate_dicts(search: Dict[str, Any], target: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find duplicate dictionaries in a target list based on a search dictionary.

    Args:
        search: The dictionary to search for within the target list.
        target: The list of dictionaries to search through.

    Returns:
        A list of dictionaries from the target that match the search criteria.
    """
    assert isinstance(search, dict), "Search parameter must be a dictionary."
    duplicates = []

    for candidate in target:
        assert isinstance(candidate, dict), "All candidates must be dictionaries."
        if set(search).issubset(set(candidate)):
            if all(candidate[key] == val for key, val in search.items()):
                duplicates.append(candidate)

    return duplicates


def list_duplicates_of(seq: List[Any], item: Any) -> List[int]:
    """Find all indices of a specified item in a list.

    Args:
        seq: The list to search within.
        item: The item to find duplicates of.

    Returns:
        A list of indices where the item is found in the sequence.
    """
    # Reference: https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def is_running_notebook() -> bool:
    """Determine if the code is running inside a Jupyter notebook.

    Returns:
        True if running in a Jupyter notebook or qtconsole, False otherwise.
    """
    try:
        from IPython import get_ipython  # Imported here to avoid issues in non-IPython environments
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False  # Probably standard Python interpreter


def merge_lists(lists: List[List[Any]]) -> List[List[Any]]:
    """Merge a list of lists into a single list with merged dictionaries.

    Deepcopy is used to avoid modifying the original lists/dicts.

    Args:
        lists: A list of lists where each sublist has the format
            [key1, key2, key3, value_dict], where key1, key2, key3 form a tuple,
            and value_dict is a dictionary.

    Returns:
        merged_list: A list of lists with the format [key1, key2, key3, merged_value_dict].
    """
    merged_dict: Dict[tuple, Dict[str, Any]] = defaultdict(dict)

    for sublist in lists:
        if len(sublist) < 4:
            raise ValueError("Each sublist must have at least 4 elements.")
        key = tuple(sublist[:3])
        value_dict = sublist[3]

        if key in merged_dict:
            # Merge dictionaries
            for k, v in value_dict.items():
                if k in merged_dict[key]:
                    if isinstance(v, list):
                        # Merge lists and remove duplicates
                        merged_dict[key][k] = list(set(merged_dict[key][k] + v))
                    else:
                        # Handle non-list values by taking the maximum
                        merged_dict[key][k] = max(merged_dict[key][k], v)
                else:
                    merged_dict[key][k] = deepcopy(v)
        else:
            merged_dict[key] = deepcopy(value_dict)

    # Convert back to the original list of lists format
    merged_list: List[List[Any]] = [[*k, v] for k, v in merged_dict.items()]

    return merged_list

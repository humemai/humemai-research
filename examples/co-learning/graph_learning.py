"""Graph learning module for co-learning."""

import os
import ast
import json
from glob import glob
from collections import defaultdict
import math
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
import bs4
from rdflib import Graph, Namespace, RDF, Literal, XSD
from tqdm.auto import tqdm
import rdflib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def parse_cp_messages(
    path: str = "./user-raw-data/new/all_cp_messages.csv", sep: str = ";"
) -> dict:
    """Parse the CP messages from the CSV file.

    Args:
        path (str): The path to the CSV file containing the CP messages.
        sep (str): The separator used in the CSV file.

    Returns:
        dict: A dictionary containing the parsed CP messages.

    """
    # Load CSV data
    data = pd.read_csv(path, sep=sep)

    # Initialize an empty dictionary to store the final output
    all_cp_messages = {}
    all_keys = []

    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        participant = row["Participant"]
        roundnr = row["Roundnr"]

        # Ensure that the participant exists in the result dictionary
        if participant not in all_cp_messages:
            all_cp_messages[participant] = {}

        # Collect all valid entries from the row, starting from the third column
        round_data = [row[col] for col in data.columns[2:] if pd.notna(row[col])]

        # Process each element in round_data and parse it
        parsed_round_data = []
        for element in round_data:
            # Convert the string to a dictionary using ast.literal_eval
            outer_dict = ast.literal_eval(element)

            # Initialize a dictionary to store parsed values from 'html'
            parsed_html = {}

            for key, value in outer_dict.items():
                all_keys.append(key)

                if key != "html":
                    parsed_html[key] = value
                elif key == "html" and value is not None:
                    html_content = json.loads(outer_dict["html"])

                    # Process 'situation' if present and not empty
                    if (
                        "situation" in html_content
                        and html_content["situation"].strip()
                    ):
                        situation_soup = BeautifulSoup(
                            html_content["situation"], "html.parser"
                        )

                        situation = []

                        for situ_box in situation_soup.find_all(
                            "div", class_="box_cp box_situ"
                        ):
                            situation_ = []
                            # Find all items within the situation box
                            for item in situ_box.find_all("div", class_="item"):
                                # Determine the type of the item
                                if "location" in item.get("class", []):
                                    item_type = "location"
                                elif "object_cp" in item.get("class", []):
                                    item_type = "object"
                                elif "actor" in item.get("class", []):
                                    item_type = "actor"
                                else:
                                    item_type = "unknown"

                                # Collect content pieces
                                content_pieces = []
                                for child in item.children:
                                    if isinstance(child, bs4.element.Tag):
                                        # If it's a paragraph, extract the text
                                        # (excluding child tags)
                                        if child.name == "p":
                                            text = "".join(
                                                child.find_all(
                                                    text=True, recursive=False
                                                )
                                            ).strip()
                                            if text:
                                                content_pieces.append(text)
                                        # If it's an 'i' tag, extract its text
                                        elif child.name == "i":
                                            i_content = child.get_text(strip=True)
                                            content_pieces.append(f"<{i_content}>")
                                        elif (
                                            child.name == "div"
                                            and "invulvakje" in child.get("class", [])
                                        ):
                                            i_tag = child.find("i")
                                            if i_tag:
                                                i_content = i_tag.get_text(strip=True)
                                                content_pieces.append(f"<{i_content}>")
                                        elif child.name == "button":
                                            i_tag = child.find("i")
                                            if i_tag:
                                                i_content = i_tag.get_text(strip=True)
                                                content_pieces.append(f"<{i_content}>")
                                # Combine content pieces into a single string
                                content = " ".join(content_pieces)
                                situation_.append(
                                    {"type": item_type, "content": content}
                                )
                            situation.append(situation_)

                        parsed_html["situation"] = situation

                    # Process 'actionA' if present and not empty
                    if "actionA" in html_content and html_content["actionA"].strip():
                        actionA_soup = BeautifulSoup(
                            html_content["actionA"], "html.parser"
                        )

                        actionA = []

                        # Find all actionA boxes
                        for box_action in actionA_soup.find_all(
                            "div", class_="box_cp box_action"
                        ):
                            actionA_ = []
                            # Process each 'item' div within the box_action
                            for item in box_action.find_all("div", class_="item"):
                                # Determine the type of the item
                                if "task" in item.get("class", []):
                                    item_type = "action"
                                elif "location" in item.get("class", []):
                                    item_type = "location"
                                elif "object_cp" in item.get("class", []):
                                    item_type = "object"
                                elif "actor" in item.get("class", []):
                                    item_type = "actor"
                                else:
                                    item_type = "unknown"

                                # Collect content pieces
                                content_pieces = []
                                for child in item.children:
                                    if isinstance(child, bs4.element.Tag):
                                        # If it's a paragraph, extract the text (excluding child tags)
                                        if child.name == "p":
                                            text = "".join(
                                                child.find_all(
                                                    text=True, recursive=False
                                                )
                                            ).strip()
                                            if text:
                                                content_pieces.append(text)
                                        # If it's an 'i' tag, extract its text
                                        elif child.name == "i":
                                            i_content = child.get_text(strip=True)
                                            content_pieces.append(f"<{i_content}>")
                                        elif (
                                            child.name == "div"
                                            and "invulvakje" in child.get("class", [])
                                        ):
                                            i_tag = child.find("i")
                                            if i_tag:
                                                i_content = i_tag.get_text(strip=True)
                                                content_pieces.append(f"<{i_content}>")
                                        elif child.name == "button":
                                            i_tag = child.find("i")
                                            if i_tag:
                                                i_content = i_tag.get_text(strip=True)
                                                content_pieces.append(f"<{i_content}>")
                                # Combine the pieces into a single content string
                                content = " ".join(content_pieces)
                                # Append the item to the actionA_ list
                                actionA_.append({"type": item_type, "content": content})
                            # Append the actionA_ list to the main actionA list
                            actionA.append(actionA_)

                        # Output the parsed actions
                        parsed_html["HumanAction"] = actionA

                    # Process 'actionB' if present and not empty
                    if "actionB" in html_content and html_content["actionB"].strip():
                        actionB_soup = BeautifulSoup(
                            html_content["actionB"], "html.parser"
                        )

                        actionB = []

                        # Find all actionB boxes
                        for box_action in actionB_soup.find_all(
                            "div", class_="box_cp box_action"
                        ):
                            actionB_ = []
                            # Process each 'item' div within the box_action
                            for item in box_action.find_all("div", class_="item"):
                                # Determine the type of the item
                                if "task" in item.get("class", []):
                                    item_type = "action"
                                elif "location" in item.get("class", []):
                                    item_type = "location"
                                elif "object_cp" in item.get("class", []):
                                    item_type = "object"
                                elif "actor" in item.get("class", []):
                                    item_type = "actor"
                                else:
                                    item_type = "unknown"

                                # Collect content pieces
                                content_pieces = []
                                for child in item.children:
                                    if isinstance(child, bs4.element.Tag):
                                        # If it's a paragraph, extract the text (excluding child tags)
                                        if child.name == "p":
                                            text = "".join(
                                                child.find_all(
                                                    text=True, recursive=False
                                                )
                                            ).strip()
                                            if text:
                                                content_pieces.append(text)
                                        # If it's an 'i' tag, extract its text
                                        elif child.name == "i":
                                            i_content = child.get_text(strip=True)
                                            content_pieces.append(f"<{i_content}>")
                                        elif (
                                            child.name == "div"
                                            and "invulvakje" in child.get("class", [])
                                        ):
                                            i_tag = child.find("i")
                                            if i_tag:
                                                i_content = i_tag.get_text(strip=True)
                                                content_pieces.append(f"<{i_content}>")
                                        elif child.name == "button":
                                            i_tag = child.find("i")
                                            if i_tag:
                                                i_content = i_tag.get_text(strip=True)
                                                content_pieces.append(f"<{i_content}>")
                                # Combine the pieces into a single content string
                                content = " ".join(content_pieces)
                                # Append the item to the actionB_ list
                                actionB_.append({"type": item_type, "content": content})
                            # Append the actionB_ list to the main actionB list
                            actionB.append(actionB_)
                        # Output the parsed actions
                        parsed_html["RobotAction"] = actionB

                    parsed_round_data.append(parsed_html)
                else:
                    pass

        # Store the parsed round data under the specific participant and round number
        all_cp_messages[participant][roundnr] = parsed_round_data

    return all_cp_messages


def parse_cp_execution(
    path: str = "./user-raw-data/new/cp_execution.csv", sep: str = ";"
) -> dict:
    """Parse the CP execution data from the CSV file.

    Args:
        path (str): The path to the CSV file containing the CP execution data.
        sep (str): The separator used in the CSV file.

    Returns:
        dict: A dictionary containing the parsed CP execution data.
    """

    # Load the CSV file
    data = pd.read_csv(path, sep=sep)

    # Initialize an empty dictionary to store the final result
    cp_execution = {}

    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        participant = row["Participant"]
        roundnr = row["Round"]

        # Ensure that the participant exists in the result dictionary
        if participant not in cp_execution:
            cp_execution[participant] = {}

        # Collect all valid entries from the row, starting from the third column
        round_data = []
        for col in data.columns[2:]:
            element = row[col]
            if pd.notna(element):
                try:
                    # Parse the element as a list
                    parsed_element = ast.literal_eval(element)
                    round_data.append(parsed_element)
                except (ValueError, SyntaxError):
                    # Handle parsing errors gracefully
                    round_data.append(element)

        # Store the round data for the specific participant and round
        cp_execution[participant][roundnr] = round_data

    # Now 'result' contains the desired dictionary structure
    return cp_execution


def match_cp_with_execution(all_cp_messages: dict, cp_execution: dict) -> dict:
    """Match the CP messages with the CP execution data.

    Args:
        all_cp_messages (dict): A dictionary containing the parsed CP messages.
        cp_execution (dict): A dictionary containing the parsed CP execution data.

    Returns:
        dict: A dictionary containing the matched CP messages and execution data.
    """

    cp_messages_execution = {}
    cps_used = 0

    for participant, rounds in cp_execution.items():
        for round_num, round_data in rounds.items():

            cp_candidates = [
                cp
                for i in range(1, round_num + 1)
                for cp in all_cp_messages[participant][i]
            ]

            for round_data_ in round_data:
                cp_name, ticks = round_data_[0], round_data_[1]
                if cp_name not in ["False", "false", False, None]:

                    # Get the latest cp_candidate with the same cp_name
                    for cp_candidate in cp_candidates[::-1]:
                        if cp_name in list(cp_candidate.values()):
                            # print(cp_candidate)
                            break

                    cps_used += 1
                    cp_candidate["ticks_lasted"] = ticks
                    cp_candidate["cp_name"] = cp_name

                    if participant not in cp_messages_execution:
                        cp_messages_execution[participant] = {}

                    if round_num not in cp_messages_execution[participant]:
                        cp_messages_execution[participant][round_num] = [cp_candidate]
                    else:
                        cp_messages_execution[participant][round_num].append(
                            cp_candidate
                        )

    return cp_messages_execution


def parse_time(date: str, time: str):
    # Regex to extract day, month, year, hour, minute, second
    date_pattern = r"date_(\d{2})d-(\d{2})m-(\d{4})y"
    time_pattern = r"time_(\d{2})h-(\d{2})m-(\d{2})s"

    # Extract date components
    date_match = re.match(date_pattern, date)
    time_match = re.match(time_pattern, time)

    if date_match and time_match:
        day, month, year = date_match.groups()
        hour, minute, second = time_match.groups()

        # Combine into a single datetime object with 'T' for xsd:dateTime format
        dt_str = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        dt_obj = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")

        # Get Unix timestamp (seconds since the epoch)
        unix_time = int(dt_obj.timestamp())

        return dt_str, unix_time

    return None, None  # Return None if the pattern doesn't match


def get_metrics(path: str = "./user-raw-data/new/data_aggregate.csv", sep: str = ","):
    data = pd.read_csv(path, sep=sep)

    # Filter the DataFrame to only include rows where Condition == "C3"
    data = data[data["Condition"] == "C3"]

    metrics = {}

    weird = 0
    # Iterate through each row in the filtered DataFrame
    for index, row in data.iterrows():
        participant = row["Participant"]
        roundnr = row["Roundnr"]
        date = row["Date"]
        time = row["Time"]

        dt_str, unix_time = parse_time(date, time)

        time_elapsed = row["corrected_tick"]
        remaining_rocks = row["remaining_rocks"]
        victim_harm = row["victim_harm"]

        # Ensure that the participant exists in the result dictionary
        if participant not in metrics:
            metrics[participant] = {}
        if roundnr not in metrics[participant]:
            metrics[participant][roundnr] = {}

        if math.isnan(time_elapsed) or math.isnan(victim_harm):
            # this is not acceptable, so we won't include it.
            weird += 1
            continue

        if math.isnan(remaining_rocks):
            remaining_rocks = 0

        metrics[participant][roundnr]["timestamp"] = dt_str
        metrics[participant][roundnr]["unix_timestamp"] = unix_time
        metrics[participant][roundnr]["time_elapsed"] = int(time_elapsed)
        metrics[participant][roundnr]["remaining_rocks"] = int(remaining_rocks)
        metrics[participant][roundnr]["victim_harm"] = int(victim_harm)

    print(f"number of weird: {weird}")

    return metrics


def get_final_data(cp_messages_execution: dict, metrics: dict):
    """Merge the CP messages and metrics into a single data structure.

    Args:
        cp_messages_execution (dict): A dictionary containing the matched CP messages
        and execution data.
        metrics (dict): A dictionary containing the parsed metrics.

    Returns:
        list: A list containing the merged data.
    """
    data = []
    cp_added = 0
    weird = 0

    for participant, rounds in cp_messages_execution.items():
        for round_num, round_data in rounds.items():
            metrics_ = metrics[participant][round_num]

            if "timestamp" not in metrics_:
                weird += 1
                continue

            for round_data_ in round_data:

                cp = {
                    "cp_num": cp_added,
                    "participant": participant,
                    "cp_name": round_data_["cp_name"],
                    "ticks_lasted": round_data_["ticks_lasted"],
                    "round_num": round_num,
                    "timestamp": metrics_["timestamp"],
                    "unix_timestamp": metrics_["unix_timestamp"],
                    "time_elapsed": metrics_["time_elapsed"],
                    "remaining_rocks": metrics_["remaining_rocks"],
                    "success": metrics_["success"],
                    "victim_harm": metrics_["victim_harm"],
                    "situation": round_data_["situation"],
                    "HumanAction": round_data_["HumanAction"],
                    "RobotAction": round_data_["RobotAction"],
                }

                data.append(cp)
                cp_added += 1

    print(f"number of cps added: {cp_added}")
    print(f"number of weird: {weird}")

    return data


def make_rdf_data_iid(
    raw_data_path: str = "raw-data.json", output_dir: str = "./rdf-data-iid"
) -> None:
    """Create RDF data from the raw data.

    Args:
        raw_data_path (str): The path to the raw data JSON file.
        output_dir (str): The directory where the RDF data will be saved.
    """

    # Load JSON data
    with open(raw_data_path, "r") as f:
        data = json.load(f)

    # Define the namespace
    CO_LEARNING = Namespace("https://humem.ai/example/co_learning#")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each CP in the data
    for cp in tqdm(data):
        cp_num = cp["cp_num"]
        cp_uri = CO_LEARNING[f"CP{cp_num}"]

        # Create a new RDF graph for each CP
        g = Graph()
        g.bind("co_learning", CO_LEARNING)

        # Add the CollaborationPattern instance
        g.add((cp_uri, RDF.type, CO_LEARNING.CollaborationPattern))
        g.add(
            (cp_uri, CO_LEARNING.hasCPNum, Literal(cp["cp_num"], datatype=XSD.integer))
        )
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasParticipantNumber,
                Literal(cp["participant"], datatype=XSD.integer),
            )
        )
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasRoundNumber,
                Literal(cp["round_num"], datatype=XSD.integer),
            )
        )
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasTimeStamp,
                Literal(cp["timestamp"], datatype=XSD.dateTime),
            )
        )
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasUnixTimeStamp,
                Literal(cp["unix_timestamp"], datatype=XSD.integer),
            )
        )
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasRemainingTime,
                Literal(cp["time_elapsed"], datatype=XSD.integer),
            )
        )
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasRemainingRocks,
                Literal(cp["remaining_rocks"], datatype=XSD.integer),
            )
        )
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasVictimHarm,
                Literal(cp["victim_harm"], datatype=XSD.integer),
            )
        )

        # Add the new properties
        # hasCPLabel
        cp_label = cp.get("cp_name", "")
        g.add((cp_uri, CO_LEARNING.hasCPLabel, Literal(cp_label, datatype=XSD.string)))

        # hasTicksLasted
        ticks_lasted = cp.get("ticks_lasted", 0)
        g.add(
            (
                cp_uri,
                CO_LEARNING.hasTicksLasted,
                Literal(ticks_lasted, datatype=XSD.integer),
            )
        )

        # Handle situations
        prev_situation_uri = None
        for idx, situation in enumerate(cp["situation"]):
            situation_uri = CO_LEARNING[f"Situation{cp_num}_{idx}"]
            g.add((situation_uri, RDF.type, CO_LEARNING.Situation))

            if idx == 0:
                g.add((cp_uri, CO_LEARNING.hasSituation, situation_uri))
            else:
                g.add((prev_situation_uri, CO_LEARNING.hasNextSituation, situation_uri))
            prev_situation_uri = situation_uri

            # Add data properties to Situation
            for elem in situation:
                elem_type = elem["type"]
                elem_content = elem["content"]

                if elem_type == "actor":
                    g.add((situation_uri, CO_LEARNING.hasActor, Literal(elem_content)))
                elif elem_type == "action":
                    g.add((situation_uri, CO_LEARNING.hasAction, Literal(elem_content)))
                elif elem_type == "location":
                    g.add(
                        (situation_uri, CO_LEARNING.hasLocation, Literal(elem_content))
                    )
                elif elem_type == "object":
                    g.add((situation_uri, CO_LEARNING.hasObject, Literal(elem_content)))

        # Handle HumanAction
        prev_action_human_uri = None
        for idx, action_human in enumerate(cp["HumanAction"]):
            action_human_uri = CO_LEARNING[f"HumanAction{cp_num}_{idx}"]
            g.add((action_human_uri, RDF.type, CO_LEARNING.HumanAction))

            if idx == 0:
                g.add((cp_uri, CO_LEARNING.hasHumanAction, action_human_uri))
            else:
                g.add(
                    (
                        prev_action_human_uri,
                        CO_LEARNING.hasNextHumanAction,
                        action_human_uri,
                    )
                )
            prev_action_human_uri = action_human_uri

            # Add data properties to HumanAction
            for elem in action_human:
                elem_type = elem["type"]
                elem_content = elem["content"]
                if elem_type == "actor":
                    g.add(
                        (action_human_uri, CO_LEARNING.hasActor, Literal(elem_content))
                    )
                elif elem_type == "action":
                    g.add(
                        (action_human_uri, CO_LEARNING.hasAction, Literal(elem_content))
                    )
                elif elem_type == "location":
                    g.add(
                        (
                            action_human_uri,
                            CO_LEARNING.hasLocation,
                            Literal(elem_content),
                        )
                    )
                elif elem_type == "object":
                    g.add(
                        (action_human_uri, CO_LEARNING.hasObject, Literal(elem_content))
                    )

        # Handle RobotAction
        prev_action_robot_uri = None
        for idx, action_robot in enumerate(cp["RobotAction"]):
            action_robot_uri = CO_LEARNING[f"RobotAction{cp_num}_{idx}"]
            g.add((action_robot_uri, RDF.type, CO_LEARNING.RobotAction))

            if idx == 0:
                g.add((cp_uri, CO_LEARNING.hasRobotAction, action_robot_uri))
            else:
                g.add(
                    (
                        prev_action_robot_uri,
                        CO_LEARNING.hasNextRobotAction,
                        action_robot_uri,
                    )
                )
            prev_action_robot_uri = action_robot_uri

            # Add data properties to RobotAction
            for elem in action_robot:
                elem_type = elem["type"]
                elem_content = elem["content"]
                if elem_type == "actor":
                    g.add(
                        (action_robot_uri, CO_LEARNING.hasActor, Literal(elem_content))
                    )
                elif elem_type == "action":
                    g.add(
                        (action_robot_uri, CO_LEARNING.hasAction, Literal(elem_content))
                    )
                elif elem_type == "location":
                    g.add(
                        (
                            action_robot_uri,
                            CO_LEARNING.hasLocation,
                            Literal(elem_content),
                        )
                    )
                elif elem_type == "object":
                    g.add(
                        (action_robot_uri, CO_LEARNING.hasObject, Literal(elem_content))
                    )

        # Format cp_num with leading zeros to three digits
        cp_num_formatted = f"{cp_num:03d}"

        # Serialize each CP's graph to a separate Turtle file named '<cp_num_formatted>.ttl'
        file_name = os.path.join(output_dir, f"{cp_num_formatted}.ttl")
        g.serialize(destination=file_name, format="turtle")
        print(f"Saved CP{cp_num} to {file_name}")


def parse_rdf_to_networkx(ttl_file: str) -> tuple:
    """Parse an RDF file into a NetworkX graph.

    Args:
        ttl_file (str): The path to the RDF file.

    Returns:
        tuple: A tuple containing the NetworkX graph, node labels, and edge labels.
    """

    # Load the RDF graph
    g = rdflib.Graph()
    g.parse(ttl_file, format="turtle")

    # Create a NetworkX graph
    G = nx.DiGraph()

    node_labels = {}
    edge_labels = {}

    # Iterate over the triples in the RDF graph
    for s, p, o in g:
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)

        # Ignore triples where the predicate is rdf:type
        if p_str.endswith("type"):
            continue

        # Check if the object is a literal or a URI
        if isinstance(o, rdflib.term.Literal):
            # Handle literals by adding them as labels on the subject node
            if s_str not in node_labels:
                node_labels[s_str] = s_str.split("#")[-1]
            predicate = p_str.split("#")[-1]
            node_labels[s_str] += f"\n{predicate}: {o_str}"

        else:
            # Add subject and object as nodes if not already present
            if s_str not in node_labels:
                node_labels[s_str] = s_str.split("#")[-1]
            if o_str not in node_labels:
                node_labels[o_str] = o_str.split("#")[-1]

            # Add an edge between the subject and object
            G.add_edge(s_str, o_str)
            edge_labels[(s_str, o_str)] = p_str.split("#")[-1]

    return G, node_labels, edge_labels


def visualize_graph(G, node_labels, edge_labels, output_dir, output_filename) -> None:
    """Visualize a NetworkX graph and save it as a PDF and PNG.

    Args:
        G (nx.DiGraph): The NetworkX graph to visualize.
        node_labels (dict): A dictionary mapping node IDs to their labels.
        edge_labels (dict): A dictionary mapping edge tuples to their labels.
        output_dir (str): The directory where the output files will be saved.
        output_filename (str): The base filename for the output files.

    """

    # Set the figure size (e.g., 12 inches by 12 inches)
    plt.figure(figsize=(10, 10))

    # Generate layout with increased spread to reduce overlap
    pos = nx.spring_layout(G, k=0.7, iterations=100)

    # Draw nodes (default color, no changes)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.9)

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", alpha=0.6
    )

    # Draw node labels (reduce font size for better readability)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_color="black")

    # Draw edge labels (with smaller font size and better positioning)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8, font_color="red", rotate=False
    )

    plt.axis("off")  # Hide the axis
    plt.title(f"Collaboration Pattern: {output_filename}")
    plt.tight_layout()

    os.makedirs(f"./{output_dir}/pdf", exist_ok=True)
    # os.makedirs(f"./{output_dir}/png", exist_ok=True)
    # Save the figure as both PDF and PNG
    plt.savefig(f"./{output_dir}/pdf/{output_filename}.pdf", format="pdf")
    # plt.savefig(
    #     f"./{output_dir}/png/{output_filename}.png", format="png", dpi=300
    # )  # 300 dpi for high resolution

    # Display the plot in Jupyter Notebook
    plt.show()

    # Clear the figure after saving to avoid overlap in the next iteration
    plt.clf()


def visualize_ttl_files_iid(
    directory="./rdf-data-iid", output_dir="graphs-visualized-iid"
) -> None:
    """Process all .ttl files in the specified directory.

    Args:
        directory (str): The directory containing the .ttl files.
        output_dir (str): The directory where the visualized graphs will be saved.

    """
    # Use glob to find all .ttl files in the directory
    ttl_files = sorted(glob(os.path.join(directory, "*.ttl")))

    for ttl_file in tqdm(ttl_files):
        # Extract the base filename (without the .ttl extension)
        base_filename = os.path.splitext(os.path.basename(ttl_file))[0]

        # Parse the RDF data
        G, node_labels, edge_labels = parse_rdf_to_networkx(ttl_file)

        # Visualize the graph and save as both PDF and PNG
        visualize_graph(G, node_labels, edge_labels, output_dir, base_filename)


def get_some_stats_iid(directory: str = "./rdf-data-iid") -> str:
    """Get some statistics from the RDF data.

    Args:
        directory (str): The directory containing the .ttl files.

    Returns:
        str: A Markdown-formatted string containing the statistics.

    """

    # Initialize lists to store statistics
    timestamps = []
    unix_timestamps = []
    time_elapseds = []
    remaining_rocks = []
    victim_harms = []
    ticks_lasted = []
    participants = defaultdict(int)  # Count collaboration patterns per participant
    non_empty_situations_per_graph = []
    non_empty_action_humans_per_graph = []
    non_empty_action_robots_per_graph = []

    CO_LEARNING_NS = "https://humem.ai/example/co_learning#"

    # Function to check if a node is non-empty
    def is_non_empty(g, node):
        for _, p, _ in g.triples((node, None, None)):
            if (
                p.endswith("hasAction")
                or p.endswith("hasLocation")
                or p.endswith("hasObject")
                or p.endswith("hasActor")
            ):
                return True
        return False

    # Loop through all .ttl files and extract stats
    for ttl_path in tqdm(glob(os.path.join(directory, "*.ttl"))):
        g = rdflib.Graph()
        g.parse(ttl_path, format="turtle")

        non_empty_situations = 0
        non_empty_action_humans = 0
        non_empty_action_robots = 0

        # Extract triples and gather data
        for s, p, o in g:
            if p.endswith("hasTimeStamp"):
                timestamps.append(o)
            if p.endswith("hasUnixTimeStamp"):
                unix_timestamps.append(int(o))
            if p.endswith("hasRemainingTime"):
                time_elapseds.append(int(o))
            if p.endswith("hasRemainingRocks"):
                remaining_rocks.append(int(o))
            if p.endswith("hasVictimHarm"):
                victim_harms.append(int(o))
            if p.endswith("hasTicksLasted"):
                ticks_lasted.append(int(o))
            if p.endswith("hasParticipantNumber"):
                participants[int(o)] += 1

        # Identify instances of Situation, HumanAction, and RobotAction
        for s, rdf_type in g.subject_objects(rdflib.RDF.type):
            rdf_type_str = str(rdf_type)
            if rdf_type_str == CO_LEARNING_NS + "Situation" and is_non_empty(g, s):
                non_empty_situations += 1
            elif rdf_type_str == CO_LEARNING_NS + "HumanAction" and is_non_empty(g, s):
                non_empty_action_humans += 1
            elif rdf_type_str == CO_LEARNING_NS + "RobotAction" and is_non_empty(g, s):
                non_empty_action_robots += 1

        non_empty_situations_per_graph.append(non_empty_situations)
        non_empty_action_humans_per_graph.append(non_empty_action_humans)
        non_empty_action_robots_per_graph.append(non_empty_action_robots)

    # Helper to calculate stats
    def calculate_stats(data):
        return {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "max": np.max(data),
            "min": np.min(data),
        }

    # Remaining time statistics
    time_elapsed_stats = calculate_stats(time_elapseds)

    # Remaining rocks statistics
    remaining_rocks_stats = calculate_stats(remaining_rocks)

    # Victim harm statistics
    victim_harm_stats = calculate_stats(victim_harms)

    # TicksLasted statistics
    ticks_lasted_stats = calculate_stats(ticks_lasted)

    # Collaboration Patterns per Participant statistics
    cps_per_participant = list(participants.values())
    cps_per_participant_stats = calculate_stats(cps_per_participant)
    num_unique_participants = len(participants)

    # Non-empty Situations, HumanAction, RobotAction statistics
    non_empty_situations_stats = calculate_stats(non_empty_situations_per_graph)
    non_empty_action_humans_stats = calculate_stats(non_empty_action_humans_per_graph)
    non_empty_action_robots_stats = calculate_stats(non_empty_action_robots_per_graph)

    # Return all stats as a dictionary
    return {
        "time_elapsed_stats": time_elapsed_stats,
        "remaining_rocks_stats": remaining_rocks_stats,
        "victim_harm_stats": victim_harm_stats,
        "ticks_lasted": ticks_lasted_stats,
        "cps_per_participant": {
            **cps_per_participant_stats,
            "num_unique_participants": num_unique_participants,
        },
        "non_empty_situations": non_empty_situations_stats,
        "non_empty_action_humans": non_empty_action_humans_stats,
        "non_empty_action_robots": non_empty_action_robots_stats,
    }

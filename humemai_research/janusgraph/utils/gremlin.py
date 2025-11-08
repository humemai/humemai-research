"""Utility functions for interacting with JanusGraph using Gremlin Python."""

import json
from gremlin_python.structure.graph import Graph, Vertex, Edge
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import P, T, Direction


def remove_all_data(g: Graph) -> None:
    """Remove all vertices and edges from the graph.

    Args:
        g (Graph): JanusGraph graph instance.
    """
    g.V().drop().iterate()


def get_all_vertices(g: Graph) -> list[Vertex]:
    """Retrieve all vertices from the graph.

    Args:
        g (Graph): JanusGraph graph instance.

    Returns:
        list of Vertex: List of all vertices in the graph.
    """

    return g.V().toList()


def get_all_edges(g: Graph) -> list[Edge]:
    """Retrieve all edges from the graph.

    Args:
        g (Graph): JanusGraph graph instance.

    Returns:
        list of Edge: List of all edges in the graph.
    """

    return g.E().toList()


def find_vertex_by_label(g: Graph, label: str) -> list[Vertex]:
    """Find a vertex by its label. It can return multiple vertices.

    Args:
        g (Graph): JanusGraph graph instance.
        label (str): Label of the vertex.

    Returns:
        list of Vertex: List of vertices with the provided label.
    """

    return g.V().hasLabel(label).toList()


def find_vertices_by_properties(
    g: Graph, include_keys: list[str], exclude_keys: list[str] = None
) -> list[Vertex]:
    """Find vertices based on included and excluded properties.

    Args:
        g (Graph): JanusGraph graph instance.
        include_keys (list of str): List of properties that must be included.
        exclude_keys (list of str, optional): List of properties that must be excluded.

    Returns:
        list of Vertex: List of vertices matching the criteria.
    """
    traversal = g.V()

    # Add filters for properties to include
    for key in include_keys:
        traversal = traversal.has(key)

    # Add filters for properties to exclude
    if exclude_keys:
        for key in exclude_keys:
            traversal = traversal.hasNot(key)

    return traversal.toList()


def remove_vertex(g: Graph, vertex: Vertex) -> None:
    """
    Removes a given vertex from the graph.

    Args:
        g (GraphTraversalSource): The Gremlin graph traversal source.
        vertex (Vertex): The vertex to remove.

    Returns:
        None
    """
    if g.V(vertex.id).hasNext():
        g.V(vertex.id).drop().iterate()
    else:
        raise ValueError(f"Vertex with ID {vertex.id} not found.")


def remove_edge(g: Graph, edge: Edge) -> None:
    """
    Removes a given edge from the graph.

    The syntax for removing an edge is different from removing a vertex. It's quite
    annoying, but it is what it is.

    Args:
        g (GraphTraversalSource): The Gremlin graph traversal source.
        edge (Edge): The edge to remove.

    Returns:
        None
    """
    if g.E(edge.id["@value"]["relationId"]).hasNext():
        g.E(edge.id["@value"]["relationId"]).drop().iterate()
    else:
        raise ValueError(f"Edge with ID {edge.id} not found.")


def find_edge_by_vertices_and_label(
    g: Graph, head: Vertex, label: str, tail: Vertex
) -> list[Edge]:
    """Find an edge by its label and property.

    Args:
        g (Graph): JanusGraph graph instance.
        head (Vertex): Head vertex of the edge.
        label (str): Label of the edge.
        tail (Vertex): Tail vertex of the edge.

    Returns:
        list of Edge: List of edges with the provided label.
    """
    return g.V(head.id).outE(label).where(__.inV().hasId(tail.id)).toList()


def find_edge_by_label(g: Graph, label: str) -> list[Edge]:
    """Find an edge by its label.

    Args:
        g (Graph): JanusGraph graph instance.
        label (str): Label of the edge.

    Returns:
        list of Edge: List of edges with the provided label.
    """

    return g.E().hasLabel(label).toList()


def find_edges_by_properties(
    g: Graph, include_keys: list[str], exclude_keys: list[str] = None
) -> list[Edge]:
    """Find edges based on included and excluded properties.

    Args:
        g (Graph): JanusGraph graph instance.
        include_keys (list of str): List of properties that must be included.
        exclude_keys (list of str, optional): List of properties that must be excluded.

    Returns:
        list of Edge: List of edges matching the criteria.
    """
    traversal = g.E()

    # Add filters for properties to include
    for key in include_keys:
        traversal = traversal.has(key)

    # Add filters for properties to exclude
    if exclude_keys:
        for key in exclude_keys:
            traversal = traversal.hasNot(key)

    return traversal.toList()


def create_vertex(g: Graph, label: str, properties: dict) -> Vertex:
    """Create a vertex with the given properties.

    Note that this does not check if the vertex already exists.

    Args:
        g (Graph): JanusGraph graph instance.
        label (str): Label of the vertex.
        properties (dict): Dictionary of properties for the vertex.

    """
    vertex = g.addV(label)
    for key, value in properties.items():
        # Serialize list or dict properties to JSON
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        vertex = vertex.property(key, value)

    return vertex.next()


def create_edge(
    g: Graph, head: Vertex, label: str, tail: Vertex, properties: dict
) -> Edge:
    """Create an edge between two vertices.

    Note that this does not check if the edge already exists.

    Args:
        g (Graph): JanusGraph graph instance.
        head (Vertex): Vertex where the edge originates.
        label (str): Label of the edge.
        tail (Vertex): Vertex where the edge terminates.
        properties (dict): Dictionary of properties for the edge.

    """
    # Create a new edge with the provided properties
    edge = g.V(head.id).addE(label).to(__.V(tail.id))  # GraphTraversal object
    for key, value in properties.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        edge = edge.property(key, value)
    return edge.next()  # Return the newly created edge


def update_vertex_properties(g: Graph, vertex: Vertex, properties: dict) -> Vertex:
    """Update the properties of an existing vertex and return the updated vertex.

    Args:
        g (Graph): JanusGraph graph instance.
        vertex (Vertex): Vertex to update.
        properties (dict): Dictionary of properties to update.

    Returns:
        Vertex: The updated vertex.
    """

    # Update the properties of the existing vertex
    for key, value in properties.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        g.V(vertex.id).property(key, value).iterate()

    # Fetch and return the updated vertex
    updated_vertex = g.V(vertex.id).next()
    return updated_vertex


def remove_vertex_properties(g: Graph, vertex: Vertex, property_keys: list) -> Vertex:
    """Remove specific properties from an existing vertex and return the updated vertex.

    Args:
        g (Graph): JanusGraph graph instance.
        vertex (Vertex): Vertex to update.
        property_keys (list): List of property keys to remove.
    """
    for key in property_keys:
        g.V(vertex.id).properties(key).drop().iterate()

    # Fetch and return the updated vertex
    updated_vertex = g.V(vertex.id).next()
    return updated_vertex


def update_edge_properties(g: Graph, edge: Edge, properties: dict) -> Edge:
    """Update the properties of an existing edge and return the updated edge.

    Args:
        g (Graph): JanusGraph graph instance.
        edge (Edge): Edge to update.
        properties (dict): Dictionary of properties to update.
    """

    # Update the properties of the existing edge
    for key, value in properties.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        g.E(edge.id["@value"]["relationId"]).property(key, value).iterate()

    # Fetch and return the updated edge
    updated_edge = g.E(edge.id["@value"]["relationId"]).next()
    return updated_edge


def remove_edge_properties(g: Graph, edge: Edge, property_keys: list) -> Edge:
    """Remove specific properties from an existing edge and return the updated edge.

    Args:
        g (Graph): JanusGraph graph instance.
        edge (Edge): Edge whose properties are to be removed.
        property_keys (list): List of property keys to remove.
    """
    for key in property_keys:
        # Drop the property if it exists
        g.E(edge.id["@value"]["relationId"]).properties(key).drop().iterate()

    # Fetch and return the updated edge
    updated_edge = g.E(edge.id["@value"]["relationId"]).next()
    return updated_edge


def get_properties(vertex_or_edge: Vertex | Edge) -> dict:
    """Retrieve all properties of a vertex or edge, decoding JSON-encoded values.

    Args:
        vertex_or_edge (Vertex | Edge): Vertex or edge to retrieve properties for.

    Returns:
        dict: Dictionary of properties for the element.
    """
    if vertex_or_edge.properties is None:
        return {}

    def try_parse_json(value):
        """Try to parse a JSON string, return the original value if it fails."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
        return value

    return {prop.key: try_parse_json(prop.value) for prop in vertex_or_edge.properties}


def get_vertices_within_hops(
    g: Graph, vertices: list[Vertex], hops: int
) -> list[Vertex]:
    """Retrieve all vertices within N hops from a starting vertex.

    Args:
        g (Graph): JanusGraph graph instance.
        vertices (list[Vertex]): List of starting vertex IDs for the traversal.
        hops (int): Number of hops to traverse from the starting vertex.

    Returns:
        list[Vertex]: List of vertices within N hops from the starting vertex.
    """
    assert hops >= 0, "Number of hops must be a non-negative integer."
    assert isinstance(vertices, list), "Vertices must be provided as a list."

    if hops == 0:
        # Directly return the vertices themselves when hops is 0
        return g.V([v.id for v in vertices]).toList()

    # Perform traversal for N hops
    vertices_within_hops = (
        g.V([v.id for v in vertices])  # Start from the provided vertex IDs
        .emit()  # Emit the starting vertex
        .repeat(__.both().simplePath())  # Traverse to neighbors
        .times(hops)  # Limit the number of hops
        .dedup()  # Avoid duplicate vertices in the result
        .toList()
    )

    return vertices_within_hops


def get_edges_between_vertices(g: Graph, vertices: list[Vertex]) -> list[Edge]:
    """Retrieve all edges between a list of vertices.

    Args:
        g (Graph): JanusGraph graph instance.
        vertices (list[Vertex]): List of vertices to find edges between.

    Returns:
        list[Edge]: List of edges between the provided vertices.
    """
    assert isinstance(vertices, list), "Vertices must be provided as a list."
    # Extract vertex IDs from the provided Vertex objects
    vertex_ids = [v.id for v in vertices]

    edges_between_vertices = (
        g.V(vertex_ids)  # Start with the given vertex IDs
        .bothE()  # Traverse all edges connected to these vertices
        .where(
            __.otherV().hasId(P.within(vertex_ids))
        )  # Ensure the other end is in the vertex set
        .dedup()  # Avoid duplicates
        .toList()  # Convert traversal result to a list
    )

    return edges_between_vertices

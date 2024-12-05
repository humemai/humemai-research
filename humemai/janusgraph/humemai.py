"""Humemai class"""

import json
import os
from datetime import datetime
import logging
import docker
import nest_asyncio
from gremlin_python.process.graph_traversal import __
from gremlin_python.structure.graph import Graph, Vertex, Edge
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversalSource
from gremlin_python.process.traversal import P, T, Direction
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from humemai.janusgraph.utils.docker import (
    start_containers,
    stop_containers,
    remove_containers,
    copy_file_from_docker,
    copy_file_to_docker,
)

from humemai.utils import is_iso8601_datetime, write_json, read_json


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Humemai:
    def __init__(
        self,
        janusgraph_container_name="janusgraph",
        gremlin_server_url="ws://localhost:8182/gremlin",
        gremlin_traversal_source="g",
        configs_dir="./configs",
        janusgraph_config="janusgraph.properties",
        gremlin_server_config="gremlin-server.yaml",
    ) -> None:
        """
        Initialize a Humemai object for connecting to JanusGraph and in-memory graph.

        Currently a persistent database, e.g., Cassandra is not supported. When we are
        production-ready, we will add support for Cassandra.

        Args:
            janusgraph_container_name (str): Name of the JanusGraph container.
            gremlin_server_url (str): URL for connecting to the Gremlin server.
            gremlin_traversal_source (str): Traversal source name for Gremlin.
            configs_dir (str): Directory containing JanusGraph and Gremlin Server
            configuration files.
            janusgraph_config (str): JanusGraph configuration file.
            gremlin_server_config (str): Gremlin Server configuration file.
        """

        self.janusgraph_container_name = janusgraph_container_name
        self.gremlin_server_url = gremlin_server_url
        self.gremlin_traversal_source = gremlin_traversal_source
        self.configs_dir = configs_dir
        self.janusgraph_config = janusgraph_config
        self.gremlin_server_config = gremlin_server_config

        # Initialize Docker client
        self.client = docker.from_env()

        # Set up Gremlin connection and traversal source (to be initialized in connect
        # method)
        self.connection = None
        self.g = None

        # Logging configuration
        self.logger = logger

    def start_containers(self, warmup_seconds: int = 10) -> None:
        """Start the JanusGraph container with optional warmup time.

        Args:
            warmup_seconds (int): Number of seconds to wait after starting the
                containers
        """
        start_containers(
            configs_dir=self.configs_dir,
            janusgraph_config=self.janusgraph_config,
            gremlin_server_config=self.gremlin_server_config,
            janusgraph_container_name=self.janusgraph_container_name,
            warmup_seconds=warmup_seconds,
        )

    def stop_containers(self) -> None:
        """Stop the JanusGraph container."""
        stop_containers(
            janusgraph_container_name=self.janusgraph_container_name,
        )

    def remove_containers(self) -> None:
        """Remove the JanusGraph container."""
        remove_containers(
            janusgraph_container_name=self.janusgraph_container_name,
        )

    def connect(self) -> None:
        """Establish a connection to the Gremlin server."""
        try:
            if not self.connection:
                # Apply nest_asyncio to allow nested event loops (useful in Jupyter
                # notebooks)
                nest_asyncio.apply()

                # Initialize Gremlin connection using GraphSON 3.0 serializer
                self.connection = DriverRemoteConnection(
                    self.gremlin_server_url,
                    self.gremlin_traversal_source,
                    message_serializer=GraphSONSerializersV3d0(),
                )
                # Set up the traversal source
                self.g = Graph().traversal().withRemote(self.connection)
                self.logger.debug("Successfully connected to the Gremlin server.")
        except Exception as e:
            self.logger.error(f"Failed to connect to the Gremlin server: {e}")
            raise

    def disconnect(self) -> None:
        """Close the connection to the Gremlin server."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.debug("Disconnected from the Gremlin server.")
            except Exception as e:
                self.logger.error(f"Failed to disconnect from the Gremlin server: {e}")
            finally:
                self.connection = None
                self.g = None

    def remove_all_data(self) -> None:
        """Remove all vertices and edges from the JanusGraph graph."""
        if self.g:
            self.g.V().drop().iterate()
        else:
            self.logger.warning("Graph traversal source (g) is not initialized.")

    def create_vertex(self, label: str, properties: dict = {}) -> Vertex:
        """Create a vertex with the given properties.

        Note that this does not check if the vertex already exists.

        Args:
            label (str): Label of the vertex.
            properties (dict): Dictionary of properties for the vertex. Defaults to {}.

        """
        vertex = self.g.addV(label)
        for key, value in properties.items():
            vertex = vertex.property(key, value)

        return vertex.next()

    def remove_vertex(self, vertex: Vertex) -> None:
        """
        Remove a vertex from the graph.

        Args:
            vertex (Vertex): The vertex to be removed.
        """
        if self.g.V(vertex.id).hasNext():
            self.g.V(vertex.id).drop().iterate()
        else:
            raise ValueError(f"Vertex with ID {vertex.id} not found.")

    def remove_vertex_properties(self, vertex: Vertex, property_keys: list) -> Vertex:
        """Remove specific properties from an existing vertex and return the updated.

        Args:
            vertex (Vertex): Vertex to update.
            property_keys (list): List of property keys to remove.
        """
        for key in property_keys:
            self.g.V(vertex.id).properties(key).drop().iterate()

        # Fetch and return the updated vertex
        updated_vertex = self.g.V(vertex.id).next()
        return updated_vertex

    def update_vertex_properties(self, vertex: Vertex, properties: dict) -> Vertex:
        """Update the properties of an existing vertex and return the updated vertex.

        Args:
            vertex (Vertex): Vertex to update.
            properties (dict): Dictionary of properties to update.

        Returns:
            Vertex: The updated vertex.
        """

        # Update the properties of the existing vertex
        for key, value in properties.items():
            self.g.V(vertex.id).property(key, value).iterate()

        # Fetch and return the updated vertex
        updated_vertex = self.g.V(vertex.id).next()

        return updated_vertex

    def get_vertices_by_properties(
        self, include_keys: list[str], exclude_keys: list[str] = []
    ) -> list[Vertex]:
        """Find vertices based on included and excluded properties.

        Args:
            include_keys (list of str): List of properties that must be included.
            exclude_keys (list of str, optional): List of properties that must be
                excluded.

        Returns:
            list of Vertex: List of vertices matching the criteria.
        """
        traversal = self.g.V()

        # Add filters for properties to include
        for key in include_keys:
            traversal = traversal.has(key)

        # Add filters for properties to exclude
        if exclude_keys:
            for key in exclude_keys:
                traversal = traversal.hasNot(key)

        return traversal.toList()

    def get_vertices_by_label_and_properties(
        self, label: str, include_keys: list[str] = [], exclude_keys: list[str] = []
    ) -> list[Vertex]:
        """
        Find vertices by label and filter them based on included and excluded
        properties.

        Args:
            label (str): The label to search for.
            include_keys (list of str, optional): List of properties that must be
                included.
            exclude_keys (list of str, optional): List of properties that must be
                excluded.

        Returns:
            list of Vertex: List of vertices matching the criteria.
        """
        traversal = self.g.V().hasLabel(label)

        # Add filters for properties to include
        if include_keys:
            for key in include_keys:
                traversal = traversal.has(key)

        # Add filters for properties to exclude
        if exclude_keys:
            for key in exclude_keys:
                traversal = traversal.hasNot(key)

        return traversal.toList()

    def get_all(self) -> tuple[list[Vertex], list[Edge]]:
        """Retrieve all vertices and edges from the graph.

        Returns:
            tuple: List of vertices and edges.
        """

        return self.g.V().toList(), self.g.E().toList()

    def create_edge(
        self, head: Vertex, label: str, tail: Vertex, properties: dict = {}
    ) -> Edge:
        """Create an edge between two vertices.

        Note that this does not check if the edge already exists.

        Args:
            head (Vertex): Vertex where the edge originates.
            label (str): Label of the edge.
            tail (Vertex): Vertex where the edge terminates.
            properties (dict): Dictionary of properties for the edge. Defaults to {}.

        """
        # Create a new edge with the provided properties
        edge = self.g.V(head.id).addE(label).to(__.V(tail.id))  # GraphTraversal object
        for key, value in properties.items():
            edge = edge.property(key, value)
        return edge.next()  # Return the newly created edge

    def remove_edge(self, edge: Edge) -> None:
        """
        Remove an edge from the graph.

        Args:
            edge (Edge): The edge to be removed.
        """

        if self.g.E(edge.id["@value"]["relationId"]).hasNext():
            self.g.E(edge.id["@value"]["relationId"]).drop().iterate()
        else:
            raise ValueError(f"Edge with ID {edge.id} not found.")

    def remove_edge_properties(self, edge: Edge, property_keys: list) -> Edge:
        """Remove specific properties from an existing edge and return the updated edge.

        Args:
            edge (Edge): Edge whose properties are to be removed.
            property_keys (list): List of property keys to remove.
        """
        for key in property_keys:
            # Drop the property if it exists
            self.g.E(edge.id["@value"]["relationId"]).properties(key).drop().iterate()

        # Fetch and return the updated edge
        updated_edge = self.g.E(edge.id["@value"]["relationId"]).next()

        return updated_edge

    def update_edge_properties(self, edge: Edge, properties: dict) -> Edge:
        """Update the properties of an existing edge and return the updated edge.

        Args:
            edge (Edge): Edge to update.
            properties (dict): Dictionary of properties to update.
        """

        # Update the properties of the existing edge
        for key, value in properties.items():
            self.g.E(edge.id["@value"]["relationId"]).property(key, value).iterate()

        # Fetch and return the updated edge
        updated_edge = self.g.E(edge.id["@value"]["relationId"]).next()

        return updated_edge

    def get_edges_by_vertices_and_label(
        self, head: Vertex, label: str, tail: Vertex
    ) -> list[Edge]:
        """Find an edge by its label and property.

        Args:
            head (Vertex): Head vertex of the edge.
            label (str): Label of the edge.
            tail (Vertex): Tail vertex of the edge.

        Returns:
            list of Edge: List of edges with the provided label.
        """
        return self.g.V(head.id).outE(label).where(__.inV().hasId(tail.id)).toList()

    def get_edges_by_label(self, label: str) -> list[Edge]:
        """Find an edge by its label.

        Args:
            label (str): Label of the edge.

        Returns:
            list of Edge: List of edges with the provided label.
        """

        return self.g.E().hasLabel(label).toList()

    def get_edges_by_properties(
        self, include_keys: list[str], exclude_keys: list[str] = []
    ) -> list[Edge]:
        """Find edges based on included and excluded properties.

        Args:
            include_keys (list of str): List of properties that must be included.
            exclude_keys (list of str, optional): List of properties that must be
                excluded.

        Returns:
            list of Edge: List of edges matching the criteria.
        """
        traversal = self.g.E()

        # Add filters for properties to include
        for key in include_keys:
            traversal = traversal.has(key)

        # Add filters for properties to exclude
        if exclude_keys:
            for key in exclude_keys:
                traversal = traversal.hasNot(key)

        return traversal.toList()

    def get_edges_by_label_and_properties(
        self, label: str, include_keys: list[str] = [], exclude_keys: list[str] = []
    ) -> list[Edge]:
        """
        Find edges by label and filter them based on included and excluded properties.

        Args:
            label (str): The label to search for.
            include_keys (list of str, optional): List of properties that must be
                included.
            exclude_keys (list of str, optional): List of properties that must be
                excluded.

        Returns:
            list of Edge: List of edges matching the criteria.
        """
        traversal = self.g.E().hasLabel(label)

        # Add filters for properties to include
        if include_keys:
            for key in include_keys:
                traversal = traversal.has(key)

        # Add filters for properties to exclude
        if exclude_keys:
            for key in exclude_keys:
                traversal = traversal.hasNot(key)

        return traversal.toList()

    def get_edges_between_vertices(self, vertices: list[Vertex]) -> list[Edge]:
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
            self.g.V(vertex_ids)  # Start with the given vertex IDs
            .bothE()  # Traverse all edges connected to these vertices
            .where(
                __.otherV().hasId(P.within(vertex_ids))
            )  # Ensure the other end is in the vertex set
            .dedup()  # Avoid duplicates
            .toList()  # Convert traversal result to a list
        )

        return edges_between_vertices

    def get_properties(self, vertex_or_edge: Vertex | Edge) -> dict:
        """Retrieve all properties of a vertex or edge, decoding JSON-encoded values.

        Args:
            vertex_or_edge (Vertex | Edge): Vertex or edge to retrieve properties for.

        Returns:
            dict: Dictionary of properties for the element.
        """
        if vertex_or_edge.properties is None:
            return {}

        return {prop.key: prop.value for prop in vertex_or_edge.properties}

    def write_time_vertex(self, timestamp: str) -> Vertex:
        """Write a time vertex to the graph.

        Args:
            timestamp (str): The timestamp (ISO 8601 with seconds) of the vertex.
        """
        assert is_iso8601_datetime(timestamp), "Timestamp must be an ISO 8601 datetime."

        vertex = self.create_vertex(timestamp, {"timestamp": True})

        return vertex

    def write_short_term_vertex(
        self, label: str, time_vertex: Vertex, properties: dict = {}
    ) -> Vertex:
        """
        Write a new short-term vertex to the graph.

        This does not check if a vertex with the same label is in the database or not.

        Args:
            label (str): Label of the vertex.
            time_vertex (Vertex): Time vertex of the vertex.
            properties (dict): Properties of the vertex.

        Returns:
            Vertex: The newly created short-term memory vertex.

        """
        # Step 1: Create a vertex with the given label and properties
        short_term_vertex = self.create_vertex(label, properties)
        self.logger.debug(f"Created vertex with ID: {short_term_vertex.id}")

        # Step 2: Connect the vertex to the time vertex
        has_short_term_memory_edge = self.create_edge(
            time_vertex, "has_short_term_memory", short_term_vertex
        )
        self.logger.debug(f"Created edge with ID: {has_short_term_memory_edge.id}")

        return short_term_vertex

    def write_edge(
        self,
        head_vertex: Vertex,
        edge_label: str,
        tail_vertex: Vertex,
        properties: dict = {},
    ) -> Edge:
        """
        Write a new short-term edge to the graph.

        This does not check if an edge with the same label is in the database or not.

        Args:
            head_vertex (Vertex): Head vertex of the edge.
            edge_label (str): Label of the edge.
            tail_vertex (Vertex): Tail vertex of the edge.
            properties (dict): Properties of the edge.

        Returns:
            Edge: The newly created edge.
        """
        edge = self.create_edge(head_vertex, edge_label, tail_vertex, properties)
        self.logger.debug(f"Created edge with ID: {edge.id}")

        return edge

    def move_short_term_vertex(self, vertex: Vertex, action: str) -> None:
        """Move the short-term vertex to another memory type.

        Args:
            vertex (Vertex): The vertex to be moved.
            action (str): The action to be taken. Choose from 'episodic' or 'semantic'

        """
        assert (
            self.g.V(vertex).inE("has_short_term_memory").hasNext()
        ), "The vertex must have an incoming edge 'has_short_term_memory'."

        short_term_edge = self.g.V(vertex).inE("has_short_term_memory").next()

        assert (
            self.g.V(vertex).inE("has_short_term_memory").outV().hasNext()
        ), "The incoming edge 'has_short_term_memory' must have a head vertex."

        time_vertex = self.g.V(vertex).inE("has_short_term_memory").outV().next()

        assert is_iso8601_datetime(time_vertex.label), "The time vertex must be valid."

        if action == "episodic":
            episodic_edge = self.create_edge(
                time_vertex, "has_episodic_memory", vertex, {"num_recalled": 0}
            )
            self.remove_edge(short_term_edge)

            self.logger.debug(f"Moved vertex to episodic memory with ID: {vertex.id}")

        elif action == "semantic":
            semantic_edge = self.create_edge(
                time_vertex, "has_semantic_memory", vertex, {"num_recalled": 0}
            )
            self.remove_edge(short_term_edge)

            self.logger.debug(f"Moved vertex to semantic memory with ID: {vertex.id}")

        else:
            self.logger.error("Invalid action. Choose from 'episodic' or 'semantic'.")
            raise ValueError("Invalid action. Choose from 'episodic' or 'semantic'.")

    def remove_all_short_term(self) -> None:
        """Remove all pure short-term vertices and edges.

        This method removes all the short-term edges.

        """
        short_term_edges = self.get_edges_by_label("has_short_term_memory")

        for edge in short_term_edges:
            self.remove_edge(edge)
            self.logger.debug(f"Removed short-term edge with ID: {edge.id}")

        # # Remove the timestamp vertices without edges
        # for vertex in self.g.V().has("timestamp", True).not_(__.bothE()).toList():
        #     self.remove_vertex(vertex)
        #     self.logger.debug(f"Removed timestamp vertex with ID: {vertex.id}")

    def write_long_term_vertex(
        self,
        label: str,
        memory_type: str,
        time_vertex: Vertex,
        properties: dict = {},
    ) -> Vertex:
        """Write a new long-term vertex to the graph.

        This is directly writing a vertex to the long-term memory.

        Args:
            label (str): Label of the vertex.
            memory_type (str): Type of memory to write to. Choose from 'episodic' or
                'semantic'.
            time_vertex (Vertex): Time vertex of the vertex.
            properties (dict): Properties of the vertex.

        Returns:
            Vertex: The updated vertex.
        """

        # Step 1: Create a vertex with the given label and properties
        vertex = self.create_vertex(label, properties)
        self.logger.debug(f"Created vertex with ID: {vertex.id}")

        # Step 2 (optional): Connect the vertex to the time vertex
        if memory_type == "episodic":
            has_episodic_memory_edge = self.create_edge(
                time_vertex, "has_episodic_memory", vertex, {"num_recalled": 0}
            )
            self.logger.debug(f"Created edge with ID: {has_episodic_memory_edge.id}")
        elif memory_type == "semantic":
            has_semantic_memory_edge = self.create_edge(
                time_vertex, "has_semantic_memory", vertex, {"num_recalled": 0}
            )
            self.logger.debug(f"Created edge with ID: {has_semantic_memory_edge.id}")
        else:
            self.logger.error("No memory type specified. The vertex is not connected.")
            raise ValueError("No memory type specified. The vertex is not connected.")

        return vertex

    def _increment_num_recalled(self, edges: list[Edge]) -> list[Edge]:
        """Helper function to increment 'num_recalled' on Edges

        Args:
            edges (list of Edge): List of edges to be updated.

        Returns:
            list of Edge: List of updated edges.

        """
        edges_updated = []
        for edge in edges:
            num_recalled = self.get_properties(edge).get("num_recalled")
            edge = self.update_edge_properties(edge, {"num_recalled": num_recalled + 1})
            edges_updated.append(edge)

        return edges_updated

    def get_working(
        self,
        include_all_long_term: bool = True,
        hops: int = None,
    ) -> tuple[list[Vertex], list[Edge], list[Vertex], list[Edge]]:
        """
        Retrieves the working memory based on the short-term memories.

        Currently, we first fetch long-term vertices whose "label" values are the same
        as those of the short-term vertices. These fetched long-term memories are then
        used as triggers. We then fetch long-term vertices and edges within N hops from
        the triggers.

        Args:
            short_term_vertices (list of Vertex): List of short-term vertices.
            include_all_long_term (bool): If True, include all long-term memories.
            hops (int): Number of hops to traverse from the trigger node.

        Returns:
            tuple: short-term vertices, short-term edges, long-term vertices, long-term
                edges.
        """
        short_term_vertices, short_term_edges = self.get_all_short_term()
        long_term_edge_labels = ["has_episodic_memory", "has_semantic_memory"]

        if len(short_term_vertices) == 0:
            self.logger.debug("Short-term memory is emtpy")
            return [], [], [], []

        if include_all_long_term:
            long_term_vertices, _ = self.get_all_long_term()

        else:
            assert (
                hops is not None
            ), "hops must be provided when include_all_long_term is False."

            short_term_vertex_labels = [vertex.label for vertex in short_term_vertices]

            long_term_trigger_vertices = (
                self.g.V()
                .hasLabel(*short_term_vertex_labels)
                .inE(*long_term_edge_labels)
                .inV()
                .toList()
            )
            if len(long_term_trigger_vertices) == 0:
                return short_term_vertices, short_term_edges, [], []
            else:
                long_term_vertices = self.get_vertices_within_hops(
                    long_term_trigger_vertices, hops, exclude_keys=["timestamp"]
                )

        # increment num_recalled for the long-term memories
        incoming_edges = (
            self.g.V(long_term_vertices)  # Start from the given vertices
            .inE(*long_term_edge_labels)
            .toList()  # Convert to a list of edges
        )
        self._increment_num_recalled(incoming_edges)

        long_term_edges = self.get_edges_between_vertices(long_term_vertices)

        return (
            short_term_vertices,
            short_term_edges,
            long_term_vertices,
            long_term_edges,
        )

    def get_vertices_within_hops(
        self,
        vertices: list[Vertex],
        hops: int,
        include_keys: list[str] = [],
        exclude_keys: list[str] = [],
    ) -> list[Vertex]:
        """Retrieve all vertices within N hops from a starting vertex.

        Args:
            vertices (list[Vertex]): List of starting vertex IDs for the traversal.
            hops (int): Number of hops to traverse from the starting vertex.
            include_keys (list of str, optional): List of properties that must be
                included.
            exclude_keys (list of str, optional): List of properties that must be
                excluded.

        Returns:
            list[Vertex]: List of vertices within N hops from the starting vertex.
        """
        assert hops >= 0, "Number of hops must be a non-negative integer."
        assert isinstance(vertices, list), "Vertices must be provided as a list."

        if hops == 0:
            # Directly return the vertices themselves when hops is 0
            return self.g.V([v.id for v in vertices]).toList()

        # Perform traversal for N hops
        traversal = (
            self.g.V([v.id for v in vertices])  # Start from the provided vertex IDs
            .emit()  # Emit the starting vertex
            .repeat(__.both().simplePath())  # Traverse to neighbors
            .times(hops)  # Limit the number of hops
            .dedup()  # Avoid duplicate vertices in the result
        )

        # Add filters for properties to include
        if include_keys:
            for key in include_keys:
                traversal = traversal.has(key)

        # Add filters for properties to exclude
        if exclude_keys:
            for key in exclude_keys:
                traversal = traversal.hasNot(key)

        # Execute the traversal and return the results
        return traversal.toList()

    def save_db_as_json(self, json_name: str = "db.json") -> None:
        """Read the database as a JSON file.

        Args:
            json_name (str): The name of the JSON file.
        """
        self.g.io(json_name).write().iterate()

        copy_file_from_docker(
            self.janusgraph_container_name, f"/opt/janusgraph/{json_name}", json_name
        )

    def load_db_from_json(self, json_name: str = "db.json") -> None:
        """Write a JSON file to the database.

        Args:
            json_name (str): The name of the JSON file.
        """
        copy_file_to_docker(
            self.janusgraph_container_name, json_name, f"/opt/janusgraph/{json_name}"
        )

        self.g.io(json_name).read().iterate()

    def get_all_short_term(self) -> tuple[list[Vertex], list[Edge]]:
        """
        Retrieve all short-term vertices and edges from the graph.

        Returns:
            tuple: List of short-term vertices and edges.
        """
        vertices = self.g.V().inE("has_short_term_memory").inV().toList()
        edges = self.get_edges_between_vertices(vertices)

        return vertices, edges

    def get_all_long_term(self) -> tuple[list[Vertex], list[Edge]]:
        """
        Retrieve all long-term vertices and edges from the graph.

        Returns:
            tuple: List of long-term vertices and edges.
        """
        vertices = (
            self.g.V().inE("has_episodic_memory", "has_semantic_memory").inV().toList()
        )
        edges = self.get_edges_between_vertices(vertices)

        return vertices, edges

    def get_all_episodic(self) -> tuple[list[Vertex], list[Edge]]:
        """
        Retrieve all episodic vertices and edges from the graph.

        Returns:
            tuple: List of episodic vertices and edges.
        """
        vertices = self.g.V().inE("has_episodic_memory").inV().toList()
        edges = self.get_edges_between_vertices(vertices)

        return vertices, edges

    def get_all_semantic(self) -> tuple[list[Vertex], list[Edge]]:
        """
        Retrieve all semantic vertices and edges from the graph.

        Returns:
            tuple: List of semantic vertices and edges.
        """
        vertices = self.g.V().inE("has_semantic_memory").inV().toList()
        edges = self.get_edges_between_vertices(vertices)

        return vertices, edges

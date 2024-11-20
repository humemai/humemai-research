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
from humemai.janusgraph.utils.gremlin import (
    remove_all_data,
    get_all_vertices,
    get_all_edges,
    find_vertex_by_label,
    find_vertices_by_properties,
    remove_vertex,
    remove_edge,
    find_edge_by_vertices_and_label,
    find_edge_by_label,
    find_edges_by_properties,
    create_vertex,
    create_edge,
    update_vertex_properties,
    remove_vertex_properties,
    update_edge_properties,
    remove_edge_properties,
    get_vertices_within_hops,
    get_edges_between_vertices,
    get_properties,
)
from humemai.janusgraph.utils.docker import (
    start_containers,
    stop_containers,
    remove_containers,
)

from humemai.utils import is_iso8601_datetime


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("humemai.janusgraph.Humemai")


class Humemai:
    def __init__(
        self,
        cassandra_container_name="cassandra",
        janusgraph_container_name="janusgraph",
        gremlin_server_url="ws://localhost:8182/gremlin",
        gremlin_traversal_source="g",
        cassandra_data_dir="./cassandra_data",
        configs_dir="./configs",
        janusgraph_config="janusgraph.properties",
        gremlin_server_config="gremlin-server.yaml",
    ) -> None:
        """
        Initialize a Humemai object for connecting to JanusGraph and Cassandra
        containers.

        Args:
            cassandra_container_name (str): Name of the Cassandra container.
            janusgraph_container_name (str): Name of the JanusGraph container.
            gremlin_server_url (str): URL for connecting to the Gremlin server.
            gremlin_traversal_source (str): Traversal source name for Gremlin.
            cassandra_data_dir (str): Directory for Cassandra data persistence.
            configs_dir (str): Directory containing JanusGraph and Gremlin Server
            configuration files.
            janusgraph_config (str): JanusGraph configuration file.
            gremlin_server_config (str): Gremlin Server configuration file.
        """

        self.cassandra_container_name = cassandra_container_name
        self.janusgraph_container_name = janusgraph_container_name
        self.gremlin_server_url = gremlin_server_url
        self.gremlin_traversal_source = gremlin_traversal_source
        self.cassandra_data_dir = cassandra_data_dir
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
        """Start the Cassandra and JanusGraph containers with optional warmup time.

        Args:
            warmup_seconds (int): Number of seconds to wait after starting the
                containers
        """
        start_containers(
            configs_dir=self.configs_dir,
            janusgraph_config=self.janusgraph_config,
            gremlin_server_config=self.gremlin_server_config,
            cassandra_data_dir=self.cassandra_data_dir,
            cassandra_container_name=self.cassandra_container_name,
            janusgraph_container_name=self.janusgraph_container_name,
            warmup_seconds=warmup_seconds,
        )

    def stop_containers(self) -> None:
        """Stop the Cassandra and JanusGraph containers."""
        stop_containers(
            cassandra_container_name=self.cassandra_container_name,
            janusgraph_container_name=self.janusgraph_container_name,
        )

    def remove_containers(self) -> None:
        """Remove the Cassandra and JanusGraph containers."""
        remove_containers(
            cassandra_container_name=self.cassandra_container_name,
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
            remove_all_data(self.g)
        else:
            self.logger.warning("Graph traversal source (g) is not initialized.")

    def write_short_term_vertex(self, label: str, properties: dict = {}) -> Vertex:
        """
        Write a new short-term vertex to the graph.

        if 'current_time' property does not exist in 'properties', it will be added.

        Args:
            label (str): Label of the vertex.
            properties (dict): Properties of the vertex.

        Returns:
            Vertex: The newly created vertex.
        """
        if "current_time" not in properties:
            properties["current_time"] = datetime.now().isoformat(timespec="seconds")
        else:
            assert is_iso8601_datetime(
                properties["current_time"]
            ), "current_time must be an ISO8601 datetime (timespec=seconds)."

        vertices = find_vertex_by_label(self.g, label)

        if len(vertices) == 0:
            vertex = create_vertex(self.g, label, properties)
            self.logger.debug(f"Created vertex with ID: {vertex.id}")

        else:
            self.logger.warning(
                f"Vertex with label '{label}' already exists. "
                f"Disambiguation might be needed."
            )
            vertex = vertices[0]
            existing_properties = get_properties(vertex)

            # merge properties and existing_properties if there are the same keys,
            # then `properties` takes over
            properties = {**existing_properties, **properties}

            vertex = update_vertex_properties(self.g, vertex, properties)
            self.logger.debug(f"Updated vertex with ID: {vertex.id}")

        return vertex

    def write_short_term_edge(
        self,
        head_vertex: Vertex,
        edge_label: str,
        tail_vertex: Vertex,
        properties: dict = {},
    ) -> Edge:
        """
        Write a new short-term edge to the graph.

        if 'current_time' property does not exist in 'properties', it will be added.

        Args:
            head_vertex (Vertex): Head vertex of the edge.
            edge_label (str): Label of the edge.
            tail_vertex (Vertex): Tail vertex of the edge.
            properties (dict): Properties of the edge.

        Returns:
            Edge: The newly created edge.
        """
        if "current_time" not in properties:
            properties["current_time"] = datetime.now().isoformat(timespec="seconds")
        else:
            assert is_iso8601_datetime(
                properties["current_time"]
            ), "current_time must be an ISO8601 datetime (timespec=seconds)."

        edges = find_edge_by_vertices_and_label(
            self.g, head_vertex, edge_label, tail_vertex
        )

        if len(edges) == 0:
            edge = create_edge(self.g, head_vertex, edge_label, tail_vertex, properties)
            self.logger.debug(f"Created edge with ID: {edge.id}")

        else:
            self.logger.warning(
                f"Edge with label '{edge_label}' already exists. "
                f"Disambiguation might be needed."
            )
            edge = edges[0]
            existing_properties = get_properties(edge)

            # merge properties and existing_properties if there are the same keys,
            # then `properties` takes over
            properties = {**existing_properties, **properties}

            edge = update_edge_properties(self.g, edge, properties)
            self.logger.debug(f"Updated edge with ID: {edge.id}")

        return edge

    def move_short_term_vertex(self, vertex: Vertex, action: str) -> Vertex | None:
        """Move the short-term vertex to another memory type.

        Args:
            vertex (Vertex): The vertex to be moved.
            action (str): The action to be taken. Choose from 'episodic' or 'semantic'

        Returns:
            Vertex: The updated vertex

        """
        new_properties = {}
        assert "current_time" in self.get_vertex_properties(
            vertex
        ), "current_time must exist."

        if action == "episodic":
            if "event_time" in get_properties(vertex):
                new_properties["event_time"] = get_properties(vertex)["event_time"] + [
                    get_properties(vertex)["current_time"]
                ]
            else:
                new_properties["event_time"] = [get_properties(vertex)["current_time"]]

            if "num_recalled" not in get_properties(vertex):
                new_properties["num_recalled"] = 0

            vertex = update_vertex_properties(self.g, vertex, new_properties)
            self.logger.debug(f"Moved vertex to episodic memory with ID: {vertex.id}")

        elif action == "semantic":
            # look for the existing known_since
            if "known_since" in get_properties(vertex):
                new_properties["known_since"] = get_properties(vertex)["known_since"]
            else:
                new_properties["known_since"] = get_properties(vertex)["current_time"]

            if "num_recalled" not in get_properties(vertex):
                new_properties["num_recalled"] = 0

            vertex = update_vertex_properties(self.g, vertex, new_properties)
            self.logger.debug(f"Moved vertex to semantic memory with ID: {vertex.id}")

        else:
            self.logger.error("Invalid action. Choose from 'episodic' or 'semantic'.")
            raise ValueError("Invalid action. Choose from 'episodic' or 'semantic'.")

        return vertex

    def move_short_term_edge(self, edge: Edge, action: str) -> Edge | None:
        """Move the short-term edge to another memory type.

        Args:
            edge (Edge): The edge to be moved.
            action (str): The action to be taken. Choose from 'episodic' or 'semantic'

        Returns:
            Edge: The updated edge.

        """
        assert "current_time" in self.get_edge_properties(
            edge
        ), "current_time must exist."
        new_properties = {}

        if action == "episodic":
            if "event_time" in get_properties(edge):
                new_properties["event_time"] = get_properties(edge)["event_time"] + [
                    get_properties(edge)["current_time"]
                ]
            else:
                new_properties["event_time"] = [get_properties(edge)["current_time"]]

            if "num_recalled" not in get_properties(edge):
                new_properties["num_recalled"] = 0

            edge = update_edge_properties(self.g, edge, new_properties)
            self.logger.debug(f"Moved edge to episodic memory with ID: {edge.id}")

        elif action == "semantic":
            # look for the existing known_since
            if "known_since" in get_properties(edge):
                new_properties["known_since"] = get_properties(edge)["known_since"]
            else:
                new_properties["known_since"] = get_properties(edge)["current_time"]

            if "num_recalled" not in get_properties(edge):
                new_properties["num_recalled"] = 0

            edge = update_edge_properties(self.g, edge, new_properties)
            self.logger.debug(f"Moved edge to semantic memory with ID: {edge.id}")

        else:
            self.logger.error("Invalid action. Choose from 'episodic' or 'semantic'.")
            raise ValueError("Invalid action. Choose from 'episodic' or 'semantic'.")

        return edge

    def remove_all_short_term(self) -> None:
        """Remove all pure short-term vertices and edges.

        This method removes all vertices and edges that have 'current_time' property,
        but not 'num_recalled', 'event_time', or 'known_since' properties, meaning that
        they are pure short-term memories. This will also remove the 'current_time'
        property from all the vertices and edges. Call this after you are done with
        moving short-term memories to other memory types.

        """
        # Step 1: Remove all pure short-term vertices and edges
        short_term_vertices = find_vertices_by_properties(
            self.g, ["current_time"], ["num_recalled"]
        )
        for vertex in short_term_vertices:
            remove_vertex(self.g, vertex)

        short_term_edges = find_edges_by_properties(
            self.g, ["current_time"], ["num_recalled"]
        )
        for edge in short_term_edges:
            remove_edge(self.g, edge)

        # Step 2: Remove 'current_time' property from all vertices and edges
        for vertex in self.get_all_short_term_vertices():
            remove_vertex_properties(self.g, vertex, ["current_time"])
        for edge in self.get_all_short_term_edges():
            remove_edge_properties(self.g, edge, ["current_time"])

    def write_long_term_vertex(self, label: str, properties: dict = {}) -> Vertex:
        """
        Write a new long-term vertex to the graph.

        This is directly writing a vertex to the long-term memory, so the vertex must
        should include the time information (e.g., 'event_time' or 'known_since'). One
        thing to note is that the Vertex might already exist in the long-term memory.
        In this case, the properties will be updated. You should add 'num_recalled=0'
        property!

        Args:
            label (str): Label of the vertex.
            properties (dict): Properties of the vertex.

        Returns:
            Vertex: The updated vertex.
        """
        assert "current_time" not in properties, "current_time must not be included."
        assert (
            "event_time" in properties or "known_since" in properties
        ), "event_time or known_since must be included."
        assert (
            "num_recalled" in properties and properties["num_recalled"] == 0
        ), "num_recalled must be included and set to 0."

        vertices = find_vertex_by_label(self.g, label)

        if len(vertices) == 0:
            vertex = create_vertex(self.g, label, properties)
            self.logger.debug(f"Created vertex with ID: {vertex.id}")

        else:
            self.logger.error(
                f"{len(vertices)} vertices found. Disambiguation might be needed."
            )
            raise ValueError(
                f"{len(vertices)} vertices found. We need to disambiguate."
            )

        return vertex

    def write_long_term_edge(
        self,
        head_vertex: Vertex,
        edge_label: str,
        tail_vertex: Vertex,
        properties: dict = {},
    ) -> Edge:
        """
        Write a new long-term edge to the graph.

        This is directly writing an edge to the long-term memory, so the edge must
        should include the time information (e.g., 'event_time' or 'known_since'). One
        thing to note is that the Edge might already exist in the long-term memory.
        In this case, the properties will be updated. If not, a new edge will be
        created, with 'num_recalled=0' property added.

        Args:
            head_vertex (Vertex): Head vertex of the edge.
            edge_label (str): Label of the edge.
            tail_vertex (Vertex): Tail vertex of the edge.
            properties (dict): Properties of the edge.

        Returns:
            Edge: The updated edge.
        """
        assert "current_time" not in properties, "current_time must not be included."
        assert (
            "event_time" in properties or "known_since" in properties
        ), "event_time or known_since must be included."

        edges = find_edge_by_vertices_and_label(
            self.g, head_vertex, edge_label, tail_vertex
        )

        if len(edges) == 0:
            # Add 'num_recalled' property
            properties["num_recalled"] = 0
            edge = create_edge(self.g, head_vertex, edge_label, tail_vertex, properties)
            self.logger.debug(f"Created edge with ID: {edge.id}")

        else:
            self.logger.error(
                f"{len(edges)} edges found. Disambiguation might be needed."
            )
            raise ValueError(f"{len(edges)} edges found. We need to disambiguate.")

        return edge

    def _increment_num_recalled_vertices_and_edges(
        self, vertices: list[Vertex], edges: list[Edge]
    ) -> tuple[list[Vertex], list[Edge]]:
        """Helper function to increment 'num_recalled' on vertices and edges

        Args:
            vertices (list of Vertex): List of vertices to be updated.
            edges (list of Edge): List of edges to be updated.

        Returns:
            tuple: A tuple of updated vertices and edges.

        """
        vertices_updated = []
        for vertex in vertices:
            num_recalled = get_properties(vertex).get("num_recalled")
            vertex = update_vertex_properties(
                self.g, vertex, {"num_recalled": num_recalled + 1}
            )
            vertices_updated.append(vertex)

        edges_updated = []
        for edge in edges:
            num_recalled = get_properties(edge).get("num_recalled")
            edge = update_edge_properties(
                self.g, edge, {"num_recalled": num_recalled + 1}
            )
            edges_updated.append(edge)

        return vertices_updated, edges_updated

    def get_working_vertices_and_edges(
        self,
        short_term_vertices: list[Vertex],
        short_term_edges: list[Edge],
        include_all_long_term: bool = True,
        hops: int = None,
    ) -> tuple[list[Vertex], list[Vertex], list[Edge], list[Edge]]:
        """
        Retrieves the working memory based on the short-term memories.

        The short-term memories are used as a trigger to retrieve the working memory.

        Args:
            short_term_vertices (list of Vertex): List of short-term vertices.
            short_term_edges (list of Edge): List of short-term edges.
            include_all_long_term (bool): If True, include all long-term memories.
            hops (int): Number of hops to traverse from the trigger node.

        Returns:
            tuple: A tuple of short-term vertices, long-term vertices, short-term edges,
                and long-term edges.
        """
        if len(short_term_vertices) == 0 or len(short_term_edges) == 0:
            self.logger.error("Short-term vertices and edges must not be empty.")
            raise ValueError("Short-term vertices and edges must not be empty.")
        if include_all_long_term:
            long_term_vertices = self.get_all_long_term_vertices()
            long_term_edges = self.get_all_long_term_edges()

        else:
            assert (
                hops is not None
            ), "hops must be provided when include_all_long_term is False."

            long_term_vertices = get_vertices_within_hops(
                self.g, short_term_vertices, hops
            )

            long_term_edges = get_edges_between_vertices(self.g, long_term_vertices)

        long_term_vertices = [
            vertex for vertex in long_term_vertices if vertex not in short_term_vertices
        ]

        long_term_edges = [
            edge for edge in long_term_edges if edge not in short_term_edges
        ]

        # Increment 'num_recalled' on long-term vertices and edges
        long_term_vertices, long_term_edges = (
            self._increment_num_recalled_vertices_and_edges(
                long_term_vertices, long_term_edges
            )
        )

        return (
            short_term_vertices,
            long_term_vertices,
            short_term_edges,
            long_term_edges,
        )

    def get_all_vertices(self) -> list[Vertex]:
        """
        Retrieve all vertices from the graph.

        Returns:
            list of Vertex: List of all vertices.
        """
        return get_all_vertices(self.g)

    def get_all_edges(self) -> list[Edge]:
        """
        Retrieve all edges from the graph.

        Returns:
            list of Edge: List of all edges.
        """
        return get_all_edges(self.g)

    def get_all_short_term_vertices(self) -> list[Vertex]:
        """
        Retrieve all short-term vertices from the graph.

        Returns:
            list of Vertex: List of short-term vertices.
        """
        return find_vertices_by_properties(self.g, ["current_time"])

    def get_all_short_term_edges(self) -> list[Edge]:
        """
        Retrieve all short-term edges from the graph.

        Returns:
            list of Edge: List of short-term edges.
        """
        return find_edges_by_properties(self.g, ["current_time"])

    def get_all_long_term_vertices(self) -> list[Vertex]:
        """
        Retrieve all long-term vertices from the graph.

        Returns:
            list of Vertex: List of long-term vertices.
        """
        return find_vertices_by_properties(self.g, ["num_recalled"])

    def get_all_long_term_edges(self) -> list[Edge]:
        """
        Retrieve all long-term edges from the graph.

        Returns:
            list of Edge: List of long-term edges.
        """
        return find_edges_by_properties(self.g, ["num_recalled"])

    def get_all_episodic_vertices(self) -> list[Vertex]:
        """
        Retrieve all episodic vertices from the graph.

        Returns:
            list of Vertex: List of episodic vertices.
        """
        return find_vertices_by_properties(self.g, ["event_time"])

    def get_all_episodic_edges(self) -> list[Edge]:
        """
        Retrieve all episodic edges from the graph.

        Returns:
            list of Edge: List of episodic edges.
        """
        return find_edges_by_properties(self.g, ["event_time"])

    def get_all_semantic_vertices(self) -> list[Vertex]:
        """
        Retrieve all semantic vertices from the graph.

        Returns:
            list of Vertex: List of semantic vertices.
        """
        return find_vertices_by_properties(self.g, ["known_since"])

    def get_all_semantic_edges(self) -> list[Edge]:
        """
        Retrieve all semantic edges from the graph.

        Returns:
            list of Edge: List of semantic edges.
        """
        return find_edges_by_properties(self.g, ["known_since"])

    def get_vertex_properties(self, vertex: Vertex) -> dict:
        """
        Retrieve the properties of a vertex.

        Args:
            vertex (Vertex): The vertex to retrieve properties from.

        Returns:
            dict: The properties of the vertex.
        """
        return get_properties(vertex)

    def get_edge_properties(self, edge: Edge) -> dict:
        """
        Retrieve the properties of an edge.

        Args:
            edge (Edge): The edge to retrieve properties from.

        Returns:
            dict: The properties of the edge.
        """
        return get_properties(edge)

    def remove_vertex(self, vertex: Vertex) -> None:
        """
        Remove a vertex from the graph.

        Args:
            vertex (Vertex): The vertex to be removed.
        """
        remove_vertex(self.g, vertex)

    def remove_edge(self, edge: Edge) -> None:
        """
        Remove an edge from the graph.

        Args:
            edge (Edge): The edge to be removed.
        """
        remove_edge(self.g, edge)

    def find_vertex_by_label(self, label: str) -> list[Vertex]:
        """
        Find vertices by label.

        Args:
            label (str): The label to search for.

        Returns:
            list of Vertex: List of vertices with the given label.
        """
        return find_vertex_by_label(self.g, label)

"""Test utils gremlin"""

import unittest
import os
from datetime import datetime
import docker
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.structure.graph import Graph

from humemai.janusgraph.utils.docker import (
    start_containers,
    stop_containers,
    remove_containers,
)

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
    get_properties,
    get_vertices_within_hops,
    get_edges_between_vertices,
)


class TestGremlinUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up resources once for all tests."""
        start_containers(
            warmup_seconds=30,
        )
        cls.graph = Graph()
        cls.connection = DriverRemoteConnection(
            "ws://localhost:8182/gremlin",
            "g",
            message_serializer=GraphSONSerializersV3d0(),
        )
        cls.g = cls.graph.traversal().withRemote(cls.connection)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests."""
        cls.connection.close()
        stop_containers(cassandra_container_name="foo", janusgraph_container_name="bar")
        remove_containers(
            cassandra_container_name="foo", janusgraph_container_name="bar"
        )

    def test_vertex_methods(self):
        remove_all_data(self.g)
        vertices = find_vertex_by_label(self.g, "Alice")
        self.assertEqual(len(vertices), 0)

        vertex = create_vertex(self.g, "Alice", {})
        vertices = get_all_vertices(self.g)

        self.assertEqual(len(vertices), 1)
        self.assertEqual(vertices[0], vertex)

        vertices = find_vertex_by_label(self.g, "Alice")
        self.assertEqual(len(vertices), 1)

        create_vertex(self.g, "Alice", {})

        vertices = find_vertex_by_label(self.g, "Alice")
        self.assertEqual(len(vertices), 2)

        remove_vertex(self.g, vertices[1])

        self.assertEqual(len(get_all_vertices(self.g)), 1)

        update_vertex_properties(self.g, vertices[0], {"foo": [1, 2, 3], "bar": "baz"})
        vertices = find_vertex_by_label(self.g, "Alice")
        self.assertEqual(len(vertices), 1)
        self.assertEqual(len(find_vertices_by_properties(self.g, ["bar"])), 1)

        vertex = vertices[0]
        props = {prop.key: prop.value for prop in vertex.properties}
        self.assertEqual(props["foo"], "[1, 2, 3]")
        self.assertEqual(props["bar"], "baz")

        update_vertex_properties(self.g, vertex, {"foo": [4, 5, 6]})
        vertices = find_vertex_by_label(self.g, "Alice")
        self.assertEqual(len(vertices), 1)

        vertex = vertices[0]
        props = {prop.key: prop.value for prop in vertex.properties}
        self.assertEqual(props["foo"], "[4, 5, 6]")
        self.assertEqual(props["bar"], "baz")

        remove_vertex_properties(self.g, vertex, ["foo"])
        vertices = find_vertex_by_label(self.g, "Alice")
        self.assertEqual(len(vertices), 1)
        props = {prop.key: prop.value for prop in vertices[0].properties}
        self.assertNotIn("foo", props)
        self.assertEqual(props["bar"], "baz")

    def test_edge_methods(self):
        remove_all_data(self.g)
        alice = create_vertex(self.g, "Alice", {})
        bob = create_vertex(self.g, "Bob", {})

        edges = find_edge_by_vertices_and_label(self.g, alice, "knows", bob)
        self.assertEqual(len(edges), 0)

        edge = create_edge(self.g, alice, "knows", bob, {"foo": [1, 2, 3]})

        edges = get_all_edges(self.g)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0], edge)

        edges = find_edge_by_vertices_and_label(self.g, alice, "knows", bob)
        self.assertEqual(len(edges), 1)

        create_edge(self.g, alice, "knows", bob, {})

        edges = find_edge_by_vertices_and_label(self.g, alice, "knows", bob)
        self.assertEqual(len(edges), 2)
        self.assertEqual(len(find_edge_by_label(self.g, "knows")), 2)

        remove_edge(self.g, edges[1])

        self.assertEqual(len(get_all_edges(self.g)), 1)

        update_edge_properties(self.g, edge, {"foo": [4, 5, 6], "bar": "baz"})
        edges = find_edge_by_vertices_and_label(self.g, alice, "knows", bob)
        self.assertEqual(len(edges), 1)

        edge = edges[0]
        props = {prop.key: prop.value for prop in edge.properties}
        self.assertEqual(props["foo"], "[4, 5, 6]")
        self.assertEqual(props["bar"], "baz")
        self.assertEqual(len(find_edges_by_properties(self.g, ["bar"])), 1)

        remove_edge_properties(self.g, edge, ["foo"])
        edges = find_edge_by_vertices_and_label(self.g, alice, "knows", bob)
        self.assertEqual(len(edges), 1)
        props = {prop.key: prop.value for prop in edges[0].properties}
        self.assertNotIn("foo", props)
        self.assertEqual(props["bar"], "baz")

    def test_hops(self):
        remove_all_data(self.g)
        vertex_a = create_vertex(
            self.g, "Alice", {"timestamps": [1, 2, 3], "bar": "baz"}
        )
        vertex_b = create_vertex(
            self.g, "Bob", {"foo": {"my_dict": "value"}, "bar": "qux"}
        )
        vertex_c = create_vertex(self.g, "Charlie", {})
        vertex_d = create_vertex(self.g, "David", {"age": 30, "type": "person"})
        vertex_e = create_vertex(
            self.g, "Eve", {"age": 25, "type": "person", "gender": "M"}
        )
        vertex_f = create_vertex(self.g, "Frank", {"age": 40, "type": "agent"})
        vertex_g = create_vertex(
            self.g,
            "Grace",
            {
                "timestamp": [
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            },
        )

        edge_da = create_edge(
            self.g,
            vertex_d,
            "likes",
            vertex_a,
            {"current_time": datetime.now().isoformat(timespec="seconds")},
        )
        edge_ab = create_edge(
            self.g, vertex_a, "kills", vertex_b, {"since": 2019, "bar": [1, 2]}
        )
        edge_ba = create_edge(
            self.g, vertex_b, "hates", vertex_a, {"since": 2018, "bar": [3, 4]}
        )

        edge_fb = create_edge(
            self.g,
            vertex_f,
            "knows",
            vertex_b,
            {
                "current_time": [
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            },
        )
        edge_cb = create_edge(self.g, vertex_c, "knows", vertex_b, {"since": 2020})
        edge_cg = create_edge(
            self.g,
            vertex_c,
            "knows",
            vertex_g,
            {
                "timestamp": [
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            },
        )

        edge_ce = create_edge(
            self.g,
            vertex_c,
            "loves",
            vertex_e,
            {
                "timestamp": [
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            },
        )
        edge_ge = create_edge(
            self.g,
            vertex_g,
            "friend_of",
            vertex_e,
            {
                "timestamp": [
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            },
        )

        # test get_vertices_within_hops()
        vertices = get_all_vertices(self.g)
        self.assertEqual(len(vertices), 7)
        edges = get_all_edges(self.g)
        self.assertEqual(len(edges), 8)

        vertices = get_vertices_within_hops(self.g, [vertex_a], 0)
        self.assertEqual(len(vertices), 1)
        self.assertEqual(vertices[0], vertex_a)

        vertices = get_vertices_within_hops(self.g, [vertex_a, vertex_c, vertex_g], 0)
        self.assertEqual(len(vertices), 3)
        self.assertEqual(set(vertices), {vertex_a, vertex_c, vertex_g})

        vertices = get_vertices_within_hops(self.g, [vertex_b, vertex_f], 1)
        self.assertEqual(len(vertices), 4)
        self.assertEqual(set(vertices), {vertex_a, vertex_b, vertex_f, vertex_c})

        vertices = get_vertices_within_hops(self.g, [vertex_a, vertex_b], 2)
        self.assertEqual(len(vertices), 7)
        self.assertEqual(
            set(vertices),
            {vertex_a, vertex_b, vertex_c, vertex_d, vertex_e, vertex_f, vertex_g},
        )

        vertices = get_vertices_within_hops(self.g, [vertex_d], 3)
        self.assertEqual(len(vertices), 5)
        self.assertEqual(
            set(vertices), {vertex_a, vertex_b, vertex_c, vertex_d, vertex_f}
        )

        vertices = get_vertices_within_hops(self.g, [vertex_f], 4)
        self.assertEqual(len(vertices), 7)
        self.assertEqual(
            set(vertices),
            {vertex_a, vertex_b, vertex_c, vertex_d, vertex_e, vertex_f, vertex_g},
        )

        # test get_edges_between_vertices()
        edges = get_edges_between_vertices(self.g, [vertex_a])
        self.assertEqual(len(edges), 0)
        edges = get_edges_between_vertices(self.g, [vertex_g])
        self.assertEqual(len(edges), 0)

        edges = get_edges_between_vertices(self.g, [vertex_a, vertex_b])
        self.assertEqual(len(edges), 2)
        self.assertEqual(
            set([edge.id["@value"]["relationId"] for edge in edges]),
            {edge_ba.id["@value"]["relationId"], edge_ab.id["@value"]["relationId"]},
        )

        edges = get_edges_between_vertices(self.g, [vertex_a, vertex_b, vertex_c])
        self.assertEqual(len(edges), 3)
        self.assertEqual(
            set([edge.id["@value"]["relationId"] for edge in edges]),
            {
                edge_ba.id["@value"]["relationId"],
                edge_ab.id["@value"]["relationId"],
                edge_cb.id["@value"]["relationId"],
            },
        )

        edges = get_edges_between_vertices(
            self.g, [vertex_a, vertex_b, vertex_c, vertex_g]
        )
        self.assertEqual(len(edges), 4)
        self.assertEqual(
            set([edge.id["@value"]["relationId"] for edge in edges]),
            {
                edge_ba.id["@value"]["relationId"],
                edge_ab.id["@value"]["relationId"],
                edge_cb.id["@value"]["relationId"],
                edge_cg.id["@value"]["relationId"],
            },
        )

        edges = get_edges_between_vertices(
            self.g, [vertex_a, vertex_b, vertex_c, vertex_g, vertex_e]
        )
        self.assertEqual(len(edges), 6)
        self.assertEqual(
            set([edge.id["@value"]["relationId"] for edge in edges]),
            {
                edge_ba.id["@value"]["relationId"],
                edge_ab.id["@value"]["relationId"],
                edge_cb.id["@value"]["relationId"],
                edge_cg.id["@value"]["relationId"],
                edge_ce.id["@value"]["relationId"],
                edge_ge.id["@value"]["relationId"],
            },
        )

        edges = get_edges_between_vertices(
            self.g, [vertex_a, vertex_b, vertex_c, vertex_g, vertex_e, vertex_f]
        )
        self.assertEqual(len(edges), 7)
        self.assertEqual(
            set([edge.id["@value"]["relationId"] for edge in edges]),
            {
                edge_ba.id["@value"]["relationId"],
                edge_ab.id["@value"]["relationId"],
                edge_cb.id["@value"]["relationId"],
                edge_cg.id["@value"]["relationId"],
                edge_ce.id["@value"]["relationId"],
                edge_ge.id["@value"]["relationId"],
                edge_fb.id["@value"]["relationId"],
            },
        )

        edges = get_edges_between_vertices(
            self.g,
            [vertex_a, vertex_b, vertex_c, vertex_g, vertex_e, vertex_f, vertex_d],
        )
        self.assertEqual(len(edges), 8)
        self.assertEqual(
            set([edge.id["@value"]["relationId"] for edge in edges]),
            {
                edge_ba.id["@value"]["relationId"],
                edge_ab.id["@value"]["relationId"],
                edge_cb.id["@value"]["relationId"],
                edge_cg.id["@value"]["relationId"],
                edge_ce.id["@value"]["relationId"],
                edge_ge.id["@value"]["relationId"],
                edge_fb.id["@value"]["relationId"],
                edge_da.id["@value"]["relationId"],
            },
        )

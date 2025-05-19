"""Unittests for Humemai basic functions."""

import unittest
from datetime import datetime

from gremlin_python.process.graph_traversal import __

from humemai.janusgraph import Humemai


class TestHumemai(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Start containers, connect to Gremlin, and initialize Humemai instance."""
        cls.humemai = Humemai(
            warmup_seconds=300,
            container_prefix="test",
            janusgraph_public_port=8182 + 10,
            cassandra_port_9042=9042 + 10,
            cassandra_port_9160=9160 + 10,
            elasticsearch_public_port=9200 + 10,
            visualizer_port_1=3000 + 10,
            visualizer_port_2=3001 + 10,
        )
        cls.humemai.connect()
        cls.humemai.remove_all_data()

    @classmethod
    def tearDownClass(cls) -> None:
        """Disconnect and stop containers after all tests."""
        cls.humemai.disconnect()
        cls.humemai.stop_docker_compose()
        cls.humemai.remove_docker_compose()

    def test_write_short_term(self):
        """Test writing a short-term vertex and index."""
        self.humemai.remove_all_data()

        properties = {"current_time": datetime.now().isoformat(timespec="seconds")}
        vertex_a = self.humemai.write_short_term_vertex("Alice", properties)
        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 1)
        self.assertEqual(len(edges), 0)

        vertex_a = self.humemai.write_short_term_vertex("Alice", properties)
        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 0)

        vertex_b = self.humemai.write_short_term_vertex("Bob", properties)

        properties["years"] = 5

        edge = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, properties
        )
        edge = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, properties
        )

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 2)

    def test_move_short_term(self):
        """Test moving short-term vertices and edges to episodic long-term."""
        self.humemai.remove_all_data()

        properties = {"current_time": datetime.now().isoformat(timespec="seconds")}

        vertex_a = self.humemai.write_short_term_vertex("Alice", properties)
        vertex_b = self.humemai.write_short_term_vertex("Bob", properties)
        vertex_c = self.humemai.write_short_term_vertex("Charlie", properties)

        edge_ab = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, properties
        )
        edge_bc = self.humemai.write_short_term_edge(
            vertex_b, "likes", vertex_c, properties
        )
        edge_cb = self.humemai.write_short_term_edge(
            vertex_c, "friend_of", vertex_b, properties
        )

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertex_a = self.humemai.move_short_term_vertex(vertex_a, "episodic")
        vertex_b = self.humemai.move_short_term_vertex(vertex_b, "episodic")
        edge_ab = self.humemai.move_short_term_edge(edge_ab, "episodic")

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 1)
        self.assertEqual(len(edges), 2)
        for vertex in vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
            self.assertNotIn("time_added", self.humemai.get_properties(vertex))
            self.assertNotIn("known_since", self.humemai.get_properties(vertex))
            self.assertNotIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
            self.assertNotIn("time_added", self.humemai.get_properties(edge))
            self.assertNotIn("known_since", self.humemai.get_properties(edge))
            self.assertNotIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)
        for vertex in vertices:
            self.assertNotIn("current_time", self.humemai.get_properties(vertex))
            self.assertIn("time_added", self.humemai.get_properties(vertex))
            self.assertNotIn("known_since", self.humemai.get_properties(vertex))
            self.assertIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertNotIn("current_time", self.humemai.get_properties(edge))
            self.assertIn("time_added", self.humemai.get_properties(edge))
            self.assertNotIn("known_since", self.humemai.get_properties(edge))
            self.assertIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)
        for vertex in vertices:
            self.assertNotIn("current_time", self.humemai.get_properties(vertex))
            self.assertIn("time_added", self.humemai.get_properties(vertex))
            self.assertNotIn("known_since", self.humemai.get_properties(vertex))
            self.assertIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertNotIn("current_time", self.humemai.get_properties(edge))
            self.assertIn("time_added", self.humemai.get_properties(edge))
            self.assertNotIn("known_since", self.humemai.get_properties(edge))
            self.assertIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        self.humemai.remove_all_short_term()

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        properties = {"current_time": datetime.now().isoformat(timespec="seconds")}

        vertex_b = self.humemai.write_short_term_vertex("Bob", properties)
        vertex_c = self.humemai.write_short_term_vertex("Charlie", properties)

        edge_bc = self.humemai.write_short_term_edge(
            vertex_b, "likes", vertex_c, properties
        )
        edge_cb = self.humemai.write_short_term_edge(
            vertex_c, "friend_of", vertex_b, properties
        )

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 2)
        for vertex in vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
            self.assertNotIn("time_added", self.humemai.get_properties(vertex))
            self.assertNotIn("known_since", self.humemai.get_properties(vertex))
            self.assertNotIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
            self.assertNotIn("time_added", self.humemai.get_properties(edge))
            self.assertNotIn("known_since", self.humemai.get_properties(edge))
            self.assertNotIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)
        for vertex in vertices:
            self.assertNotIn("current_time", self.humemai.get_properties(vertex))
            self.assertIn("time_added", self.humemai.get_properties(vertex))
            self.assertNotIn("known_since", self.humemai.get_properties(vertex))
            self.assertIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertNotIn("current_time", self.humemai.get_properties(edge))
            self.assertIn("time_added", self.humemai.get_properties(edge))
            self.assertNotIn("known_since", self.humemai.get_properties(edge))
            self.assertIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)
        for vertex in vertices:
            self.assertNotIn("current_time", self.humemai.get_properties(vertex))
            self.assertIn("time_added", self.humemai.get_properties(vertex))
            self.assertNotIn("known_since", self.humemai.get_properties(vertex))
            self.assertIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertNotIn("current_time", self.humemai.get_properties(edge))
            self.assertIn("time_added", self.humemai.get_properties(edge))
            self.assertNotIn("known_since", self.humemai.get_properties(edge))
            self.assertIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        self.humemai.move_short_term_vertex(vertex_b, "semantic")
        self.humemai.move_short_term_vertex(vertex_c, "semantic")
        edge_bc = self.humemai.move_short_term_edge(edge_bc, "semantic")
        edge_cb = self.humemai.move_short_term_edge(edge_cb, "semantic")

        self.humemai.remove_all_short_term()

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(edges), 3)
        for vertex in vertices:
            self.assertNotIn("current_time", self.humemai.get_properties(vertex))
            self.assertIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertNotIn("current_time", self.humemai.get_properties(edge))
            self.assertIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)
        for vertex in vertices:
            self.assertNotIn("current_time", self.humemai.get_properties(vertex))
            self.assertIn("time_added", self.humemai.get_properties(vertex))
            self.assertNotIn("known_since", self.humemai.get_properties(vertex))
            self.assertIn("num_recalled", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertNotIn("current_time", self.humemai.get_properties(edge))
            self.assertIn("time_added", self.humemai.get_properties(edge))
            self.assertNotIn("known_since", self.humemai.get_properties(edge))
            self.assertIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 2)
        for vertex in vertices:
            self.assertNotIn("current_time", self.humemai.get_properties(vertex))
            self.assertNotIn("time_added", self.humemai.get_properties(vertex))
            self.assertIn("num_recalled", self.humemai.get_properties(vertex))
            self.assertIn("known_since", self.humemai.get_properties(vertex))
        for edge in edges:
            self.assertNotIn("current_time", self.humemai.get_properties(edge))
            self.assertNotIn("time_added", self.humemai.get_properties(edge))
            self.assertIn("num_recalled", self.humemai.get_properties(edge))
            self.assertIn("known_since", self.humemai.get_properties(edge))

    def test_write_remove_long_term(self):
        """Test writing and removing long-term vertices and edges."""
        self.humemai.remove_all_data()

        timestamp = datetime.now().isoformat(timespec="seconds")

        vertex_a = self.humemai.write_long_term_vertex(
            "Alice", {"time_added": timestamp, "foo": 123}
        )
        vertex_b = self.humemai.write_long_term_vertex(
            "Bob", {"time_added": timestamp, "known_since": timestamp, "foo": 234}
        )
        vertex_c = self.humemai.write_long_term_vertex(
            "Charlie", {"known_since": timestamp, "foo": 345}
        )

        edge_ab = self.humemai.write_long_term_edge(
            vertex_a,
            "knows",
            vertex_b,
            {"time_added": timestamp, "foo": 234},
        )
        edge_bc = self.humemai.write_long_term_edge(
            vertex_b,
            "likes",
            vertex_c,
            {"time_added": timestamp, "known_since": timestamp, "foo": 234},
        )

        edge_cb = self.humemai.write_long_term_edge(
            vertex_c,
            "friend_of",
            vertex_b,
            {"known_since": timestamp, "foo": 345},
        )

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)
        for vertice in vertices:
            self.assertIn("num_recalled", self.humemai.get_properties(vertice))
        for edge in edges:
            self.assertIn("num_recalled", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 2)
        for vertice in vertices:
            self.assertIn("num_recalled", self.humemai.get_properties(vertice))
            self.assertIn("time_added", self.humemai.get_properties(vertice))
        for edge in edges:
            self.assertIn("num_recalled", self.humemai.get_properties(edge))
            self.assertIn("time_added", self.humemai.get_properties(edge))

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 2)
        for vertice in vertices:
            self.assertIn("num_recalled", self.humemai.get_properties(vertice))
            self.assertIn("known_since", self.humemai.get_properties(vertice))
        for edge in edges:
            self.assertIn("num_recalled", self.humemai.get_properties(edge))
            self.assertIn("known_since", self.humemai.get_properties(edge))

    def test_get_all_long_term_in_time_range(self):
        """Test getting all semantic vertices and edges in a time range."""

        self.humemai.remove_all_data()
        vertex_alice = self.humemai.write_long_term_vertex(
            "Alice", {"time_added": "2021-01-01T00:00:00"}
        )
        vertex_bob = self.humemai.write_long_term_vertex(
            "Bob", {"time_added": "2021-01-02T01:10:00"}
        )
        vertex_charlie = self.humemai.write_long_term_vertex(
            "Charlie", {"time_added": "2021-01-03T02:20:00"}
        )
        vertex_david = self.humemai.write_long_term_vertex(
            "David", {"time_added": "2021-01-04T03:30:00"}
        )

        edge_ab = self.humemai.write_long_term_edge(
            vertex_alice, "knows", vertex_bob, {"time_added": "2021-01-01T00:00:00"}
        )

        start_time = "2021-01-01T00:00:00"
        end_time = "2021-01-03T00:00:00"
        vertices, edges = self.humemai.get_all_episodic_in_time_range(
            start_time, end_time
        )
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        start_time = "2021-01-02T01:10:01"
        end_time = "2021-01-04T03:30:00"
        vertices, edges = self.humemai.get_all_episodic_in_time_range(
            start_time, end_time
        )
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 0)

        vertex_alice = self.humemai.write_long_term_vertex(
            "Alice", {"known_since": "2021-01-01T00:00:00"}
        )
        vertex_bob = self.humemai.write_long_term_vertex(
            "Bob", {"known_since": "2021-01-02T01:10:00"}
        )
        vertex_charlie = self.humemai.write_long_term_vertex(
            "Charlie", {"known_since": "2021-01-03T02:20:00"}
        )
        vertex_david = self.humemai.write_long_term_vertex(
            "David", {"known_since": "2021-01-04T03:30:00"}
        )

        edge_cd = self.humemai.write_long_term_edge(
            vertex_charlie,
            "knows",
            vertex_david,
            {"known_since": "2021-01-03T02:20:00"},
        )

        start_time = "2021-01-01T00:00:00"
        end_time = "2021-01-03T00:00:00"
        vertices, edges = self.humemai.get_all_semantic_in_time_range(
            start_time, end_time
        )
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 0)

        start_time = "2021-01-02T01:10:01"
        end_time = "2021-01-04T03:30:00"
        vertices, edges = self.humemai.get_all_semantic_in_time_range(
            start_time, end_time
        )
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        start_time = "2021-01-01T00:00:00"
        end_time = "2021-01-03T00:00:00"
        vertices, edges = self.humemai.get_all_long_term_in_time_range(
            start_time, end_time
        )
        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(edges), 1)

        start_time = "2021-01-02T01:10:01"
        end_time = "2021-01-04T03:30:00"
        vertices, edges = self.humemai.get_all_long_term_in_time_range(
            start_time, end_time
        )
        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(edges), 1)

    def test_get_within_hops(self) -> None:
        """Test getting vertices within a certain number of hops."""
        self.humemai.remove_all_data()

        current_time = datetime.now().isoformat(timespec="seconds")

        vertex_d = self.humemai.write_vertex("D", {"current_time": current_time})
        vertex_a = self.humemai.write_vertex("A", {"time_added": current_time})
        vertex_b = self.humemai.write_vertex("B", {"current_time": current_time})
        vertex_f = self.humemai.write_vertex(
            "F", {"known_since": current_time, "current_time": current_time}
        )
        vertex_c = self.humemai.write_vertex("C", {"current_time": current_time})
        vertex_e = self.humemai.write_vertex("E", {"current_time": current_time})
        vertex_g = self.humemai.write_vertex("G", {"known_since": current_time})

        edge_da = self.humemai.write_edge(
            vertex_d, "works_at", vertex_a, {"time_added": current_time, "foo": 123}
        )
        edge_ab = self.humemai.write_edge(
            vertex_a, "owns", vertex_b, {"current_time": current_time}
        )
        edge_ba = self.humemai.write_edge(
            vertex_b, "owned_by", vertex_a, {"known_since": current_time}
        )
        edge_fb = self.humemai.write_edge(
            vertex_f, "works_at", vertex_b, {"current_time": current_time}
        )
        edge_cb = self.humemai.write_edge(
            vertex_c, "works_at", vertex_b, {"current_time": current_time}
        )
        edge_ce = self.humemai.write_edge(
            vertex_c,
            "knows",
            vertex_e,
            {
                "time_added": current_time,
                "known_since": current_time,
                "current_time": current_time,
            },
        )
        edge_ge = self.humemai.write_edge(
            vertex_g, "created_by", vertex_e, {"current_time": current_time}
        )
        edge_cg = self.humemai.write_edge(
            vertex_c, "created", vertex_g, {"current_time": current_time}
        )

        with self.assertRaises(AssertionError):
            self.humemai.get_within_hops([vertex_d], 0)

        vertices, edges = self.humemai.get_within_hops([vertex_d], 1)
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_within_hops([vertex_d], 2)
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        vertices, edges = self.humemai.get_within_hops([vertex_d], 3)
        self.assertEqual(len(vertices), 5)
        self.assertEqual(len(edges), 5)

        vertices, edges = self.humemai.get_within_hops([vertex_c], 2)
        self.assertEqual(len(vertices), 6)
        self.assertEqual(len(edges), 7)

        vertices, edges = self.humemai.get_within_hops([vertex_a, vertex_b], 1)
        self.assertEqual(len(vertices), 5)
        self.assertEqual(len(edges), 5)

    def test_connect_duplicate_vertices(self) -> None:
        """Test connecting duplicate vertices."""
        self.humemai.remove_all_data()
        current_time = datetime.now().isoformat(timespec="seconds")

        vertex_a = self.humemai.write_short_term_vertex(
            "A", {"current_time": current_time}
        )
        vertex_b = self.humemai.write_short_term_vertex(
            "B", {"current_time": current_time}
        )
        vertex_c = self.humemai.write_short_term_vertex(
            "C", {"current_time": current_time}
        )
        vertex_d = self.humemai.write_short_term_vertex(
            "D", {"current_time": current_time}
        )

        edge_ab = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, {"current_time": current_time}
        )
        edge_bc = self.humemai.write_short_term_edge(
            vertex_b, "likes", vertex_c, {"current_time": current_time}
        )
        edge_cd = self.humemai.write_short_term_edge(
            vertex_c, "is_friend_of", vertex_d, {"current_time": current_time}
        )

        vertex_e = self.humemai.write_short_term_vertex(
            "E", {"current_time": current_time}
        )
        vertex_f = self.humemai.write_short_term_vertex(
            "F", {"current_time": current_time}
        )
        vertex_g = self.humemai.write_short_term_vertex(
            "G", {"current_time": current_time}
        )

        edge_ef = self.humemai.write_short_term_edge(
            vertex_e, "knows", vertex_f, {"current_time": current_time}
        )
        edge_fg = self.humemai.write_short_term_edge(
            vertex_f, "likes", vertex_g, {"current_time": current_time}
        )

        vertex_a_prime = self.humemai.write_short_term_vertex(
            "A", {"current_time": current_time}
        )
        vertex_b_prime = self.humemai.write_short_term_vertex(
            "B", {"current_time": current_time}
        )

        edge_ab_prime = self.humemai.write_short_term_edge(
            vertex_a_prime, "knows", vertex_b_prime, {"current_time": current_time}
        )

        with self.assertRaises(NotImplementedError):
            self.humemai.connect_duplicate_vertices(match_logic="fuzzy")

        self.humemai.connect_duplicate_vertices(match_logic="exact_label")

        vertices = self.humemai.get_vertices_by_partial_label("meta_")
        self.assertEqual(len(vertices), 2)

        edges = self.humemai.get_edges_by_label("has_meta_node")
        self.assertEqual(len(edges), 4)

        vertices = self.humemai.g.V().where(__.outE("has_meta_node")).toList()
        self.assertEqual(len(vertices), 4)
        for vertex in vertices:
            self.assertIn(vertex.label, ["A", "B"])

    def test_get_working(self):
        """Test getting working memory. THIS IS A VERY IMPORTANT FUNCTION."""
        self.humemai.remove_all_data()

        with self.assertRaises(NotImplementedError):
            self.humemai.get_working(match_logic="fuzzy", hops=1)

        with self.assertRaises(AssertionError):
            self.humemai.get_working(match_logic="exact_label", hops=0)

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(match_logic="exact_label", hops=1)
        )
        self.assertEqual(len(short_term_vertices), 0)
        self.assertEqual(len(short_term_edges), 0)
        self.assertEqual(len(long_term_vertices), 0)
        self.assertEqual(len(long_term_edges), 0)

        current_time = datetime.now().isoformat(timespec="seconds")

        vertex_a = self.humemai.write_short_term_vertex(
            "A", {"current_time": current_time}
        )
        vertex_b = self.humemai.write_short_term_vertex(
            "B", {"current_time": current_time}
        )
        edge_ab = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, {"current_time": current_time}
        )

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(match_logic="exact_label", hops=1)
        )
        self.assertEqual(len(short_term_vertices), 2)
        self.assertEqual(len(short_term_edges), 1)
        self.assertEqual(len(long_term_vertices), 0)
        self.assertEqual(len(long_term_edges), 0)

        self.humemai.move_short_term_vertex(vertex_a, "episodic")
        self.humemai.move_short_term_vertex(vertex_b, "episodic")
        self.humemai.move_short_term_edge(edge_ab, "episodic")

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(match_logic="exact_label", hops=1)
        )
        self.assertEqual(len(short_term_vertices), 0)
        self.assertEqual(len(short_term_edges), 0)
        self.assertEqual(len(long_term_vertices), 0)
        self.assertEqual(len(long_term_edges), 0)

        vertex_c = self.humemai.write_short_term_vertex(
            "C", {"current_time": current_time}
        )
        vertex_d = self.humemai.write_short_term_vertex(
            "D", {"current_time": current_time}
        )
        edge_cd = self.humemai.write_short_term_edge(
            vertex_c, "knows", vertex_d, {"current_time": current_time}
        )

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(match_logic="exact_label", hops=1)
        )
        self.assertEqual(len(short_term_vertices), 2)
        self.assertEqual(len(short_term_edges), 1)
        self.assertEqual(len(long_term_vertices), 0)
        self.assertEqual(len(long_term_edges), 0)

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=True
            )
        )
        self.assertEqual(len(short_term_vertices), 2)
        self.assertEqual(len(short_term_edges), 1)
        self.assertEqual(len(long_term_vertices), 2)
        self.assertEqual(len(long_term_edges), 1)
        for vertex in short_term_vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
        for edge in short_term_edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
        for vertex in long_term_vertices:
            self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 1)
        for edge in long_term_edges:
            self.assertEqual(self.humemai.get_properties(edge)["num_recalled"], 1)

        self.humemai.move_short_term_vertex(vertex_c, "semantic")
        self.humemai.move_short_term_vertex(vertex_d, "semantic")
        self.humemai.move_short_term_edge(edge_cd, "semantic")

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=True
            )
        )
        self.assertEqual(len(short_term_vertices), 0)
        self.assertEqual(len(short_term_edges), 0)
        self.assertEqual(len(long_term_vertices), 0)
        self.assertEqual(len(long_term_edges), 0)

        vertex_d_prime = self.humemai.write_short_term_vertex(
            "D", {"current_time": current_time}
        )
        vertex_e = self.humemai.write_short_term_vertex(
            "E", {"current_time": current_time}
        )
        vertex_f = self.humemai.write_short_term_vertex(
            "F", {"current_time": current_time}
        )
        edge_de = self.humemai.write_short_term_edge(
            vertex_d_prime, "knows", vertex_e, {"current_time": current_time}
        )
        edge_ef = self.humemai.write_short_term_edge(
            vertex_e, "likes", vertex_f, {"current_time": current_time}
        )

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=False, hops=2
            )
        )

        self.assertEqual(len(short_term_vertices), 3)
        self.assertEqual(len(short_term_edges), 2)
        self.assertEqual(len(long_term_vertices), 2)
        self.assertEqual(len(long_term_edges), 1)
        for vertex in short_term_vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
        for edge in short_term_edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
        for vertex in long_term_vertices:
            self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 1)
            self.assertIn(vertex.label, ["C", "D"])
        for edge in long_term_edges:
            self.assertEqual(self.humemai.get_properties(edge)["num_recalled"], 1)

        self.humemai.move_short_term_vertex(vertex_d_prime, "semantic")
        self.humemai.move_short_term_vertex(vertex_e, "semantic")
        self.humemai.move_short_term_vertex(vertex_f, "semantic")
        self.humemai.move_short_term_edge(edge_de, "semantic")
        self.humemai.move_short_term_edge(edge_ef, "semantic")

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=False, hops=2
            )
        )
        self.assertEqual(len(short_term_vertices), 0)
        self.assertEqual(len(short_term_edges), 0)
        self.assertEqual(len(long_term_vertices), 0)
        self.assertEqual(len(long_term_edges), 0)

        self.humemai.connect_duplicate_vertices()

        vertex_e_prime = self.humemai.write_short_term_vertex(
            "E", {"current_time": current_time}
        )
        vertex_g = self.humemai.write_short_term_vertex(
            "G", {"current_time": current_time}
        )

        edge_ge = self.humemai.write_short_term_edge(
            vertex_e_prime, "knows", vertex_g, {"current_time": current_time}
        )

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=True, hops=1
            )
        )

        self.assertEqual(len(short_term_vertices), 2)
        self.assertEqual(len(short_term_edges), 1)
        self.assertEqual(len(long_term_vertices), 7)
        self.assertEqual(len(long_term_edges), 4)
        for vertex in short_term_vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
        for edge in short_term_edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
        for vertex in long_term_vertices:
            if vertex.label in ["A", "B", "C"]:
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 2)
            elif vertex.label == "D":
                self.assertIn(
                    self.humemai.get_properties(vertex)["num_recalled"], [1, 2]
                )
            elif vertex.label in ["E", "F"]:
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 1)
            else:
                raise ValueError("Unexpected vertex label.")
        for edge in long_term_edges:
            self.assertIn(self.humemai.get_properties(edge)["num_recalled"], [1, 2])

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=False, hops=1
            )
        )
        self.assertEqual(len(short_term_vertices), 2)
        self.assertEqual(len(short_term_edges), 1)
        self.assertEqual(len(long_term_vertices), 4)
        self.assertEqual(len(long_term_edges), 2)
        for vertex in short_term_vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
        for edge in short_term_edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
        for vertex in long_term_vertices:
            if vertex.label == "D":
                self.assertIn(
                    self.humemai.get_properties(vertex)["num_recalled"], [2, 3]
                )
            elif vertex.label in ["E", "F"]:
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 2)
            else:
                raise ValueError("Unexpected vertex label.")
        for edge in long_term_edges:
            self.assertEqual(self.humemai.get_properties(edge)["num_recalled"], 2)

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=False, hops=3
            )
        )
        self.assertEqual(len(short_term_vertices), 2)
        self.assertEqual(len(short_term_edges), 1)
        self.assertEqual(len(long_term_vertices), 5)
        self.assertEqual(len(long_term_edges), 3)
        for vertex in short_term_vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
        for edge in short_term_edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
        for vertex in long_term_vertices:
            if vertex.label == "D":
                self.assertIn(
                    self.humemai.get_properties(vertex)["num_recalled"], [3, 4]
                )
            elif vertex.label in ["C", "E", "F"]:
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 3)
            else:
                raise ValueError("Unexpected vertex label.")
        for edge in long_term_edges:
            self.assertEqual(self.humemai.get_properties(edge)["num_recalled"], 3)

        self.humemai.move_short_term_vertex(vertex_e_prime, "episodic")
        self.humemai.move_short_term_vertex(vertex_g, "episodic")
        self.humemai.move_short_term_edge(edge_ge, "episodic")

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=True, hops=3
            )
        )
        self.assertEqual(len(short_term_vertices), 0)
        self.assertEqual(len(short_term_edges), 0)
        self.assertEqual(len(long_term_vertices), 0)
        self.assertEqual(len(long_term_edges), 0)

        self.humemai.connect_duplicate_vertices()

        current_time = datetime.now().isoformat(timespec="seconds")

        g_prime = self.humemai.write_short_term_vertex(
            "G", {"current_time": current_time}
        )
        h = self.humemai.write_short_term_vertex("H", {"current_time": current_time})
        i = self.humemai.write_short_term_vertex("I", {"current_time": current_time})

        edge_hg = self.humemai.write_short_term_edge(
            h, "knows", g_prime, {"current_time": current_time}
        )
        edge_hi = self.humemai.write_short_term_edge(
            h, "likes", i, {"current_time": current_time}
        )

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=False, hops=1
            )
        )
        self.assertEqual(len(short_term_vertices), 3)
        self.assertEqual(len(short_term_edges), 2)
        self.assertEqual(len(long_term_vertices), 3)
        self.assertEqual(len(long_term_edges), 1)
        for vertex in short_term_vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
        for edge in short_term_edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
        for vertex in long_term_vertices:
            if vertex.label == "G":
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 1)
            elif vertex.label == "E":
                self.assertIn(
                    self.humemai.get_properties(vertex)["num_recalled"], [1, 4]
                )
            else:
                raise ValueError("Unexpected vertex label.")
        for edge in long_term_edges:
            self.assertEqual(self.humemai.get_properties(edge)["num_recalled"], 1)

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(
                match_logic="exact_label", include_all_long_term=False, hops=3
            )
        )
        self.assertEqual(len(short_term_vertices), 3)
        self.assertEqual(len(short_term_edges), 2)
        self.assertEqual(len(long_term_vertices), 7)
        self.assertEqual(len(long_term_edges), 4)
        for vertex in short_term_vertices:
            self.assertIn("current_time", self.humemai.get_properties(vertex))
        for edge in short_term_edges:
            self.assertIn("current_time", self.humemai.get_properties(edge))
        for vertex in long_term_vertices:
            if vertex.label == "C":
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 4)
            elif vertex.label == "D":
                self.assertIn(
                    self.humemai.get_properties(vertex)["num_recalled"], [4, 5]
                )
            elif vertex.label == "E":
                self.assertIn(
                    self.humemai.get_properties(vertex)["num_recalled"], [2, 5]
                )
            elif vertex.label == "F":
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 4)
            elif vertex.label == "G":
                self.assertEqual(self.humemai.get_properties(vertex)["num_recalled"], 2)
            else:
                raise ValueError("Unexpected vertex label.")

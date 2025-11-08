"""Unittests for Humemai basic functions."""

import unittest
from datetime import datetime
from humemai_research.janusgraph import Humemai


class TestHumemai(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Start containers, connect to Gremlin, and initialize Humemai instance."""
        cls.humemai = Humemai()
        cls.humemai.start_containers(warmup_seconds=30)
        cls.humemai.connect()
        cls.humemai.remove_all_data()

    @classmethod
    def tearDownClass(cls) -> None:
        """Disconnect and stop containers after all tests."""
        cls.humemai.disconnect()
        cls.humemai.stop_containers()
        cls.humemai.remove_containers()

    def test_write_short_term(self):
        """Test writing a short-term vertex and index."""
        self.humemai.remove_all_data()
        vertex_a = self.humemai.write_short_term_vertex("Alice", {"age": 30})
        self.assertEqual(self.humemai.get_all_short_term_vertices()[0], vertex_a)
        vertex_a = self.humemai.write_short_term_vertex("Alice", {"age": 30})
        self.assertEqual(self.humemai.get_all_short_term_vertices()[0], vertex_a)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 1)

        vertex_a = self.humemai.get_all_short_term_vertices()[0]
        self.assertIn("current_time", self.humemai.get_vertex_properties(vertex_a))

        with self.assertRaises(AssertionError):
            self.humemai.write_short_term_vertex(
                "Alice", {"age": 30, "current_time": "2021-01-01T00:00:00:00"}
            )

        vertex_b = self.humemai.write_short_term_vertex("Bob")
        edge = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, {"years": 5}
        )
        edge = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, {"years": 5}
        )
        self.assertEqual(self.humemai.get_all_short_term_edges()[0], edge)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_edges()), 1)

    def test_move_short_term_episodic(self):
        """Test moving short-term vertices and edges to episodic long-term."""
        self.humemai.remove_all_data()

        vertex_a = self.humemai.write_short_term_vertex("Alice")
        vertex_b = self.humemai.write_short_term_vertex("Bob", {})
        vertex_c = self.humemai.write_short_term_vertex("Charlie", {})

        edge_ab = self.humemai.write_short_term_edge(vertex_a, "knows", vertex_b)
        edge_bc = self.humemai.write_short_term_edge(vertex_b, "likes", vertex_c)
        edge_cb = self.humemai.write_short_term_edge(vertex_c, "friend_of", vertex_b)

        self.assertEqual(len(self.humemai.get_all_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 0)

        vertex_a = self.humemai.move_short_term_vertex(vertex_a, "episodic")
        vertex_b = self.humemai.move_short_term_vertex(vertex_b, "episodic")
        edge_ab = self.humemai.move_short_term_edge(edge_ab, "episodic")

        self.assertEqual(len(self.humemai.get_all_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 0)

        self.assertIn("num_recalled", self.humemai.get_vertex_properties(vertex_a))
        self.assertIn("current_time", self.humemai.get_vertex_properties(vertex_a))
        self.assertIn("event_time", self.humemai.get_vertex_properties(vertex_a))
        self.assertIsInstance(
            self.humemai.get_vertex_properties(vertex_a)["event_time"], list
        )
        self.assertNotIn("known_since", self.humemai.get_vertex_properties(vertex_a))

        self.assertIn("num_recalled", self.humemai.get_vertex_properties(vertex_b))
        self.assertIn("current_time", self.humemai.get_vertex_properties(vertex_b))
        self.assertIn("event_time", self.humemai.get_vertex_properties(vertex_b))
        self.assertIsInstance(
            self.humemai.get_vertex_properties(vertex_b)["event_time"], list
        )
        self.assertNotIn("known_since", self.humemai.get_vertex_properties(vertex_b))

        self.assertNotIn("num_recalled", self.humemai.get_vertex_properties(vertex_c))
        self.assertIn("current_time", self.humemai.get_vertex_properties(vertex_c))
        self.assertNotIn("event_time", self.humemai.get_vertex_properties(vertex_c))
        self.assertNotIn("known_since", self.humemai.get_vertex_properties(vertex_c))

        self.assertIn("num_recalled", self.humemai.get_edge_properties(edge_ab))
        self.assertIn("current_time", self.humemai.get_edge_properties(edge_ab))
        self.assertIn("event_time", self.humemai.get_edge_properties(edge_ab))
        self.assertIsInstance(
            self.humemai.get_edge_properties(edge_ab)["event_time"], list
        )
        self.assertNotIn("known_since", self.humemai.get_edge_properties(edge_ab))

        self.assertNotIn("num_recalled", self.humemai.get_edge_properties(edge_bc))
        self.assertIn("current_time", self.humemai.get_edge_properties(edge_bc))
        self.assertNotIn("event_time", self.humemai.get_edge_properties(edge_bc))
        self.assertNotIn("known_since", self.humemai.get_edge_properties(edge_bc))

        self.assertNotIn("num_recalled", self.humemai.get_edge_properties(edge_cb))
        self.assertIn("current_time", self.humemai.get_edge_properties(edge_cb))
        self.assertNotIn("event_time", self.humemai.get_edge_properties(edge_cb))
        self.assertNotIn("known_since", self.humemai.get_edge_properties(edge_cb))

        self.humemai.remove_all_short_term()

        self.assertEqual(len(self.humemai.get_all_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 0)

        vertices = self.humemai.get_all_vertices()
        for vertice in vertices:
            self.assertIn("num_recalled", self.humemai.get_vertex_properties(vertice))
            self.assertNotIn(
                "current_time", self.humemai.get_vertex_properties(vertice)
            )
            self.assertIn("event_time", self.humemai.get_vertex_properties(vertice))
            self.assertIsInstance(
                self.humemai.get_vertex_properties(vertice)["event_time"], list
            )
            self.assertNotIn("known_since", self.humemai.get_vertex_properties(vertice))

        edges = self.humemai.get_all_edges()
        for edge in edges:
            self.assertIn("num_recalled", self.humemai.get_edge_properties(edge))
            self.assertNotIn("current_time", self.humemai.get_edge_properties(edge))
            self.assertIn("event_time", self.humemai.get_edge_properties(edge))
            self.assertIsInstance(
                self.humemai.get_edge_properties(edge)["event_time"], list
            )
            self.assertNotIn("known_since", self.humemai.get_edge_properties(edge))

    def test_move_short_term_semantic(self):
        """Test moving short-term vertices and edges to semantic long-term."""
        self.humemai.remove_all_data()

        vertex_a = self.humemai.write_short_term_vertex("Alice")
        vertex_b = self.humemai.write_short_term_vertex("Bob", {})
        vertex_c = self.humemai.write_short_term_vertex("Charlie", {"foo": 243})

        edge_ab = self.humemai.write_short_term_edge(
            vertex_a, "knows", vertex_b, {"foo": 123}
        )
        edge_bc = self.humemai.write_short_term_edge(vertex_b, "likes", vertex_c)
        edge_cb = self.humemai.write_short_term_edge(
            vertex_c, "friend_of", vertex_b, {"baz": 1}
        )

        self.assertEqual(len(self.humemai.get_all_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 0)

        vertex_b = self.humemai.move_short_term_vertex(vertex_b, "semantic")
        vertex_c = self.humemai.move_short_term_vertex(vertex_c, "semantic")
        edge_cb = self.humemai.move_short_term_edge(edge_cb, "semantic")

        self.assertEqual(len(self.humemai.get_all_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 1)

        self.assertNotIn("num_recalled", self.humemai.get_vertex_properties(vertex_a))
        self.assertIn("current_time", self.humemai.get_vertex_properties(vertex_a))
        self.assertNotIn("event_time", self.humemai.get_vertex_properties(vertex_a))
        self.assertNotIn("known_since", self.humemai.get_vertex_properties(vertex_a))

        self.assertIn("num_recalled", self.humemai.get_vertex_properties(vertex_b))
        self.assertIn("current_time", self.humemai.get_vertex_properties(vertex_b))
        self.assertNotIn("event_time", self.humemai.get_vertex_properties(vertex_b))
        self.assertIn("known_since", self.humemai.get_vertex_properties(vertex_b))
        self.assertIsInstance(
            self.humemai.get_vertex_properties(vertex_b)["known_since"], str
        )

        self.assertIn("num_recalled", self.humemai.get_vertex_properties(vertex_c))
        self.assertIn("current_time", self.humemai.get_vertex_properties(vertex_c))
        self.assertNotIn("event_time", self.humemai.get_vertex_properties(vertex_c))
        self.assertIn("known_since", self.humemai.get_vertex_properties(vertex_c))
        self.assertIsInstance(
            self.humemai.get_vertex_properties(vertex_c)["known_since"], str
        )

        self.assertNotIn("num_recalled", self.humemai.get_edge_properties(edge_ab))
        self.assertIn("current_time", self.humemai.get_edge_properties(edge_ab))
        self.assertNotIn("event_time", self.humemai.get_edge_properties(edge_ab))
        self.assertNotIn("known_since", self.humemai.get_edge_properties(edge_ab))

        self.assertNotIn("num_recalled", self.humemai.get_edge_properties(edge_bc))
        self.assertIn("current_time", self.humemai.get_edge_properties(edge_bc))
        self.assertNotIn("event_time", self.humemai.get_edge_properties(edge_bc))
        self.assertNotIn("known_since", self.humemai.get_edge_properties(edge_bc))

        self.assertIn("num_recalled", self.humemai.get_edge_properties(edge_cb))
        self.assertIn("current_time", self.humemai.get_edge_properties(edge_cb))
        self.assertNotIn("event_time", self.humemai.get_edge_properties(edge_cb))
        self.assertIn("known_since", self.humemai.get_edge_properties(edge_cb))
        self.assertIsInstance(
            self.humemai.get_edge_properties(edge_cb)["known_since"], str
        )

        self.humemai.remove_all_short_term()

        self.assertEqual(len(self.humemai.get_all_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 1)

        vertices = self.humemai.get_all_vertices()
        for vertice in vertices:
            self.assertIn("num_recalled", self.humemai.get_vertex_properties(vertice))
            self.assertNotIn(
                "current_time", self.humemai.get_vertex_properties(vertice)
            )
            self.assertNotIn("event_time", self.humemai.get_vertex_properties(vertice))
            self.assertIn("known_since", self.humemai.get_vertex_properties(vertice))

        edges = self.humemai.get_all_edges()
        for edge in edges:
            self.assertIn("num_recalled", self.humemai.get_edge_properties(edge))
            self.assertNotIn("current_time", self.humemai.get_edge_properties(edge))
            self.assertNotIn("event_time", self.humemai.get_edge_properties(edge))
            self.assertIn("known_since", self.humemai.get_edge_properties(edge))

    def test_forget_all_short_term(self):
        """Test forgetting all short-term vertices and edges."""
        self.humemai.remove_all_data()

        vertex_a = self.humemai.write_short_term_vertex("Alice")
        vertex_b = self.humemai.write_short_term_vertex("Bob", {})
        vertex_c = self.humemai.write_short_term_vertex("Charlie", {})

        edge_ab = self.humemai.write_short_term_edge(vertex_a, "knows", vertex_b)
        edge_bc = self.humemai.write_short_term_edge(vertex_b, "likes", vertex_c)
        edge_cb = self.humemai.write_short_term_edge(vertex_c, "friend_of", vertex_b)

        self.humemai.remove_all_short_term()

        self.assertEqual(len(self.humemai.get_all_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 0)

    def test_write_remove_long_term(self):
        """Test writing and removing long-term vertices and edges."""
        self.humemai.remove_all_data()

        with self.assertRaises(AssertionError):
            self.humemai.write_long_term_vertex(
                "Alice",
                {
                    "current_time": datetime.now().isoformat(timespec="seconds"),
                    "age": 35,
                    "num_recalled": 0,
                },
            )
        with self.assertRaises(AssertionError):
            self.humemai.write_long_term_vertex("Alice", {})

        vertex_a = self.humemai.write_long_term_vertex(
            "Alice", {"event_time": ["2020-01-01T00:00:00"], "num_recalled": 0}
        )
        vertex_b = self.humemai.write_long_term_vertex(
            "Bob",
            {
                "event_time": ["2020-01-01T00:00:00", "2021-01-01T00:00:00"],
                "known_since": "2021-01-31T00:00:00",
                "num_recalled": 0,
            },
        )
        vertex_c = self.humemai.write_long_term_vertex(
            "Charlie",
            {"event_time": ["2021-01-01T00:00:00"], "age": 30, "num_recalled": 0},
        )

        edge_ab = self.humemai.write_long_term_edge(
            vertex_a,
            "knows",
            vertex_b,
            {
                "event_time": ["2021-01-01T00:00:00", "2022-01-01T00:00:00"],
                "num_recalled": 0,
            },
        )
        edge_bc = self.humemai.write_long_term_edge(
            vertex_b,
            "likes",
            vertex_c,
            {"known_since": "2000-01-01T00:00:00", "num_recalled": 0},
        )
        edge_cb = self.humemai.write_long_term_edge(
            vertex_c,
            "friend_of",
            vertex_b,
            {
                "event_time": ["2021-01-01T00:00:00"],
                "known_since": "2000-01-01T00:00:00",
                "foo": 123,
                "num_recalled": 0,
            },
        )

        self.assertEqual(len(self.humemai.get_all_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 1)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 2)

        self.humemai.remove_all_short_term()

        self.assertEqual(len(self.humemai.get_all_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 1)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 2)

        self.humemai.remove_vertex(vertex_a)
        self.humemai.remove_edge(edge_bc)

        self.assertEqual(len(self.humemai.get_all_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 1)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_episodic_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_semantic_edges()), 1)

    def test_hops(self):
        """Test traversing the graph with a specific number of hops."""
        self.humemai.remove_all_data()
        vertex_d = self.humemai.write_short_term_vertex("D", {"type": "Person"})
        vertex_a = self.humemai.write_short_term_vertex(
            "A", {"type": "Organization"}
        )
        vertex_b = self.humemai.write_short_term_vertex("B", {"type": "Organization"})
        vertex_f = self.humemai.write_short_term_vertex(
            "F", {"type": "Person", "age": 30}
        )
        vertex_c = self.humemai.write_short_term_vertex("C", {"type": "Person"})
        vertex_e = self.humemai.write_short_term_vertex("E", {"type": "Person"})
        vertex_g = self.humemai.write_short_term_vertex(
            "G", {"type": "document", "title": "Document 1"}
        )

        edge_da = self.humemai.write_short_term_edge(
            vertex_d, "works_at", vertex_a, {"role": "CEO"}
        )
        edge_ab = self.humemai.write_short_term_edge(vertex_a, "owns", vertex_b)
        edge_ba = self.humemai.write_short_term_edge(
            vertex_b, "owned_by", vertex_a, {"foo": 2010}
        )
        edge_fb = self.humemai.write_short_term_edge(
            vertex_f, "works_at", vertex_b, {"role": "CTO"}
        )
        edge_cb = self.humemai.write_short_term_edge(
            vertex_c, "works_at", vertex_b, {"role": "CFO"}
        )
        edge_ce = self.humemai.write_short_term_edge(vertex_c, "knows", vertex_e)
        edge_ge = self.humemai.write_short_term_edge(vertex_g, "created_by", vertex_e)
        edge_cg = self.humemai.write_short_term_edge(vertex_c, "created", vertex_g)

        self.assertEqual(len(self.humemai.get_all_vertices()), 7)
        self.assertEqual(len(self.humemai.get_all_edges()), 8)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 7)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 8)

        short_term_vertices, long_term_vertices, short_term_edges, long_term_edges = (
            self.humemai.get_working_vertices_and_edges(
                self.humemai.get_all_short_term_vertices(),
                self.humemai.get_all_short_term_edges(),
                include_all_long_term=True,
            )
        )
        self.assertEqual(len(short_term_vertices + long_term_vertices), 7)
        self.assertEqual(len(short_term_edges + long_term_edges), 8)
        self.assertEqual(len(self.humemai.get_all_vertices()), 7)
        self.assertEqual(len(self.humemai.get_all_edges()), 8)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 7)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 8)

        short_term_vertices, long_term_vertices, short_term_edges, long_term_edges = (
            self.humemai.get_working_vertices_and_edges(
                self.humemai.get_all_short_term_vertices(),
                self.humemai.get_all_short_term_edges(),
                include_all_long_term=False,
                hops=2,
            )
        )
        self.assertEqual(len(short_term_vertices + long_term_vertices), 7)
        self.assertEqual(len(short_term_edges + long_term_edges), 8)
        self.assertEqual(len(self.humemai.get_all_vertices()), 7)
        self.assertEqual(len(self.humemai.get_all_edges()), 8)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 7)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 8)

        self.humemai.move_short_term_vertex(vertex_d, "episodic")
        self.humemai.move_short_term_vertex(vertex_a, "episodic")
        self.humemai.move_short_term_edge(edge_da, "episodic")
        self.humemai.move_short_term_vertex(vertex_c, "semantic")
        self.humemai.move_short_term_vertex(vertex_e, "semantic")
        self.humemai.move_short_term_vertex(vertex_g, "semantic")
        self.humemai.move_short_term_edge(edge_ce, "semantic")
        self.humemai.move_short_term_edge(edge_ge, "semantic")

        self.humemai.remove_all_short_term()

        self.assertEqual(len(self.humemai.get_all_vertices()), 5)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_edges()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 5)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 3)

        for vertice in self.humemai.get_all_vertices():
            self.assertIn("num_recalled", self.humemai.get_vertex_properties(vertice))
            self.assertEqual(
                self.humemai.get_vertex_properties(vertice)["num_recalled"], 0
            )
            self.assertNotIn(
                "current_time", self.humemai.get_vertex_properties(vertice)
            )
        for edge in self.humemai.get_all_edges():
            self.assertIn("num_recalled", self.humemai.get_edge_properties(edge))
            self.assertNotIn("current_time", self.humemai.get_edge_properties(edge))
            self.assertEqual(self.humemai.get_edge_properties(edge)["num_recalled"], 0)

        vertex_h = self.humemai.write_short_term_vertex(
            "H", {"type": "Person", "hobby": "reading"}
        )
        vertex_g = self.humemai.write_short_term_vertex(
            "G", {"type": "document", "title": "Document 1"}
        )
        edge_hg = self.humemai.write_short_term_edge(
            vertex_h, "likes", vertex_g, {"foo": 111}
        )

        short_term_vertices, long_term_vertices, short_term_edges, long_term_edges = (
            self.humemai.get_working_vertices_and_edges(
                self.humemai.get_all_short_term_vertices(),
                self.humemai.get_all_short_term_edges(),
                include_all_long_term=True,
            )
        )
        self.assertEqual(len(short_term_vertices + long_term_vertices), 6)
        self.assertEqual(len(short_term_edges + long_term_edges), 4)
        self.assertEqual(len(self.humemai.get_all_vertices()), 6)
        self.assertEqual(len(self.humemai.get_all_edges()), 4)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 5)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 3)

        short_term_vertices, long_term_vertices, short_term_edges, long_term_edges = (
            self.humemai.get_working_vertices_and_edges(
                self.humemai.get_all_short_term_vertices(),
                self.humemai.get_all_short_term_edges(),
                include_all_long_term=False,
                hops=2,
            )
        )
        self.assertEqual(len(short_term_vertices + long_term_vertices), 4)
        self.assertEqual(len(short_term_edges + long_term_edges), 3)
        self.assertEqual(len(self.humemai.get_all_vertices()), 6)
        self.assertEqual(len(self.humemai.get_all_edges()), 4)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 1)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 5)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 2)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 3)

        self.humemai.move_short_term_vertex(vertex_h, "episodic")
        self.humemai.move_short_term_vertex(vertex_g, "episodic")
        self.humemai.move_short_term_edge(edge_hg, "episodic")

        self.humemai.remove_all_short_term()

        with self.assertRaises(ValueError):
            (
                short_term_vertices,
                long_term_vertices,
                short_term_edges,
                long_term_edges,
            ) = self.humemai.get_working_vertices_and_edges(
                self.humemai.get_all_short_term_vertices(),
                self.humemai.get_all_short_term_edges(),
                include_all_long_term=False,
                hops=2,
            )

        self.assertEqual(len(self.humemai.get_all_vertices()), 6)
        self.assertEqual(len(self.humemai.get_all_edges()), 4)
        self.assertEqual(len(self.humemai.get_all_short_term_vertices()), 0)
        self.assertEqual(len(self.humemai.get_all_short_term_edges()), 0)
        self.assertEqual(len(self.humemai.get_all_long_term_vertices()), 6)
        self.assertEqual(len(self.humemai.get_all_episodic_vertices()), 4)
        self.assertEqual(len(self.humemai.get_all_semantic_vertices()), 3)
        self.assertEqual(len(self.humemai.get_all_long_term_edges()), 4)

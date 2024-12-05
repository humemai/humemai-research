"""Unittests for Humemai basic functions."""

import unittest
from datetime import datetime
from humemai.janusgraph import Humemai


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

    def test_write_time_vertex(self):
        """Test writing a time vertex."""
        self.humemai.remove_all_data()
        with self.assertRaises(AssertionError):
            self.humemai.write_time_vertex("ddd")

        vertex = self.humemai.write_time_vertex("2021-01-01T00:00:00")

        self.assertTrue(self.humemai.get_properties(vertex)["timestamp"])
        self.assertEqual(vertex.label, "2021-01-01T00:00:00")

    def test_write_short_term(self):
        """Test writing a short-term vertex and index."""
        self.humemai.remove_all_data()

        time_vertex = self.humemai.write_time_vertex(
            datetime.now().isoformat(timespec="seconds")
        )

        vertex_a = self.humemai.write_short_term_vertex("Alice", time_vertex)
        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 1)
        self.assertEqual(len(edges), 0)

        vertex_a = self.humemai.write_short_term_vertex("Alice", time_vertex)
        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 0)

        vertex_b = self.humemai.write_short_term_vertex("Bob", time_vertex)

        edge = self.humemai.write_edge(vertex_a, "knows", vertex_b, {"years": 5})
        edge = self.humemai.write_edge(vertex_a, "knows", vertex_b, {"years": 5})

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 2)

    def test_move_short_term_episodic(self):
        """Test moving short-term vertices and edges to episodic long-term."""
        self.humemai.remove_all_data()

        time_vertex = self.humemai.write_time_vertex(
            datetime.now().isoformat(timespec="seconds")
        )
        vertex_a = self.humemai.write_short_term_vertex("Alice", time_vertex)
        vertex_b = self.humemai.write_short_term_vertex("Bob", time_vertex, {})
        vertex_c = self.humemai.write_short_term_vertex("Charlie", time_vertex, {})

        edge_ab = self.humemai.write_edge(vertex_a, "knows", vertex_b)
        edge_bc = self.humemai.write_edge(vertex_b, "likes", vertex_c)
        edge_cb = self.humemai.write_edge(vertex_c, "friend_of", vertex_b)

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

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 1)
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

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 1)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_label("has_episodic_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_label("has_semantic_memory")
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

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_label("has_episodic_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_label("has_semantic_memory")
        self.assertEqual(len(edges), 0)

    def test_move_short_term_semantic(self):
        """Test moving short-term vertices and edges to semantic long-term."""
        self.humemai.remove_all_data()

        time_vertex = self.humemai.write_time_vertex(
            datetime.now().isoformat(timespec="seconds")
        )
        vertex_a = self.humemai.write_short_term_vertex("Alice", time_vertex)
        vertex_b = self.humemai.write_short_term_vertex(
            "Bob", time_vertex, {"foo": 123}
        )
        vertex_c = self.humemai.write_short_term_vertex("Charlie", time_vertex, {})

        edge_ab = self.humemai.write_edge(vertex_a, "knows", vertex_b)
        edge_bc = self.humemai.write_edge(vertex_b, "likes", vertex_c)
        edge_cb = self.humemai.write_edge(vertex_c, "friend_of", vertex_b)

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

        vertex_a = self.humemai.move_short_term_vertex(vertex_a, "semantic")
        vertex_b = self.humemai.move_short_term_vertex(vertex_b, "semantic")
        vertex_c = self.humemai.move_short_term_vertex(vertex_c, "semantic")

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_episodic_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_label("has_semantic_memory")
        self.assertEqual(len(edges), 3)

        self.humemai.remove_all_short_term()

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_episodic_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_label("has_semantic_memory")
        self.assertEqual(len(edges), 3)

    def test_write_remove_long_term(self):
        """Test writing and removing long-term vertices and edges."""
        self.humemai.remove_all_data()

        time_vertex = self.humemai.write_time_vertex(
            datetime.now().isoformat(timespec="seconds")
        )

        vertex_a = self.humemai.write_long_term_vertex(
            "Alice", "episodic", time_vertex, {"age": 30}
        )
        vertex_b = self.humemai.write_long_term_vertex(
            "Bob", "episodic", time_vertex, {"foo": 123}
        )
        vertex_c = self.humemai.write_long_term_vertex(
            "Charlie", "semantic", time_vertex, {"bar": "baz"}
        )

        edge_ab = self.humemai.write_edge(
            vertex_a,
            "knows",
            vertex_b,
            {},
        )
        edge_bc = self.humemai.write_edge(
            vertex_b,
            "likes",
            vertex_c,
        )
        edge_cb = self.humemai.write_edge(
            vertex_c,
            "friend_of",
            vertex_b,
            {"years": 5},
        )

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 1)
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_episodic_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_label("has_semantic_memory")
        self.assertEqual(len(edges), 1)

    def test_hops(self):
        """Test traversing the graph with a specific number of hops."""
        self.humemai.remove_all_data()

        time_vertex = self.humemai.write_time_vertex(
            datetime.now().isoformat(timespec="seconds")
        )

        vertex_d = self.humemai.write_short_term_vertex(
            "D", time_vertex, {"type": "Person"}
        )
        vertex_a = self.humemai.write_short_term_vertex(
            "A", time_vertex, {"type": "Organization"}
        )
        vertex_b = self.humemai.write_short_term_vertex(
            "B", time_vertex, {"type": "Organization"}
        )
        vertex_f = self.humemai.write_short_term_vertex(
            "F", time_vertex, {"type": "Person", "age": 30}
        )
        vertex_c = self.humemai.write_short_term_vertex(
            "C", time_vertex, {"type": "Person"}
        )
        vertex_e = self.humemai.write_short_term_vertex(
            "E", time_vertex, {"type": "Person"}
        )
        vertex_g = self.humemai.write_short_term_vertex(
            "G", time_vertex, {"type": "document", "title": "Document 1"}
        )

        edge_da = self.humemai.write_edge(
            vertex_d, "works_at", vertex_a, {"role": "CEO"}
        )
        edge_ab = self.humemai.write_edge(vertex_a, "owns", vertex_b)
        edge_ba = self.humemai.write_edge(vertex_b, "owned_by", vertex_a, {"foo": 2010})
        edge_fb = self.humemai.write_edge(
            vertex_f, "works_at", vertex_b, {"role": "CTO"}
        )
        edge_cb = self.humemai.write_edge(
            vertex_c, "works_at", vertex_b, {"role": "CFO"}
        )
        edge_ce = self.humemai.write_edge(vertex_c, "knows", vertex_e)
        edge_ge = self.humemai.write_edge(vertex_g, "created_by", vertex_e)
        edge_cg = self.humemai.write_edge(vertex_c, "created", vertex_g)

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 7)
        self.assertEqual(len(edges), 8)
        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(include_all_long_term=True)
        )
        self.assertEqual(len(short_term_vertices), 7)
        self.assertEqual(len(long_term_vertices), 0)

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(include_all_long_term=False, hops=2)
        )
        self.assertEqual(len(short_term_vertices), 7)
        self.assertEqual(len(long_term_vertices), 0)

        self.humemai.move_short_term_vertex(vertex_d, "episodic")
        self.humemai.move_short_term_vertex(vertex_a, "episodic")
        self.humemai.move_short_term_vertex(vertex_c, "semantic")
        self.humemai.move_short_term_vertex(vertex_e, "semantic")
        self.humemai.move_short_term_vertex(vertex_g, "semantic")

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 5)

        for vertex in [vertex_d, vertex_a, vertex_c, vertex_e, vertex_g]:
            self.assertEqual(
                self.humemai.g.V(vertex)
                .inE("has_episodic_memory", "has_semantic_memory")
                .values("num_recalled")
                .next(),
                0,
            )

        edges = self.humemai.get_edges_by_label("has_episodic_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_label("has_semantic_memory")
        self.assertEqual(len(edges), 3)

        self.humemai.remove_all_short_term()

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 5)
        self.assertEqual(len(edges), 4)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 5)

        for vertex in [vertex_d, vertex_a, vertex_c, vertex_e, vertex_g]:
            self.assertEqual(
                self.humemai.g.V(vertex)
                .inE("has_episodic_memory", "has_semantic_memory")
                .values("num_recalled")
                .next(),
                0,
            )

        edges = self.humemai.get_edges_by_label("has_episodic_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_label("has_semantic_memory")
        self.assertEqual(len(edges), 3)

        time_vertex = self.humemai.write_time_vertex(
            datetime.now().isoformat(timespec="seconds")
        )

        vertex_h = self.humemai.write_short_term_vertex(
            "H", time_vertex, {"type": "Person", "hobby": "reading"}
        )
        vertex_g = self.humemai.write_short_term_vertex(
            "G", time_vertex, {"type": "document", "title": "Document 1"}
        )
        edge_hg = self.humemai.write_edge(vertex_h, "likes", vertex_g, {"foo": 111})

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(include_all_long_term=True)
        )
        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 5)
        self.assertEqual(len(edges), 4)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 5)

        for vertex in [long_term_vertices]:
            self.assertEqual(
                self.humemai.g.V(vertex)
                .inE("has_episodic_memory", "has_semantic_memory")
                .values("num_recalled")
                .next(),
                1,
            )

        short_term_vertices, short_term_edges, long_term_vertices, long_term_edges = (
            self.humemai.get_working(include_all_long_term=False, hops=2)
        )

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 5)
        self.assertEqual(len(edges), 4)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 2)
        self.assertEqual(len(edges), 1)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 2)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 5)

        for vertex in [long_term_vertices]:
            self.assertEqual(
                self.humemai.g.V(vertex)
                .inE("has_episodic_memory", "has_semantic_memory")
                .values("num_recalled")
                .next(),
                2,
            )

        self.humemai.move_short_term_vertex(vertex_h, "episodic")
        self.humemai.move_short_term_vertex(vertex_g, "episodic")

        self.humemai.remove_all_short_term()

        vertices, edges = self.humemai.get_all_short_term()
        self.assertEqual(len(vertices), 0)
        self.assertEqual(len(edges), 0)

        vertices, edges = self.humemai.get_all_long_term()
        self.assertEqual(len(vertices), 7)
        self.assertEqual(len(edges), 5)

        vertices, edges = self.humemai.get_all_episodic()
        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(edges), 2)

        vertices, edges = self.humemai.get_all_semantic()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(len(edges), 3)

        edges = self.humemai.get_edges_by_label("has_short_term_memory")
        self.assertEqual(len(edges), 0)

        edges = self.humemai.get_edges_by_properties(include_keys=["num_recalled"])
        self.assertEqual(len(edges), 7)

        self.assertEqual(
            self.humemai.g.V()
            .hasLabel("G")
            .inE("likes")
            .inV()
            .inE("has_episodic_memory")
            .values("num_recalled")
            .next(),
            0,
        )

        self.assertEqual(
            self.humemai.g.V()
            .hasLabel("H")
            .inE("has_episodic_memory")
            .values("num_recalled")
            .next(),
            0,
        )

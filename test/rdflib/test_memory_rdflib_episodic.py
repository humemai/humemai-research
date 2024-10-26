"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestAddEpisodic(unittest.TestCase):

    def setUp(self) -> None:
        """Initialize the Memory and populate it with episodic, semantic, and short-term memories."""
        # Initialize the memory system
        self.memory: Memory = Humemai()

        # Define multiple triples
        self.triples: list[tuple[URIRef, URIRef, URIRef]] = [
            (
                URIRef("https://example.org/person/Alice"),
                URIRef("https://example.org/event/met"),
                URIRef("https://example.org/person/Bob"),
            ),
            (
                URIRef("https://example.org/person/Bob"),
                URIRef("https://example.org/event/met"),
                URIRef("https://example.org/person/Charlie"),
            ),
            (
                URIRef("https://example.org/person/Alice"),
                URIRef("https://example.org/relationship/knows"),
                URIRef("https://example.org/person/David"),
            ),
            (
                URIRef("https://example.org/person/Charlie"),
                URIRef("https://example.org/event/met"),
                URIRef("https://example.org/person/Eve"),
            ),
            (
                URIRef("https://example.org/person/David"),
                URIRef("https://example.org/event/invited"),
                URIRef("https://example.org/person/Alice"),
            ),
            (
                URIRef("https://example.org/person/Eve"),
                URIRef("https://example.org/relationship/worksWith"),
                URIRef("https://example.org/person/Bob"),
            ),
            (
                URIRef("https://example.org/person/Charlie"),
                URIRef("https://example.org/event/attended"),
                URIRef("https://example.org/person/Bob"),
            ),
            (
                URIRef("https://example.org/entity/Dog"),
                URIRef("https://example.org/relationship/is"),
                URIRef("https://example.org/entity/Animal"),
            ),
            (
                URIRef("https://example.org/person/Alice"),
                URIRef("https://example.org/relationship/owns"),
                URIRef("https://example.org/entity/Dog"),
            ),
            (
                URIRef("https://example.org/person/Eve"),
                URIRef("https://example.org/event/met"),
                URIRef("https://example.org/person/Alice"),
            ),
            (
                URIRef("https://example.org/person/John"),
                URIRef("https://example.org/relationship/loves"),
                URIRef("https://example.org/entity/Animal"),
            ),
        ]

        # Define qualifiers for episodic memories using URIRef and Literal
        episodic_qualifiers_1: dict[URIRef, Literal] = {
            humemai.location: Literal("New York"),
            humemai.eventTime: Literal(
                "2024-04-27T15:00:00",
                datatype=XSD.dateTime,
            ),
            humemai.emotion: Literal("happy"),
            humemai.event: Literal("Coffee meeting"),
        }
        episodic_qualifiers_2: dict[URIRef, Literal] = {
            humemai.location: Literal("London"),
            humemai.eventTime: Literal(
                "2024-05-01T10:00:00",
                datatype=XSD.dateTime,
            ),
            humemai.emotion: Literal("excited"),
            humemai.event: Literal("Conference meeting"),
        }
        episodic_qualifiers_3: dict[URIRef, Literal] = {
            humemai.location: Literal("Paris"),
            humemai.eventTime: Literal(
                "2024-05-03T14:00:00",
                datatype=XSD.dateTime,
            ),
            humemai.emotion: Literal("curious"),
            humemai.event: Literal("Workshop"),
        }

        # Add episodic memories
        self.memory.add_episodic_memory(
            [self.triples[0]], qualifiers=episodic_qualifiers_1
        )
        self.memory.add_episodic_memory(
            [self.triples[1]], qualifiers=episodic_qualifiers_2
        )
        self.memory.add_episodic_memory(
            [self.triples[2]], qualifiers=episodic_qualifiers_3
        )
        self.memory.add_episodic_memory(
            [self.triples[3]], qualifiers=episodic_qualifiers_1
        )
        self.memory.add_episodic_memory(
            [self.triples[4]], qualifiers=episodic_qualifiers_2
        )
        self.memory.add_episodic_memory(
            [self.triples[5]], qualifiers=episodic_qualifiers_3
        )
        self.memory.add_episodic_memory(
            [self.triples[6]], qualifiers=episodic_qualifiers_1
        )
        self.memory.add_episodic_memory(
            [self.triples[7]], qualifiers=episodic_qualifiers_2
        )
        self.memory.add_episodic_memory(
            [self.triples[8]], qualifiers=episodic_qualifiers_3
        )
        self.memory.add_episodic_memory(
            [self.triples[9]], qualifiers=episodic_qualifiers_1
        )
        self.memory.add_episodic_memory(
            [self.triples[10]], qualifiers=episodic_qualifiers_2
        )

        # Add semantic memories
        semantic_qualifiers: dict[URIRef, Literal] = {
            humemai.knownSince: Literal(
                "2023-01-01T00:00:00",
                datatype=XSD.dateTime,
            ),
            humemai.derivedFrom: Literal("animal_research"),
            humemai.strength: Literal(5, datatype=XSD.integer),
        }
        self.memory.add_semantic_memory(
            [self.triples[7]], qualifiers=semantic_qualifiers
        )
        self.memory.add_semantic_memory(
            [self.triples[7]], qualifiers=semantic_qualifiers
        )
        self.memory.add_semantic_memory(
            [self.triples[10]], qualifiers=semantic_qualifiers
        )
        self.memory.add_semantic_memory(
            [self.triples[10]], qualifiers=semantic_qualifiers
        )

        # Add short-term memories
        self.memory.add_short_term_memory(
            [self.triples[8]], qualifiers={humemai.location: Literal("Alice's home")}
        )
        self.memory.add_short_term_memory(
            [self.triples[8]], qualifiers={humemai.location: Literal("Paris Cafe")}
        )

        # Add another short-term memory
        self.memory.add_short_term_memory(
            [self.triples[9]], qualifiers={humemai.location: Literal("Paris Cafe")}
        )
        self.memory.add_short_term_memory(
            [self.triples[9]], qualifiers={humemai.location: Literal("Paris Cafe")}
        )

    def test_working_memory_hop_0(self) -> None:
        """Test that hop=0 only retrieves short-term memories involving Alice."""
        trigger_node: URIRef = URIRef("https://example.org/person/Alice")
        working_memory = self.memory.get_working_memory(
            trigger_node=trigger_node, hops=0
        )

        self.assertEqual(working_memory.get_main_triple_count_except_event(), 2)
        self.assertEqual(working_memory.get_memory_count(), 4)

    def test_working_memory_hop_1(self) -> None:
        """Test that hop=1 retrieves immediate neighbors' relationships."""
        trigger_node: URIRef = URIRef("https://example.org/person/Alice")
        working_memory = self.memory.get_working_memory(
            trigger_node=trigger_node, hops=1
        )

        self.assertEqual(working_memory.get_main_triple_count_except_event(), 5)
        self.assertEqual(working_memory.get_memory_count(), 9)

    def test_working_memory_hop_2(self) -> None:
        """Test that hop=2 retrieves 2-hop neighbors' relationships."""
        trigger_node: URIRef = URIRef("https://example.org/person/Alice")
        working_memory = self.memory.get_working_memory(
            trigger_node=trigger_node, hops=2
        )

        self.assertEqual(working_memory.get_main_triple_count_except_event(), 10)
        self.assertEqual(working_memory.get_memory_count(), 16)

    def test_working_memory_hop_3(self) -> None:
        """Test that hop=3 retrieves 3-hop neighbors' relationships."""
        trigger_node: URIRef = URIRef("https://example.org/person/Alice")
        working_memory = self.memory.get_working_memory(
            trigger_node=trigger_node, hops=3
        )

        self.assertEqual(working_memory.get_main_triple_count_except_event(), 11)
        self.assertEqual(working_memory.get_memory_count(), 19)

    def test_working_memory_include_all_long_term(self) -> None:
        """Test that all long-term memories are included when include_all_long_term=True."""
        working_memory = self.memory.get_working_memory(include_all_long_term=True)

        self.assertEqual(working_memory.get_main_triple_count_except_event(), 11)
        self.assertEqual(working_memory.get_memory_count(), 19)

    def test_recalled_value_increment(self) -> None:
        """Test that the recalled value increments correctly."""
        trigger_node: URIRef = URIRef("https://example.org/person/Alice")

        # Retrieve working memory at different hops
        self.memory.get_working_memory(trigger_node=trigger_node, hops=1)
        self.memory.get_working_memory(trigger_node=trigger_node, hops=2)

        # Check that recalled has incremented accordingly
        working_memory = self.memory.get_working_memory(
            trigger_node=trigger_node, hops=3
        )

        # Find an example of a statement and assert its recall count
        memory_statements = working_memory.graph.triples((None, None, None))
        for triple in memory_statements:
            if str(triple[1]) == "met" and str(triple[2]) == "Bob":
                # Ensure the 'recalled' qualifier has increased properly (should be 3)
                qualifiers: list[tuple[URIRef, Literal]] = list(
                    working_memory.graph.predicate_objects(triple[0])
                )
                for pred, obj in qualifiers:
                    if str(pred) == str(humemai.recalled):
                        self.assertEqual(int(obj), 3)

    def test_empty_memory(self) -> None:
        """Test that working memory handles empty memory cases."""
        empty_memory_system: Memory = Humemai()
        working_memory = empty_memory_system.get_working_memory(
            URIRef("https://example.org/person/Alice"), hops=1
        )

        self.assertEqual(working_memory.get_main_triple_count_except_event(), 0)
        self.assertEqual(working_memory.get_memory_count(), 0)

    def test_invalid_trigger_node(self) -> None:
        """Test behavior when an invalid trigger node is provided."""
        with self.assertRaises(ValueError):
            self.memory.get_working_memory(trigger_node=None, hops=1)


class TestMemoryMethods(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a new Memory instance before each test."""
        self.memory: Memory = Humemai()

        # Add a short-term memory to the memory system
        self.short_term_triplet_1: tuple[URIRef, URIRef, URIRef] = (
            URIRef("https://example.org/Alice"),
            URIRef("https://example.org/meet"),
            URIRef("https://example.org/Bob"),
        )
        self.short_term_triplet_2: tuple[URIRef, URIRef, URIRef] = (
            URIRef("https://example.org/Bob"),
            URIRef("https://example.org/travel"),
            URIRef("https://example.org/Paris"),
        )

        # Add these triples to short-term memory
        self.memory.add_short_term_memory(
            [self.short_term_triplet_1],
            qualifiers={
                humemai.location: Literal("Paris"),
                humemai.currentTime: Literal(
                    "2023-05-05T00:00:00", datatype=XSD.dateTime
                ),
            },
        )
        self.memory.add_short_term_memory(
            [self.short_term_triplet_2],
            qualifiers={
                humemai.location: Literal("Paris"),
                humemai.currentTime: Literal(
                    "2023-05-06T00:00:00", datatype=XSD.dateTime
                ),
            },
        )

    def test_move_short_term_to_long_term(self) -> None:
        """
        Test that a specific short-term memory is moved to long-term memory.
        """
        # Initially, ensure there are short-term memories
        short_term_memories = self.memory.get_short_term_memories()
        self.assertEqual(short_term_memories.get_memory_count(), 2)

        # Move short-term memory with ID 0 to long-term memory
        self.memory.move_short_term_to_episodic(Literal(0))

        # Check that the short-term memory count is now 1 (after moving one to long-term)
        short_term_memories = self.memory.get_short_term_memories()
        self.assertEqual(short_term_memories.get_memory_count(), 1)

        # Check that the long-term memory count is now 1 (since one was moved)
        long_term_memories = self.memory.get_long_term_memories()
        self.assertEqual(long_term_memories.get_memory_count(), 1)

        # Check that the moved long-term memory contains the correct triple
        found_memory: bool = False
        for subj, pred, obj, qualifiers in long_term_memories.iterate_memories():
            if (subj, pred, obj) == self.short_term_triplet_1:
                found_memory = True
        self.assertTrue(
            found_memory, "Moved short-term memory was not found in long-term memory."
        )

    def test_clear_short_term_memories(self) -> None:
        """
        Test that all short-term memories are cleared after calling clear_short_term_memories.
        """
        # Ensure there are initially 2 short-term memories
        short_term_memories = self.memory.get_short_term_memories()
        self.assertEqual(short_term_memories.get_memory_count(), 2)

        # Clear all short-term memories
        self.memory.clear_short_term_memories()

        # Check that there are no short-term memories remaining
        short_term_memories = self.memory.get_short_term_memories()
        self.assertEqual(short_term_memories.get_memory_count(), 0)

        # Ensure long-term memory is unaffected by clearing short-term memories
        long_term_memories = self.memory.get_long_term_memories()
        self.assertEqual(long_term_memories.get_memory_count(), 0)

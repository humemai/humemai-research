"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemoryLongTerm(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_add_single_episodic_memory(self) -> None:
        """
        Test adding a single episodic memory with location, time, and emotion.
        """
        # Define the sample triple
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        # Define the qualifiers
        qualifiers = {
            self.humemai.location: Literal("Paris"),
            self.humemai.emotion: Literal("happy"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
        }

        # Add the episodic long-term memory
        self.memory.add_episodic_memory([triple], qualifiers=qualifiers)

        # Verify that the memory was added with the correct qualifiers
        result = self.memory.print_memories(True)
        self.assertIn("Alice", result)
        self.assertIn("met", result)
        self.assertIn("Bob", result)
        self.assertIn("location", result)
        self.assertIn("Paris", result)
        self.assertIn("eventTime", result)
        self.assertIn("2024-04-27T15:00:00", result)
        self.assertIn("emotion", result)
        self.assertIn("happy", result)

    def test_add_single_semantic_memory(self) -> None:
        """
        Test adding a single semantic memory with strength and derivedFrom qualifiers.
        """
        # Define the sample triple
        triple = (
            URIRef("https://example.org/entity/Dog"),
            URIRef("https://example.org/relationship/is"),
            URIRef("https://example.org/entity/Animal"),
        )

        # Define the qualifiers
        qualifiers = {
            self.humemai.strength: Literal(5, datatype=XSD.integer),
            self.humemai.derivedFrom: Literal("textbook"),
        }
        knownSince = Literal("2024-04-27T15:00:00", datatype=XSD.dateTime)

        # Add the semantic long-term memory
        self.memory.add_semantic_memory(
            [triple], qualifiers={self.humemai.knownSince: knownSince, **qualifiers}
        )

        # Verify that the memory was added with the correct qualifiers
        result = self.memory.print_memories(True)
        self.assertIn("Dog", result)
        self.assertIn("is", result)
        self.assertIn("Animal", result)
        self.assertIn("strength", result)
        self.assertIn("5", result)
        self.assertIn("derivedFrom", result)
        self.assertIn("textbook", result)

    def test_add_multiple_episodic_memories_same_triple(self) -> None:
        """
        Test adding multiple episodic long-term memories with the same triple but different qualifiers.
        """
        # Define the sample triple
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        # Define the first set of qualifiers
        qualifiers_1 = {
            self.humemai.location: Literal("Paris"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
        }

        # Define the second set of qualifiers
        qualifiers_2 = {
            self.humemai.location: Literal("London"),
            self.humemai.eventTime: Literal(
                "2024-05-01T09:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("excited"),
        }

        # Add both episodic long-term memories with different qualifiers
        self.memory.add_episodic_memory([triple], qualifiers=qualifiers_1)
        self.memory.add_episodic_memory([triple], qualifiers=qualifiers_2)

        # Verify that both memories were added with the correct qualifiers
        result = self.memory.print_memories(True)

        # Check first memory
        self.assertIn("Paris", result)
        self.assertIn("2024-04-27T15:00:00", result)
        self.assertIn("happy", result)

        # Check second memory
        self.assertIn("London", result)
        self.assertIn("2024-05-01T09:00:00", result)
        self.assertIn("excited", result)

    def test_invalid_qualifiers_for_memory_types(self) -> None:
        """
        Test that invalid qualifiers raise an error when added to episodic or semantic
        memories.
        """
        # Define the sample triple for both tests
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        # Try adding an invalid strength qualifier to episodic memory
        with self.assertRaises(ValueError):
            self.memory.add_episodic_memory(
                [triple],
                qualifiers={
                    self.humemai.location: Literal("Paris"),
                    self.humemai.eventTime: Literal(
                        "2024-04-27T15:30:00", datatype=XSD.dateTime
                    ),
                    self.humemai.strength: Literal(
                        10, datatype=XSD.integer
                    ),  # Invalid for episodic memory
                },
            )

        # Try adding an invalid location qualifier to semantic memory
        with self.assertRaises(ValueError):
            self.memory.add_semantic_memory(
                [triple],
                qualifiers={
                    self.humemai.location: Literal(
                        "Paris"
                    ),  # Invalid for semantic memory
                    self.humemai.knownSince: Literal(
                        "2024-04-27T15:00:00", datatype=XSD.dateTime
                    ),
                },
            )


class TestMemoryMoveShortTermToLongTerm(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the Memory for testing with some initial short-term memories.
        """
        self.memory: Memory = Humemai()

        # Add a short-term memory to the memory system
        self.memory.add_short_term_memory(
            [
                (
                    URIRef("https://example.org/Alice"),
                    URIRef("https://example.org/met"),
                    URIRef("https://example.org/Bob"),
                )
            ],
            qualifiers={
                humemai.location: Literal("Paris"),
                humemai.currentTime: Literal(
                    "2023-05-05T00:00:00", datatype=XSD.dateTime
                ),
            },
        )

        # Add another short-term memory
        self.memory.add_short_term_memory(
            [
                (
                    URIRef("https://example.org/Charlie"),
                    URIRef("https://example.org/saw"),
                    URIRef("https://example.org/Alice"),
                )
            ],
            qualifiers={
                humemai.location: Literal("London"),
                humemai.currentTime: Literal(
                    "2023-05-06T00:00:00", datatype=XSD.dateTime
                ),
            },
        )

    def test_move_short_term_to_long_term_episodic(self) -> None:
        """
        Test moving a short-term memory to long-term episodic memory with emotion and event qualifiers.
        """
        # Move short-term memory to long-term episodic memory
        self.memory.move_short_term_to_episodic(
            memory_id_to_move=Literal(0),
            qualifiers={
                humemai.emotion: Literal("excited"),
                humemai.event: Literal("AI Conference"),
            },
        )

        # Check that the memory was moved to long-term and has correct qualifiers
        long_term_memories = self.memory.get_long_term_memories()
        episodic_memories: list[tuple[URIRef, URIRef, URIRef, dict]] = list(
            self.memory.iterate_memories(memory_type="episodic")
        )

        # Assert that the long-term memory contains the correct triple and qualifiers
        self.assertEqual(len(episodic_memories), 1)
        subj, pred, obj, qualifiers = episodic_memories[0]
        self.assertEqual(subj, URIRef("https://example.org/Alice"))
        self.assertEqual(pred, URIRef("https://example.org/met"))
        self.assertEqual(obj, URIRef("https://example.org/Bob"))
        self.assertEqual(qualifiers.get(humemai.location), Literal("Paris"))
        self.assertEqual(
            qualifiers.get(humemai.eventTime),
            Literal("2023-05-05T00:00:00", datatype=XSD.dateTime),
        )
        self.assertEqual(qualifiers.get(humemai.emotion), Literal("excited"))
        self.assertEqual(qualifiers.get(humemai.event), Literal("AI Conference"))

    def test_move_short_term_to_long_term_semantic(self) -> None:
        """
        Test moving a short-term memory to long-term semantic memory with strength and derivedFrom qualifiers.
        """
        # Move short-term memory to long-term semantic memory
        self.memory.move_short_term_to_semantic(
            memory_id_to_move=Literal(1),
            qualifiers={
                humemai.strength: Literal(5),
                humemai.derivedFrom: Literal("Observation"),
            },
        )

        # Check that the memory was moved to long-term and has correct qualifiers
        semantic_memories: list[tuple[URIRef, URIRef, URIRef, dict]] = list(
            self.memory.iterate_memories(memory_type="semantic")
        )

        # Assert that the long-term memory contains the correct triple and qualifiers
        self.assertEqual(len(semantic_memories), 1)
        subj, pred, obj, qualifiers = semantic_memories[0]
        self.assertEqual(subj, URIRef("https://example.org/Charlie"))
        self.assertEqual(pred, URIRef("https://example.org/saw"))
        self.assertEqual(obj, URIRef("https://example.org/Alice"))
        self.assertEqual(qualifiers.get(humemai.strength), Literal(5))
        self.assertEqual(qualifiers.get(humemai.derivedFrom), Literal("Observation"))

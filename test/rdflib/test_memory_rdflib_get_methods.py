"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemoryGetMemories(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

        # Define sample triples
        self.triple1 = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        self.triple2 = (
            URIRef("https://example.org/entity/Cat"),
            URIRef("https://example.org/relationship/is"),
            URIRef("https://example.org/entity/Animal"),
        )

        self.triple3 = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/likes"),
            Literal("Chocolate"),
        )

        # Add memories to the graph with proper URIRef and Literal formats
        qualifiers_episodic = {
            self.humemai.location: Literal("Paris"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
        }
        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic)

        qualifiers_semantic = {
            self.humemai.strength: Literal(8, datatype=XSD.integer),
            self.humemai.derivedFrom: Literal("study"),
            self.humemai.knownSince: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_semantic_memory([self.triple2], qualifiers=qualifiers_semantic)

        # Short-term memory with correct qualifier structure
        short_term_qualifiers = {
            self.humemai.location: Literal("Berlin"),
            self.humemai.currentTime: Literal(
                "2024-04-27T18:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_short_term_memory([self.triple3], short_term_qualifiers)

    def test_basic_memory_retrieval(self) -> None:
        """
        Test basic retrieval of all memories without filters.
        """
        retrieved_memory = self.memory.get_memories()

        # Check if all triples are retrieved
        result = retrieved_memory.print_memories(True)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertIn("Cat", result)
        self.assertIn("Chocolate", result)

    def test_qualifiers_association(self) -> None:
        """
        Test that qualifiers are correctly associated with their triples.
        """
        retrieved_memory = self.memory.get_memories()

        # Ensure the qualifiers are correctly attached to the main triples
        result = retrieved_memory.print_memories(True)

        # Check for Alice's memory qualifiers
        self.assertIn("location", result)
        self.assertIn("Paris", result)
        self.assertIn("eventTime", result)
        self.assertIn("2024-04-27T15:00:00", result)
        self.assertIn("emotion", result)
        self.assertIn("happy", result)

        # Check for Cat's memory qualifiers
        self.assertIn("derivedFrom", result)
        self.assertIn("study", result)
        self.assertIn("strength", result)
        self.assertIn("8", result)

    def test_memory_retrieval_with_subject_filter(self) -> None:
        """
        Test memory retrieval with a subject filter.
        """
        # Retrieve only memories with Alice as the subject
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/person/Alice")
        )

        result = retrieved_memory.print_memories(True)
        self.assertIn("Alice", result)
        self.assertNotIn("https://example.org/entity/Cat", result)
        self.assertNotIn("Chocolate", result)

    def test_memory_retrieval_with_object_filter(self) -> None:
        """
        Test memory retrieval with an object filter.
        """
        # Retrieve only memories where Bob is the object
        retrieved_memory = self.memory.get_memories(
            object_=URIRef("https://example.org/person/Bob")
        )

        result = retrieved_memory.print_memories(True)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertNotIn("https://example.org/entity/Cat", result)
        self.assertNotIn("Chocolate", result)

    def test_unique_blank_nodes(self) -> None:
        """
        Ensure that each reified statement is associated with a unique blank node.
        """
        retrieved_memory = self.memory.get_memories()

        # Extract all blank nodes (reified statements)
        blank_nodes = set()
        for subj, pred, obj in retrieved_memory.graph.triples(
            (None, RDF.type, RDF.Statement)
        ):
            blank_nodes.add(subj)

        # Ensure we have unique blank nodes for each reified statement
        self.assertEqual(len(blank_nodes), 3)  # One for each main triple

    def test_qualifiers_with_blank_nodes(self) -> None:
        """
        Ensure that qualifiers are correctly assigned to the reified blank nodes.
        """
        retrieved_memory = self.memory.get_memories()

        # Extract qualifiers for each blank node
        for statement, _, _ in retrieved_memory.graph.triples(
            (None, RDF.type, RDF.Statement)
        ):
            qualifiers = list(retrieved_memory.graph.predicate_objects(statement))
            self.assertGreater(
                len(qualifiers), 1
            )  # Each blank node should have qualifiers


class TestMemoryGetMemoryCount(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

        # Define the triples
        self.triple1 = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        self.triple2 = (
            URIRef("https://example.org/entity/Cat"),
            URIRef("https://example.org/relationship/is"),
            URIRef("https://example.org/entity/Animal"),
        )

        self.triple3 = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/likes"),
            Literal("Chocolate"),
        )

    def test_single_memory_count(self) -> None:
        """
        Test that the count is correct when one unique memory is added.
        """
        # Add one memory
        qualifiers_episodic = {
            self.humemai.location: Literal("Paris"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
        }
        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic)

        # Test memory count
        memory_count = self.memory.get_main_triple_count_except_event()
        self.assertEqual(memory_count, 1, "Memory count should be 1.")

    def test_duplicate_memory_with_different_qualifiers(self) -> None:
        """
        Test that duplicate memories with different qualifiers are only counted once.
        """
        # Add the same triple multiple times with different qualifiers
        qualifiers_episodic1 = {
            self.humemai.location: Literal("Paris"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
        }

        qualifiers_episodic2 = {
            self.humemai.location: Literal("London"),
            self.humemai.eventTime: Literal(
                "2024-05-01T09:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("excited"),
        }

        qualifiers_episodic3 = {
            self.humemai.location: Literal("Berlin"),
            self.humemai.eventTime: Literal(
                "2024-05-10T10:30:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("nervous"),
        }

        # Add the same triple with different qualifiers
        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic1)
        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic2)
        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic3)

        # Test memory count (should still be 1 since the triple is the same)
        memory_count = self.memory.get_main_triple_count_except_event()
        self.assertEqual(
            memory_count,
            1,
            "Memory count should still be 1 for the same triple with different qualifiers.",
        )

    def test_unique_memory_count(self) -> None:
        """
        Test that the count is correct when multiple unique triples are added.
        """
        # Add multiple unique triples
        qualifiers_episodic = {
            self.humemai.location: Literal("Paris"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
        }

        qualifiers_semantic = {
            self.humemai.strength: Literal(8, datatype=XSD.integer),
            self.humemai.derivedFrom: Literal("study"),
            self.humemai.knownSince: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
        }

        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic)
        self.memory.add_semantic_memory([self.triple2], qualifiers=qualifiers_semantic)

        # Short-term memory
        short_term_qualifiers = {
            self.humemai.location: Literal("Berlin"),
            self.humemai.currentTime: Literal(
                "2024-04-27T18:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_short_term_memory([self.triple3], short_term_qualifiers)

        # Test memory count (should be 3 because all triples are unique)
        memory_count = self.memory.get_main_triple_count_except_event()
        self.assertEqual(
            memory_count, 3, "Memory count should be 3 for three unique triples."
        )

    def test_mixed_duplicate_and_unique_triples(self) -> None:
        """
        Test memory count with a mix of duplicate and unique triples.
        """
        # Add the same triple multiple times with different qualifiers
        qualifiers_episodic1 = {
            self.humemai.location: Literal("Paris"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
        }

        qualifiers_episodic2 = {
            self.humemai.location: Literal("London"),
            self.humemai.eventTime: Literal(
                "2024-05-01T09:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("excited"),
        }

        qualifiers_semantic = {
            self.humemai.strength: Literal(8, datatype=XSD.integer),
            self.humemai.derivedFrom: Literal("study"),
            self.humemai.knownSince: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
        }

        # Add duplicate and unique triples
        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic1)
        self.memory.add_episodic_memory([self.triple1], qualifiers=qualifiers_episodic2)
        self.memory.add_semantic_memory([self.triple2], qualifiers=qualifiers_semantic)

        short_term_qualifiers = {
            self.humemai.location: Literal("Berlin"),
            self.humemai.currentTime: Literal(
                "2024-04-27T18:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_short_term_memory([self.triple3], short_term_qualifiers)

        # Test memory count (should be 3: one for the duplicate triple, and two for the unique ones)
        memory_count = self.memory.get_main_triple_count_except_event()
        self.assertEqual(
            memory_count,
            3,
            "Memory count should be 3 for a mix of one duplicate and two unique triples.",
        )

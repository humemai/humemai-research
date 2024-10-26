"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemoryCounts(unittest.TestCase):
    """
    Test cases for get_memory_count and get_main_triple_count_except_event methods in the Memory class.
    """

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

        # Define sample triples
        self.triple1 = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
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

    def test_no_memories(self) -> None:
        """
        Test that counts are zero when no memories have been added.
        """
        self.assertEqual(
            self.memory.get_memory_count(),
            0,
            "Memory count should be 0 when no memories are added.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            0,
            "Triple count should be 0 when no triples are added.",
        )

    def test_single_memory_single_reified_statement(self) -> None:
        """
        Test counts after adding a single memory with one reified statement.
        """
        qualifiers = {
            self.humemai.currentTime: Literal("2024-04-27T10:00:00"),
            self.humemai.location: Literal("New York"),
        }
        self.memory.add_memory([self.triple1], qualifiers)

        self.assertEqual(
            self.memory.get_memory_count(),
            1,
            "Memory count should be 1 after adding one reified statement.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            1,
            "Triple count should be 1 after adding one unique triple.",
        )

    def test_single_memory_multiple_reified_statements(self) -> None:
        """
        Test counts after adding the same triple multiple times with different qualifiers.
        """
        qualifiers1 = {
            self.humemai.currentTime: Literal("2024-04-27T10:00:00"),
            self.humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            self.humemai.currentTime: Literal("2024-04-27T12:00:00"),
            self.humemai.location: Literal("London"),
        }
        qualifiers3 = {
            self.humemai.currentTime: Literal("2024-04-27T14:00:00"),
            self.humemai.location: Literal("Paris"),
        }

        self.memory.add_memory([self.triple1], qualifiers1)
        self.memory.add_memory([self.triple1], qualifiers2)
        self.memory.add_memory([self.triple1], qualifiers3)

        self.assertEqual(
            self.memory.get_memory_count(),
            3,
            "Memory count should be 3 after adding three reified statements.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            1,
            "Triple count should be 1 since all statements share the same triple.",
        )

    def test_multiple_unique_triples_multiple_reified_statements(self) -> None:
        """
        Test counts after adding multiple unique triples with multiple reified statements each.
        """
        qualifiers1 = {
            self.humemai.currentTime: Literal("2024-04-27T10:00:00"),
            self.humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            self.humemai.currentTime: Literal("2024-04-27T12:00:00"),
            self.humemai.location: Literal("London"),
        }
        qualifiers3 = {
            self.humemai.currentTime: Literal("2024-04-27T14:00:00"),
            self.humemai.location: Literal("Paris"),
        }

        # Add multiple reified statements for triple1
        self.memory.add_memory([self.triple1], qualifiers1)
        self.memory.add_memory([self.triple1], qualifiers2)

        # Add multiple reified statements for triple2
        self.memory.add_memory([self.triple2], qualifiers1)
        self.memory.add_memory([self.triple2], qualifiers3)

        # Add a single reified statement for triple3
        self.memory.add_memory([self.triple3], qualifiers2)

        expected_memory_count = 5  # 2 for triple1, 2 for triple2, 1 for triple3
        expected_triple_count = 3  # triple1, triple2, triple3

        self.assertEqual(
            self.memory.get_memory_count(),
            expected_memory_count,
            f"Memory count should be {expected_memory_count} after adding five reified statements.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            expected_triple_count,
            f"Triple count should be {expected_triple_count} after adding three unique triples.",
        )

    def test_deleting_reified_statements(self) -> None:
        """
        Test that deleting a triple removes all its reified statements.
        """
        qualifiers1 = {
            self.humemai.currentTime: Literal("2024-04-27T10:00:00"),
            self.humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            self.humemai.currentTime: Literal("2024-04-27T12:00:00"),
            self.humemai.location: Literal("London"),
        }

        # Add multiple reified statements for triple1
        self.memory.add_memory([self.triple1], qualifiers1)
        self.memory.add_memory([self.triple1], qualifiers2)

        # Add a reified statement for triple2
        self.memory.add_memory([self.triple2], qualifiers1)

        # Initial counts
        self.assertEqual(
            self.memory.get_memory_count(), 3, "Initial memory count should be 3."
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            2,
            "Initial triple count should be 2.",
        )

        # Delete triple1
        self.memory.delete_triple(*self.triple1)

        # Counts after deletion
        self.assertEqual(
            self.memory.get_memory_count(),
            1,
            "Memory count should be 1 after deleting triple1.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            1,
            "Triple count should be 1 after deleting triple1.",
        )

    def test_deleting_non_existent_triple(self) -> None:
        """
        Test that deleting a non-existent triple does not affect the counts.
        """
        qualifiers = {
            self.humemai.currentTime: Literal("2024-04-27T10:00:00"),
            self.humemai.location: Literal("New York"),
        }
        self.memory.add_memory([self.triple1], qualifiers)

        # Initial counts
        self.assertEqual(
            self.memory.get_memory_count(), 1, "Initial memory count should be 1."
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            1,
            "Initial triple count should be 1.",
        )

        # Define a non-existent triple
        non_existent_triple = (
            URIRef("https://example.org/person/David"),
            URIRef("https://example.org/relationship/hates"),
            Literal("Broccoli"),
        )

        # Attempt to delete the non-existent triple
        self.memory.delete_triple(*non_existent_triple)

        # Counts should remain unchanged
        self.assertEqual(
            self.memory.get_memory_count(),
            1,
            "Memory count should remain 1 after attempting to delete a non-existent triple.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            1,
            "Triple count should remain 1 after attempting to delete a non-existent triple.",
        )

    def test_triple_count_with_no_reified_statements(self) -> None:
        """
        Test that triples without any reified statements are not counted.
        """
        # Add a triple without qualifiers (no reified statements should be created)
        simple_triple = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/likes"),
            Literal("Ice Cream"),
        )
        self.memory.add_memory([simple_triple], {})  # Empty qualifiers

        self.assertEqual(
            self.memory.get_memory_count(),
            1,
            "Memory count should be 1 after adding one reified statement with empty qualifiers.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            1,
            "Triple count should be 1 after adding one unique triple.",
        )

    def test_triple_count_after_adding_and_deleting(self) -> None:
        """
        Test triple and memory counts after adding and deleting multiple triples.
        """
        qualifiers1 = {
            self.humemai.currentTime: Literal("2024-04-27T10:00:00"),
            self.humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            self.humemai.currentTime: Literal("2024-04-27T12:00:00"),
            self.humemai.location: Literal("London"),
        }

        # Add multiple triples
        self.memory.add_memory([self.triple1], qualifiers1)
        self.memory.add_memory([self.triple2], qualifiers2)
        self.memory.add_memory(
            [self.triple1], qualifiers2
        )  # Duplicate triple with different qualifiers

        # Initial counts
        self.assertEqual(
            self.memory.get_memory_count(), 3, "Initial memory count should be 3."
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            2,
            "Initial triple count should be 2.",
        )

        # Delete triple1
        self.memory.delete_triple(*self.triple1)

        # Counts after deletion
        self.assertEqual(
            self.memory.get_memory_count(),
            1,
            "Memory count should be 1 after deleting triple1.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            1,
            "Triple count should be 1 after deleting triple1.",
        )

        # Delete triple2
        self.memory.delete_triple(*self.triple2)

        # Counts after deleting all triples
        self.assertEqual(
            self.memory.get_memory_count(),
            0,
            "Memory count should be 0 after deleting all triples.",
        )
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(),
            0,
            "Triple count should be 0 after deleting all triples.",
        )

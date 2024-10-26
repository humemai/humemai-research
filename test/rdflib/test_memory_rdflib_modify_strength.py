"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemoryModifyStrength(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the Memory object and add some semantic and episodic memories.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

        # Define a semantic triple
        self.triple_semantic = (
            URIRef("https://example.org/entity/Cat"),
            URIRef("https://example.org/relationship/is"),
            URIRef("https://example.org/entity/Animal"),
        )

        # Define an episodic triple (which should not be affected by strength modification)
        self.triple_episodic = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        # Add a semantic memory with strength
        qualifiers_semantic = {
            self.humemai.strength: Literal(8, datatype=XSD.integer),
            self.humemai.derivedFrom: Literal("study"),
            self.humemai.knownSince: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_semantic_memory(
            [self.triple_semantic], qualifiers=qualifiers_semantic
        )

        # Add an episodic memory without strength (it should not have a strength value)
        qualifiers_episodic = {
            self.humemai.location: Literal("Paris"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
        }
        self.memory.add_episodic_memory(
            [self.triple_episodic], qualifiers=qualifiers_episodic
        )

    def test_increment_strength(self) -> None:
        """
        Test that the strength is incremented correctly by 5.
        """
        # Increment strength by 5
        self.memory.modify_strength(
            filters={RDF.subject: URIRef("https://example.org/entity/Cat")},
            increment_by=5,
        )

        # Retrieve the updated memory and verify the new strength
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "13",
            retrieved_memory.print_memories(True),
            "Strength should be incremented to 13",
        )

    def test_multiply_strength(self) -> None:
        """
        Test that the strength is multiplied correctly by 2.
        """
        # Multiply strength by 2
        self.memory.modify_strength(
            filters={RDF.subject: URIRef("https://example.org/entity/Cat")},
            multiply_by=2,
        )

        # Retrieve the updated memory and verify the new strength
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "16",
            retrieved_memory.print_memories(True),
            "Strength should be multiplied to 16",
        )

    def test_increment_and_multiply_strength(self) -> None:
        """
        Test that the strength is incremented and then multiplied correctly.
        """
        # First increment by 5
        self.memory.modify_strength(
            filters={RDF.subject: URIRef("https://example.org/entity/Cat")},
            increment_by=5,
        )

        # Then multiply by 2
        self.memory.modify_strength(
            filters={RDF.subject: URIRef("https://example.org/entity/Cat")},
            multiply_by=2,
        )

        # Retrieve the updated memory and verify the final strength
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "26",
            retrieved_memory.print_memories(True),
            "Strength should be 26 after incrementing and multiplying",
        )

    def test_no_strength_for_episodic(self) -> None:
        """
        Test that episodic memories without a strength qualifier are not affected.
        """
        # Try to modify strength of an episodic memory (should not change anything)
        self.memory.modify_strength(
            filters={RDF.subject: URIRef("https://example.org/person/Alice")},
            increment_by=5,
        )

        # Retrieve the episodic memory and ensure no strength was added
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/person/Alice")
        )
        self.assertNotIn(
            "strength",
            retrieved_memory.print_memories(True),
            "Episodic memory should not have a strength qualifier",
        )

    def test_modify_multiple_statements(self) -> None:
        """
        Test that all reified statements for the same triple are updated.
        """
        # Add a second reified statement for the same semantic triple with a different strength
        qualifiers_semantic_2 = {
            self.humemai.strength: Literal(10, datatype=XSD.integer),
            self.humemai.derivedFrom: Literal("research"),
            self.humemai.knownSince: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_semantic_memory(
            [self.triple_semantic], qualifiers=qualifiers_semantic_2
        )

        # Increment strength by 5 for all reified statements
        self.memory.modify_strength(
            filters={RDF.subject: URIRef("https://example.org/entity/Cat")},
            increment_by=5,
        )

        # Verify that both statements were updated
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "13",
            retrieved_memory.print_memories(True),
            "First strength should be incremented to 13",
        )
        self.assertIn(
            "15",
            retrieved_memory.print_memories(True),
            "Second strength should be incremented to 15",
        )

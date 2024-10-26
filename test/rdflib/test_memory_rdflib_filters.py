"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemoryTimeFilters(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_filter_memories_by_time_range(self) -> None:
        """
        Test retrieving memories based on time range filtering.
        """
        # Define sample triples
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )

        qualifiers1 = {
            self.humemai.currentTime: Literal(
                "2024-04-27T10:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            self.humemai.currentTime: Literal(
                "2024-04-27T12:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("London"),
        }

        # Add memories with different times
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Retrieve memories within a time range
        lower_time_bound = Literal("2024-04-27T09:00:00", datatype=XSD.dateTime)
        upper_time_bound = Literal("2024-04-27T11:00:00", datatype=XSD.dateTime)
        filtered_memory = self.memory.get_memories(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
        )

        # Verify only the memory within the time range is retrieved
        result = filtered_memory.print_memories(True)
        self.assertIn("2024-04-27T10:00:00", result)
        self.assertNotIn("2024-04-27T12:00:00", result)

    def test_filter_memories_outside_time_range(self) -> None:
        """
        Test that no memories are returned if they fall outside the specified time range.
        """
        triple = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/likes"),
            Literal("Ice Cream"),
        )
        qualifiers = {
            self.humemai.currentTime: Literal(
                "2024-04-27T18:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("Berlin"),
        }

        # Add memory outside the desired time range
        self.memory.add_memory([triple], qualifiers)

        # Retrieve memories within a time range where this memory does not fall
        lower_time_bound = Literal("2024-04-27T09:00:00", datatype=XSD.dateTime)
        upper_time_bound = Literal("2024-04-27T17:00:00", datatype=XSD.dateTime)
        filtered_memory = self.memory.get_memories(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
        )

        # Ensure no memories are returned
        result = filtered_memory.print_memories(True)
        self.assertNotIn("Ice Cream", result)


class TestMemoryLocationFilters(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_filter_memories_by_location(self) -> None:
        """
        Test retrieving memories based on location filtering.
        """
        triple = (
            URIRef("https://example.org/person/David"),
            URIRef("https://example.org/relationship/met"),
            URIRef("https://example.org/person/Eve"),
        )
        qualifiers1 = {
            self.humemai.currentTime: Literal(
                "2024-04-27T10:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            self.humemai.currentTime: Literal(
                "2024-04-27T12:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("Berlin"),
        }

        # Add memory with different locations
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Retrieve memories based on location filter
        filtered_memory = self.memory.get_memories(
            qualifiers={self.humemai.location: Literal("New York")}
        )

        # Verify only the memory with the specified location is retrieved
        result = filtered_memory.print_memories(True)
        self.assertIn("New York", result)
        self.assertNotIn("Berlin", result)

    def test_filter_memories_by_invalid_location(self) -> None:
        """
        Test that no memories are returned when an invalid location is specified.
        """
        triple = (
            URIRef("https://example.org/person/Frank"),
            URIRef("https://example.org/relationship/met"),
            URIRef("https://example.org/person/Grace"),
        )
        qualifiers = {
            self.humemai.currentTime: Literal(
                "2024-04-27T10:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("Paris"),
        }

        # Add memory with a location
        self.memory.add_memory([triple], qualifiers)

        # Retrieve memories based on an invalid location filter
        filtered_memory = self.memory.get_memories(
            qualifiers={self.humemai.location: Literal("Tokyo")}
        )

        # Verify no memories are returned
        result = filtered_memory.print_memories(True)
        self.assertNotIn("Paris", result)
        self.assertNotIn("Frank", result)


class TestMemoryEmotionFilters(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_filter_memories_by_emotion(self) -> None:
        """
        Test retrieving memories based on emotion filtering.
        """
        triple = (
            URIRef("https://example.org/person/Gary"),
            URIRef("https://example.org/relationship/felt"),
            URIRef("https://example.org/event/happy"),
        )
        qualifiers1 = {
            self.humemai.emotion: Literal("happy"),
            self.humemai.currentTime: Literal(
                "2024-04-27T14:00:00", datatype=XSD.dateTime
            ),
        }
        qualifiers2 = {
            self.humemai.emotion: Literal("sad"),
            self.humemai.currentTime: Literal(
                "2024-04-27T16:00:00", datatype=XSD.dateTime
            ),
        }

        # Add memories with different emotions
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Retrieve memories based on emotion filter
        filtered_memory = self.memory.get_memories(
            qualifiers={self.humemai.emotion: Literal("happy")}
        )

        # Verify only the memory with the specified emotion is retrieved
        result = filtered_memory.print_memories(True)
        self.assertIn("happy", result)
        self.assertNotIn("sad", result)

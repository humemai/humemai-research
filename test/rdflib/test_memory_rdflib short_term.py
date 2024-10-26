"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemoryShortTerm(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_add_single_short_term_memory(self) -> None:
        """
        Test adding a single short-term memory with current location and current time.
        """
        # Define the sample triple
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )
        currentTime = Literal("2024-04-27T10:00:00", datatype=XSD.dateTime)
        qualifiers = {
            self.humemai.location: Literal("New York"),
            self.humemai.currentTime: currentTime,
        }

        # Add the short-term memory
        self.memory.add_short_term_memory([triple], qualifiers)

        # Verify that the memory was added with the correct qualifiers
        result = self.memory.print_memories(True)
        self.assertIn("Alice", result)

        # Check for 'currentTime' and 'location'
        self.assertIn("currentTime", result)
        self.assertIn("2024-04-27T10:00:00", result)
        self.assertIn("location", result)
        self.assertIn("New York", result)

    def test_add_short_term_memory_with_default_time(self) -> None:
        """
        Test adding a short-term memory without specifying the current time.
        The system should use the current time automatically.
        """
        # Define the sample triple
        triple = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/likes"),
            Literal("Ice Cream"),
        )
        location = Literal("Berlin")
        qualifiers = {self.humemai.location: location}

        # Capture the current time before adding the memory
        before_time = datetime.now()

        # Add the short-term memory without specifying time
        self.memory.add_short_term_memory([triple], qualifiers)

        # Capture the result and verify
        result = self.memory.print_memories(True)
        self.assertIn("Charlie", result)
        self.assertIn("location", result)
        self.assertIn("Berlin", result)

        # Ensure the currentTime was added automatically
        self.assertIn("currentTime", result)

        # Extract all dynamically assigned currentTime values from the graph
        current_times_in_graph = list(
            self.memory.graph.objects(None, self.humemai.currentTime)
        )

        # Verify that one of the currentTime values is close to the current time
        after_time = datetime.now()

        self.assertTrue(
            any(
                before_time <= datetime.fromisoformat(str(current_time)) <= after_time
                for current_time in current_times_in_graph
            ),
            "The currentTime should be within the expected range.",
        )

    def test_add_multiple_short_term_memories(self) -> None:
        """
        Test adding multiple triples as short-term memory.
        """
        # Define multiple triples
        triples = [
            (
                URIRef("https://example.org/person/David"),
                URIRef("https://example.org/relationship/knows"),
                URIRef("https://example.org/person/Eve"),
            ),
            (
                URIRef("https://example.org/person/Eve"),
                URIRef("https://example.org/relationship/likes"),
                Literal("Chocolate"),
            ),
        ]
        currentTime = Literal("2024-04-28T09:00:00", datatype=XSD.dateTime)
        qualifiers = {
            self.humemai.location: Literal("Tokyo"),
            self.humemai.currentTime: currentTime,
        }

        # Add the short-term memory with multiple triples
        self.memory.add_short_term_memory(triples, qualifiers)

        # Verify both triples were added correctly
        result = self.memory.print_memories(True)
        self.assertIn("David", result)
        self.assertIn("Eve", result)
        self.assertIn("Chocolate", result)
        self.assertIn("currentTime", result)
        self.assertIn("2024-04-28T09:00:00", result)
        self.assertIn("location", result)
        self.assertIn("Tokyo", result)


class TestGetShortTerm(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test case with a Memory instance and populate the RDF graph."""
        # Initialize a Memory instance and use an actual Graph instance
        self.memory = Humemai()
        self.memory.graph = Graph()

        # Short-term triple
        self.subj = URIRef("https://example.org/person/Alice")
        self.pred = URIRef("https://example.org/event/met")
        self.obj = URIRef("https://example.org/person/Bob")

        # Qualifiers for the short-term memory
        self.currentTime_qualifier = URIRef("https://humem.ai/ontology#currentTime")
        self.location_qualifier = URIRef("https://humem.ai/ontology#location")
        self.emotion_qualifier = URIRef("https://humem.ai/ontology#emotion")

        self.currentTime_literal = Literal("2024-10-03T15:00:00", datatype=XSD.dateTime)
        self.location_literal = Literal("New York", datatype=XSD.string)
        self.emotion_literal = Literal("happy", datatype=XSD.string)

        # Create a reified statement for the triple (Alice, met, Bob)
        self.reified_statement = BNode()

        # Add the reified statement and its qualifiers to the graph
        self.memory.graph.add((self.reified_statement, RDF.type, RDF.Statement))
        self.memory.graph.add((self.reified_statement, RDF.subject, self.subj))
        self.memory.graph.add((self.reified_statement, RDF.predicate, self.pred))
        self.memory.graph.add((self.reified_statement, RDF.object, self.obj))
        self.memory.graph.add(
            (
                self.reified_statement,
                self.currentTime_qualifier,
                self.currentTime_literal,
            )
        )
        self.memory.graph.add(
            (self.reified_statement, self.location_qualifier, self.location_literal)
        )
        self.memory.graph.add(
            (self.reified_statement, self.emotion_qualifier, self.emotion_literal)
        )

    def testget_short_term_memories(self) -> None:
        """Test that short-term memories with currentTime are correctly retrieved and added to the Memory object."""
        # Call the method to retrieve short-term memories
        short_term_memory = self.memory.get_short_term_memories()

        # Assertions to check if the triples and their qualifiers were added to the short-term memory
        # Check if the triple (Alice, met, Bob) was added
        self.assertIn((self.subj, self.pred, self.obj), short_term_memory.graph)

        # Check if the reified statement with qualifiers (currentTime, location, emotion) was added
        reified_statements = list(
            short_term_memory.graph.triples((None, RDF.type, RDF.Statement))
        )
        self.assertEqual(len(reified_statements), 1)  # Only one reified statement

        reified_statement = reified_statements[0][0]

        # Check if reified statement has the correct subject, predicate, and object
        self.assertIn(
            (reified_statement, RDF.subject, self.subj), short_term_memory.graph
        )
        self.assertIn(
            (reified_statement, RDF.predicate, self.pred), short_term_memory.graph
        )
        self.assertIn(
            (reified_statement, RDF.object, self.obj), short_term_memory.graph
        )

        # Check if the correct qualifiers were added
        self.assertIn(
            (reified_statement, self.currentTime_qualifier, self.currentTime_literal),
            short_term_memory.graph,
        )
        self.assertIn(
            (reified_statement, self.location_qualifier, self.location_literal),
            short_term_memory.graph,
        )
        self.assertIn(
            (reified_statement, self.emotion_qualifier, self.emotion_literal),
            short_term_memory.graph,
        )

    def test_empty_short_term_memory(self) -> None:
        """Test that the method handles an empty graph correctly."""
        # Use an empty graph
        self.memory.graph = Graph()

        # Call the method to retrieve short-term memories
        short_term_memory = self.memory.get_short_term_memories()

        # Ensure the graph is empty in this case
        self.assertEqual(
            len(list(short_term_memory.graph.triples((None, None, None)))), 0
        )

    def test_partial_qualifiers_in_short_term_memory(self) -> None:
        """Test that short-term memories handle cases where some qualifiers are missing."""
        # Clear the graph and set up a partial reified statement with only some qualifiers (no emotion)
        self.memory.graph = Graph()

        # Add the reified statement with only currentTime and location qualifiers
        self.memory.graph.add((self.reified_statement, RDF.type, RDF.Statement))
        self.memory.graph.add((self.reified_statement, RDF.subject, self.subj))
        self.memory.graph.add((self.reified_statement, RDF.predicate, self.pred))
        self.memory.graph.add((self.reified_statement, RDF.object, self.obj))
        self.memory.graph.add(
            (
                self.reified_statement,
                self.currentTime_qualifier,
                self.currentTime_literal,
            )
        )
        self.memory.graph.add(
            (self.reified_statement, self.location_qualifier, self.location_literal)
        )

        # Call the method to retrieve short-term memories
        short_term_memory = self.memory.get_short_term_memories()

        # Assertions to check if the triple and available qualifiers were added
        reified_statements = list(
            short_term_memory.graph.triples((None, RDF.type, RDF.Statement))
        )
        self.assertEqual(len(reified_statements), 1)
        reified_statement = reified_statements[0][0]

        # Ensure the triple (Alice, met, Bob) exists
        self.assertIn((self.subj, self.pred, self.obj), short_term_memory.graph)

        # Ensure the reified statement has the correct subject, predicate, object
        self.assertIn(
            (reified_statement, RDF.subject, self.subj), short_term_memory.graph
        )
        self.assertIn(
            (reified_statement, RDF.predicate, self.pred), short_term_memory.graph
        )
        self.assertIn(
            (reified_statement, RDF.object, self.obj), short_term_memory.graph
        )

        # Check the available qualifiers (no emotion)
        self.assertIn(
            (reified_statement, self.currentTime_qualifier, self.currentTime_literal),
            short_term_memory.graph,
        )
        self.assertIn(
            (reified_statement, self.location_qualifier, self.location_literal),
            short_term_memory.graph,
        )
        self.assertNotIn(
            (reified_statement, self.emotion_qualifier, self.emotion_literal),
            short_term_memory.graph,
        )

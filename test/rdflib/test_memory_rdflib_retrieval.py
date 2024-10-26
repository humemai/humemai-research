"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemoryRetrievalAndDeletion(unittest.TestCase):

    def setUp(self) -> None:
        """Set up the memory system and add initial long-term memories."""
        self.memory = Humemai()

        # Define multiple triples
        self.triples = [
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
        ]

        # Define qualifiers with URIRef keys
        self.episodic_qualifiers_1 = {
            URIRef("https://humem.ai/ontology#location"): Literal("New York"),
            URIRef("https://humem.ai/ontology#eventTime"): Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            URIRef("https://humem.ai/ontology#emotion"): Literal("happy"),
            URIRef("https://humem.ai/ontology#event"): Literal("Coffee meeting"),
        }

        self.episodic_qualifiers_2 = {
            URIRef("https://humem.ai/ontology#location"): Literal("London"),
            URIRef("https://humem.ai/ontology#eventTime"): Literal(
                "2024-05-01T10:00:00", datatype=XSD.dateTime
            ),
            URIRef("https://humem.ai/ontology#emotion"): Literal("excited"),
            URIRef("https://humem.ai/ontology#event"): Literal("Conference meeting"),
        }

        # Add long-term episodic memories using add_memory
        self.memory.add_memory(
            [self.triples[0]], self.episodic_qualifiers_1
        )  # Memory ID 0
        self.memory.add_memory(
            [self.triples[1]], self.episodic_qualifiers_2
        )  # Memory ID 1

    def test_retrieve_memories(self) -> None:
        """Test retrieval of memories by qualifiers and ensure correct memory IDs are returned."""
        # Retrieve memories with the filter location="New York"
        location_qualifier = {
            URIRef("https://humem.ai/ontology#location"): Literal("New York")
        }
        retrieved_memories = self.memory.get_memories(qualifiers=location_qualifier)

        # Extract memory IDs from the retrieved memories
        memory_ids = [
            int(retrieved_memories.graph.value(statement, humemai.memoryID))
            for statement in retrieved_memories.graph.subjects(RDF.type, RDF.Statement)
        ]

        # Check if the correct memory was retrieved (Memory ID 0 should be retrieved)
        self.assertIn(0, memory_ids)
        self.assertNotIn(1, memory_ids)

    def test_delete_memory_by_retrieved_id(self) -> None:
        """Test deleting a memory by retrieving its ID and verifying it is deleted."""
        # Retrieve memories with location="New York"
        location_qualifier = {
            URIRef("https://humem.ai/ontology#location"): Literal("New York")
        }
        retrieved_memories = self.memory.get_memories(qualifiers=location_qualifier)

        # Extract memory IDs from the retrieved memories
        memory_ids_to_delete = [
            int(retrieved_memories.graph.value(statement, humemai.memoryID))
            for statement in retrieved_memories.graph.subjects(RDF.type, RDF.Statement)
        ]

        # Delete the retrieved memory from the original memory system
        for memory_id in memory_ids_to_delete:
            self.memory.delete_memory(Literal(memory_id, datatype=XSD.integer))

        # Verify that the memory has been deleted
        for memory_id in memory_ids_to_delete:
            deleted_memory = self.memory.get_memory_by_id(
                Literal(memory_id, datatype=XSD.integer)
            )
            self.assertIsNone(
                deleted_memory,
                f"Memory ID {memory_id} should have been deleted but was not.",
            )

    def test_memory_deletion_does_not_affect_others(self) -> None:
        """Test that deleting one memory does not affect other memories."""
        # Retrieve and delete memories with location="New York"
        location_qualifier = {
            URIRef("https://humem.ai/ontology#location"): Literal("New York")
        }
        retrieved_memories = self.memory.get_memories(qualifiers=location_qualifier)
        memory_ids_to_delete = [
            int(retrieved_memories.graph.value(statement, humemai.memoryID))
            for statement in retrieved_memories.graph.subjects(RDF.type, RDF.Statement)
        ]
        for memory_id in memory_ids_to_delete:
            self.memory.delete_memory(Literal(memory_id, datatype=XSD.integer))

        # Ensure memory with ID 1 still exists (the one for "London")
        remaining_memory = self.memory.get_memory_by_id(
            Literal(1, datatype=XSD.integer)
        )
        self.assertIsNotNone(remaining_memory, "Memory ID 1 should still exist.")

    def test_delete_non_existent_memory(self) -> None:
        """Test that deleting a non-existent memory ID does not raise an error."""
        non_existent_memory_id = Literal(
            999, datatype=XSD.integer
        )  # Choose an ID that doesn't exist
        try:
            self.memory.delete_memory(non_existent_memory_id)
        except Exception as e:
            self.fail(
                f"delete_memory raised an exception on a non-existent memory ID: {e}"
            )


class TestMemoryRetrievalMethods(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a memory object with both short-term and long-term memories
        for testing.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")
        self.ex = Namespace("https://example.org/")

        # Add short-term memory
        triples_short = [
            (
                self.ex.Alice,
                self.ex.meet,
                self.ex.Bob,
            )
        ]
        current_time = Literal(datetime.now().isoformat(), datatype=XSD.dateTime)
        location = Literal("Paris")
        qualifiers_short = {
            self.humemai.currentTime: current_time,
            self.humemai.location: location,
        }
        self.memory.add_memory(triples_short, qualifiers_short)

        # Add long-term episodic memory
        triples_episodic = [
            (
                self.ex.Bob,
                self.ex.visit,
                self.ex.Paris,
            )
        ]
        time_episodic = Literal("2022-05-05T10:00:00", datatype=XSD.dateTime)
        location_episodic = Literal("Paris")
        qualifiers_episodic = {
            self.humemai.eventTime: time_episodic,
            self.humemai.location: location_episodic,
        }
        self.memory.add_memory(triples_episodic, qualifiers_episodic)

        # Add long-term semantic memory
        triples_semantic = [
            (
                self.ex.Charlie,
                self.ex.hasType,
                self.ex.Human,
            )
        ]
        derived_from = Literal("research_paper_1")
        strength = Literal(5, datatype=XSD.integer)
        qualifiers_semantic = {
            self.humemai.derivedFrom: derived_from,
            self.humemai.strength: strength,
            self.humemai.knownSince: Literal(
                "2023-01-01T10:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_memory(triples_semantic, qualifiers_semantic)

    def test_get_short_term_memories(self) -> None:
        """
        Test that get_short_term_memories() retrieves only short-term memories.
        """
        short_term_memories = self.memory.get_short_term_memories()
        short_term_count = short_term_memories.get_memory_count()

        # Check that the short-term memory count is 1
        self.assertEqual(short_term_count, 1)

        # Iterate over reified statements and validate that we get the correct subject, predicate, object
        for statement in short_term_memories.graph.subjects(RDF.type, RDF.Statement):
            subj = short_term_memories.graph.value(statement, RDF.subject)
            pred = short_term_memories.graph.value(statement, RDF.predicate)
            obj = short_term_memories.graph.value(statement, RDF.object)
            current_time = short_term_memories.graph.value(
                statement, self.humemai.currentTime
            )

            # Ensure this is a short-term memory
            self.assertIsNotNone(current_time)

            # Validate that the subject, predicate, and object are correctly extracted
            self.assertEqual(subj, self.ex.Alice)
            self.assertEqual(pred, self.ex.meet)
            self.assertEqual(obj, self.ex.Bob)

    def test_get_long_term_memories(self) -> None:
        """
        Test that get_long_term_memories() retrieves only long-term memories.
        """
        long_term_memories = self.memory.get_long_term_memories()
        long_term_count = long_term_memories.get_memory_count()

        # Check that the long-term memory count matches the expected number
        self.assertEqual(long_term_count, 2)

        # Validate the presence of episodic and semantic memories
        episodic_found = False
        semantic_found = False

        # Iterate over reified statements and check qualifiers to determine episodic vs. semantic
        for statement in long_term_memories.graph.subjects(RDF.type, RDF.Statement):
            subj = long_term_memories.graph.value(statement, RDF.subject)
            pred = long_term_memories.graph.value(statement, RDF.predicate)
            obj = long_term_memories.graph.value(statement, RDF.object)
            event_time = long_term_memories.graph.value(
                statement, self.humemai.eventTime
            )
            current_time = long_term_memories.graph.value(
                statement, self.humemai.currentTime
            )
            derived_from = long_term_memories.graph.value(
                statement, self.humemai.derivedFrom
            )

            # Ensure that there is no currentTime for long-term memories
            self.assertIsNone(current_time)

            if event_time:
                episodic_found = True
                self.assertEqual(subj, self.ex.Bob)
                self.assertEqual(pred, self.ex.visit)
                self.assertEqual(obj, self.ex.Paris)
            if derived_from:
                semantic_found = True
                self.assertEqual(subj, self.ex.Charlie)
                self.assertEqual(pred, self.ex.hasType)
                self.assertEqual(obj, self.ex.Human)

        # Assert that we found both an episodic and a semantic memory
        self.assertTrue(episodic_found, "Episodic memory not found.")
        self.assertTrue(semantic_found, "Semantic memory not found.")

    def test_memory_counts(self) -> None:
        """
        Test that the memory counts align correctly.
        """
        short_term_count = self.memory.get_short_term_memories().get_memory_count()
        long_term_count = self.memory.get_long_term_memories().get_memory_count()
        total_count = self.memory.get_memory_count()

        # Ensure the total count matches short-term + long-term
        self.assertEqual(total_count, short_term_count + long_term_count)

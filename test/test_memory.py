"""Test Memory class"""

import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock

from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef

from humemai import Memory

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology/")


class TestMemory(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)

    def test_add_single_memory_with_qualifiers(self):
        """
        Test adding a single triple with qualifiers and check if it's correctly stored.
        """
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )
        qualifiers = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }

        # Add memory
        self.memory.add_memory([triple], qualifiers)

        # Verify memory has been added correctly
        result = repr(self.memory)

        expected = "[https://example.org/person/Alice, https://example.org/relationship/knows, https://example.org/person/Bob, {'https://humem.ai/ontology/memoryID': '0', 'https://humem.ai/ontology/currentTime': '2024-04-27T10:00:00', 'https://humem.ai/ontology/location': 'New York'}]"

        self.assertIn(expected, result)

    def test_add_duplicate_memory_with_different_qualifiers(self):
        """
        Test adding the same triple multiple times with different qualifiers.
        Ensure that both reified statements are stored with different qualifiers.
        """
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )

        qualifiers1 = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        qualifiers2 = {
            humemai.currentTime: "2024-04-27T12:00:00",
            humemai.location: "London",
        }

        # Add memory twice with different qualifiers
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Verify that both reified statements are stored
        result = repr(self.memory)

        expected1 = "[https://example.org/person/Alice, https://example.org/relationship/knows, https://example.org/person/Bob, {'https://humem.ai/ontology/memoryID': '0', 'https://humem.ai/ontology/currentTime': '2024-04-27T10:00:00', 'https://humem.ai/ontology/location': 'New York'}]"
        expected2 = "[https://example.org/person/Alice, https://example.org/relationship/knows, https://example.org/person/Bob, {'https://humem.ai/ontology/memoryID': '1', 'https://humem.ai/ontology/currentTime': '2024-04-27T12:00:00', 'https://humem.ai/ontology/location': 'London'}]"

        # Ensure both entries are present
        self.assertIn(expected1, result)
        self.assertIn(expected2, result)

    def test_main_triple_not_duplicated(self):
        """
        Test that adding the same main triple multiple times doesn't duplicate the main triple,
        only the reified statements are different.
        """
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )

        qualifiers1 = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        qualifiers2 = {
            humemai.currentTime: "2024-04-27T12:00:00",
            humemai.location: "London",
        }

        # Add memory twice with different qualifiers
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Check that there is only one main triple
        triples = list(
            self.memory.graph.triples(
                (
                    URIRef("https://example.org/person/Alice"),
                    URIRef("https://example.org/relationship/knows"),
                    URIRef("https://example.org/person/Bob"),
                )
            )
        )

        # There should only be one occurrence of the main triple
        self.assertEqual(len(triples), 1)


class TestMemoryDelete(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

        # Define sample triples
        self.triple1 = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )

        self.triple2 = (
            URIRef("https://example.org/person/Bob"),
            URIRef("https://example.org/relationship/likes"),
            Literal("Chocolate"),
        )

        # Define qualifiers for the first triple
        self.qualifiers1 = {
            self.humemai.currentTime: "2024-04-27T10:00:00",
            self.humemai.location: "New York",
        }

        # Define qualifiers for the second triple
        self.qualifiers2 = {
            self.humemai.currentTime: "2024-04-27T11:00:00",
            self.humemai.location: "Paris",
        }

        # Add sample triples to memory
        self.memory.add_memory([self.triple1], self.qualifiers1)
        self.memory.add_memory([self.triple2], self.qualifiers2)

    def test_delete_existing_memory(self):
        """
        Test deleting a memory that exists in the graph.
        """
        # Ensure the memory is there before deletion
        before_delete = repr(self.memory)
        self.assertIn("https://example.org/person/Alice", before_delete)
        self.assertIn("https://humem.ai/ontology/location", before_delete)

        # Delete the triple (Alice knows Bob)
        self.memory.delete_triple(*self.triple1)

        # Ensure the memory is no longer in the graph after deletion
        after_delete = repr(self.memory)

        # Check that the specific triple (Alice, knows, Bob) and its qualifiers are gone
        self.assertNotIn("https://example.org/person/Alice", after_delete)
        self.assertNotIn("https://example.org/relationship/knows", after_delete)
        self.assertNotIn(
            "New York", after_delete
        )  # The specific qualifier for this triple

        # Ensure that the other memory still exists (Bob likes Chocolate)
        self.assertIn("https://example.org/person/Bob", after_delete)
        self.assertIn(
            "https://humem.ai/ontology/location", after_delete
        )  # The other qualifier

    def test_delete_triple_with_multiple_qualifiers(self):
        """
        Test deleting a memory with multiple qualifiers.
        Ensure that the main triple and all associated reified statements are removed.
        """
        # Add the same triple with different qualifiers (simulating multiple reified statements)
        qualifiers_additional = {
            self.humemai.currentTime: "2024-04-27T14:00:00",
            self.humemai.location: "London",
        }

        self.memory.add_memory([self.triple1], qualifiers_additional)

        # Verify that both versions of the triple are present before deletion
        before_delete = repr(self.memory)
        self.assertIn("New York", before_delete)
        self.assertIn("London", before_delete)

        # Delete the triple (Alice knows Bob)
        self.memory.delete_triple(*self.triple1)

        # Verify that both versions of the triple are removed
        after_delete = repr(self.memory)
        self.assertNotIn("New York", after_delete)
        self.assertNotIn("London", after_delete)
        self.assertNotIn("https://example.org/person/Alice", after_delete)

    def test_delete_non_existent_memory(self):
        """
        Test deleting a memory that does not exist in the graph.
        Ensure that attempting to delete a non-existent triple doesn't affect the graph.
        """
        # Define a non-existent triple
        non_existent_triple = (
            URIRef("https://example.org/person/David"),
            URIRef("https://example.org/relationship/hates"),
            Literal("Broccoli"),
        )

        # Verify that the non-existent triple is not in the graph before deletion
        before_delete = repr(self.memory)
        self.assertNotIn("https://example.org/person/David", before_delete)

        # Attempt to delete the non-existent triple
        self.memory.delete_triple(*non_existent_triple)

        # Verify that the graph is unchanged after the attempt
        after_delete = repr(self.memory)
        self.assertEqual(before_delete, after_delete)

    def test_delete_triple_with_no_qualifiers(self):
        """
        Test deleting a memory that has no qualifiers (just a main triple).
        Ensure that only the main triple is deleted.
        """
        # Add a simple triple without qualifiers
        simple_triple = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/David"),
        )
        self.memory.add_memory([simple_triple], {})

        # Verify that the triple exists before deletion
        before_delete = repr(self.memory)
        self.assertIn("https://example.org/person/Charlie", before_delete)

        # Delete the simple triple
        self.memory.delete_triple(*simple_triple)

        # Verify that the triple is deleted
        after_delete = repr(self.memory)
        self.assertNotIn("https://example.org/person/Charlie", after_delete)


class TestMemoryShortTerm(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_add_single_short_term_memory(self):
        """
        Test adding a single short-term memory with current location and current time.
        """
        # Define the sample triple
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )
        location = "New York"
        currentTime = "2024-04-27T10:00:00"

        # Add the short-term memory
        self.memory.add_short_term_memory([triple], location, currentTime)

        # Verify that the memory was added with the correct qualifiers
        result = repr(self.memory)
        self.assertIn("https://example.org/person/Alice", result)

        # Instead of checking for the full URI, check for 'currentTime' and 'location'
        self.assertIn("currentTime", result)
        self.assertIn("2024-04-27T10:00:00", result)
        self.assertIn("location", result)
        self.assertIn("New York", result)

    def test_add_short_term_memory_with_default_time(self):
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
        location = "Berlin"

        # Capture the current time
        before_time = datetime.now().isoformat()

        # Add the short-term memory without specifying time
        self.memory.add_short_term_memory([triple], location)

        # Capture the result and verify
        result = repr(self.memory)
        self.assertIn("https://example.org/person/Charlie", result)
        self.assertIn("location", result)
        self.assertIn("Berlin", result)

        # Ensure the currentTime was added automatically
        self.assertIn("currentTime", result)

        # Verify that the time is close to the current time (tolerance for small delay)
        after_time = datetime.now().isoformat()
        currentTime_in_result = result.split("currentTime': '")[1].split("'")[
            0
        ]  # Extract the current time from result
        self.assertTrue(
            before_time <= currentTime_in_result <= after_time,
            "The time should be the current time.",
        )

    def test_add_multiple_short_term_memories(self):
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
        location = "Tokyo"
        currentTime = "2024-04-28T09:00:00"

        # Add the short-term memory with multiple triples
        self.memory.add_short_term_memory(triples, location, currentTime)

        # Verify both triples were added correctly
        result = repr(self.memory)
        self.assertIn("https://example.org/person/David", result)
        self.assertIn("https://example.org/person/Eve", result)
        self.assertIn("Chocolate", result)
        self.assertIn("currentTime", result)
        self.assertIn("2024-04-28T09:00:00", result)
        self.assertIn("location", result)
        self.assertIn("Tokyo", result)


class TestMemoryLongTerm(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_add_single_episodic_memory(self):
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
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }

        # Add the episodic long-term memory
        self.memory.add_long_term_memory("episodic", [triple], **qualifiers)

        # Verify that the memory was added with the correct qualifiers
        result = repr(self.memory)
        self.assertIn("https://example.org/person/Alice", result)
        self.assertIn("https://example.org/event/met", result)
        self.assertIn("https://example.org/person/Bob", result)
        self.assertIn("location", result)
        self.assertIn("Paris", result)
        self.assertIn("time", result)
        self.assertIn("2024-04-27T15:00:00", result)
        self.assertIn("emotion", result)
        self.assertIn("happy", result)

    def test_add_single_semantic_memory(self):
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
        qualifiers = {"strength": 5, "derivedFrom": "textbook"}

        # Add the semantic long-term memory
        self.memory.add_long_term_memory("semantic", [triple], **qualifiers)

        # Verify that the memory was added with the correct qualifiers
        result = repr(self.memory)
        self.assertIn("https://example.org/entity/Dog", result)
        self.assertIn("https://example.org/relationship/is", result)
        self.assertIn("https://example.org/entity/Animal", result)
        self.assertIn("strength", result)
        self.assertIn("5", result)
        self.assertIn("derivedFrom", result)
        self.assertIn("textbook", result)

    def test_add_multiple_episodic_memories_same_triple(self):
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
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }

        # Define the second set of qualifiers
        qualifiers_2 = {
            "location": "London",
            "time": "2024-05-01T09:00:00",
            "emotion": "excited",
        }

        # Add both episodic long-term memories with different qualifiers
        self.memory.add_long_term_memory("episodic", [triple], **qualifiers_1)
        self.memory.add_long_term_memory("episodic", [triple], **qualifiers_2)

        # Verify that both memories were added with the correct qualifiers
        result = repr(self.memory)

        # Check first memory
        self.assertIn("Paris", result)
        self.assertIn("2024-04-27T15:00:00", result)
        self.assertIn("happy", result)

        # Check second memory
        self.assertIn("London", result)
        self.assertIn("2024-05-01T09:00:00", result)
        self.assertIn("excited", result)

    def test_invalid_qualifiers_for_memory_types(self):
        """
        Test that invalid qualifiers raise an error when added to episodic or semantic memories.
        """
        # Define the sample triple for both tests
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        # Try adding an invalid strength qualifier to episodic memory
        with self.assertRaises(ValueError):
            self.memory.add_long_term_memory(
                "episodic",
                [triple],
                location="bar",
                time="2024-04-27T15:30:00",
                strength=10,
            )

        # Try adding an invalid location qualifier to semantic memory
        with self.assertRaises(ValueError):
            self.memory.add_long_term_memory("semantic", [triple], location="Paris")


class TestMemoryGetMemories(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

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

        # Add memories to the graph
        qualifiers_episodic = {
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic
        )

        qualifiers_semantic = {"strength": 8, "derivedFrom": "study"}
        self.memory.add_long_term_memory(
            "semantic", [self.triple2], **qualifiers_semantic
        )

        # Short-term memory
        self.memory.add_short_term_memory(
            [self.triple3], "Berlin", "2024-04-27T18:00:00"
        )

    def test_basic_memory_retrieval(self):
        """
        Test basic retrieval of all memories without filters.
        """
        retrieved_memory = self.memory.get_memories()

        # Check if all triples are retrieved
        result = repr(retrieved_memory)
        self.assertIn("https://example.org/person/Alice", result)
        self.assertIn("https://example.org/person/Bob", result)
        self.assertIn("https://example.org/entity/Cat", result)
        self.assertIn("Chocolate", result)

    def test_qualifiers_association(self):
        """
        Test that qualifiers are correctly associated with their triples.
        """
        retrieved_memory = self.memory.get_memories()

        # Ensure the qualifiers are correctly attached to the main triples
        result = repr(retrieved_memory)

        # Check for Alice's memory qualifiers
        self.assertIn("location", result)
        self.assertIn("Paris", result)
        self.assertIn("time", result)
        self.assertIn("2024-04-27T15:00:00", result)
        self.assertIn("emotion", result)
        self.assertIn("happy", result)

        # Check for Cat's memory qualifiers
        self.assertIn("derivedFrom", result)
        self.assertIn("study", result)
        self.assertIn("strength", result)
        self.assertIn("8", result)

    def test_memory_retrieval_with_subject_filter(self):
        """
        Test memory retrieval with a subject filter.
        """
        # Retrieve only memories with Alice as the subject
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/person/Alice")
        )

        result = repr(retrieved_memory)
        self.assertIn("https://example.org/person/Alice", result)
        self.assertNotIn("https://example.org/entity/Cat", result)
        self.assertNotIn("Chocolate", result)

    def test_memory_retrieval_with_object_filter(self):
        """
        Test memory retrieval with an object filter.
        """
        # Retrieve only memories where Bob is the object
        retrieved_memory = self.memory.get_memories(
            object_=URIRef("https://example.org/person/Bob")
        )

        result = repr(retrieved_memory)
        self.assertIn("https://example.org/person/Alice", result)
        self.assertIn("https://example.org/person/Bob", result)
        self.assertNotIn("https://example.org/entity/Cat", result)
        self.assertNotIn("Chocolate", result)

    def test_unique_blank_nodes(self):
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

    def test_qualifiers_with_blank_nodes(self):
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

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

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

    def test_single_memory_count(self):
        """
        Test that the count is correct when one unique memory is added.
        """
        # Add one memory
        qualifiers_episodic = {
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic
        )

        # Test memory count
        memory_count = self.memory.get_triple_count()
        self.assertEqual(memory_count, 1, "Memory count should be 1.")

    def test_duplicate_memory_with_different_qualifiers(self):
        """
        Test that duplicate memories with different qualifiers are only counted once.
        """
        # Add the same triple multiple times with different qualifiers
        qualifiers_episodic1 = {
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }

        qualifiers_episodic2 = {
            "location": "London",
            "time": "2024-05-01T09:00:00",
            "emotion": "excited",
        }

        qualifiers_episodic3 = {
            "location": "Berlin",
            "time": "2024-05-10T10:30:00",
            "emotion": "nervous",
        }

        # Add the same triple with different qualifiers
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic1
        )
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic2
        )
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic3
        )

        # Test memory count (should still be 1 since the triple is the same)
        memory_count = self.memory.get_triple_count()
        self.assertEqual(
            memory_count,
            1,
            "Memory count should still be 1 for the same triple with different qualifiers.",
        )

    def test_unique_memory_count(self):
        """
        Test that the count is correct when multiple unique triples are added.
        """
        # Add multiple unique triples
        qualifiers_episodic = {
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }

        qualifiers_semantic = {"strength": 8, "derivedFrom": "study"}

        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic
        )
        self.memory.add_long_term_memory(
            "semantic", [self.triple2], **qualifiers_semantic
        )
        self.memory.add_short_term_memory(
            [self.triple3], "Berlin", "2024-04-27T18:00:00"
        )

        # Test memory count (should be 3 because all triples are unique)
        memory_count = self.memory.get_triple_count()
        self.assertEqual(
            memory_count, 3, "Memory count should be 3 for three unique triples."
        )

    def test_mixed_duplicate_and_unique_triples(self):
        """
        Test memory count with a mix of duplicate and unique triples.
        """
        # Add the same triple multiple times with different qualifiers
        qualifiers_episodic1 = {
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }

        qualifiers_episodic2 = {
            "location": "London",
            "time": "2024-05-01T09:00:00",
            "emotion": "excited",
        }

        qualifiers_semantic = {"strength": 8, "derivedFrom": "study"}

        # Add duplicate and unique triples
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic1
        )
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **qualifiers_episodic2
        )
        self.memory.add_long_term_memory(
            "semantic", [self.triple2], **qualifiers_semantic
        )
        self.memory.add_short_term_memory(
            [self.triple3], "Berlin", "2024-04-27T18:00:00"
        )

        # Test memory count (should be 3: one for the duplicate triple, and two for the unique ones)
        memory_count = self.memory.get_triple_count()
        self.assertEqual(
            memory_count,
            3,
            "Memory count should be 3 for a mix of one duplicate and two unique triples.",
        )


class TestMemoryModifyStrength(unittest.TestCase):
    def setUp(self):
        """
        Set up the Memory object and add some semantic and episodic memories.
        """
        self.memory = Memory(verbose_repr=True)

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
        qualifiers_semantic = {"strength": 8, "derivedFrom": "study"}
        self.memory.add_long_term_memory(
            "semantic", [self.triple_semantic], **qualifiers_semantic
        )

        # Add an episodic memory without strength (it should not have a strength value)
        qualifiers_episodic = {
            "location": "Paris",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
        }
        self.memory.add_long_term_memory(
            "episodic", [self.triple_episodic], **qualifiers_episodic
        )

    def test_increment_strength(self):
        """
        Test that the strength is incremented correctly by 5.
        """
        # Increment strength by 5
        self.memory.modify_strength(
            filters={"subject": URIRef("https://example.org/entity/Cat")},
            increment_by=5,
        )

        # Retrieve the updated memory and verify the new strength
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "13", repr(retrieved_memory), "Strength should be incremented to 13"
        )

    def test_multiply_strength(self):
        """
        Test that the strength is multiplied correctly by 2.
        """
        # Multiply strength by 2
        self.memory.modify_strength(
            filters={"subject": URIRef("https://example.org/entity/Cat")}, multiply_by=2
        )

        # Retrieve the updated memory and verify the new strength
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "16", repr(retrieved_memory), "Strength should be multiplied to 16"
        )

    def test_increment_and_multiply_strength(self):
        """
        Test that the strength is incremented and then multiplied correctly.
        """
        # First increment by 5
        self.memory.modify_strength(
            filters={"subject": URIRef("https://example.org/entity/Cat")},
            increment_by=5,
        )

        # Then multiply by 2
        self.memory.modify_strength(
            filters={"subject": URIRef("https://example.org/entity/Cat")}, multiply_by=2
        )

        # Retrieve the updated memory and verify the final strength
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "26",
            repr(retrieved_memory),
            "Strength should be 26 after incrementing and multiplying",
        )

    def test_no_strength_for_episodic(self):
        """
        Test that episodic memories without a strength qualifier are not affected.
        """
        # Try to modify strength of an episodic memory (should not change anything)
        self.memory.modify_strength(
            filters={"subject": URIRef("https://example.org/person/Alice")},
            increment_by=5,
        )

        # Retrieve the episodic memory and ensure no strength was added
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/person/Alice")
        )
        self.assertNotIn(
            "strength",
            repr(retrieved_memory),
            "Episodic memory should not have a strength qualifier",
        )

    def test_modify_multiple_statements(self):
        """
        Test that all reified statements for the same triple are updated.
        """
        # Add a second reified statement for the same semantic triple with a different strength
        qualifiers_semantic_2 = {"strength": 10, "derivedFrom": "research"}
        self.memory.add_long_term_memory(
            "semantic", [self.triple_semantic], **qualifiers_semantic_2
        )

        # Increment strength by 5 for all reified statements
        self.memory.modify_strength(
            filters={"subject": URIRef("https://example.org/entity/Cat")},
            increment_by=5,
        )

        # Verify that both statements were updated
        retrieved_memory = self.memory.get_memories(
            subject=URIRef("https://example.org/entity/Cat")
        )
        self.assertIn(
            "13", repr(retrieved_memory), "First strength should be incremented to 13"
        )
        self.assertIn(
            "15", repr(retrieved_memory), "Second strength should be incremented to 15"
        )


class TestMemoryTimeFilters(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_filter_memories_by_time_range(self):
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
            self.humemai.currentTime: "2024-04-27T10:00:00",
            self.humemai.location: "New York",
        }
        qualifiers2 = {
            self.humemai.currentTime: "2024-04-27T12:00:00",
            self.humemai.location: "London",
        }

        # Add memories with different times
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Retrieve memories within a time range
        filtered_memory = self.memory.get_memories(
            lower_time_bound="2024-04-27T09:00:00",
            upper_time_bound="2024-04-27T11:00:00",
        )

        # Verify only the memory within the time range is retrieved
        result = repr(filtered_memory)
        self.assertIn("2024-04-27T10:00:00", result)
        self.assertNotIn("2024-04-27T12:00:00", result)

    def test_filter_memories_outside_time_range(self):
        """
        Test that no memories are returned if they fall outside the specified time range.
        """
        triple = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/likes"),
            Literal("Ice Cream"),
        )
        qualifiers = {
            self.humemai.currentTime: "2024-04-27T18:00:00",
            self.humemai.location: "Berlin",
        }

        # Add memory outside the desired time range
        self.memory.add_memory([triple], qualifiers)

        # Retrieve memories within a time range where this memory does not fall
        filtered_memory = self.memory.get_memories(
            lower_time_bound="2024-04-27T09:00:00",
            upper_time_bound="2024-04-27T17:00:00",
        )

        # Ensure no memories are returned
        result = repr(filtered_memory)
        self.assertNotIn("Ice Cream", result)


class TestMemoryLocationFilters(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_filter_memories_by_location(self):
        """
        Test retrieving memories based on location filtering.
        """
        triple = (
            URIRef("https://example.org/person/David"),
            URIRef("https://example.org/relationship/met"),
            URIRef("https://example.org/person/Eve"),
        )
        qualifiers1 = {
            self.humemai.currentTime: "2024-04-27T10:00:00",
            self.humemai.location: "New York",
        }
        qualifiers2 = {
            self.humemai.currentTime: "2024-04-27T12:00:00",
            self.humemai.location: "Berlin",
        }

        # Add memory with different locations
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Retrieve memories based on location filter
        filtered_memory = self.memory.get_memories(location="New York")

        # Verify only the memory with the specified location is retrieved
        result = repr(filtered_memory)
        self.assertIn("New York", result)
        self.assertNotIn("Berlin", result)

    def test_filter_memories_by_invalid_location(self):
        """
        Test that no memories are returned when an invalid location is specified.
        """
        triple = (
            URIRef("https://example.org/person/Frank"),
            URIRef("https://example.org/relationship/met"),
            URIRef("https://example.org/person/Grace"),
        )
        qualifiers = {
            self.humemai.currentTime: "2024-04-27T10:00:00",
            self.humemai.location: "Paris",
        }

        # Add memory with a location
        self.memory.add_memory([triple], qualifiers)

        # Retrieve memories based on an invalid location filter
        filtered_memory = self.memory.get_memories(location="Tokyo")

        # Verify no memories are returned
        result = repr(filtered_memory)
        self.assertNotIn("Paris", result)
        self.assertNotIn("Frank", result)


class TestMemoryEmotionFilters(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_filter_memories_by_emotion(self):
        """
        Test retrieving memories based on emotion filtering.
        """
        triple = (
            URIRef("https://example.org/person/Gary"),
            URIRef("https://example.org/relationship/felt"),
            URIRef("https://example.org/event/happy"),
        )
        qualifiers1 = {
            self.humemai.emotion: "happy",
            self.humemai.currentTime: "2024-04-27T14:00:00",
        }
        qualifiers2 = {
            self.humemai.emotion: "sad",
            self.humemai.currentTime: "2024-04-27T16:00:00",
        }

        # Add memories with different emotions
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Retrieve memories based on emotion filter
        filtered_memory = self.memory.get_memories(emotion="happy")

        # Verify only the memory with the specified emotion is retrieved
        result = repr(filtered_memory)
        self.assertIn("happy", result)
        self.assertNotIn("sad", result)


class TestInvalidInputHandling(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_invalid_time_format(self):
        """
        Test that invalid time formats raise a ValueError.
        """
        triple = (
            URIRef("https://example.org/person/Helen"),
            URIRef("https://example.org/relationship/met"),
            URIRef("https://example.org/person/Ivan"),
        )
        invalid_qualifiers = {
            self.humemai.currentTime: "04-27-2024 10:00:00",  # Invalid format
            self.humemai.location: "Paris",
        }

        with self.assertRaises(ValueError):
            self.memory.add_memory([triple], invalid_qualifiers)


class TestMemoryDeleteWithTime(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_delete_triple_with_time_filter(self):
        """
        Test deleting a memory that has a time qualifier.
        """
        triple = (
            URIRef("https://example.org/person/Luke"),
            URIRef("https://example.org/relationship/met"),
            URIRef("https://example.org/person/Mary"),
        )
        qualifiers = {
            self.humemai.currentTime: "2024-04-27T12:00:00",
            self.humemai.location: "Paris",
        }

        # Add the memory
        self.memory.add_memory([triple], qualifiers)

        # Verify the memory is there before deletion
        result_before = repr(self.memory)
        self.assertIn("Paris", result_before)
        self.assertIn("Luke", result_before)

        # Delete the memory
        self.memory.delete_triple(*triple)

        # Verify the memory is no longer there after deletion
        result_after = repr(self.memory)
        self.assertNotIn("Paris", result_after)
        self.assertNotIn("Luke", result_after)


class TestMemoryInvalidQualifiers(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

    def test_missing_qualifiers_in_long_term_memory_episodic(self):
        """
        Test that missing required qualifiers in episodic long-term memory raise a ValueError.
        """
        triple = (
            URIRef("https://example.org/person/Jack"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Kate"),
        )

        # Missing 'time'
        with self.assertRaises(ValueError) as context:
            self.memory.add_long_term_memory(
                "episodic", [triple], location="Paris", time=None, emotion="happy"
            )
        self.assertIn(
            "Missing required qualifier for episodic memory: time",
            str(context.exception),
        )


class TestMemoryCounts(unittest.TestCase):
    """
    Test cases for get_memory_count and get_triple_count methods in the Memory class.
    """

    def setUp(self):
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Memory(verbose_repr=True)

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

    def test_no_memories(self):
        """
        Test that counts are zero when no memories have been added.
        """
        self.assertEqual(
            self.memory.get_memory_count(),
            0,
            "Memory count should be 0 when no memories are added.",
        )
        self.assertEqual(
            self.memory.get_triple_count(),
            0,
            "Triple count should be 0 when no triples are added.",
        )

    def test_single_memory_single_reified_statement(self):
        """
        Test counts after adding a single memory with one reified statement.
        """
        qualifiers = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        self.memory.add_memory([self.triple1], qualifiers)

        self.assertEqual(
            self.memory.get_memory_count(),
            1,
            "Memory count should be 1 after adding one reified statement.",
        )
        self.assertEqual(
            self.memory.get_triple_count(),
            1,
            "Triple count should be 1 after adding one unique triple.",
        )

    def test_single_memory_multiple_reified_statements(self):
        """
        Test counts after adding the same triple multiple times with different qualifiers.
        """
        qualifiers1 = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        qualifiers2 = {
            humemai.currentTime: "2024-04-27T12:00:00",
            humemai.location: "London",
        }
        qualifiers3 = {
            humemai.currentTime: "2024-04-27T14:00:00",
            humemai.location: "Paris",
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
            self.memory.get_triple_count(),
            1,
            "Triple count should be 1 since all statements share the same triple.",
        )

    def test_multiple_unique_triples_multiple_reified_statements(self):
        """
        Test counts after adding multiple unique triples with multiple reified statements each.
        """
        qualifiers1 = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        qualifiers2 = {
            humemai.currentTime: "2024-04-27T12:00:00",
            humemai.location: "London",
        }
        qualifiers3 = {
            humemai.currentTime: "2024-04-27T14:00:00",
            humemai.location: "Paris",
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
            self.memory.get_triple_count(),
            expected_triple_count,
            f"Triple count should be {expected_triple_count} after adding three unique triples.",
        )

    def test_deleting_reified_statements(self):
        """
        Test that deleting a triple removes all its reified statements.
        """
        qualifiers1 = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        qualifiers2 = {
            humemai.currentTime: "2024-04-27T12:00:00",
            humemai.location: "London",
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
            self.memory.get_triple_count(), 2, "Initial triple count should be 2."
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
            self.memory.get_triple_count(),
            1,
            "Triple count should be 1 after deleting triple1.",
        )

    def test_deleting_non_existent_triple(self):
        """
        Test that deleting a non-existent triple does not affect the counts.
        """
        qualifiers = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        self.memory.add_memory([self.triple1], qualifiers)

        # Initial counts
        self.assertEqual(
            self.memory.get_memory_count(), 1, "Initial memory count should be 1."
        )
        self.assertEqual(
            self.memory.get_triple_count(), 1, "Initial triple count should be 1."
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
            self.memory.get_triple_count(),
            1,
            "Triple count should remain 1 after attempting to delete a non-existent triple.",
        )

    def test_triple_count_with_no_reified_statements(self):
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

        # Depending on implementation, this might still create a reified statement even with no qualifiers
        # If no reified statement is created, counts should be zero
        # However, based on your `add_memory` method, it always creates a reified statement
        # So expect one reified statement and one triple

        self.assertEqual(
            self.memory.get_memory_count(),
            1,
            "Memory count should be 1 after adding one reified statement with empty qualifiers.",
        )
        self.assertEqual(
            self.memory.get_triple_count(),
            1,
            "Triple count should be 1 after adding one unique triple.",
        )

    def test_triple_count_after_adding_and_deleting(self):
        """
        Test triple and memory counts after adding and deleting multiple triples.
        """
        qualifiers1 = {
            humemai.currentTime: "2024-04-27T10:00:00",
            humemai.location: "New York",
        }
        qualifiers2 = {
            humemai.currentTime: "2024-04-27T12:00:00",
            humemai.location: "London",
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
            self.memory.get_triple_count(), 2, "Initial triple count should be 2."
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
            self.memory.get_triple_count(),
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
            self.memory.get_triple_count(),
            0,
            "Triple count should be 0 after deleting all triples.",
        )


class TestModifyEpisodicEvent(unittest.TestCase):

    def setUp(self):
        """
        Set up the Memory object with some episodic, semantic, and short-term memories.
        """
        self.memory = Memory(verbose_repr=True)

        # Define some triples
        self.triple1 = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        self.triple2 = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/David"),
        )

        # Define semantic triples (which should not be affected by the modification)
        self.semantic_triple1 = (
            URIRef("https://example.org/entity/Cat"),
            URIRef("https://example.org/relationship/is"),
            URIRef("https://example.org/entity/Animal"),
        )

        self.semantic_triple2 = (
            URIRef("https://example.org/entity/Tree"),
            URIRef("https://example.org/relationship/has"),
            URIRef("https://example.org/entity/Leaves"),
        )

        # Define episodic memories with qualifiers
        episodic_qualifiers1 = {
            "location": "New York",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
            "event": "Meeting for coffee",
        }

        episodic_qualifiers2 = {
            "location": "London",
            "time": "2024-04-27T17:00:00",
            "emotion": "excited",
            "event": "Dinner event",
        }

        # Define semantic memories with qualifiers
        semantic_qualifiers1 = {"strength": 5, "derivedFrom": "study"}

        semantic_qualifiers2 = {"strength": 7, "derivedFrom": "observation"}

        # Add episodic long-term memories
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **episodic_qualifiers1
        )
        self.memory.add_long_term_memory(
            "episodic", [self.triple2], **episodic_qualifiers2
        )

        # Add semantic long-term memories (these should not be affected by modify_episodic_event)
        self.memory.add_long_term_memory(
            "semantic", [self.semantic_triple1], **semantic_qualifiers1
        )
        self.memory.add_long_term_memory(
            "semantic", [self.semantic_triple2], **semantic_qualifiers2
        )

        # Add short-term memories (these should not be affected by modify_episodic_event)
        self.memory.add_short_term_memory(
            [self.triple1], location="Berlin", currentTime="2024-04-27T18:00:00"
        )
        self.memory.add_short_term_memory(
            [self.triple2], location="Tokyo", currentTime="2024-04-27T19:00:00"
        )

    def test_modify_event_for_episodic_memories(self):
        """
        Test that only episodic memories within the time range have their event modified.
        """
        lower_time_bound = "2024-04-27T00:00:00"
        upper_time_bound = "2024-04-28T00:00:00"
        new_event = "Updated Event for Conference"

        # Modify the event for episodic memories within the time range
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve all memories to check
        result = repr(self.memory)

        # Check that episodic memories are updated
        self.assertIn("Updated Event for Conference", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertNotIn("Dinner event", result)

        # Check that semantic memories are NOT updated
        self.assertIn("study", result)
        self.assertIn("observation", result)

        # Check that short-term memories are NOT affected
        self.assertIn("Berlin", result)
        self.assertIn("Tokyo", result)

    def test_no_effect_on_memories_outside_time_range(self):
        """
        Test that memories outside the specified time range are not modified.
        """
        lower_time_bound = "2024-04-26T00:00:00"
        upper_time_bound = "2024-04-27T12:00:00"  # Before any episodic memory time
        new_event = "Updated Event"

        # Modify the event for episodic memories (which shouldn't modify anything)
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve all memories to check
        result = repr(self.memory)

        # Ensure that no events have been modified
        self.assertIn("Meeting for coffee", result)
        self.assertIn("Dinner event", result)
        self.assertNotIn("Updated Event", result)

    def test_only_episodic_memories_are_modified(self):
        """
        Test that only episodic long-term memories are modified, semantic and short-term memories are unaffected.
        """
        lower_time_bound = "2024-04-27T00:00:00"
        upper_time_bound = "2024-04-28T00:00:00"
        new_event = "Conference Event Update"

        # Modify episodic memories
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve memories and check that only episodic ones are updated
        result = repr(self.memory)

        # Check that episodic events are updated
        self.assertIn("Conference Event Update", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertNotIn("Dinner event", result)

        # Semantic memories should remain unchanged
        self.assertIn("study", result)
        self.assertIn("observation", result)

        # Short-term memories should remain unchanged
        self.assertIn("Berlin", result)
        self.assertIn("Tokyo", result)

    def test_modify_multiple_episodic_memories(self):
        """
        Test that multiple episodic memories within the time range are modified.
        """
        lower_time_bound = "2024-04-27T00:00:00"
        upper_time_bound = "2024-04-28T00:00:00"
        new_event = "Global Event Update"

        # Modify the event for episodic memories
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve memories and check that both episodic events are updated
        result = repr(self.memory)

        # Check that both episodic memories are updated
        self.assertIn("Global Event Update", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertNotIn("Dinner event", result)

    def test_event_modification_with_additional_filters(self):
        """
        Test that episodic memories can be modified with additional filters (e.g., subject, location).
        """
        lower_time_bound = "2024-04-27T00:00:00"
        upper_time_bound = "2024-04-28T00:00:00"
        new_event = "Filtered Event Update"

        # Modify the event for episodic memories with a subject filter (only Alice's memory)
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
            subject=URIRef("https://example.org/person/Alice"),
        )

        # Retrieve memories and check that only Alice's event was updated
        result = repr(self.memory)

        # Check that only Alice's episodic memory is updated
        self.assertIn("Filtered Event Update", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertIn(
            "Dinner event", result
        )  # Charlie's memory should remain unchanged


class TestIncrementRecalled(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance with episodic and semantic memories.
        """
        self.memory = Memory(verbose_repr=True)
        self.humemai = Namespace("https://humem.ai/ontology/")

        # Define some triples
        self.triple1 = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Bob"),
        )

        self.semantic_triple1 = (
            URIRef("https://example.org/entity/Cat"),
            URIRef("https://example.org/relationship/is"),
            URIRef("https://example.org/entity/Animal"),
        )

        # Define episodic and semantic qualifiers
        episodic_qualifiers = {
            "location": "New York",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
            "event": "Meeting for coffee",
        }

        semantic_qualifiers = {"strength": 5, "derivedFrom": "study"}

        # Add episodic and semantic memories
        self.memory.add_long_term_memory(
            "episodic", [self.triple1], **episodic_qualifiers
        )
        self.memory.add_long_term_memory(
            "semantic", [self.semantic_triple1], **semantic_qualifiers
        )

    def test_multiple_increments_recalled(self):
        """
        Test incrementing the recalled value multiple times for both episodic and semantic memories.
        """
        # Increment recalled values once
        self.memory.increment_recalled()

        # Retrieve the updated memory system after first increment
        result = repr(self.memory)

        # Check that the 'recalled' value has been incremented to 1 for both episodic and semantic memories
        self.assertIn("'https://humem.ai/ontology/recalled': '1'", result)

        # Increment recalled values again
        self.memory.increment_recalled()

        # Retrieve the updated memory system after second increment
        result = repr(self.memory)

        # Check that the 'recalled' value has been incremented to 2 for both episodic and semantic memories
        self.assertIn("'https://humem.ai/ontology/recalled': '2'", result)

        # Increment recalled values a third time
        self.memory.increment_recalled()

        # Retrieve the updated memory system after third increment
        result = repr(self.memory)

        # Check that the 'recalled' value has been incremented to 3 for both episodic and semantic memories
        self.assertIn("'https://humem.ai/ontology/recalled': '3'", result)


class TestMemoryDelete(unittest.TestCase):
    def setUp(self):
        """Set up the memory and add test data before each test."""
        self.memory = Memory()

        # Define multiple triples for episodic and semantic memories
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
            (
                URIRef("https://example.org/person/Alice"),
                URIRef("https://example.org/relationship/knows"),
                URIRef("https://example.org/person/David"),
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
        ]

        # Define episodic and semantic qualifiers
        self.episodic_qualifiers_1 = {
            "location": "New York",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
            "event": "Coffee meeting",
        }
        self.episodic_qualifiers_2 = {
            "location": "London",
            "time": "2024-05-01T10:00:00",
            "emotion": "excited",
            "event": "Conference meeting",
        }
        self.episodic_qualifiers_3 = {
            "location": "Paris",
            "time": "2024-05-05T18:00:00",
            "emotion": "curious",
            "event": "Workshop",
        }

        self.semantic_qualifiers_1 = {"derivedFrom": "animal_research", "strength": 5}
        self.semantic_qualifiers_2 = {"derivedFrom": "pet_database", "strength": 10}

        # Add episodic memories and semantic memories
        self.memory.add_long_term_memory(
            "episodic", [self.triples[0]], **self.episodic_qualifiers_1
        )  # Memory ID 0
        self.memory.add_long_term_memory(
            "episodic", [self.triples[0]], **self.episodic_qualifiers_2
        )  # Memory ID 1
        self.memory.add_long_term_memory(
            "episodic", [self.triples[1]], **self.episodic_qualifiers_1
        )  # Memory ID 2
        self.memory.add_long_term_memory(
            "episodic", [self.triples[2]], **self.episodic_qualifiers_3
        )  # Memory ID 3

        self.memory.add_long_term_memory(
            "semantic", [self.triples[3]], **self.semantic_qualifiers_1
        )  # Memory ID 4
        self.memory.add_long_term_memory(
            "semantic", [self.triples[3]], **self.semantic_qualifiers_2
        )  # Memory ID 5

        # Add a short-term memory
        self.memory.add_short_term_memory(
            [self.triples[4]], location="Alice's home"
        )  # Memory ID 6

    def test_memory_retrieval_by_id(self):
        """Test retrieving memories by ID."""
        memory_0 = self.memory.get_memory_by_id(0)
        self.assertIn(
            "New York", memory_0["qualifiers"]["https://humem.ai/ontology/location"]
        )  # Check in the qualifiers

        memory_1 = self.memory.get_memory_by_id(1)
        self.assertIn(
            "London", memory_1["qualifiers"]["https://humem.ai/ontology/location"]
        )  # Check in the qualifiers

        memory_4 = self.memory.get_memory_by_id(4)
        self.assertIn(
            "animal_research",
            memory_4["qualifiers"]["https://humem.ai/ontology/derivedFrom"],
        )  # Check semantic memory

        memory_6 = self.memory.get_memory_by_id(6)
        self.assertIn(
            "Alice's home", memory_6["qualifiers"]["https://humem.ai/ontology/location"]
        )  # Check short-term memory

    def test_memory_deletion_by_id(self):
        """Test deleting a memory by ID."""
        self.memory.delete_memory(1)  # Delete memory ID 1
        deleted_memory = self.memory.get_memory_by_id(1)
        self.assertIsNone(deleted_memory)  # Ensure memory ID 1 is deleted

        # Ensure other memories are still present
        memory_0 = self.memory.get_memory_by_id(0)
        self.assertIsNotNone(memory_0)

    def test_triple_deletion(self):
        """Test deleting a triple and all associated memories."""
        self.memory.delete_triple(*self.triples[0])

        # Both Memory ID 0 and 1 refer to the triple (Alice, met, Bob), so they should be deleted
        self.assertIsNone(self.memory.get_memory_by_id(0))
        self.assertIsNone(self.memory.get_memory_by_id(1))

        # Ensure other memories are not affected
        memory_2 = self.memory.get_memory_by_id(2)
        memory_3 = self.memory.get_memory_by_id(3)
        self.assertIsNotNone(memory_2)
        self.assertIsNotNone(memory_3)

    def test_count_triples_and_memories(self):
        """Test counting triples and reified memories."""
        self.assertEqual(self.memory.get_triple_count(), 5)  # 5 unique triples
        self.assertEqual(self.memory.get_memory_count(), 7)  # 7 reified memories

    def test_delete_triple_and_memory_count(self):
        """Test memory and triple count after deleting a triple."""
        # Delete triple (Alice, met, Bob) and ensure memory count is updated
        self.memory.delete_triple(*self.triples[0])
        self.assertEqual(self.memory.get_triple_count(), 4)  # 1 triple removed
        self.assertEqual(self.memory.get_memory_count(), 5)  # 2 memories removed


class TestMemoryRetrievalAndDeletion(unittest.TestCase):

    def setUp(self):
        """Set up the memory system and add initial long-term memories."""
        self.memory = Memory()

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

        # Define qualifiers
        self.episodic_qualifiers_1 = {
            "location": "New York",
            "time": "2024-04-27T15:00:00",
            "emotion": "happy",
            "event": "Coffee meeting",
        }

        self.episodic_qualifiers_2 = {
            "location": "London",
            "time": "2024-05-01T10:00:00",
            "emotion": "excited",
            "event": "Conference meeting",
        }

        # Add long-term episodic memories
        self.memory.add_long_term_memory(
            "episodic", [self.triples[0]], **self.episodic_qualifiers_1
        )  # Memory ID 0
        self.memory.add_long_term_memory(
            "episodic", [self.triples[1]], **self.episodic_qualifiers_2
        )  # Memory ID 1

    def test_retrieve_memories(self):
        """Test retrieval of memories by qualifiers and ensure correct memory IDs are returned."""
        # Retrieve memories with the filter location="New York"
        retrieved_memories = self.memory.get_memories(location="New York")

        # Extract memory IDs from the retrieved memories
        memory_ids = [
            int(retrieved_memories.graph.value(statement, humemai.memoryID))
            for statement in retrieved_memories.graph.subjects(RDF.type, RDF.Statement)
        ]

        # Check if the correct memory was retrieved (Memory ID 0 should be retrieved)
        self.assertIn(0, memory_ids)
        self.assertNotIn(1, memory_ids)

    def test_delete_memory_by_retrieved_id(self):
        """Test deleting a memory by retrieving its ID and verifying it is deleted."""
        # Retrieve memories with location="New York"
        retrieved_memories = self.memory.get_memories(location="New York")

        # Extract memory IDs from the retrieved memories
        memory_ids_to_delete = [
            int(retrieved_memories.graph.value(statement, humemai.memoryID))
            for statement in retrieved_memories.graph.subjects(RDF.type, RDF.Statement)
        ]

        # Delete the retrieved memory from the original memory system
        for memory_id in memory_ids_to_delete:
            self.memory.delete_memory(memory_id)

        # Verify that the memory has been deleted
        for memory_id in memory_ids_to_delete:
            deleted_memory = self.memory.get_memory_by_id(memory_id)
            self.assertIsNone(
                deleted_memory,
                f"Memory ID {memory_id} should have been deleted but was not.",
            )

    def test_memory_deletion_does_not_affect_others(self):
        """Test that deleting one memory does not affect other memories."""
        # Retrieve and delete memories with location="New York"
        retrieved_memories = self.memory.get_memories(location="New York")
        memory_ids_to_delete = [
            int(retrieved_memories.graph.value(statement, humemai.memoryID))
            for statement in retrieved_memories.graph.subjects(RDF.type, RDF.Statement)
        ]
        for memory_id in memory_ids_to_delete:
            self.memory.delete_memory(memory_id)

        # Ensure memory with ID 1 still exists (the one for "London")
        remaining_memory = self.memory.get_memory_by_id(1)
        self.assertIsNotNone(remaining_memory, "Memory ID 1 should still exist.")

    def test_delete_non_existent_memory(self):
        """Test that deleting a non-existent memory ID does not raise an error."""
        non_existent_memory_id = 999  # Choose an ID that doesn't exist
        try:
            self.memory.delete_memory(non_existent_memory_id)
        except Exception as e:
            self.fail(
                f"delete_memory raised an exception on a non-existent memory ID: {e}"
            )


class TestRefiedMemory(unittest.TestCase):
    def setUp(self):
        # Initialize the Memory instance
        self.memory = Memory()

        # Mock the graph object to simulate RDF triples and statements
        self.memory.graph = MagicMock()

        # Example URIs for the test
        self.subj = URIRef("https://example.org/person/Alice")
        self.pred = URIRef("https://example.org/event/met")
        self.obj = URIRef("https://example.org/person/Bob")

        # Example reified statement as a blank node
        self.reified_statement = BNode()

        # Working memory instance for adding triples
        self.working_memory = Memory()
        self.working_memory.graph = MagicMock()

    def test_add_reified_statement_and_increment_recall(self):
        """Test that the reified statement is added to working memory and recalled is incremented."""

        # Mock the subjects method to return the reified statement we are interested in
        self.memory.graph.subjects.return_value = [self.reified_statement]

        # Mock the value method to return the correct subject, predicate, and object
        # The method calls value three times for each reified statement (subject, predicate, object)
        self.memory.graph.value.side_effect = [self.subj, self.pred, self.obj]

        # Simulate an initial recalled value of 1
        self.memory.graph.triples.return_value = [
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(1, datatype=XSD.integer),
            )
        ]

        # Define the side effect function for predicate_objects to return updated 'recalled' value
        def predicate_objects_side_effect(statement):
            if statement == self.reified_statement:
                return [
                    (
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(2, datatype=XSD.integer),
                    )
                ]
            return []

        self.memory.graph.predicate_objects.side_effect = predicate_objects_side_effect

        # Call the method under test
        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
            self.subj, self.pred, self.obj, self.working_memory
        )

        # Check that the recall value was incremented to 2
        self.memory.graph.set.assert_called_with(
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(2, datatype=XSD.integer),
            )
        )

        # Ensure the reified statement was added to the working memory with the updated recall
        self.working_memory.graph.add.assert_any_call(
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(2, datatype=XSD.integer),
            )
        )

    def test_add_reified_statement_with_no_initial_recall(self):
        """Test that if there is no recall value, it starts from 0 and increments to 1."""

        # Mock the subjects method to return the reified statement
        self.memory.graph.subjects.return_value = [self.reified_statement]

        # Mock the value method to return the correct subject, predicate, and object
        self.memory.graph.value.side_effect = [self.subj, self.pred, self.obj]

        # Simulate no initial recall value
        self.memory.graph.triples.return_value = []

        # Define the side effect function for predicate_objects to return updated 'recalled' value
        def predicate_objects_side_effect(statement):
            if statement == self.reified_statement:
                return [
                    (
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(1, datatype=XSD.integer),
                    )
                ]
            return []

        self.memory.graph.predicate_objects.side_effect = predicate_objects_side_effect

        # Call the method under test
        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
            self.subj, self.pred, self.obj, self.working_memory
        )

        # Check that the recall value was set to 1
        self.memory.graph.set.assert_called_with(
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

        # Ensure the reified statement with recall value 1 was added to the working memory
        self.working_memory.graph.add.assert_any_call(
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

    def test_specific_statement_handling(self):
        """Test that when a specific reified statement is provided, only that statement is processed."""

        # Mock a specific reified statement that we will process
        specific_statement = BNode()

        # Mock the subjects method to return multiple statements
        self.memory.graph.subjects.return_value = [
            self.reified_statement,
            specific_statement,
        ]

        # Mock the value method to return subject, predicate, object for both statements
        self.memory.graph.value.side_effect = [
            self.subj,
            self.pred,
            self.obj,  # For self.reified_statement
            self.subj,
            self.pred,
            self.obj,  # For specific_statement
        ]

        # Simulate no initial recall values
        self.memory.graph.triples.return_value = []

        # Define the side effect function for predicate_objects to return updated 'recalled' value for specific_statement
        def predicate_objects_side_effect(statement):
            if statement == specific_statement:
                return [
                    (
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(1, datatype=XSD.integer),
                    )
                ]
            return []

        self.memory.graph.predicate_objects.side_effect = predicate_objects_side_effect

        # Call the method with a specific statement
        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
            self.subj,
            self.pred,
            self.obj,
            self.working_memory,
            specific_statement=specific_statement,
        )

        # Check that only the specific statement was processed
        self.memory.graph.set.assert_called_once_with(
            (
                specific_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

        # Ensure only the specific statement was added to the working memory
        self.working_memory.graph.add.assert_called_once_with(
            (
                specific_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

    def test_no_reified_statements(self):
        """Test that if no reified statements match, nothing is processed."""

        # Mock subjects to return no matching reified statements
        self.memory.graph.subjects.return_value = []

        # Call the method under test
        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
            self.subj, self.pred, self.obj, self.working_memory
        )

        # Check that set was never called since no reified statements exist
        self.memory.graph.set.assert_not_called()

        # Check that nothing was added to the working memory
        self.working_memory.graph.add.assert_not_called()

    def test_multiple_reified_statements(self):
        """Test that multiple reified statements for the same triple are processed correctly."""

        # Create additional reified statements for the same triple
        reified_statement_2 = BNode()

        # Mock the subjects method to return multiple reified statements
        self.memory.graph.subjects.return_value = [
            self.reified_statement,
            reified_statement_2,
        ]

        # Mock the value method to return subject, predicate, object for both statements
        self.memory.graph.value.side_effect = [
            self.subj,
            self.pred,
            self.obj,  # For self.reified_statement
            self.subj,
            self.pred,
            self.obj,  # For reified_statement_2
        ]

        # Simulate an initial recall value of 1 for both statements
        self.memory.graph.triples.side_effect = [
            [
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
            [
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
        ]

        # Define the side effect function for predicate_objects to return updated 'recalled' values
        def predicate_objects_side_effect(statement):
            if statement in [self.reified_statement, reified_statement_2]:
                return [
                    (
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(2, datatype=XSD.integer),
                    )
                ]
            return []

        self.memory.graph.predicate_objects.side_effect = predicate_objects_side_effect

        # Call the method under test
        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
            self.subj, self.pred, self.obj, self.working_memory
        )

        # Check that the recall value was incremented to 2 for both statements
        expected_calls = [
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
        ]
        self.memory.graph.set.assert_has_calls(expected_calls, any_order=True)

        # Ensure both reified statements were added to the working memory with updated recall
        expected_add_calls = [
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
        ]
        self.working_memory.graph.add.assert_has_calls(
            expected_add_calls, any_order=True
        )

    def test_set_called_correctly_with_qualifiers(self):
        """Test that qualifiers are correctly added to the working memory."""

        # Mock the subjects method to return the reified statement
        self.memory.graph.subjects.return_value = [self.reified_statement]

        # Mock the value method to return the correct subject, predicate, and object
        self.memory.graph.value.side_effect = [self.subj, self.pred, self.obj]

        # Define the side effect function for triples to return the recalled value separately
        def triples_side_effect(query):
            if query == (
                self.reified_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                None,
            ):
                return [
                    (
                        self.reified_statement,
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(1, datatype=XSD.integer),
                    )
                ]
            return []

        # Mock the triples method to return recalled value when specifically queried
        self.memory.graph.triples.side_effect = triples_side_effect

        # Define the side effect function for predicate_objects to return updated 'recalled' and qualifiers
        def predicate_objects_side_effect(statement):
            if statement == self.reified_statement:
                return [
                    (
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(2, datatype=XSD.integer),
                    ),
                    (
                        URIRef("https://humem.ai/ontology/location"),
                        Literal("New York", datatype=XSD.string),
                    ),
                    (
                        URIRef("https://humem.ai/ontology/time"),
                        Literal("2024-04-27T15:00:00", datatype=XSD.dateTime),
                    ),
                    (
                        URIRef("https://humem.ai/ontology/emotion"),
                        Literal("happy", datatype=XSD.string),
                    ),
                    (
                        URIRef("https://humem.ai/ontology/event"),
                        Literal("Coffee meeting", datatype=XSD.string),
                    ),
                ]
            return []

        # Mock predicate_objects to return the qualifiers and updated recalled value
        self.memory.graph.predicate_objects.side_effect = predicate_objects_side_effect

        # Call the method under test
        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
            self.subj, self.pred, self.obj, self.working_memory
        )

        # Check that the recall value was incremented to 2
        self.memory.graph.set.assert_called_with(
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology/recalled"),
                Literal(2, datatype=XSD.integer),
            )
        )

        # Ensure all qualifiers are added to the working memory
        expected_add_calls = [
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/location"),
                    Literal("New York", datatype=XSD.string),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/time"),
                    Literal("2024-04-27T15:00:00", datatype=XSD.dateTime),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/emotion"),
                    Literal("happy", datatype=XSD.string),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/event"),
                    Literal("Coffee meeting", datatype=XSD.string),
                )
            ),
        ]
        self.working_memory.graph.add.assert_has_calls(
            expected_add_calls, any_order=True
        )

    def test_add_reified_statement_and_increment_recall_no_specific_statement(self):
        """Test that without a specific statement, all matching reified statements are processed."""

        # Create additional reified statement
        reified_statement_2 = BNode()

        # Mock the subjects method to return multiple reified statements
        self.memory.graph.subjects.return_value = [
            self.reified_statement,
            reified_statement_2,
        ]

        # Mock the value method to return subject, predicate, object for both statements
        self.memory.graph.value.side_effect = [
            self.subj,
            self.pred,
            self.obj,  # For self.reified_statement
            self.subj,
            self.pred,
            self.obj,  # For reified_statement_2
        ]

        # Simulate an initial recall value of 1 for both statements
        self.memory.graph.triples.side_effect = [
            [
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
            [
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
        ]

        # Define the side effect function for predicate_objects to return updated 'recalled' values
        def predicate_objects_side_effect(statement):
            if statement in [self.reified_statement, reified_statement_2]:
                return [
                    (
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(2, datatype=XSD.integer),
                    )
                ]
            return []

        self.memory.graph.predicate_objects.side_effect = predicate_objects_side_effect

        # Call the method without specifying a specific_statement
        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
            self.subj, self.pred, self.obj, self.working_memory
        )

        # Check that the recall value was incremented to 2 for both statements
        expected_calls = [
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
        ]
        self.memory.graph.set.assert_has_calls(expected_calls, any_order=True)

        # Ensure both reified statements were added to the working memory with updated recall
        expected_add_calls = [
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
        ]
        self.working_memory.graph.add.assert_has_calls(
            expected_add_calls, any_order=True
        )


class TestGetShortTerm(unittest.TestCase):
    def setUp(self):
        """Set up the test case with a Memory instance and populate the RDF graph."""
        # Initialize a Memory instance and use an actual Graph instance
        self.memory = Memory(verbose_repr=True)
        self.memory.graph = Graph()

        # Short-term triple
        self.subj = URIRef("https://example.org/person/Alice")
        self.pred = URIRef("https://example.org/event/met")
        self.obj = URIRef("https://example.org/person/Bob")

        # Qualifiers for the short-term memory
        self.current_time_qualifier = URIRef("https://humem.ai/ontology/currentTime")
        self.location_qualifier = URIRef("https://humem.ai/ontology/location")
        self.emotion_qualifier = URIRef("https://humem.ai/ontology/emotion")

        self.current_time_literal = Literal(
            "2024-10-03T15:00:00", datatype=XSD.dateTime
        )
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
                self.current_time_qualifier,
                self.current_time_literal,
            )
        )
        self.memory.graph.add(
            (self.reified_statement, self.location_qualifier, self.location_literal)
        )
        self.memory.graph.add(
            (self.reified_statement, self.emotion_qualifier, self.emotion_literal)
        )

    def testget_short_term_memories(self):
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
            (reified_statement, self.current_time_qualifier, self.current_time_literal),
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

    def test_empty_short_term_memory(self):
        """Test that the method handles an empty graph correctly."""
        # Use an empty graph
        self.memory.graph = Graph()

        # Call the method to retrieve short-term memories
        short_term_memory = self.memory.get_short_term_memories()

        # Ensure the graph is empty in this case
        self.assertEqual(
            len(list(short_term_memory.graph.triples((None, None, None)))), 0
        )

    def test_partial_qualifiers_in_short_term_memory(self):
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
                self.current_time_qualifier,
                self.current_time_literal,
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
            (reified_statement, self.current_time_qualifier, self.current_time_literal),
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


class TestMemorySaveLoad(unittest.TestCase):
    def setUp(self):
        """Set up a Memory object and add some test data"""
        # Create an instance of the Memory class
        self.memory = Memory(verbose_repr=True)
        self.test_file = "test_memory.ttl"

        # Define the humemai and example namespaces
        humemai = Namespace("https://humem.ai/ontology/")
        ex = Namespace("https://example.org/")

        # Add multiple short-term memories
        self.memory.add_short_term_memory(
            [(ex.Alice, ex.met, ex.Bob)],
            location="New York",
            currentTime="2024-04-27T15:00:00",
        )
        self.memory.add_short_term_memory(
            [(ex.Alice, ex.met, ex.Bob)],
            location="New York",
            currentTime="2024-04-27T16:00:00",
        )

        self.memory.add_short_term_memory(
            [(ex.Bob, ex.knows, ex.Alice)],
            location="Paris",
            currentTime="2024-05-01T10:00:00",
        )

        # Add multiple long-term episodic memories
        self.memory.add_long_term_memory(
            "episodic",
            [(ex.Alice, ex.attended, ex.Conference)],
            location="London",
            time="2023-09-15T10:00:00",
            emotion="excited",
            event="AI Conference",
        )
        self.memory.add_long_term_memory(
            "episodic",
            [(ex.Alice, ex.spokeWith, ex.Charlie)],
            location="London",
            time="2023-09-15T11:00:00",
            emotion="happy",
            event="AI Conference",
        )
        self.memory.add_long_term_memory(
            "episodic",
            [(ex.Alice, ex.spokeWith, ex.Charlie)],
            location="London",
            time="2023-09-15T11:00:00",
            emotion="sad",
            event="AI Conference",
        )

        # Add multiple long-term semantic memories
        self.memory.add_long_term_memory(
            "semantic",
            [(ex.Dog, ex.hasType, ex.Animal)],
            derivedFrom="research_paper_1",
            strength=5,
        )
        self.memory.add_long_term_memory(
            "semantic",
            [(ex.Cat, ex.hasType, ex.Animal)],
            derivedFrom="research_paper_2",
            strength=4,
        )
        self.memory.add_long_term_memory(
            "semantic",
            [(ex.Cat, ex.hasType, ex.Animal)],
            derivedFrom="research_paper_2",
            strength=4,
        )
        self.memory.add_long_term_memory(
            "semantic",
            [(ex.Cat, ex.hasType, ex.Animal)],
            derivedFrom="research_paper_2",
            strength=4,
        )

        # Store the triples before wiping
        self.original_triples = self._get_triples_without_bnodes(self.memory.graph)

    def _get_triples_without_bnodes(self, graph):
        """
        Get a set of triples from the graph, with BNodes replaced by a generic placeholder.
        This is to ignore the differences in BNode IDs.
        """
        triples_without_bnodes = set()
        for s, p, o in graph:
            s = "BNode" if isinstance(s, BNode) else s
            o = "BNode" if isinstance(o, BNode) else o
            triples_without_bnodes.add((s, p, o))
        return triples_without_bnodes

    def test_save_and_load_ttl(self):
        """Test saving and loading memory to/from TTL"""
        # Step 1: Save the memory to a TTL file
        self.memory.save_to_ttl(self.test_file)

        # Step 2: Wipe the memory (clear the graph)
        self.memory.graph = self.memory.graph.__class__()

        # Ensure the memory is wiped
        self.assertEqual(
            len(self.memory.graph), 0, "Memory should be empty after wiping."
        )

        # Step 3: Load the memory back from the TTL file
        self.memory.load_from_ttl(self.test_file)

        # Step 4: Compare the triples before and after loading
        loaded_triples = self._get_triples_without_bnodes(self.memory.graph)
        self.assertEqual(
            self.original_triples,
            loaded_triples,
            "Memory mismatch after loading from TTL",
        )

    def tearDown(self):
        """Clean up after tests"""
        # Remove the test file if it exists
        if os.path.exists(self.test_file):
            os.remove(self.test_file)


class TestMemoryCounts(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh Memory instance for each test.
        """
        self.memory = Memory(verbose_repr=False)

    def test_memory_counts(self):
        """
        Test to verify memory counts and ensure relationships between total, short-term,
        and long-term memories are consistent.
        """
        # Add some short-term memories
        triples_1 = [
            (
                URIRef("https://example.org/Alice"),
                URIRef("https://example.org/knows"),
                URIRef("https://example.org/Bob"),
            )
        ]
        triples_2 = [
            (
                URIRef("https://example.org/Charlie"),
                URIRef("https://example.org/spokeWith"),
                URIRef("https://example.org/Alice"),
            )
        ]

        self.memory.add_short_term_memory(
            triples_1, location="Paris", currentTime="2024-05-01T10:00:00"
        )
        self.memory.add_short_term_memory(
            triples_2, location="London", currentTime="2024-05-02T12:00:00"
        )

        # Add some long-term episodic memories
        self.memory.add_long_term_memory(
            memory_type="episodic",
            triples=triples_1,
            location="Paris",
            time="2024-05-01T10:00:00",
            emotion="happy",
            event="AI Conference",
        )

        # Add some long-term semantic memories
        self.memory.add_long_term_memory(
            memory_type="semantic",
            triples=triples_2,
            derivedFrom="research_paper_1",
            strength=5,
        )

        # Retrieve counts
        total_memories = self.memory.get_memory_count()
        short_term_memories = self.memory.get_short_term_memory_count()
        long_term_memories = self.memory.get_long_term_memory_count()
        episodic_memories = self.memory.get_long_term_episodic_memory_count()
        semantic_memories = self.memory.get_long_term_semantic_memory_count()

        # Check that total memory count matches the sum of short-term and long-term memories
        self.assertEqual(
            total_memories,
            short_term_memories + long_term_memories,
            "Total memory count should equal short-term + long-term memory count",
        )

        # Check that long-term memory count matches the sum of episodic and semantic memories
        self.assertEqual(
            long_term_memories,
            episodic_memories + semantic_memories,
            "Long-term memory count should equal episodic + semantic memory count",
        )


class TestMemoryRetrievalMethods(unittest.TestCase):

    def setUp(self):
        """
        Set up a memory object with both short-term and long-term memories
        for testing.
        """
        self.memory = Memory()

        # Add short-term memory
        triples_short = [
            (
                URIRef("https://example.org/Alice"),
                URIRef("https://example.org/meet"),
                URIRef("https://example.org/Bob"),
            )
        ]
        current_time = datetime.now().isoformat()
        location = "Paris"
        self.memory.add_short_term_memory(
            triples_short, location, currentTime=current_time
        )

        # Add long-term episodic memory
        triples_episodic = [
            (
                URIRef("https://example.org/Bob"),
                URIRef("https://example.org/visit"),
                URIRef("https://example.org/Paris"),
            )
        ]
        time_episodic = "2022-05-05T10:00:00"
        location_episodic = "Paris"
        self.memory.add_long_term_memory(
            memory_type="episodic",
            triples=triples_episodic,
            time=time_episodic,
            location=location_episodic,
        )

        # Add long-term semantic memory
        triples_semantic = [
            (
                URIRef("https://example.org/Charlie"),
                URIRef("https://example.org/hasType"),
                URIRef("https://example.org/Human"),
            )
        ]
        derived_from = "research_paper_1"
        strength = 5
        self.memory.add_long_term_memory(
            memory_type="semantic",
            triples=triples_semantic,
            derivedFrom=derived_from,
            strength=strength,
        )

    def test_get_short_term_memories(self):
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
                statement, humemai.currentTime
            )

            # Ensure this is a short-term memory
            self.assertIsNotNone(current_time)

            # Validate that the subject, predicate, and object are correctly extracted
            self.assertEqual(subj, URIRef("https://example.org/Alice"))
            self.assertEqual(pred, URIRef("https://example.org/meet"))
            self.assertEqual(obj, URIRef("https://example.org/Bob"))

    def test_get_long_term_memories(self):
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
            time = long_term_memories.graph.value(statement, humemai.time)
            current_time = long_term_memories.graph.value(
                statement, humemai.currentTime
            )
            derived_from = long_term_memories.graph.value(
                statement, humemai.derivedFrom
            )

            # Debug print to inspect the triples and qualifiers
            print(f"Statement: {statement}")
            print(f"Subject: {subj}, Predicate: {pred}, Object: {obj}")
            print(f"Time (episodic): {time}")
            print(f"DerivedFrom (semantic): {derived_from}")
            print(f"CurrentTime (should be None for long-term): {current_time}")
            print("---")

            # Ensure that there is no currentTime for long-term memories
            self.assertIsNone(current_time)

            if time:
                episodic_found = True
                self.assertEqual(subj, URIRef("https://example.org/Bob"))
                self.assertEqual(pred, URIRef("https://example.org/visit"))
                self.assertEqual(obj, URIRef("https://example.org/Paris"))
            if derived_from:
                semantic_found = True
                self.assertEqual(subj, URIRef("https://example.org/Charlie"))
                self.assertEqual(pred, URIRef("https://example.org/hasType"))
                self.assertEqual(obj, URIRef("https://example.org/Human"))

        # Assert that we found both an episodic and a semantic memory
        self.assertTrue(episodic_found, "Episodic memory not found.")
        self.assertTrue(semantic_found, "Semantic memory not found.")

    def test_memory_counts(self):
        """
        Test that the memory counts align correctly.
        """
        short_term_count = self.memory.get_short_term_memories().get_memory_count()
        long_term_count = self.memory.get_long_term_memories().get_memory_count()
        total_count = self.memory.get_memory_count()

        # Ensure the total count matches short-term + long-term
        self.assertEqual(total_count, short_term_count + long_term_count)


class TestMemoryIteration(unittest.TestCase):

    def setUp(self):
        """
        Set up the Memory object and add short-term, episodic, and semantic memories.
        """
        self.memory = Memory()

        # Add a short-term memory
        triples_short_term = [
            (
                URIRef("https://example.org/Alice"),
                URIRef("https://example.org/meet"),
                URIRef("https://example.org/Bob"),
            )
        ]
        self.memory.add_short_term_memory(
            triples=triples_short_term,
            location="Paris",
            currentTime="2023-05-05T10:00:00",
        )

        # Add a long-term episodic memory
        triples_episodic = [
            (
                URIRef("https://example.org/Bob"),
                URIRef("https://example.org/visit"),
                URIRef("https://example.org/Paris"),
            )
        ]
        self.memory.add_long_term_memory(
            memory_type="episodic",
            triples=triples_episodic,
            location="Paris",
            time="2023-05-06T10:00:00",
            emotion="happy",
        )

        # Add a long-term semantic memory
        triples_semantic = [
            (
                URIRef("https://example.org/Charlie"),
                URIRef("https://example.org/hasType"),
                URIRef("https://example.org/Human"),
            )
        ]
        self.memory.add_long_term_memory(
            memory_type="semantic",
            triples=triples_semantic,
            derivedFrom="research_paper_1",
            strength=5,
        )

    def test_iterate_short_term_memories(self):
        """
        Test iteration over short-term memories.
        """
        short_term_count = 0
        for subj, pred, obj, qualifiers in self.memory.iterate_memories(
            memory_type="short_term"
        ):
            short_term_count += 1
            self.assertEqual(subj, URIRef("https://example.org/Alice"))
            self.assertEqual(pred, URIRef("https://example.org/meet"))
            self.assertEqual(obj, URIRef("https://example.org/Bob"))
            self.assertIn("https://humem.ai/ontology/currentTime", qualifiers)

        self.assertEqual(
            short_term_count, 1, "There should be exactly 1 short-term memory."
        )

    def test_iterate_long_term_memories(self):
        """
        Test iteration over long-term memories.
        """
        long_term_count = 0
        for subj, pred, obj, qualifiers in self.memory.iterate_memories(
            memory_type="long_term"
        ):
            long_term_count += 1
            self.assertIn(
                subj,
                [
                    URIRef("https://example.org/Bob"),
                    URIRef("https://example.org/Charlie"),
                ],
            )

        self.assertEqual(
            long_term_count, 2, "There should be exactly 2 long-term memories."
        )

    def test_iterate_episodic_memories(self):
        """
        Test iteration over episodic long-term memories.
        """
        episodic_count = 0
        for subj, pred, obj, qualifiers in self.memory.iterate_memories(
            memory_type="episodic"
        ):
            episodic_count += 1
            self.assertEqual(subj, URIRef("https://example.org/Bob"))
            self.assertEqual(pred, URIRef("https://example.org/visit"))
            self.assertEqual(obj, URIRef("https://example.org/Paris"))
            self.assertIn("https://humem.ai/ontology/time", qualifiers)

        self.assertEqual(
            episodic_count, 1, "There should be exactly 1 episodic memory."
        )

    def test_iterate_semantic_memories(self):
        """
        Test iteration over semantic long-term memories.
        """
        semantic_count = 0
        for subj, pred, obj, qualifiers in self.memory.iterate_memories(
            memory_type="semantic"
        ):
            semantic_count += 1
            self.assertEqual(subj, URIRef("https://example.org/Charlie"))
            self.assertEqual(pred, URIRef("https://example.org/hasType"))
            self.assertEqual(obj, URIRef("https://example.org/Human"))
            self.assertIn("https://humem.ai/ontology/derivedFrom", qualifiers)

        self.assertEqual(
            semantic_count, 1, "There should be exactly 1 semantic memory."
        )

    def test_iterate_all_memories(self):
        """
        Test iteration over all memories (short-term + long-term).
        """
        all_count = 0
        for subj, pred, obj, qualifiers in self.memory.iterate_memories(
            memory_type="all"
        ):
            all_count += 1

        self.assertEqual(all_count, 3, "There should be exactly 3 memories in total.")

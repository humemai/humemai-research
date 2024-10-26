"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestMemory(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a fresh Memory instance before each test."""
        self.memory = Humemai()

    def test_add_single_memory_with_qualifiers(self) -> None:
        """Test adding a single triple with qualifiers and check if it's correctly stored."""
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )
        qualifiers = {
            humemai.currentTime: Literal("2024-04-27T10:00:00", datatype=XSD.dateTime),
            humemai.location: Literal("New York"),
        }

        # Add memory
        self.memory.add_memory([triple], qualifiers)

        # Verify memory has been added correctly
        result = self.memory.print_memories(True)
        expected = "(Alice, knows, Bob, {'memoryID': '0', 'currentTime': '2024-04-27T10:00:00', 'location': 'New York'})"

        self.assertIn(expected, result)

    def test_add_duplicate_memory_with_different_qualifiers(self) -> None:
        """Test adding the same triple multiple times with different qualifiers."""
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )

        qualifiers1 = {
            humemai.currentTime: Literal("2024-04-27T10:00:00", datatype=XSD.dateTime),
            humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            humemai.currentTime: Literal("2024-04-27T12:00:00", datatype=XSD.dateTime),
            humemai.location: Literal("London"),
        }

        # Add memory twice with different qualifiers
        self.memory.add_memory([triple], qualifiers1)
        self.memory.add_memory([triple], qualifiers2)

        # Verify that both reified statements are stored
        result = self.memory.print_memories(True)

        expected1 = "(Alice, knows, Bob, {'memoryID': '0', 'currentTime': '2024-04-27T10:00:00', 'location': 'New York'})"
        expected2 = "(Alice, knows, Bob, {'memoryID': '1', 'currentTime': '2024-04-27T12:00:00', 'location': 'London'})"

        # Ensure both entries are present
        self.assertIn(expected1, result)
        self.assertIn(expected2, result)

    def test_main_triple_not_duplicated(self) -> None:
        """Test that adding the same main triple multiple times doesn't duplicate the main triple."""
        triple = (
            URIRef("https://example.org/person/Alice"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/Bob"),
        )

        qualifiers1 = {
            humemai.currentTime: Literal("2024-04-27T10:00:00", datatype=XSD.dateTime),
            humemai.location: Literal("New York"),
        }
        qualifiers2 = {
            humemai.currentTime: Literal("2024-04-27T12:00:00", datatype=XSD.dateTime),
            humemai.location: Literal("London"),
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
    def setUp(self) -> None:
        """Set up a fresh Memory instance before each test."""
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

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

        # Define qualifiers for the first triple with Literal values
        self.qualifiers1 = {
            self.humemai.currentTime: Literal(
                "2024-04-27T10:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("New York"),
        }

        # Define qualifiers for the second triple with Literal values
        self.qualifiers2 = {
            self.humemai.currentTime: Literal(
                "2024-04-27T11:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("Paris"),
        }

        # Add sample triples to memory
        self.memory.add_memory([self.triple1], self.qualifiers1)
        self.memory.add_memory([self.triple2], self.qualifiers2)

    def test_delete_existing_memory(self) -> None:
        """Test deleting a memory that exists in the graph."""
        before_delete = self.memory.print_memories(True)
        self.assertIn("Alice", before_delete)
        self.assertIn("location", before_delete)

        # Delete the triple (Alice knows Bob)
        self.memory.delete_triple(*self.triple1)

        after_delete = self.memory.print_memories(True)
        self.assertNotIn("Alice", after_delete)
        self.assertNotIn("knows", after_delete)
        self.assertNotIn("New York", after_delete)

        # Ensure that the other memory still exists (Bob likes Chocolate)
        self.assertIn("Bob", after_delete)
        self.assertIn("location", after_delete)

    def test_delete_triple_with_multiple_qualifiers(self) -> None:
        """Test deleting a memory with multiple qualifiers."""
        qualifiers_additional = {
            self.humemai.currentTime: Literal(
                "2024-04-27T14:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("London"),
        }
        self.memory.add_memory([self.triple1], qualifiers_additional)

        before_delete = self.memory.print_memories(True)
        self.assertIn("New York", before_delete)
        self.assertIn("London", before_delete)

        # Delete the triple (Alice knows Bob)
        self.memory.delete_triple(*self.triple1)

        after_delete = self.memory.print_memories(True)
        self.assertNotIn("New York", after_delete)
        self.assertNotIn("London", after_delete)
        self.assertNotIn("Alice", after_delete)

    def test_delete_non_existent_memory(self) -> None:
        """Test deleting a memory that does not exist in the graph."""
        non_existent_triple = (
            URIRef("https://example.org/person/David"),
            URIRef("https://example.org/relationship/hates"),
            Literal("Broccoli"),
        )

        before_delete = self.memory.print_memories(True)
        self.assertNotIn("David", before_delete)

        # Attempt to delete the non-existent triple
        self.memory.delete_triple(*non_existent_triple)

        # Verify that the graph is unchanged
        after_delete = self.memory.print_memories(True)
        self.assertEqual(before_delete, after_delete)

    def test_delete_triple_with_no_qualifiers(self) -> None:
        """Test deleting a memory that has no qualifiers (just a main triple)."""
        simple_triple = (
            URIRef("https://example.org/person/Charlie"),
            URIRef("https://example.org/relationship/knows"),
            URIRef("https://example.org/person/David"),
        )
        self.memory.add_memory([simple_triple], {})

        before_delete = self.memory.print_memories(True)
        self.assertIn("Charlie", before_delete)

        self.memory.delete_triple(*simple_triple)

        after_delete = self.memory.print_memories(True)
        self.assertNotIn("Charlie", after_delete)


class TestMemoryDelete(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the memory and add test data before each test."""
        self.memory = Humemai()

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

        # Define episodic and semantic qualifiers with URIRef keys
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
        self.episodic_qualifiers_3 = {
            URIRef("https://humem.ai/ontology#location"): Literal("Paris"),
            URIRef("https://humem.ai/ontology#eventTime"): Literal(
                "2024-05-05T18:00:00", datatype=XSD.dateTime
            ),
            URIRef("https://humem.ai/ontology#emotion"): Literal("curious"),
            URIRef("https://humem.ai/ontology#event"): Literal("Workshop"),
        }

        self.semantic_qualifiers_1 = {
            URIRef("https://humem.ai/ontology#derivedFrom"): Literal("animal_research"),
            URIRef("https://humem.ai/ontology#strength"): Literal(
                5, datatype=XSD.integer
            ),
        }
        self.semantic_qualifiers_2 = {
            URIRef("https://humem.ai/ontology#derivedFrom"): Literal("pet_database"),
            URIRef("https://humem.ai/ontology#strength"): Literal(
                10, datatype=XSD.integer
            ),
        }

        # Add episodic memories using add_memory (no add_long_term_memory)
        self.memory.add_memory(
            [self.triples[0]], self.episodic_qualifiers_1
        )  # Memory ID 0
        self.memory.add_memory(
            [self.triples[0]], self.episodic_qualifiers_2
        )  # Memory ID 1
        self.memory.add_memory(
            [self.triples[1]], self.episodic_qualifiers_1
        )  # Memory ID 2
        self.memory.add_memory(
            [self.triples[2]], self.episodic_qualifiers_3
        )  # Memory ID 3

        # Add semantic memories using add_memory (no add_long_term_memory)
        self.memory.add_memory(
            [self.triples[3]], self.semantic_qualifiers_1
        )  # Memory ID 4
        self.memory.add_memory(
            [self.triples[3]], self.semantic_qualifiers_2
        )  # Memory ID 5

        # Add a short-term memory
        self.memory.add_short_term_memory(
            [self.triples[4]],
            {URIRef("https://humem.ai/ontology#location"): Literal("Alice's home")},
        )  # Memory ID 6

    def test_memory_retrieval_by_id(self) -> None:
        """Test retrieving memories by ID."""
        memory_0 = self.memory.get_memory_by_id(Literal(0, datatype=XSD.integer))
        self.assertIn(
            "New York",
            memory_0["qualifiers"][URIRef("https://humem.ai/ontology#location")],
        )  # Check in the qualifiers

        memory_1 = self.memory.get_memory_by_id(Literal(1, datatype=XSD.integer))
        self.assertIn(
            "London",
            memory_1["qualifiers"][URIRef("https://humem.ai/ontology#location")],
        )  # Check in the qualifiers

        memory_4 = self.memory.get_memory_by_id(Literal(4, datatype=XSD.integer))
        self.assertIn(
            "animal_research",
            memory_4["qualifiers"][URIRef("https://humem.ai/ontology#derivedFrom")],
        )  # Check semantic memory

        memory_6 = self.memory.get_memory_by_id(Literal(6, datatype=XSD.integer))
        self.assertIn(
            "Alice's home",
            memory_6["qualifiers"][URIRef("https://humem.ai/ontology#location")],
        )  # Check short-term memory

    def test_memory_deletion_by_id(self) -> None:
        """Test deleting a memory by ID."""
        self.memory.delete_memory(
            Literal(1, datatype=XSD.integer)
        )  # Delete memory ID 1
        deleted_memory = self.memory.get_memory_by_id(Literal(1, datatype=XSD.integer))
        self.assertIsNone(deleted_memory)  # Ensure memory ID 1 is deleted

        # Ensure other memories are still present
        memory_0 = self.memory.get_memory_by_id(Literal(0, datatype=XSD.integer))
        self.assertIsNotNone(memory_0)

    def test_triple_deletion(self) -> None:
        """Test deleting a triple and all associated memories."""
        self.memory.delete_triple(*self.triples[0])

        # Both Memory ID 0 and 1 refer to the triple (Alice, met, Bob), so they should be deleted
        self.assertIsNone(
            self.memory.get_memory_by_id(Literal(0, datatype=XSD.integer))
        )
        self.assertIsNone(
            self.memory.get_memory_by_id(Literal(1, datatype=XSD.integer))
        )

        # Ensure other memories are not affected
        memory_2 = self.memory.get_memory_by_id(Literal(2, datatype=XSD.integer))
        memory_3 = self.memory.get_memory_by_id(Literal(3, datatype=XSD.integer))
        self.assertIsNotNone(memory_2)
        self.assertIsNotNone(memory_3)

    def test_count_triples_and_memories(self) -> None:
        """Test counting triples and reified memories."""
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(), 5
        )  # 5 unique triples
        self.assertEqual(self.memory.get_memory_count(), 7)  # 7 reified memories

    def test_delete_triple_and_memory_count(self) -> None:
        """Test memory and triple count after deleting a triple."""
        # Delete triple (Alice, met, Bob) and ensure memory count is updated
        self.memory.delete_triple(*self.triples[0])
        self.assertEqual(
            self.memory.get_main_triple_count_except_event(), 4
        )  # 1 triple removed
        self.assertEqual(self.memory.get_memory_count(), 5)  # 2 memories removed


class TestMemoryDeleteWithTime(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_delete_triple_with_time_filter(self) -> None:
        """
        Test deleting a memory that has a time qualifier.
        """
        triple = (
            URIRef("https://example.org/person/Luke"),
            URIRef("https://example.org/relationship/met"),
            URIRef("https://example.org/person/Mary"),
        )
        qualifiers = {
            self.humemai.currentTime: Literal(
                "2024-04-27T12:00:00", datatype=XSD.dateTime
            ),
            self.humemai.location: Literal("Paris"),
        }

        # Add the memory
        self.memory.add_memory([triple], qualifiers)

        # Verify the memory is there before deletion
        result_before = self.memory.print_memories(True)
        self.assertIn("Paris", result_before)
        self.assertIn("Luke", result_before)

        # Delete the memory
        self.memory.delete_triple(*triple)

        # Verify the memory is no longer there after deletion
        result_after = self.memory.print_memories(True)
        self.assertNotIn("Paris", result_after)
        self.assertNotIn("Luke", result_after)


class TestInvalidInputHandling(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_invalid_time_format(self) -> None:
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


class TestMemoryInvalidQualifiers(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance before each test.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

    def test_missing_qualifiers_in_long_term_memory_episodic(self) -> None:
        """
        Test that missing required qualifiers in episodic long-term memory raise a ValueError.
        """
        triple = (
            URIRef("https://example.org/person/Jack"),
            URIRef("https://example.org/event/met"),
            URIRef("https://example.org/person/Kate"),
        )

        # Missing 'eventTime' which is essential for episodic memory
        qualifiers = {
            self.humemai.location: Literal("Paris"),
            self.humemai.emotion: Literal("happy"),
        }

        # Adding an episodic memory without the required 'eventTime' qualifier should raise a ValueError
        with self.assertRaises(ValueError) as context:
            self.memory.add_episodic_memory([triple], qualifiers=qualifiers)

        self.assertIn(
            "Missing required qualifier: eventTime",
            str(context.exception),
        )


class TestMemoryIteration(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the Memory object and add short-term, episodic, and semantic memories.
        """
        self.memory = Humemai()

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
            qualifiers={
                humemai.location: Literal("Paris"),
                humemai.currentTime: Literal(
                    "2023-05-05T10:00:00", datatype=XSD.dateTime
                ),
            },
        )

        # Add a long-term episodic memory
        triples_episodic = [
            (
                URIRef("https://example.org/Bob"),
                URIRef("https://example.org/visit"),
                URIRef("https://example.org/Paris"),
            )
        ]
        self.memory.add_episodic_memory(
            triples=triples_episodic,
            qualifiers={
                humemai.location: Literal("Paris"),
                humemai.eventTime: Literal(
                    "2023-05-06T10:00:00", datatype=XSD.dateTime
                ),
                humemai.emotion: Literal("happy"),
            },
        )

        # Add a long-term semantic memory with required qualifiers
        triples_semantic = [
            (
                URIRef("https://example.org/Charlie"),
                URIRef("https://example.org/hasType"),
                URIRef("https://example.org/Human"),
            )
        ]
        self.memory.add_semantic_memory(
            triples=triples_semantic,
            qualifiers={
                humemai.knownSince: Literal(
                    "2023-01-01T00:00:00", datatype=XSD.dateTime
                ),
                humemai.derivedFrom: Literal("research_paper_1"),
                humemai.strength: Literal(5, datatype=XSD.integer),
            },
        )

    def test_iterate_short_term_memories(self) -> None:
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
            self.assertIn(humemai.currentTime, qualifiers)

        self.assertEqual(
            short_term_count, 1, "There should be exactly 1 short-term memory."
        )

    def test_iterate_long_term_memories(self) -> None:
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

    def test_iterate_episodic_memories(self) -> None:
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
            self.assertIn(humemai.eventTime, qualifiers)

        self.assertEqual(
            episodic_count, 1, "There should be exactly 1 episodic memory."
        )

    def test_iterate_semantic_memories(self) -> None:
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
            self.assertIn(humemai.derivedFrom, qualifiers)

        self.assertEqual(
            semantic_count, 1, "There should be exactly 1 semantic memory."
        )

    def test_iterate_all_memories(self) -> None:
        """
        Test iteration over all memories (short-term + long-term).
        """
        all_count = 0
        for subj, pred, obj, qualifiers in self.memory.iterate_memories(
            memory_type="all"
        ):
            all_count += 1

        self.assertEqual(all_count, 3, "There should be exactly 3 memories in total.")

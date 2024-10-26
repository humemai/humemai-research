"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestIncrementRecalled(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a fresh Memory instance with episodic and semantic memories.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

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
            self.humemai.location: Literal("New York"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
            self.humemai.event: Literal("Meeting for coffee"),
        }

        semantic_qualifiers = {
            self.humemai.knownSince: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.strength: Literal(5, datatype=XSD.integer),
            self.humemai.derivedFrom: Literal("study"),
        }

        # Add episodic and semantic memories
        self.memory.add_episodic_memory([self.triple1], qualifiers=episodic_qualifiers)
        self.memory.add_semantic_memory(
            [self.semantic_triple1], qualifiers=semantic_qualifiers
        )

    def test_multiple_increments_recalled(self) -> None:
        """
        Test incrementing the recalled value multiple times for both episodic and semantic memories.
        """
        # Increment recalled values once
        self.memory.increment_recalled()

        # Retrieve the updated memory system after first increment
        result = self.memory.print_memories(True)

        # Check that the 'recalled' value has been incremented to 1 for both episodic and semantic memories
        self.assertIn("'1'", result)

        # Increment recalled values again
        self.memory.increment_recalled()

        # Retrieve the updated memory system after second increment
        result = self.memory.print_memories(True)

        # Check that the 'recalled' value has been incremented to 2 for both episodic and semantic memories
        self.assertIn("'2'", result)

        # Increment recalled values a third time
        self.memory.increment_recalled()

        # Retrieve the updated memory system after third increment
        result = self.memory.print_memories(True)

        # Check that the 'recalled' value has been incremented to 3 for both episodic and semantic memories
        self.assertIn(
            "'3'",
            result,
        )


class TestRefiedMemory(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize the Memory instance
        self.memory = Humemai()

        # Mock the graph object to simulate RDF triples and statements
        self.memory.graph = MagicMock()

        # Example URIs for the test
        self.subj = URIRef("https://example.org/person/Alice")
        self.pred = URIRef("https://example.org/event/met")
        self.obj = URIRef("https://example.org/person/Bob")

        # Example reified statement as a blank node
        self.reified_statement = BNode()

        # Working memory instance for adding triples
        self.working_memory = Humemai()
        self.working_memory.graph = MagicMock()

    def test_add_reified_statement_and_increment_recall(self) -> None:
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
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(1, datatype=XSD.integer),
            )
        ]

        # Define the side effect function for predicate_objects to return updated 'recalled' value
        def predicate_objects_side_effect(statement):
            if statement == self.reified_statement:
                return [
                    (
                        URIRef("https://humem.ai/ontology#recalled"),
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
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(2, datatype=XSD.integer),
            )
        )

        # Ensure the reified statement was added to the working memory with the updated recall
        self.working_memory.graph.add.assert_any_call(
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(2, datatype=XSD.integer),
            )
        )

    def test_add_reified_statement_with_no_initial_recall(self) -> None:
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
                        URIRef("https://humem.ai/ontology#recalled"),
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
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

        # Ensure the reified statement with recall value 1 was added to the working memory
        self.working_memory.graph.add.assert_any_call(
            (
                self.reified_statement,
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

    def test_specific_statement_handling(self) -> None:
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
                        URIRef("https://humem.ai/ontology#recalled"),
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
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

        # Ensure only the specific statement was added to the working memory
        self.working_memory.graph.add.assert_called_once_with(
            (
                specific_statement,
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(1, datatype=XSD.integer),
            )
        )

    def test_no_reified_statements(self) -> None:
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

    def test_multiple_reified_statements(self) -> None:
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
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
            [
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
        ]

        # Define the side effect function for predicate_objects to return updated 'recalled' values
        def predicate_objects_side_effect(statement):
            if statement in [self.reified_statement, reified_statement_2]:
                return [
                    (
                        URIRef("https://humem.ai/ontology#recalled"),
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
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology#recalled"),
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
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
        ]
        self.working_memory.graph.add.assert_has_calls(
            expected_add_calls, any_order=True
        )

    def test_set_called_correctly_with_qualifiers(self) -> None:
        """Test that qualifiers are correctly added to the working memory."""

        # Mock the subjects method to return the reified statement
        self.memory.graph.subjects.return_value = [self.reified_statement]

        # Mock the value method to return the correct subject, predicate, and object
        self.memory.graph.value.side_effect = [self.subj, self.pred, self.obj]

        # Define the side effect function for triples to return the recalled value separately
        def triples_side_effect(query):
            if query == (
                self.reified_statement,
                URIRef("https://humem.ai/ontology#recalled"),
                None,
            ):
                return [
                    (
                        self.reified_statement,
                        URIRef("https://humem.ai/ontology#recalled"),
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
                        URIRef("https://humem.ai/ontology#recalled"),
                        Literal(2, datatype=XSD.integer),
                    ),
                    (
                        URIRef("https://humem.ai/ontology#location"),
                        Literal("New York", datatype=XSD.string),
                    ),
                    (
                        URIRef("https://humem.ai/ontology#time"),
                        Literal("2024-04-27T15:00:00", datatype=XSD.dateTime),
                    ),
                    (
                        URIRef("https://humem.ai/ontology#emotion"),
                        Literal("happy", datatype=XSD.string),
                    ),
                    (
                        URIRef("https://humem.ai/ontology#event"),
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
                URIRef("https://humem.ai/ontology#recalled"),
                Literal(2, datatype=XSD.integer),
            )
        )

        # Ensure all qualifiers are added to the working memory
        expected_add_calls = [
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology#location"),
                    Literal("New York", datatype=XSD.string),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology#time"),
                    Literal("2024-04-27T15:00:00", datatype=XSD.dateTime),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology#emotion"),
                    Literal("happy", datatype=XSD.string),
                )
            ),
            unittest.mock.call(
                (
                    self.reified_statement,
                    URIRef("https://humem.ai/ontology#event"),
                    Literal("Coffee meeting", datatype=XSD.string),
                )
            ),
        ]
        self.working_memory.graph.add.assert_has_calls(
            expected_add_calls, any_order=True
        )

    def test_add_reified_statement_and_increment_recall_no_specific_statement(
        self,
    ) -> None:
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
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
            [
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(1, datatype=XSD.integer),
                )
            ],
        ]

        # Define the side effect function for predicate_objects to return updated 'recalled' values
        def predicate_objects_side_effect(statement):
            if statement in [self.reified_statement, reified_statement_2]:
                return [
                    (
                        URIRef("https://humem.ai/ontology#recalled"),
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
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology#recalled"),
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
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
            unittest.mock.call(
                (
                    reified_statement_2,
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(2, datatype=XSD.integer),
                )
            ),
        ]
        self.working_memory.graph.add.assert_has_calls(
            expected_add_calls, any_order=True
        )


class TestMemorySaveLoad(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a Memory object and add some test data"""
        # Create an instance of the Memory class
        self.memory = Humemai()
        self.test_file = "test_memory.ttl"

        # Define the humemai and example namespaces
        humemai = Namespace("https://humem.ai/ontology#")
        ex = Namespace("https://example.org/")

        # Add multiple short-term memories
        self.memory.add_short_term_memory(
            [(ex.Alice, ex.met, ex.Bob)],
            {
                humemai.location: Literal("New York"),
                humemai.currentTime: Literal(
                    "2024-04-27T15:00:00", datatype=XSD.dateTime
                ),
            },
        )
        self.memory.add_short_term_memory(
            [(ex.Alice, ex.met, ex.Bob)],
            {
                humemai.location: Literal("New York"),
                humemai.currentTime: Literal(
                    "2024-04-27T16:00:00", datatype=XSD.dateTime
                ),
            },
        )

        self.memory.add_short_term_memory(
            [(ex.Bob, ex.knows, ex.Alice)],
            {
                humemai.location: Literal("Paris"),
                humemai.currentTime: Literal(
                    "2024-05-01T10:00:00", datatype=XSD.dateTime
                ),
            },
        )

        # Add multiple long-term episodic memories
        self.memory.add_memory(
            [(ex.Alice, ex.attended, ex.Conference)],
            {
                humemai.location: Literal("London"),
                humemai.eventTime: Literal(
                    "2023-09-15T10:00:00", datatype=XSD.dateTime
                ),
                humemai.emotion: Literal("excited"),
                humemai.event: Literal("AI Conference"),
            },
        )
        self.memory.add_memory(
            [(ex.Alice, ex.spokeWith, ex.Charlie)],
            {
                humemai.location: Literal("London"),
                humemai.eventTime: Literal(
                    "2023-09-15T11:00:00", datatype=XSD.dateTime
                ),
                humemai.emotion: Literal("happy"),
                humemai.event: Literal("AI Conference"),
            },
        )
        self.memory.add_memory(
            [(ex.Alice, ex.spokeWith, ex.Charlie)],
            {
                humemai.location: Literal("London"),
                humemai.eventTime: Literal(
                    "2023-09-15T11:00:00", datatype=XSD.dateTime
                ),
                humemai.emotion: Literal("sad"),
                humemai.event: Literal("AI Conference"),
            },
        )

        # Add multiple long-term semantic memories
        self.memory.add_memory(
            [(ex.Dog, ex.hasType, ex.Animal)],
            {
                humemai.derivedFrom: Literal("research_paper_1"),
                humemai.strength: Literal(5, datatype=XSD.integer),
                humemai.knownSince: Literal(
                    "2023-09-15T10:00:00", datatype=XSD.dateTime
                ),
            },
        )
        self.memory.add_memory(
            [(ex.Cat, ex.hasType, ex.Animal)],
            {
                humemai.derivedFrom: Literal("research_paper_2"),
                humemai.strength: Literal(4, datatype=XSD.integer),
                humemai.knownSince: Literal(
                    "2023-09-15T10:00:00", datatype=XSD.dateTime
                ),
            },
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

    def test_save_and_load_ttl(self) -> None:
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

    def tearDown(self) -> None:
        """Clean up after tests"""
        # Remove the test file if it exists
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

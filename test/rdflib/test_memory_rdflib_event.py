"""Test Memory class"""

import unittest
import os
from datetime import datetime
from unittest.mock import MagicMock
from rdflib import RDF, XSD, BNode, Graph, Literal, Namespace, URIRef
from humemai.rdflib import Humemai

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class TestModifyEpisodicEvent(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up the Memory object with some episodic, semantic, and short-term memories.
        """
        self.memory = Humemai()
        self.humemai = Namespace("https://humem.ai/ontology#")

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

        # Define episodic memories with qualifiers
        episodic_qualifiers1 = {
            self.humemai.location: Literal("New York"),
            self.humemai.eventTime: Literal(
                "2024-04-27T15:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("happy"),
            self.humemai.event: Literal("Meeting for coffee"),
        }

        episodic_qualifiers2 = {
            self.humemai.location: Literal("London"),
            self.humemai.eventTime: Literal(
                "2024-04-27T17:00:00", datatype=XSD.dateTime
            ),
            self.humemai.emotion: Literal("excited"),
            self.humemai.event: Literal("Dinner event"),
        }

        # Add episodic long-term memories
        self.memory.add_episodic_memory([self.triple1], qualifiers=episodic_qualifiers1)
        self.memory.add_episodic_memory([self.triple2], qualifiers=episodic_qualifiers2)

        # Add short-term memories (these should not be affected by modify_episodic_event)
        short_term_qualifiers1 = {
            self.humemai.location: Literal("Berlin"),
            self.humemai.currentTime: Literal(
                "2024-04-27T18:00:00", datatype=XSD.dateTime
            ),
        }
        short_term_qualifiers2 = {
            self.humemai.location: Literal("Tokyo"),
            self.humemai.currentTime: Literal(
                "2024-04-27T19:00:00", datatype=XSD.dateTime
            ),
        }
        self.memory.add_short_term_memory([self.triple1], short_term_qualifiers1)
        self.memory.add_short_term_memory([self.triple2], short_term_qualifiers2)

    def test_modify_event_for_episodic_memories(self) -> None:
        """
        Test that only episodic memories within the time range have their event modified.
        """
        lower_time_bound = Literal("2024-04-27T00:00:00", datatype=XSD.dateTime)
        upper_time_bound = Literal("2024-04-28T00:00:00", datatype=XSD.dateTime)
        new_event = Literal("Updated Event for Conference")

        # Modify the event for episodic memories within the time range
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve all memories to check
        result = self.memory.print_memories(True)

        # Check that episodic memories are updated
        self.assertIn("Updated Event for Conference", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertNotIn("Dinner event", result)

        # Check that short-term memories are NOT affected
        self.assertIn("Berlin", result)
        self.assertIn("Tokyo", result)

    def test_no_effect_on_memories_outside_time_range(self) -> None:
        """
        Test that memories outside the specified time range are not modified.
        """
        lower_time_bound = Literal("2024-04-26T00:00:00", datatype=XSD.dateTime)
        upper_time_bound = Literal("2024-04-27T12:00:00", datatype=XSD.dateTime)
        new_event = Literal("Updated Event")

        # Modify the event for episodic memories (which shouldn't modify anything)
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve all memories to check
        result = self.memory.print_memories(True)

        # Ensure that no events have been modified
        self.assertIn("Meeting for coffee", result)
        self.assertIn("Dinner event", result)
        self.assertNotIn("Updated Event", result)

    def test_only_episodic_memories_are_modified(self) -> None:
        """
        Test that only episodic long-term memories are modified, and short-term memories are unaffected.
        """
        lower_time_bound = Literal("2024-04-27T00:00:00", datatype=XSD.dateTime)
        upper_time_bound = Literal("2024-04-28T00:00:00", datatype=XSD.dateTime)
        new_event = Literal("Conference Event Update")

        # Modify episodic memories
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve memories and check that only episodic ones are updated
        result = self.memory.print_memories(True)

        # Check that episodic events are updated
        self.assertIn("Conference Event Update", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertNotIn("Dinner event", result)

        # Short-term memories should remain unchanged
        self.assertIn("Berlin", result)
        self.assertIn("Tokyo", result)

    def test_modify_multiple_episodic_memories(self) -> None:
        """
        Test that multiple episodic memories within the time range are modified.
        """
        lower_time_bound = Literal("2024-04-27T00:00:00", datatype=XSD.dateTime)
        upper_time_bound = Literal("2024-04-28T00:00:00", datatype=XSD.dateTime)
        new_event = Literal("Global Event Update")

        # Modify the event for episodic memories
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
        )

        # Retrieve memories and check that both episodic events are updated
        result = self.memory.print_memories(True)

        # Check that both episodic memories are updated
        self.assertIn("Global Event Update", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertNotIn("Dinner event", result)

    def test_event_modification_with_additional_filters(self) -> None:
        """
        Test that episodic memories can be modified with additional filters (e.g., subject).
        """
        lower_time_bound = Literal("2024-04-27T00:00:00", datatype=XSD.dateTime)
        upper_time_bound = Literal("2024-04-28T00:00:00", datatype=XSD.dateTime)
        new_event = Literal("Filtered Event Update")

        # Modify the event for episodic memories with a subject filter (only Alice's memory)
        self.memory.modify_episodic_event(
            lower_time_bound=lower_time_bound,
            upper_time_bound=upper_time_bound,
            new_event=new_event,
            subject=URIRef("https://example.org/person/Alice"),
        )

        # Retrieve memories and check that only Alice's event was updated
        result = self.memory.print_memories(True)

        # Check that only Alice's episodic memory is updated
        self.assertIn("Filtered Event Update", result)
        self.assertNotIn("Meeting for coffee", result)
        self.assertIn(
            "Dinner event", result
        )  # Charlie's memory should remain unchanged


class TestEvent(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the Memory instance for testing.
        """
        self.memory = Humemai()

    def test_add_episodic_memory(self) -> None:
        """
        Test adding an episodic memory with an event, qualifiers, and event properties.
        """
        triples = [
            (
                URIRef("https://example.org/Alice"),
                URIRef("https://example.org/met"),
                URIRef("https://example.org/Bob"),
            )
        ]
        event_time = "2023-10-01T10:00:00"
        event = URIRef("https://humem.ai/ontology#event/AI_Conference")  # Event URI
        qualifiers = {
            humemai.location: Literal("Paris"),
            humemai.emotion: Literal("excited"),
            humemai.event: event,
        }
        event_properties = {
            URIRef("https://humem.ai/ontology#location"): Literal("Paris"),
            URIRef("https://humem.ai/ontology#duration"): Literal("3 hours"),
        }

        # Call add_episodic_memory method with humemai.event as key
        self.memory.add_episodic_memory(
            triples=triples,
            qualifiers={
                humemai.eventTime: Literal(event_time, datatype=XSD.dateTime),
                **qualifiers,
            },
            event_properties=event_properties,
        )

        # Verify the triple is added
        results = list(
            self.memory.graph.triples(
                (
                    URIRef("https://example.org/Alice"),
                    URIRef("https://example.org/met"),
                    URIRef("https://example.org/Bob"),
                )
            )
        )
        self.assertEqual(len(results), 1, "Triple should be added")

        # Verify the reified statement with qualifiers
        statements = list(self.memory.graph.subjects(RDF.type, RDF.Statement))
        self.assertEqual(len(statements), 1, "There should be one reified statement")
        statement = statements[0]

        # Check eventTime qualifier
        event_time_literal = self.memory.graph.value(statement, humemai.eventTime)
        self.assertIsNotNone(
            event_time_literal, "Reified statement should have eventTime"
        )
        self.assertEqual(
            str(event_time_literal),
            event_time,
            "eventTime should match the provided time",
        )

        # Check location qualifier
        location_literal = self.memory.graph.value(statement, humemai.location)
        self.assertIsNotNone(location_literal, "Reified statement should have location")
        self.assertEqual(
            str(location_literal), "Paris", "Location qualifier should be 'Paris'"
        )

        # Check emotion qualifier
        emotion_literal = self.memory.graph.value(statement, humemai.emotion)
        self.assertIsNotNone(emotion_literal, "Reified statement should have emotion")
        self.assertEqual(
            str(emotion_literal), "excited", "Emotion qualifier should be 'excited'"
        )

        # Check event qualifier
        event_literal = self.memory.graph.value(statement, humemai.event)
        self.assertIsNotNone(event_literal, "Reified statement should have event")
        self.assertEqual(
            str(event_literal),
            str(event),
            "Event qualifier should match the event name",
        )

        # Verify the event node exists with correct properties
        self.assertTrue(
            (event, RDF.type, humemai.Event) in self.memory.graph,
            "Event node should be created",
        )

        # Verify event properties
        for prop, value in event_properties.items():
            self.assertTrue(
                (event, prop, value) in self.memory.graph,
                f"Event property '{prop}' should be '{value}'",
            )

    def test_add_event(self) -> None:
        """
        Test creating an event node in the graph.
        """
        event = URIRef("https://humem.ai/ontology#event/AI_Conference")

        # Call add_event
        self.memory.add_event(event)

        # Verify the event node exists
        self.assertTrue(
            (event, RDF.type, humemai.Event) in self.memory.graph,
            "Event node should be created",
        )

    def test_add_event_properties(self) -> None:
        """
        Test adding custom properties to an event node.
        """
        event = URIRef("https://humem.ai/ontology#event/AI_Conference")
        event_properties = {
            URIRef("https://humem.ai/ontology#location"): Literal("Paris"),
            URIRef("https://humem.ai/ontology#duration"): Literal("3 hours"),
        }

        # Create the event node first
        self.memory.add_event(event)

        # Add event properties
        self.memory.add_event_properties(event, event_properties)

        # Verify the properties are added to the event node
        for prop, value in event_properties.items():
            self.assertTrue(
                (event, prop, value) in self.memory.graph,
                f"Event property '{prop}' should be '{value}'",
            )


class TestEventCount(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up a memory instance before each test.
        """
        self.memory = Humemai()

    def test_get_event_count(self) -> None:
        """
        Test the get_event_count method by adding event nodes and verifying the count.
        """
        # Add some event nodes to the graph
        event_1 = URIRef("https://humem.ai/ontology#event/AI_Conference")
        event_2 = URIRef("https://humem.ai/ontology#event/Workshop_2023")
        event_3 = URIRef("https://humem.ai/ontology#event/Hackathon")

        # Add the events as instances of humemai:Event
        self.memory.graph.add((event_1, RDF.type, humemai.Event))
        self.memory.graph.add((event_2, RDF.type, humemai.Event))
        self.memory.graph.add((event_3, RDF.type, humemai.Event))

        # Check the count of events
        count = self.memory.get_event_count()
        self.assertEqual(count, 3, "The count of events should be 3")

    def test_no_events(self) -> None:
        """
        Test the get_event_count method when no events are present.
        """
        # Initially, there should be no events
        count = self.memory.get_event_count()
        self.assertEqual(
            count, 0, "The count of events should be 0 when no events are present"
        )

    def test_one_event(self) -> None:
        """
        Test the get_event_count method when there is only one event.
        """
        # Add one event node to the graph
        event = URIRef("https://humem.ai/ontology#event/Seminar")
        self.memory.graph.add((event, RDF.type, humemai.Event))

        # Check the count of events
        count = self.memory.get_event_count()
        self.assertEqual(count, 1, "The count of events should be 1")


class TestEventMethods(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up a memory instance before each test.
        """
        self.memory = Humemai()

    def test_iterate_events(self) -> None:
        """
        Test the iterate_events method by adding event nodes and verifying the iteration.
        """
        # Add some event nodes to the graph
        event_1 = URIRef("https://humem.ai/ontology#event/AI_Conference")
        event_2 = URIRef("https://humem.ai/ontology#event/Workshop_2023")
        event_3 = URIRef("https://humem.ai/ontology#event/Hackathon")

        # Add the events as instances of humemai:Event
        self.memory.graph.add((event_1, RDF.type, humemai.Event))
        self.memory.graph.add((event_2, RDF.type, humemai.Event))
        self.memory.graph.add((event_3, RDF.type, humemai.Event))

        # Collect the events from iterate_events
        events = list(self.memory.iterate_events())
        self.assertEqual(len(events), 3, "There should be 3 events")
        self.assertIn(event_1, events, "AI_Conference event should be in the list")
        self.assertIn(event_2, events, "Workshop_2023 event should be in the list")
        self.assertIn(event_3, events, "Hackathon event should be in the list")

    def test_iterate_no_events(self) -> None:
        """
        Test the iterate_events method when there are no events.
        """
        # No events added, so iterate_events should return an empty list
        events = list(self.memory.iterate_events())
        self.assertEqual(len(events), 0, "There should be no events")

    def test_iterate_one_event(self) -> None:
        """
        Test the iterate_events method when there is only one event.
        """
        # Add one event node to the graph
        event = URIRef("https://humem.ai/ontology#event/Seminar")
        self.memory.graph.add((event, RDF.type, humemai.Event))

        # Collect the events from iterate_events
        events = list(self.memory.iterate_events())
        self.assertEqual(len(events), 1, "There should be 1 event")
        self.assertIn(event, events, "Seminar event should be in the list")


class TestMemoryEvents(unittest.TestCase):

    def setUp(self) -> None:
        """Set up the memory system before each test."""
        self.memory = Humemai()

        # Example URIs and Literals for subjects, predicates, and objects
        self.alice = URIRef("https://example.org/Alice")
        self.bob = URIRef("https://example.org/Bob")
        self.attended = URIRef("https://example.org/attended")
        self.location = URIRef("https://example.org/location")
        self.date = URIRef("https://example.org/date")
        self.event = URIRef("https://example.org/AI_Conference")
        self.paris = Literal("Paris")
        self.date_literal = Literal("2023-05-05", datatype=XSD.date)

    def convert_to_str(self, triples) -> None:
        """Helper method to convert RDFLib objects to string form."""
        return {(str(subj), str(pred), str(obj)) for subj, pred, obj in triples}

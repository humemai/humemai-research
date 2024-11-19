import unittest
from datetime import datetime, timedelta
from humemai.memory import (
    Memory,
    ShortMemory,
    LongMemory,
    EpisodicMemory,
    SemanticMemory,
)


class TestMemory(unittest.TestCase):
    def test_memory_initialization(self):
        """Test that a Memory instance is initialized correctly with given properties."""
        memory = Memory(
            head_label="Person",
            tail_label="Person",
            edge_label="knows",
            head_properties={"name": "Alice", "age": 30},
            tail_properties={"name": "Bob", "age": 35},
            edge_properties={"since": 2020},
        )

        self.assertEqual(memory.head_label, "Person")
        self.assertEqual(memory.tail_label, "Person")
        self.assertEqual(memory.edge_label, "knows")
        self.assertEqual(memory.head_properties, {"name": "Alice", "age": 30})
        self.assertEqual(memory.tail_properties, {"name": "Bob", "age": 35})
        self.assertEqual(memory.edge_properties, {"since": 2020})

    def test_memory_to_dict(self):
        """Test that to_dict returns the correct dictionary representation of a Memory instance."""
        memory = Memory(
            head_label="Person",
            tail_label="Person",
            edge_label="knows",
            head_properties={"name": "Alice"},
            tail_properties={"name": "Bob"},
            edge_properties={"since": 2020},
        )

        expected_dict = {
            "head": {"label": "Person", "properties": {"name": "Alice"}},
            "tail": {"label": "Person", "properties": {"name": "Bob"}},
            "edge": {"label": "knows", "properties": {"since": 2020}},
        }

        self.assertEqual(memory.to_dict(), expected_dict)


class TestShortMemory(unittest.TestCase):
    def test_short_memory_initialization_with_current_time(self):
        """Test that a ShortMemory instance initializes with provided current_time in edge_properties."""
        current_time = datetime.now().isoformat(timespec="seconds")
        edge_properties = {"current_time": current_time, "strength": "strong"}

        short_memory = ShortMemory(
            head_label="Person",
            tail_label="Person",
            edge_label="knows",
            head_properties={"name": "Alice"},
            tail_properties={"name": "Bob"},
            edge_properties=edge_properties,
        )

        self.assertEqual(short_memory.edge_properties["current_time"], current_time)
        self.assertEqual(short_memory.edge_properties["strength"], "strong")

    def test_short_memory_initialization_without_current_time(self):
        """Test that a ShortMemory instance sets current_time in edge_properties if not provided."""
        short_memory = ShortMemory(
            head_label="Person",
            tail_label="Person",
            edge_label="knows",
            head_properties={"name": "Alice"},
            tail_properties={"name": "Bob"},
        )

        # Check that 'current_time' is in edge_properties and is in ISO format
        self.assertIn("current_time", short_memory.edge_properties)
        current_time = short_memory.edge_properties["current_time"]
        datetime.fromisoformat(current_time)  # Ensures it parses as ISO 8601

    def test_short_memory_to_dict(self):
        """Test the to_dict method of ShortMemory."""
        current_time = datetime.now().isoformat(timespec="seconds")
        short_memory = ShortMemory(
            head_label="Person",
            tail_label="Person",
            edge_label="knows",
            head_properties={"name": "Alice"},
            tail_properties={"name": "Bob"},
            edge_properties={"current_time": current_time},
        )

        expected_dict = {
            "head": {"label": "Person", "properties": {"name": "Alice"}},
            "tail": {"label": "Person", "properties": {"name": "Bob"}},
            "edge": {
                "label": "knows",
                "properties": {"current_time": current_time},
            },
        }

        self.assertEqual(short_memory.to_dict(), expected_dict)

    def test_short_memory_invalid_current_time(self):
        """Test that an invalid current_time raises a ValueError."""
        with self.assertRaises(ValueError):
            ShortMemory(
                head_label="Person",
                tail_label="Person",
                edge_label="knows",
                edge_properties={
                    "current_time": 12345
                },  # Invalid format, should be ISO string
            )


class TestLongMemory(unittest.TestCase):
    def test_long_memory_initialization(self):
        """Test that a LongMemory instance initializes with the `recalled` property set to 0."""
        long_memory = LongMemory(
            head_label="Person",
            tail_label="Event",
            edge_label="remembers",
            head_properties={"name": "Alice"},
            tail_properties={"event_name": "Conference"},
            edge_properties={"importance": "high"},
        )

        # Check all labels and properties
        self.assertEqual(long_memory.head_label, "Person")
        self.assertEqual(long_memory.tail_label, "Event")
        self.assertEqual(long_memory.edge_label, "remembers")
        self.assertEqual(
            long_memory.head_properties, {"name": "Alice", "num_recalled": 0}
        )
        self.assertEqual(
            long_memory.tail_properties, {"event_name": "Conference", "num_recalled": 0}
        )

        # Ensure `recalled` is in edge_properties and set to 0
        self.assertIn("num_recalled", long_memory.edge_properties)
        self.assertEqual(long_memory.edge_properties["num_recalled"], 0)
        # Also check the other edge properties are intact
        self.assertEqual(long_memory.edge_properties["importance"], "high")

    def test_long_memory_to_dict(self):
        """Test the to_dict method of LongMemory to ensure it includes the `recalled` property."""
        long_memory = LongMemory(
            head_label="Person",
            tail_label="Event",
            edge_label="remembers",
            head_properties={"name": "Alice"},
            tail_properties={"event_name": "Workshop"},
        )

        expected_dict = {
            "head": {
                "label": "Person",
                "properties": {"name": "Alice", "num_recalled": 0},
            },
            "tail": {
                "label": "Event",
                "properties": {"event_name": "Workshop", "num_recalled": 0},
            },
            "edge": {"label": "remembers", "properties": {"num_recalled": 0}},
        }

        self.assertEqual(long_memory.to_dict(), expected_dict)


class TestEpisodicMemory(unittest.TestCase):
    def test_episodic_memory_initialization(self):
        """Test that EpisodicMemory initializes with the required `event_time` and `recalled` fields."""
        event_time = datetime.now().isoformat(timespec="seconds")
        episodic_memory = EpisodicMemory(
            head_label="Person",
            tail_label="Event",
            edge_label="experienced",
            head_properties={"name": "Alice"},
            tail_properties={"event_name": "Concert"},
            edge_properties={"event_time": [event_time]},
        )

        # Ensure that `event_time` and `recalled` properties are set correctly
        self.assertEqual(episodic_memory.edge_properties["event_time"], [event_time])
        self.assertEqual(episodic_memory.edge_properties["num_recalled"], 0)

    def test_episodic_memory_missing_event_time(self):
        """Test that EpisodicMemory raises an AssertionError if `event_time` is missing."""
        with self.assertRaises(AssertionError) as context:
            EpisodicMemory(
                head_label="Person",
                tail_label="Event",
                edge_label="experienced",
                edge_properties={"description": "Alice attended the concert"},
            )
        self.assertEqual(
            str(context.exception), "Edge property 'event_time' is required"
        )

    def test_episodic_memory_invalid_event_time(self):
        """Test that EpisodicMemory raises a ValueError if `event_time` is not an ISO 8601 string."""
        with self.assertRaises(ValueError) as context:
            EpisodicMemory(
                head_label="Person",
                tail_label="Event",
                edge_label="experienced",
                edge_properties={"event_time": 12345},
            )


class TestSemanticMemory(unittest.TestCase):
    def test_semantic_memory_initialization(self):
        """Test that SemanticMemory initializes with the required `known_since`, `derived_from`, and `recalled` fields."""
        known_since = datetime.now().isoformat(timespec="seconds")
        semantic_memory = SemanticMemory(
            head_label="Person",
            tail_label="Knowledge",
            edge_label="knows",
            head_properties={"name": "Alice"},
            tail_properties={"fact": "Python programming"},
            edge_properties={"known_since": known_since, "derived_from": "CS course"},
        )

        # Ensure that `known_since`, `derived_from`, and `recalled` properties are set correctly
        self.assertEqual(semantic_memory.edge_properties["known_since"], known_since)
        self.assertEqual(semantic_memory.edge_properties["derived_from"], "CS course")
        self.assertEqual(semantic_memory.edge_properties["num_recalled"], 0)

    def test_semantic_memory_missing_known_since(self):
        """Test that SemanticMemory raises an AssertionError if `known_since` is missing."""
        with self.assertRaises(AssertionError) as context:
            SemanticMemory(
                head_label="Person",
                tail_label="Knowledge",
                edge_label="knows",
                edge_properties={"derived_from": "Experience"},
            )
        self.assertEqual(
            str(context.exception), "Edge property 'known_since' is required"
        )

    def test_semantic_memory_missing_derived_from(self):
        """Test that SemanticMemory raises an AssertionError if `derived_from` is missing."""
        known_since = datetime.now().isoformat(timespec="seconds")
        with self.assertRaises(AssertionError) as context:
            SemanticMemory(
                head_label="Person",
                tail_label="Knowledge",
                edge_label="knows",
                edge_properties={"known_since": known_since},
            )
        self.assertEqual(
            str(context.exception), "Edge property 'derived_from' is required"
        )

    def test_semantic_memory_invalid_known_since(self):
        """Test that SemanticMemory raises a ValueError if `known_since` is not an ISO 8601 string."""
        with self.assertRaises(ValueError) as context:
            SemanticMemory(
                head_label="Person",
                tail_label="Knowledge",
                edge_label="knows",
                edge_properties={"known_since": 12345, "derived_from": "Book"},
            )
        self.assertEqual(
            str(context.exception),
            "The 'known_since' in edge_properties must be an ISO 8601 string.",
        )

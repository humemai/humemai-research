import random
import unittest
from copy import deepcopy

from humemai_research.memory import LongMemory, Memory, MemorySystems, ShortMemory


class TestMemory(unittest.TestCase):

    def setUp(self):
        self.memory = Memory(capacity=5)

    def test_initialization(self):
        self.assertEqual(len(self.memory), 0)
        self.assertEqual(self.memory.capacity, 5)
        self.assertTrue(self.memory.is_empty)
        self.assertFalse(self.memory.is_full)
        self.assertFalse(self.memory.is_frozen)

    def test_add_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertEqual(len(self.memory), 1)
        self.assertTrue(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )

    def test_capacity(self):
        self.assertEqual(self.memory.size, 0)
        self.assertFalse(self.memory.is_full)
        self.memory.increase_capacity(10)
        self.assertEqual(self.memory.capacity, 15)
        self.memory.decrease_capacity(5)
        self.assertEqual(self.memory.capacity, 10)

    def test_freeze_memory(self):
        self.assertFalse(self.memory.is_frozen)
        self.memory.freeze()
        self.assertTrue(self.memory.is_frozen)
        with self.assertRaises(ValueError):
            self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.memory.unfreeze()
        self.assertFalse(self.memory.is_frozen)
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertTrue(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )

    def test_forget_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertTrue(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )
        self.memory.forget(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertFalse(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )

    def test_query_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.memory.add(["Alice", "loves", "Charlie", {"type": "episodic"}])
        result = self.memory.query(["Alice", "?", "?", {"type": "episodic"}])
        self.assertEqual(len(result), 2)
        self.assertTrue(
            ["Alice", "likes", "Bob", {"type": "episodic"}] in result.to_list()
        )
        self.assertTrue(
            ["Alice", "loves", "Charlie", {"type": "episodic"}] in result.to_list()
        )

    def test_retrieve_random_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        random_memory = self.memory.retrieve_random_memory()
        self.assertTrue(random_memory in self.memory.to_list())

    def test_memory_operations(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertEqual(self.memory.size, 1)
        self.memory.forget_all()
        self.assertEqual(self.memory.size, 0)

    def tearDown(self):
        del self.memory


class TestShortMemory(unittest.TestCase):

    def setUp(self):
        self.short_memory = ShortMemory(capacity=5)

    def test_add_memory(self):
        self.short_memory.add(["Alice", "likes", "Bob", {"current_time": 12345}])
        self.assertEqual(len(self.short_memory), 1)
        self.assertTrue(
            self.short_memory.has_memory(
                ["Alice", "likes", "Bob", {"current_time": 12345}]
            )
        )

    def test_ob2short_conversion(self):
        ob = ["Alice", "likes", "Bob", 12345]
        short_mem = ShortMemory.ob2short(ob)
        self.assertEqual(short_mem, ["Alice", "likes", "Bob", {"current_time": 12345}])

    def tearDown(self):
        del self.short_memory


class TestLongMemory(unittest.TestCase):

    def setUp(self):
        self.long_memory = LongMemory(capacity=10)

    def test_add_memory(self):
        self.long_memory.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        self.assertEqual(len(self.long_memory), 1)
        self.assertTrue(
            self.long_memory.has_memory(
                ["Alice", "likes", "Bob", {"timestamp": [12345]}]
            )
        )

    def test_pretrain_semantic(self):
        semantic_knowledge = [
            ["desk", "atlocation", "office"],
            ["chair", "atlocation", "office"],
        ]
        self.long_memory.pretrain_semantic(semantic_knowledge)
        self.assertEqual(len(self.long_memory), 2)

    def tearDown(self):
        del self.long_memory


class TestMemorySystems(unittest.TestCase):

    def setUp(self):
        short_memory = ShortMemory(capacity=5)
        long_memory = LongMemory(capacity=10)
        self.memory_systems = MemorySystems(short_memory, long_memory)

    def test_get_working_memory(self):
        self.memory_systems.short.add(
            ["Alice", "likes", "Bob", {"current_time": 12345}]
        )
        self.memory_systems.long.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        working_memory = self.memory_systems.get_working_memory()
        self.assertEqual(len(working_memory), 1)

        self.memory_systems.long.add(
            ["Alice", "likes", "Alice", {"timestamp": [12345]}]
        )
        working_memory = self.memory_systems.get_working_memory()
        self.assertEqual(len(working_memory), 2)

    def tearDown(self):
        del self.memory_systems

    def test_get_working_memory_capacity_handling(self):

        # Fill up the short memory completely
        for i in range(5):
            self.memory_systems.short.add(
                ["Person" + str(i), "likes", "Object" + str(i), {"current_time": 12345}]
            )

        # Fill up the long memory almost completely (leave space for one more entry)
        for i in range(9):
            self.memory_systems.long.add(
                ["Person" + str(i), "likes", "Object" + str(i), {"timestamp": [12345]}]
            )

        # Add one more entry to long memory, exceeding the combined capacity
        self.memory_systems.long.add(
            ["Person9", "likes", "Object9", {"timestamp": [12345]}]
        )

        # Retrieve working memory and check total count
        working_memory = self.memory_systems.get_working_memory()
        self.assertEqual(
            len(working_memory), 10
        )  # 5 from short + 10 from long - 5 same


class TestSameMemory(unittest.TestCase):

    def setUp(self):
        self.memories = [
            ["A", "related_to", "B", {"timestamp": [1, 3, 5], "strength": 10}],
            ["C", "related_to", "D", {"timestamp": [1, 2, 4], "strength": 5}],
            ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
            ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
            ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
        ]
        self.long_memory = LongMemory(capacity=10, memories=self.memories)

    def test_retrieve_memory_by_qualifier_with_ties(self):
        # Test retrieving memory with max strength (expect random choice between
        # strength 10 memories)
        memories = []
        random.seed(0)  # Setting the seed for reproducibility in the test
        for _ in range(100):
            memory = self.long_memory.retrieve_memory_by_qualifier(
                "strength", "int", "max"
            )
            self.assertIn(
                memory,
                [
                    ["A", "related_to", "B", {"timestamp": [1, 3, 5], "strength": 10}],
                    ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
                ],
            )

            memories.append(memory)

        memories = set([foo[0] for foo in memories])

        self.assertEqual(len(memories), 2)

        # Test retrieving memory with min timestamp (expect random choice between
        # timestamp 1 memories)
        memories = []
        random.seed(1)  # Setting the seed for reproducibility in the test
        for _ in range(100):
            memory = self.long_memory.retrieve_memory_by_qualifier(
                "timestamp", "list", "min", "min"
            )
            self.assertIn(
                memory,
                [
                    ["A", "related_to", "B", {"timestamp": [1, 3, 5], "strength": 10}],
                    ["C", "related_to", "D", {"timestamp": [1, 2, 4], "strength": 5}],
                ],
            )

            memories.append(memory)
        memories = set([foo[0] for foo in memories])

        self.assertEqual(len(memories), 2)

    def test_retrieve_memory_by_qualifier_with_complex_timestamps(self):
        # Test retrieving memory with max timestamp (expect random choice between
        # timestamp [5, 8] memories)
        memories = []
        random.seed(2)  # Setting the seed for reproducibility in the test
        for _ in range(100):
            memory = self.long_memory.retrieve_memory_by_qualifier(
                "timestamp", "list", "max", "max"
            )
            self.assertIn(
                memory,
                [
                    ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
                ],
            )

            memories.append(memory)

        memories = set([foo[0] for foo in memories])

        self.assertEqual(len(memories), 1)


class TestForgetBySelection(unittest.TestCase):

    def test_forget_by_selection_oldest(self):
        random.seed(0)  # Setting the seed for reproducibility in the test
        memories = [
            ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 10}],
            ["C", "related_to", "D", {"timestamp": [1, 2, 5], "strength": 5}],
            ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
            ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
            ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
        ]
        long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

        long_memory.forget_by_selection("oldest")
        remaining_memories = long_memory.to_list()
        self.assertEqual(remaining_memories, memories[1:])

    def test_forget_by_selection_oldest_hard(self):
        deleted_all = []
        random.seed(0)  # Setting the seed for reproducibility in the test
        for _ in range(100):
            memories = [
                ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 10}],
                ["C", "related_to", "D", {"timestamp": [1, 2, 4], "strength": 5}],
                ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
                ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
                ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
            ]
            long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

            long_memory.forget_by_selection("oldest")
            remaining_memories = long_memory.to_list()
            self.assertEqual(len(remaining_memories), 4)

            deleted_memory = [foo for foo in memories if foo not in remaining_memories]
            deleted_all.append(deleted_memory)

        self.assertEqual(len(set([foo[0][0] for foo in deleted_all])), 2)

    def test_forget_by_selection_latest(self):
        random.seed(0)  # Setting the seed for reproducibility in the test
        memories = [
            ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 10}],
            ["C", "related_to", "D", {"timestamp": [1, 2, 5], "strength": 5}],
            ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
            ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
            ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
        ]
        long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

        long_memory.forget_by_selection("latest")
        remaining_memories = long_memory.to_list()
        self.assertEqual(remaining_memories, memories[:-1])

    def test_forget_by_selection_latest_hard(self):
        deleted_all = []
        random.seed(0)  # Setting the seed for reproducibility in the test
        for _ in range(100):
            memories = [
                ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 10}],
                ["C", "related_to", "D", {"timestamp": [1, 2, 4], "strength": 5}],
                ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
                ["G", "related_to", "H", {"timestamp": [3, 7, 8], "strength": 8}],
                ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
            ]
            long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

            long_memory.forget_by_selection("latest")
            remaining_memories = long_memory.to_list()
            self.assertEqual(len(remaining_memories), 4)

            deleted_memory = [foo for foo in memories if foo not in remaining_memories]
            deleted_all.append(deleted_memory)

        self.assertEqual(len(set([foo[0][0] for foo in deleted_all])), 2)

    def test_forget_by_selection_weakest(self):
        random.seed(0)  # Setting the seed for reproducibility in the test
        memories = [
            ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 10}],
            ["C", "related_to", "D", {"timestamp": [1, 2, 5], "strength": 4}],
            ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
            ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
            ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
        ]
        long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

        long_memory.forget_by_selection("weakest")
        remaining_memories = long_memory.to_list()
        self.assertEqual(remaining_memories, memories[:1] + memories[2:])

    def test_forget_by_selection_weakest_hard(self):
        deleted_all = []
        random.seed(0)  # Setting the seed for reproducibility in the test
        for _ in range(100):
            memories = [
                ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 10}],
                ["C", "related_to", "D", {"timestamp": [1, 2, 4], "strength": 5}],
                ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
                ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
                ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
            ]
            long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

            long_memory.forget_by_selection("weakest")
            remaining_memories = long_memory.to_list()
            self.assertEqual(len(remaining_memories), 4)

            deleted_memory = [foo for foo in memories if foo not in remaining_memories]
            deleted_all.append(deleted_memory)

        self.assertEqual(len(set([foo[0][0] for foo in deleted_all])), 2)

    def test_forget_by_selection_strongest(self):
        random.seed(0)  # Setting the seed for reproducibility in the test
        memories = [
            ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 11}],
            ["C", "related_to", "D", {"timestamp": [1, 2, 5], "strength": 4}],
            ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
            ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
            ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
        ]
        long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

        long_memory.forget_by_selection("strongest")
        remaining_memories = long_memory.to_list()
        self.assertEqual(remaining_memories, memories[1:])

    def test_forget_by_selection_strongest_hard(self):
        deleted_all = []
        random.seed(0)  # Setting the seed for reproducibility in the test
        for _ in range(100):
            memories = [
                ["A", "related_to", "B", {"timestamp": [1, 3, 4], "strength": 10}],
                ["C", "related_to", "D", {"timestamp": [1, 2, 4], "strength": 5}],
                ["E", "related_to", "F", {"timestamp": [2, 6], "strength": 5}],
                ["G", "related_to", "H", {"timestamp": [3, 7], "strength": 8}],
                ["I", "related_to", "J", {"timestamp": [4, 8], "strength": 10}],
            ]
            long_memory = LongMemory(capacity=10, memories=deepcopy(memories))

            long_memory.forget_by_selection("strongest")
            remaining_memories = long_memory.to_list()
            self.assertEqual(len(remaining_memories), 4)

            deleted_memory = [foo for foo in memories if foo not in remaining_memories]
            deleted_all.append(deleted_memory)

        self.assertEqual(len(set([foo[0][0] for foo in deleted_all])), 2)


class TestMemoryCanBeAdded(unittest.TestCase):

    def test_can_be_added_with_non_frozen_memory(self):
        memory = Memory(capacity=5)
        mem = ["A", "related_to", "B", {"timestamp": [1]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)

    def test_can_be_added_with_frozen_memory(self):
        memory = Memory(capacity=5)
        memory.freeze()
        mem = ["A", "related_to", "B", {"timestamp": [1]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory system is frozen!")

    def test_can_be_added_with_zero_capacity(self):
        memory = Memory(capacity=0)
        mem = ["A", "related_to", "B", {"timestamp": [1]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory system capacity is 0!")

    def test_can_be_added_with_full_memory(self):
        memory = Memory(capacity=2)
        memory.add(["A", "related_to", "B", {"timestamp": [1]}])
        memory.add(["C", "related_to", "D", {"timestamp": [2]}])
        mem = ["E", "related_to", "F", {"timestamp": [3]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)


class TestShortMemoryCanBeAdded(unittest.TestCase):

    def test_can_be_added_with_current_time(self):
        memory = ShortMemory(capacity=1)
        mem = ["A", "related_to", "B", {"current_time": 1}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)
        memory.add(mem)

        mem = ["A", "related_to", "B", {"current_time": 1}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)
        memory.add(mem)

        mem = ["A", "related_to", "B", {"current_time": 2}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory system is full!")

        with self.assertRaises(AssertionError):
            memory.add(mem)

    def test_can_be_added_without_current_time(self):
        memory = ShortMemory(capacity=5)
        mem = ["A", "related_to", "B", {}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory should have current_time!")

    def test_can_be_added_with_full_memory(self):
        memory = ShortMemory(capacity=2)
        memory.add(["A", "related_to", "B", {"current_time": 1}])
        memory.add(["C", "related_to", "D", {"current_time": 2}])
        mem = ["E", "related_to", "F", {"current_time": 3}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory system is full!")


class TestLongMemoryCanBeAdded(unittest.TestCase):

    def test_can_be_added_with_timestamp(self):
        memory = LongMemory(capacity=5)
        mem = ["A", "related_to", "B", {"timestamp": [1]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)

    def test_can_be_added_with_strength(self):
        memory = LongMemory(capacity=5)
        mem = ["A", "related_to", "B", {"strength": 1}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)

    def test_can_be_added_without_qualifiers(self):
        memory = LongMemory(capacity=5)
        mem = ["A", "related_to", "B", {}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory should have timestamp or strength!")

    def test_can_be_added_with_full_memory(self):
        memory = LongMemory(capacity=2)
        memory.add(["A", "related_to", "B", {"timestamp": [1]}])
        memory.add(["C", "related_to", "D", {"strength": 2}])
        mem = ["E", "related_to", "F", {"timestamp": [3]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory system is full!")

        with self.assertRaises(AssertionError):
            memory.add(mem)

        mem = ["A", "related_to", "B", {"timestamp": [3]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)
        memory.add(mem)

        self.assertEqual(
            memory.to_list(),
            [
                ["A", "related_to", "B", {"timestamp": [1, 3]}],
                ["C", "related_to", "D", {"strength": 2}],
            ],
        )

        mem = ["C", "related_to", "D", {"strength": 3}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertTrue(can_be_added)
        self.assertIsNone(error_msg)
        memory.add(mem)

        self.assertEqual(
            memory.to_list(),
            [
                ["A", "related_to", "B", {"timestamp": [1, 3]}],
                ["C", "related_to", "D", {"strength": 5}],
            ],
        )

    def test_can_be_added_with_frozen_memory(self):
        memory = LongMemory(capacity=5)
        memory.freeze()
        mem = ["A", "related_to", "B", {"timestamp": [1]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory system is frozen!")

    def test_can_be_added_with_zero_capacity(self):
        memory = LongMemory(capacity=0)
        mem = ["A", "related_to", "B", {"timestamp": [1]}]
        can_be_added, error_msg = memory.can_be_added(mem)
        self.assertFalse(can_be_added)
        self.assertEqual(error_msg, "The memory system capacity is 0!")


class TestMemoryAdditional(unittest.TestCase):

    def setUp(self):
        self.memory = Memory(capacity=5)

    def test_add_memory_over_capacity(self):
        for i in range(5):
            self.memory.add([f"head{i}", "relation", f"tail{i}", {"type": "episodic"}])
        with self.assertRaises(ValueError):
            self.memory.add(["overflow", "relation", "tail", {"type": "episodic"}])

    def test_forget_non_existent_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        with self.assertRaises(ValueError):
            self.memory.forget(
                ["Non-existent", "relation", "memory", {"type": "episodic"}]
            )

    def test_query_memory_no_results(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        result = self.memory.query(["?", "hates", "?", {"type": "episodic"}])
        self.assertEqual(len(result), 0)

    def test_query_memory_with_partial_match(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.memory.add(["Alice", "loves", "Charlie", {"type": "episodic"}])
        result = self.memory.query(["Alice", "?", "?", "?"])
        self.assertEqual(len(result), 2)
        self.assertTrue(
            ["Alice", "likes", "Bob", {"type": "episodic"}] in result.to_list()
        )
        self.assertTrue(
            ["Alice", "loves", "Charlie", {"type": "episodic"}] in result.to_list()
        )

    def test_freeze_memory_operations(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.forget(["Alice", "likes", "Bob", {"type": "episodic"}])
        with self.assertRaises(ValueError):
            self.memory.forget_all()
        self.memory.unfreeze()
        self.memory.forget_all()
        self.assertTrue(self.memory.is_empty)


class TestShortMemoryAdditional(unittest.TestCase):

    def setUp(self):
        self.short_memory = ShortMemory(capacity=5)

    def test_ob2short_conversion_invalid_input(self):
        ob = ["Alice", "likes", "Bob"]
        with self.assertRaises(AssertionError):
            ShortMemory.ob2short(ob)

    def test_short2epi_conversion(self):
        short_mem = ["Alice", "likes", "Bob", {"current_time": 12345}]
        epi_mem = ShortMemory.short2epi(short_mem)
        self.assertEqual(epi_mem, ["Alice", "likes", "Bob", {"timestamp": [12345]}])

    def test_short2sem_conversion(self):
        short_mem = ["Alice", "likes", "Bob", {"current_time": 12345}]
        sem_mem = ShortMemory.short2sem(short_mem)
        self.assertEqual(sem_mem, ["Alice", "likes", "Bob", {"strength": 1}])


class TestLongMemoryAdditional(unittest.TestCase):

    def setUp(self):
        self.long_memory = LongMemory(capacity=10)

    def test_add_memory_with_mixed_qualifiers(self):
        self.long_memory.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        self.long_memory.add(["Alice", "likes", "Bob", {"strength": 2}])
        self.assertEqual(
            self.long_memory.to_list(),
            [["Alice", "likes", "Bob", {"timestamp": [12345], "strength": 2}]],
        )

    def test_add_memory_duplicate_timestamps(self):
        self.long_memory.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        self.long_memory.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        self.assertEqual(
            self.long_memory.to_list(),
            [["Alice", "likes", "Bob", {"timestamp": [12345, 12345]}]],
        )

    def test_forget_memory_by_selection_empty_memory(self):
        with self.assertRaises(ValueError):
            self.long_memory.forget_by_selection("oldest")

    def test_count_memories(self):
        self.long_memory.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        self.long_memory.add(["Alice", "likes", "Charlie", {"strength": 2}])
        self.assertEqual(self.long_memory.count_memories(), (1, 1))


class TestMemorySystemsAdditional(unittest.TestCase):

    def setUp(self):
        short_memory = ShortMemory(capacity=5)
        long_memory = LongMemory(capacity=10)
        self.memory_systems = MemorySystems(short_memory, long_memory)

    def test_forget_all_memories(self):
        self.memory_systems.short.add(
            ["Alice", "likes", "Bob", {"current_time": 12345}]
        )
        self.memory_systems.long.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        self.memory_systems.forget_all()
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertTrue(self.memory_systems.long.is_empty)

    def test_get_working_memory_with_duplicates(self):
        self.memory_systems.short.add(
            ["Alice", "likes", "Bob", {"current_time": 12345}]
        )
        self.memory_systems.long.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        working_memory = self.memory_systems.get_working_memory()
        self.assertEqual(len(working_memory), 1)


class TestLongMemoryDecay(unittest.TestCase):

    def setUp(self):
        # Set up the LongMemory instance with a decay factor and some memories
        self.memories = [
            ["bob", "knows", "alice", {"strength": 10}],
            ["car", "is", "blue", {"strength": 5}],
            ["sun", "is", "bright", {"timestamp": [1, 2, 3], "strength": 8}],
            ["sun", "is", "dark", {"timestamp": [1, 2]}],
        ]
        self.capacity = 10
        self.decay_factor = 0.5
        self.min_strength = 1
        self.long_memory = LongMemory(
            self.capacity, self.memories, self.decay_factor, self.min_strength
        )

    def test_decay_reduces_strength(self):
        self.long_memory.decay()
        expected_strengths = [5, 2.5, 4]
        for mem, expected_strength in zip(
            self.long_memory.entries[:3], expected_strengths
        ):
            self.assertEqual(mem[-1]["strength"], expected_strength)

    def test_decay_strength_not_below_minimum(self):
        # Set strengths close to minimum and apply decay
        self.long_memory.entries[0][-1]["strength"] = 1.5
        self.long_memory.entries[1][-1]["strength"] = 1.0
        self.long_memory.entries[2][-1]["strength"] = 2.0
        self.long_memory.decay()
        self.assertEqual(self.long_memory.entries[0][-1]["strength"], self.min_strength)
        self.assertEqual(self.long_memory.entries[1][-1]["strength"], self.min_strength)
        self.assertEqual(self.long_memory.entries[2][-1]["strength"], 1)

    def test_decay_does_not_affect_non_strength_memories(self):
        # Check that non-strength memories are unaffected
        initial_timestamp = self.long_memory.entries[3][-1]["timestamp"]
        self.long_memory.decay()
        self.assertEqual(
            self.long_memory.entries[3][-1]["timestamp"], initial_timestamp
        )

        initial_timestamp = self.long_memory.entries[2][-1]["timestamp"]
        self.long_memory.decay()
        self.assertEqual(
            self.long_memory.entries[2][-1]["timestamp"], initial_timestamp
        )
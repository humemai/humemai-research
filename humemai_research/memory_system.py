"""MemorySystem class"""

import collections
import logging

from rdflib import Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

from .memory import Memory

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("humemai.memory_system")

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class MemorySystem:
    """
    MemorySystem class that encapsulates the memory storage.
    Provides methods to interact with the overall memory system.
    """

    def __init__(self):
        self.memory = Memory()

    def is_reified_statement_short_term(self, statement) -> bool:
        """
        Check if a given reified statement is a short-term memory by verifying if it
        has a 'currentTime' qualifier.

        Args:
            statement (BNode or URIRef): The reified statement to check.

        Returns:
            bool: True if it's a short-term memory, False otherwise.
        """
        return self.memory.is_reified_statement_short_term(statement)

    def get_working_memory(
        self,
        trigger_node: URIRef = None,
        hops: int = 0,
        include_all_long_term: bool = False,
    ) -> "Memory":
        """
        Retrieve working memory based on a trigger node and a specified number of hops.
        It fetches all triples within N hops from the trigger node in the long-term
        memory, including their qualifiers. Also includes all short-term memories.

        If `include_all_long_term` is True, all long-term memories will be included,
        regardless of the BFS traversal or hops.

        For each long-term memory retrieved, the 'recalled' qualifier is incremented by 1.

        Args:
            trigger_node (URIRef, optional): The starting node for memory traversal.
            hops (int, optional): The number of hops for BFS traversal (default: 0).
            include_all_long_term (bool, optional): Include all long-term memories
            (default: False).

        Returns:
            Memory: A new Memory object containing the working memory (short-term +
            relevant long-term memories).
        """
        working_memory = Memory()
        processed_statements = set()  # Keep track of processed reified statements

        logger.info(
            f"Initializing working memory. Trigger node: {trigger_node}, Hops: {hops}, Include all long-term: {include_all_long_term}"
        )

        # Add short-term memories to working memory
        short_term = self.memory.get_short_term_memories()
        for s, p, o in short_term.graph:
            working_memory.graph.add((s, p, o))

        for s, p, o in short_term.graph.triples((None, RDF.type, RDF.Statement)):
            for qualifier_pred, qualifier_obj in short_term.graph.predicate_objects(s):
                if qualifier_pred not in (
                    RDF.type,
                    RDF.subject,
                    RDF.predicate,
                    RDF.object,
                ):
                    working_memory.graph.add((s, qualifier_pred, qualifier_obj))

        # If include_all_long_term is True, add all long-term memories to working memory
        if include_all_long_term:
            logger.info("Including all long-term memories into working memory.")

            # Get all long-term memories and add them to the working memory graph
            for statement in self.memory.graph.subjects(RDF.type, RDF.Statement):
                if not self.is_reified_statement_short_term(statement):
                    subj = self.memory.graph.value(statement, RDF.subject)
                    pred = self.memory.graph.value(statement, RDF.predicate)
                    obj = self.memory.graph.value(statement, RDF.object)

                    if statement not in processed_statements:
                        working_memory.graph.add((subj, pred, obj))
                        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
                            subj,
                            pred,
                            obj,
                            working_memory,
                            specific_statement=statement,
                        )
                        processed_statements.add(statement)

            return working_memory

        else:
            if trigger_node is None:
                raise ValueError(
                    "trigger_node must be provided when include_all_long_term is False"
                )

        # Proceed with BFS traversal
        queue = collections.deque()
        queue.append((trigger_node, 0))
        visited = set()
        visited.add(trigger_node)

        while queue:
            current_node, current_hop = queue.popleft()

            if current_hop >= hops:
                continue

            # Explore outgoing triples
            for p, o in self.memory.graph.predicate_objects(current_node):
                # Retrieve all reified statements for this triple
                reified_statements = [
                    stmt
                    for stmt in self.memory.graph.subjects(RDF.type, RDF.Statement)
                    if self.memory.graph.value(stmt, RDF.subject) == current_node
                    and self.memory.graph.value(stmt, RDF.predicate) == p
                    and self.memory.graph.value(stmt, RDF.object) == o
                ]

                for statement in reified_statements:
                    if self.is_reified_statement_short_term(statement):
                        continue  # Skip short-term memories

                    # Only increment and add the statement if it hasn't been processed yet
                    if statement not in processed_statements:
                        # Add the triple to the working memory
                        working_memory.graph.add((current_node, p, o))

                        # Add the reified statement and increment 'recalled'
                        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
                            current_node,
                            p,
                            o,
                            working_memory,
                            specific_statement=statement,
                        )

                        processed_statements.add(statement)

                    # Enqueue the object node if it's a URIRef and not visited
                    if isinstance(o, URIRef) and o not in visited:
                        queue.append((o, current_hop + 1))
                        visited.add(o)

            # Explore incoming triples
            for s, p in self.memory.graph.subject_predicates(current_node):
                # Retrieve all reified statements for this triple
                reified_statements = [
                    stmt
                    for stmt in self.memory.graph.subjects(RDF.type, RDF.Statement)
                    if self.memory.graph.value(stmt, RDF.subject) == s
                    and self.memory.graph.value(stmt, RDF.predicate) == p
                    and self.memory.graph.value(stmt, RDF.object) == current_node
                ]

                for statement in reified_statements:
                    if self.is_reified_statement_short_term(statement):
                        continue  # Skip short-term memories

                    # Only increment and add the statement if it hasn't been processed yet
                    if statement not in processed_statements:
                        # Add the triple to the working memory
                        working_memory.graph.add((s, p, current_node))

                        # Add the reified statement and increment 'recalled'
                        self.memory._add_reified_statement_to_working_memory_and_increment_recall(
                            s,
                            p,
                            current_node,
                            working_memory,
                            specific_statement=statement,
                        )

                        processed_statements.add(statement)

                    # Enqueue the subject node if it's a URIRef and not visited
                    if isinstance(s, URIRef) and s not in visited:
                        queue.append((s, current_hop + 1))
                        visited.add(s)

        return working_memory

    def move_short_term_to_episodic(
        self, memory_id_to_move: Literal, qualifiers: dict = {}
    ):
        """
        Move the specified short-term memory to long-term episodic memory.

        Args:
            memory_id_to_move (Literal): The memory ID to move from short-term to long-term.
        """
        if memory_id_to_move.datatype != XSD.integer:
            raise ValueError("Memory ID must be an integer.")

        # Iterate through the short-term memories
        for subj, pred, obj, qualifiers_ in self.memory.iterate_memories("short_term"):
            memory_id = qualifiers_.get(humemai.memoryID)

            # Check if the memory ID matches
            if memory_id == memory_id_to_move:
                location = qualifiers_.get(humemai.location)
                currentTime = qualifiers_.get(humemai.currentTime)

                qualifiers[humemai.eventTime] = currentTime

                if location:
                    qualifiers[humemai.location] = location

                # Move to long-term episodic memory
                self.memory.add_episodic_memory(
                    triples=[(subj, pred, obj)], qualifiers=qualifiers
                )

                # Remove the short-term memory after moving it to long-term
                self.memory.delete_memory(memory_id)

                logger.debug(
                    f"Moved short-term memory with ID {memory_id_to_move} to episodic long-term memory."
                )
                break

    def move_short_term_to_semantic(
        self, memory_id_to_move: Literal, qualifiers: dict = {}
    ):
        """
        Move the specified short-term memory to long-term semantic memory.

        Args:
            memory_id_to_move (Literal): The memory ID to move from short-term to long-term.
        """
        if memory_id_to_move.datatype != XSD.integer:
            raise ValueError("Memory ID must be an integer.")

        # Iterate through the short-term memories
        for subj, pred, obj, qualifiers_ in self.memory.iterate_memories("short_term"):
            memory_id = qualifiers_.get(humemai.memoryID)

            # Check if the memory ID matches
            if memory_id == memory_id_to_move:
                currentTime = qualifiers_.get(humemai.currentTime)

                qualifiers[humemai.knownSince] = currentTime

                # Move to long-term semantic memory
                self.memory.add_semantic_memory(
                    triples=[(subj, pred, obj)], qualifiers=qualifiers
                )

                # Remove the short-term memory after moving it to long-term
                self.memory.delete_memory(memory_id)
                logger.debug(
                    f"Moved short-term memory with ID {memory_id_to_move} to semantic long-term memory."
                )
                break

    def clear_short_term_memories(self):
        """
        Clear all short-term memories from the memory system.
        """
        for subj, pred, obj, qualifiers in self.memory.iterate_memories("short_term"):
            memory_id = qualifiers.get(humemai.memoryID)

            self.memory.delete_memory(memory_id)
            logger.debug(f"Cleared short-term memory with ID {memory_id}.")

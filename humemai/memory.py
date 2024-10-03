"""Memory Class with RDFLib"""

import collections
import logging
import os
from datetime import datetime

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from .utils import validate_iso_format

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("humemai.memory")

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology/")


class Memory:
    """
    Memory class for managing both short-term and long-term memories.
    Provides methods to add, retrieve, delete, cluster, and manage memories in the RDF graph.
    """

    def __init__(self, verbose_repr: bool = False):
        # Initialize RDF graph for memory storage
        self.graph = Graph()
        self.graph.bind("humemai", humemai)
        self.verbose_repr = verbose_repr
        self.current_statement_id = 0  # Counter to track the next unique ID

    def add_memory(self, triples: list, qualifiers: dict):
        """
        Add a reified statement to the RDF graph, with the main triple and optional
        qualifiers.

        Args:
            triples (list): A list of triples (subject, predicate, object) to be added.
            qualifiers (dict): A dictionary of qualifiers (e.g., location, currentTime).
        """
        for subj, pred, obj in triples:
            logger.debug(f"Adding triple: ({subj}, {pred}, {obj})")

            # Add the main triple [subject, predicate, object] if it doesn't already exist
            if not (subj, pred, obj) in self.graph:
                self.graph.add((subj, pred, obj))
                logger.debug(f"Main triple added: ({subj}, {pred}, {obj})")
            else:
                logger.debug(f"Main triple already exists: ({subj}, {pred}, {obj})")

            # Create a new reified statement to attach the qualifiers, and assign a unique ID
            statement = BNode()  # Blank node to represent the new reified statement
            unique_id = self.current_statement_id  # Get the current ID
            self.current_statement_id += 1  # Increment for the next memory

            # Add the reified statement and unique ID
            self.graph.add((statement, RDF.type, RDF.Statement))
            self.graph.add((statement, RDF.subject, subj))
            self.graph.add((statement, RDF.predicate, pred))
            self.graph.add((statement, RDF.object, obj))
            self.graph.add(
                (statement, humemai.memoryID, Literal(unique_id, datatype=XSD.integer))
            )  # Add the unique ID

            logger.debug(f"Reified statement created: {statement} with ID {unique_id}")

            # Add the provided qualifiers with the correct data types
            for key, value in qualifiers.items():
                if key in [humemai.currentTime, humemai.time]:
                    if validate_iso_format(value):
                        self.graph.add(
                            (
                                statement,
                                URIRef(key),
                                Literal(value, datatype=XSD.dateTime),
                            )
                        )
                        logger.debug(
                            f"Added qualifier: ({statement}, {key}, {value}) with datatype xsd:dateTime"
                        )
                    else:
                        raise ValueError("Invalid date format. Please use ISO format.")

                elif key == humemai.strength:
                    self.graph.add(
                        (statement, URIRef(key), Literal(value, datatype=XSD.integer))
                    )
                    logger.debug(
                        f"Added qualifier: ({statement}, {key}, {value}) with datatype xsd:integer"
                    )
                elif key == humemai.recalled:
                    self.graph.add(
                        (statement, URIRef(key), Literal(value, datatype=XSD.integer))
                    )
                    logger.debug(
                        f"Added qualifier: ({statement}, {key}, {value}) with datatype xsd:integer"
                    )
                else:
                    self.graph.add((statement, URIRef(key), Literal(value)))
                    logger.debug(
                        f"Added qualifier: ({statement}, {key}, {value}) as plain literal"
                    )

    def delete_memory(self, memory_id: int):
        """
        Delete a memory (reified statement) by its unique ID, including all associated qualifiers.

        Args:
            memory_id (int): The unique ID of the memory to be deleted.
        """
        logger.debug(f"Deleting memory with ID: {memory_id}")

        # Find the reified statement with the given ID
        statement = None
        for stmt in self.graph.subjects(
            humemai.memoryID, Literal(memory_id, datatype=XSD.integer)
        ):
            statement = stmt
            break

        if statement is None:
            logger.error(f"No memory found with ID {memory_id}")
            return

        # Get the main triple (subject, predicate, object) from the reified statement
        subj = self.graph.value(statement, RDF.subject)
        pred = self.graph.value(statement, RDF.predicate)
        obj = self.graph.value(statement, RDF.object)

        if subj is None or pred is None or obj is None:
            logger.error(
                f"Invalid memory statement {statement}. Cannot find associated triple."
            )
            return

        logger.debug(f"Deleting main triple: ({subj}, {pred}, {obj})")

        # Remove the main triple
        self.graph.remove((subj, pred, obj))

        # Remove all qualifiers associated with this reified statement
        for p, o in list(self.graph.predicate_objects(statement)):
            self.graph.remove((statement, p, o))
            logger.debug(f"Removed qualifier triple: ({statement}, {p}, {o})")

        # Remove the reified statement itself
        self.graph.remove((statement, RDF.type, RDF.Statement))
        self.graph.remove((statement, RDF.subject, subj))
        self.graph.remove((statement, RDF.predicate, pred))
        self.graph.remove((statement, RDF.object, obj))

        logger.info(f"Memory with ID {memory_id} deleted successfully.")

    def get_memory_by_id(self, memory_id: int) -> dict:
        """
        Retrieve a memory (reified statement) by its unique ID and return its details.

        Args:
            memory_id (int): The unique ID of the memory to retrieve.

        Returns:
            dict: A dictionary with the memory details (subject, predicate, object, qualifiers).
        """
        # Find the reified statement with the given ID
        for stmt in self.graph.subjects(
            humemai.memoryID, Literal(memory_id, datatype=XSD.integer)
        ):
            subj = self.graph.value(stmt, RDF.subject)
            pred = self.graph.value(stmt, RDF.predicate)
            obj = self.graph.value(stmt, RDF.object)
            qualifiers = {}

            # Collect all qualifiers for this memory
            for q_pred, q_obj in self.graph.predicate_objects(stmt):
                if q_pred not in (
                    RDF.type,
                    RDF.subject,
                    RDF.predicate,
                    RDF.object,
                ):
                    qualifiers[str(q_pred)] = str(q_obj)

            return {
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "qualifiers": qualifiers,
            }

        logger.error(f"No memory found with ID {memory_id}")
        return None

    def delete_triple(self, subject: URIRef, predicate: URIRef, object_: URIRef):
        """
        Delete a triple from the RDF graph, including all of its qualifiers.

        Args:
            subject (URIRef): The subject of the memory triple.
            predicate (URIRef): The predicate of the memory triple.
            object_ (URIRef): The object of the memory triple.
        """
        # Remove the main triple
        self.graph.remove((subject, predicate, object_))
        logger.debug(f"Removed triple: ({subject}, {predicate}, {object_})")

        # Find all reified statements for this triple
        for statement in list(self.graph.subjects(RDF.type, RDF.Statement)):
            s = self.graph.value(statement, RDF.subject)
            p = self.graph.value(statement, RDF.predicate)
            o = self.graph.value(statement, RDF.object)
            if s == subject and p == predicate and o == object_:
                logger.debug(f"Removing qualifiers for statement: {statement}")
                # Remove all triples related to this statement
                for _, _, obj in list(self.graph.triples((statement, None, None))):
                    self.graph.remove((statement, _, obj))
                    logger.debug(f"Removed qualifier triple: ({statement}, _, {obj})")

    def add_short_term_memory(
        self, triples: list, location: str = None, currentTime: str = None
    ):
        """
        Add short-term memories to the RDF graph, enforcing required qualifiers.

        Args:
            triples (list): A list of triples to add.
            location (str): The current location.
            currentTime (str, optional): The current time in ISO 8601 format.
        """
        if currentTime is None:
            currentTime = datetime.now().isoformat()

        # Ensure currentTime is in ISO 8601 format
        if not validate_iso_format(currentTime):
            raise ValueError(f"Invalid currentTime format: {currentTime}")

        # Check for required qualifiers
        if not currentTime:
            raise ValueError("Missing required qualifier: currentTime")

        if location is None:
            qualifiers = {humemai.currentTime: currentTime}
        else:
            qualifiers = {
                humemai.currentTime: currentTime,
                humemai.location: location,
            }

        self.add_memory(triples, qualifiers)

    def add_long_term_memory(
        self,
        memory_type: str,
        triples: list,
        location: str = None,
        time: str = None,
        emotion: str = None,
        derivedFrom: str = None,
        strength: int = None,
        event: str = None,
    ):
        """
        Add long-term memories to the RDF graph, enforcing required qualifiers.

        Args:
            memory_type (str): The type of memory. Should be either "episodic" or "semantic".
            triples (list): A list of triples to add.
            location (str, optional): The location associated with the memory.
            time (str, optional): The time associated with the memory in ISO 8601 format.
            emotion (str, optional): The emotion associated with the memory.
            derivedFrom (str, optional): The source from which the memory was derived.
            strength (int, optional): The strength of the memory.
            event (str, optional): The event associated with the memory.
        """
        qualifiers = {}

        if memory_type == "episodic":
            # Required qualifiers for episodic memories
            if not time:
                raise ValueError("Missing required qualifier for episodic memory: time")

            if not validate_iso_format(time):
                raise ValueError(f"Invalid time format: {time}")

            if strength or derivedFrom:
                raise ValueError(
                    "Invalid qualifiers for episodic memory. Use 'semantic' memory type instead."
                )

            # Add required qualifiers
            qualifiers[humemai.location] = location
            qualifiers[humemai.time] = time

            # Add optional qualifiers
            if emotion is not None:
                qualifiers[humemai.emotion] = emotion
            if event is not None:
                qualifiers[humemai.event] = event

        elif memory_type == "semantic":
            # Required qualifiers for semantic memories
            if location or time or emotion:
                raise ValueError(
                    "Invalid qualifiers for semantic memory. Use 'episodic' memory type instead."
                )

            # Add optional qualifiers
            if strength is not None:
                qualifiers[humemai.strength] = strength
            if derivedFrom is not None:
                qualifiers[humemai.derivedFrom] = derivedFrom

        else:
            raise ValueError("memory_type must be either 'episodic' or 'semantic'")

        # Optional qualifiers
        if event:
            qualifiers[humemai.event] = event

        qualifiers[humemai.recalled] = 0  # Initialize 'recalled' qualifier to 0

        self.add_memory(triples, qualifiers)

    def get_memories(
        self,
        subject: URIRef = None,
        predicate: URIRef = None,
        object_: URIRef = None,
        location: str = None,
        emotion: str = None,
        derivedFrom: str = None,
        strength: int = None,
        recalled: int = None,
        event: str = None,
        lower_time_bound: str = None,
        upper_time_bound: str = None,
    ) -> "Memory":
        """
        Retrieve memories with optional filtering based on the qualifiers and triple values, including time bounds.

        Args:
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            location (str, optional): Filter by location value.
            emotion (str, optional): Filter by emotion value.
            derivedFrom (str, optional): Filter by derivedFrom value.
            strength (int, optional): Filter by strength value.
            recalled (int, optional): Filter by recalled value.
            event (str, optional): Filter by event value.
            lower_time_bound (str, optional): Lower bound for time filtering (ISO format).
            upper_time_bound (str, optional): Upper bound for time filtering (ISO format).

        Returns:
            Memory: A new Memory object containing the filtered memories.
        """

        # SPARQL query to retrieve memories with optional filters
        query = """
        PREFIX humemai: <https://humem.ai/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?subject ?predicate ?object
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object .
        """

        # Add filters dynamically based on input
        if subject is not None:
            query += f"FILTER(?subject = <{subject}>) .\n"
        if predicate is not None:
            query += f"FILTER(?predicate = <{predicate}>) .\n"
        if object_ is not None:
            query += f"FILTER(?object = <{object_}>) .\n"
        if location is not None:
            query += f'?statement humemai:location "{location}" .\n'
        if emotion is not None:
            query += f'?statement humemai:emotion "{emotion}" .\n'
        if derivedFrom is not None:
            query += f'?statement humemai:derivedFrom "{derivedFrom}" .\n'
        if strength is not None:
            query += f'?statement humemai:strength "{strength}" .\n'
        if recalled is not None:
            query += f'?statement humemai:recalled "{recalled}" .\n'
        if event is not None:
            query += f'?statement humemai:event "{event}" .\n'

        # Add time filtering logic (both currentTime and time)
        if lower_time_bound and upper_time_bound:
            time_filter = f"""
            OPTIONAL {{ ?statement humemai:currentTime ?currentTime }}
            OPTIONAL {{ ?statement humemai:time ?time }}
            FILTER((?currentTime >= "{lower_time_bound}"^^xsd:dateTime && ?currentTime <= "{upper_time_bound}"^^xsd:dateTime) ||
                (?time >= "{lower_time_bound}"^^xsd:dateTime && ?time <= "{upper_time_bound}"^^xsd:dateTime)) .
            """
            query += time_filter

        # Close the WHERE block
        query += "}"

        logger.debug(f"Executing SPARQL query:\n{query}")

        # Execute the SPARQL query
        results = self.graph.query(query)

        # To store reified statements and their corresponding qualifiers
        statement_dict = {}

        # Iterate through the results and organize them
        for row in results:
            subj = row.subject
            pred = row.predicate
            obj = row.object

            # The reified statement
            statement = row.statement

            # Create a key from the reified statement (not just the triple)
            if statement not in statement_dict:
                statement_dict[statement] = {
                    "triple": (subj, pred, obj),
                    "qualifiers": {},
                }

            # Add all the qualifiers related to the reified statement
            for qualifier_pred, qualifier_obj in self.graph.predicate_objects(
                statement
            ):
                if qualifier_pred not in (
                    RDF.type,
                    RDF.subject,
                    RDF.predicate,
                    RDF.object,
                ):
                    statement_dict[statement]["qualifiers"][str(qualifier_pred)] = str(
                        qualifier_obj
                    )

        # Create a new Memory object to store the filtered results
        filtered_memory = Memory(self.verbose_repr)

        # Populate the Memory object with the main triples and their qualifiers
        for statement, data in statement_dict.items():
            subj, pred, obj = data["triple"]
            qualifiers = data["qualifiers"]

            # Add the main triple to the graph
            filtered_memory.graph.add((subj, pred, obj))

            # Create a reified statement (blank node)
            new_statement = BNode()
            filtered_memory.graph.add((new_statement, RDF.type, RDF.Statement))
            filtered_memory.graph.add((new_statement, RDF.subject, subj))
            filtered_memory.graph.add((new_statement, RDF.predicate, pred))
            filtered_memory.graph.add((new_statement, RDF.object, obj))

            # Add the qualifiers for the reified statement
            for qualifier_pred, qualifier_obj in qualifiers.items():
                filtered_memory.graph.add(
                    (new_statement, URIRef(qualifier_pred), Literal(qualifier_obj))
                )

        return filtered_memory

    def get_triple_count(self) -> int:
        """
        Count the number of unique memories (subject-predicate-object triples) in the
        graph. This does not count reified statements.

        Returns:
            int: The count of unique memories.
        """
        unique_memories = set()

        # Iterate over reified statements and extract subject-predicate-object triples
        for s, p, o in self.graph.triples((None, RDF.type, RDF.Statement)):
            subj = self.graph.value(s, RDF.subject)
            pred = self.graph.value(s, RDF.predicate)
            obj = self.graph.value(s, RDF.object)

            unique_memories.add((subj, pred, obj))

        return len(unique_memories)

    def get_memory_count(self) -> int:
        """
        Count the number of reified statements (RDF statements) in the graph.
        This counts the reified statements instead of just the main triples.
        """
        return sum(1 for _ in self.graph.subjects(RDF.type, RDF.Statement))

    def get_short_term_memory_count(self) -> int:
        """
        Count the number of short-term memories in the graph.
        Short-term memories are reified statements that have the 'currentTime' qualifier.

        Returns:
            int: The count of short-term memories.
        """
        short_term_count = 0
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            if self.graph.value(
                statement, URIRef("https://humem.ai/ontology/currentTime")
            ):
                short_term_count += 1
        return short_term_count

    def get_long_term_memory_count(self) -> int:
        """
        Count the number of long-term memories in the graph.
        Long-term memories are reified statements that do NOT have the 'currentTime' qualifier.

        Returns:
            int: The count of long-term memories.
        """
        long_term_count = 0
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            if not self.graph.value(
                statement, URIRef("https://humem.ai/ontology/currentTime")
            ):
                long_term_count += 1
        return long_term_count

    def get_long_term_episodic_memory_count(self) -> int:
        """
        Count the number of long-term episodic memories in the graph.
        Long-term episodic memories are reified statements that have the 'time' qualifier.

        Returns:
            int: The count of long-term episodic memories.
        """
        episodic_count = 0
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            if not self.graph.value(
                statement, URIRef("https://humem.ai/ontology/currentTime")
            ):
                if self.graph.value(
                    statement, URIRef("https://humem.ai/ontology/time")
                ):
                    episodic_count += 1
        return episodic_count

    def get_long_term_semantic_memory_count(self) -> int:
        """
        Count the number of long-term semantic memories in the graph.
        Long-term semantic memories are reified statements that do NOT have the 'time' qualifier.

        Returns:
            int: The count of long-term semantic memories.
        """
        semantic_count = 0
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            if not self.graph.value(
                statement, URIRef("https://humem.ai/ontology/currentTime")
            ):
                if not self.graph.value(
                    statement, URIRef("https://humem.ai/ontology/time")
                ):
                    semantic_count += 1
        return semantic_count

    def modify_strength(
        self, filters: dict, increment_by: int = None, multiply_by: float = None
    ):
        """
        Modify the strength of long-term memories by incrementing/decrementing or multiplying.

        Args:
            filters (dict): Filters to identify the memory, including subject, predicate, object, and any qualifiers.
            increment_by (int, optional): Increment or decrement the strength value by this amount.
            multiply_by (float, optional): Multiply the strength value by this factor. Rounded to the nearest integer.
        """
        logger.debug(
            f"Modifying strength with filters: {filters}, increment_by: {increment_by}, multiply_by: {multiply_by}"
        )

        subject_filter = filters.get("subject")

        # SPARQL query to retrieve all reified statements with the same subject, predicate, and object, that have a strength qualifier
        query = f"""
        PREFIX humemai: <https://humem.ai/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?subject ?predicate ?object ?strength
        WHERE {{
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object ;
                    humemai:strength ?strength .
            FILTER(?subject = <{subject_filter}>)
        }}
        """

        logger.debug(f"Executing SPARQL query:\n{query}")

        # Execute the query
        results = self.graph.query(query)

        # Iterate over the matching results and modify the strength for each reified statement
        for row in results:
            statement = row.statement
            current_strength = int(row.strength)
            new_strength = current_strength

            logger.debug(
                f"Processing statement: {statement}, current strength: {current_strength}"
            )

            # Apply increment/decrement if specified
            if increment_by is not None:
                new_strength += increment_by
                logger.debug(
                    f"Strength incremented by {increment_by}. New value: {new_strength}"
                )

            # Apply multiplication if specified
            if multiply_by is not None:
                new_strength = round(current_strength * multiply_by)
                logger.debug(
                    f"Strength multiplied by {multiply_by}. New value: {new_strength}"
                )

            # Ensure the strength remains a positive integer
            new_strength = max(new_strength, 0)

            # Update the graph with the new strength value for this statement
            self.graph.set(
                (
                    statement,
                    URIRef("https://humem.ai/ontology/strength"),
                    Literal(new_strength, datatype=XSD.integer),
                )
            )
            logger.debug(
                f"Updated strength for statement: {statement} to {new_strength}"
            )

    def modify_episodic_event(
        self,
        upper_time_bound: str,
        lower_time_bound: str,
        new_event: str,
        subject: URIRef = None,
        predicate: URIRef = None,
        object_: URIRef = None,
        location: str = None,
        emotion: str = None,
    ):
        """
        Modify the event value for episodic memories that fall within a specific time range and optional filters.

        Args:
            upper_time_bound (str): The upper bound for time filtering (ISO format).
            lower_time_bound (str): The lower bound for time filtering (ISO format).
            new_event (str): The new value for the event qualifier.
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            location (str, optional): Filter by location value.
            emotion (str, optional): Filter by emotion value.
        """
        # SPARQL query to find all reified statements with filters and time bounds
        query = """
        PREFIX humemai: <https://humem.ai/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?subject ?predicate ?object
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object ;
                    humemai:time ?time .
        """

        # Add filters dynamically based on input
        if subject is not None:
            query += f"FILTER(?subject = <{subject}>) .\n"
        if predicate is not None:
            query += f"FILTER(?predicate = <{predicate}>) .\n"
        if object_ is not None:
            query += f"FILTER(?object = <{object_}>) .\n"
        if location is not None:
            query += f'?statement humemai:location "{location}" .\n'
        if emotion is not None:
            query += f'?statement humemai:emotion "{emotion}" .\n'

        # Add time range filter
        query += f"""
        FILTER(?time >= "{lower_time_bound}"^^xsd:dateTime && ?time <= "{upper_time_bound}"^^xsd:dateTime) .
        }}

        """

        logger.debug(f"Executing SPARQL query to find episodic memories:\n{query}")

        # Execute the SPARQL query
        results = self.graph.query(query)

        # Modify the event value for all matching reified statements
        for row in results:
            statement = row.statement

            # Log the statement that will be modified
            logger.debug(f"Modifying event for statement: {statement}")

            # Set the new event value for each matching reified statement
            self.graph.set(
                (
                    statement,
                    URIRef("https://humem.ai/ontology/event"),
                    Literal(new_event),
                )
            )
            logger.debug(f"Set new event '{new_event}' for statement: {statement}")

    def increment_recalled(
        self,
        subject: URIRef = None,
        predicate: URIRef = None,
        object_: URIRef = None,
        location: str = None,
        emotion: str = None,
        derivedFrom: str = None,
        strength: int = None,
        lower_time_bound: str = None,
        upper_time_bound: str = None,
    ):
        """
        Increment the 'recalled' value for memories (episodic or semantic) that match the filters.

        Args:
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            location (str, optional): Filter by location value.
            emotion (str, optional): Filter by emotion value.
            derivedFrom (str, optional): Filter by derivedFrom value.
            strength (int, optional): Filter by strength value.
            lower_time_bound (str, optional): Lower bound for time filtering (ISO format).
            upper_time_bound (str, optional): Upper bound for time filtering (ISO format).
        """

        # SPARQL query to find reified statements with optional filters
        query = """
        PREFIX humemai: <https://humem.ai/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?subject ?predicate ?object ?recalled
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object .
        """

        # Add filters dynamically based on input
        if subject is not None:
            query += f"FILTER(?subject = <{subject}>) .\n"
        if predicate is not None:
            query += f"FILTER(?predicate = <{predicate}>) .\n"
        if object_ is not None:
            query += f"FILTER(?object = <{object_}>) .\n"
        if location is not None:
            query += f'?statement humemai:location "{location}" .\n'
        if emotion is not None:
            query += f'?statement humemai:emotion "{emotion}" .\n'
        if derivedFrom is not None:
            query += f'?statement humemai:derivedFrom "{derivedFrom}" .\n'
        if strength is not None:
            query += f'?statement humemai:strength "{strength}" .\n'

        # Add time filtering logic (both currentTime and time)
        if lower_time_bound and upper_time_bound:
            time_filter = f"""
            OPTIONAL {{ ?statement humemai:currentTime ?currentTime }}
            OPTIONAL {{ ?statement humemai:time ?time }}
            FILTER((?currentTime >= "{lower_time_bound}"^^xsd:dateTime && ?currentTime <= "{upper_time_bound}"^^xsd:dateTime) ||
                (?time >= "{lower_time_bound}"^^xsd:dateTime && ?time <= "{upper_time_bound}"^^xsd:dateTime)) .
            """
            query += time_filter

        # Add OPTIONAL to retrieve the recalled value (if it exists)
        query += """
        OPTIONAL { ?statement humemai:recalled ?recalled }
        }
        """

        logger.debug(f"Executing SPARQL query:\n{query}")

        # Execute the SPARQL query to retrieve matching reified statements
        results = self.graph.query(query)

        # Iterate through the results to increment the recalled value
        for row in results:
            statement = row.statement
            current_recalled_value = row.recalled

            if current_recalled_value is not None:
                # If 'recalled' exists, increment it by 1
                new_recalled_value = int(current_recalled_value) + 1
            else:
                # If 'recalled' does not exist, initialize it to 1 (default is 0)
                new_recalled_value = 1

            # Update or add the new 'recalled' value in the graph
            self.graph.set(
                (
                    statement,
                    URIRef("https://humem.ai/ontology/recalled"),
                    Literal(new_recalled_value, datatype=XSD.integer),
                )
            )

            logger.debug(
                f"Updated recalled for statement {statement} to {new_recalled_value}"
            )

    def __repr__(self) -> str:
        if self.verbose_repr:
            memory_strings = []
            for statement in self.graph.subjects(RDF.type, RDF.Statement):
                subj = self.graph.value(statement, RDF.subject)
                pred = self.graph.value(statement, RDF.predicate)
                obj = self.graph.value(statement, RDF.object)
                qualifiers = {}

                for q_pred, q_obj in self.graph.predicate_objects(statement):
                    if q_pred not in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                        qualifiers[str(q_pred)] = str(q_obj)

                memory_strings.append(f"[{subj}, {pred}, {obj}, {qualifiers}]")

            return "\n".join(memory_strings)
        else:
            memory_strings = []
            for statement in self.graph.subjects(RDF.type, RDF.Statement):
                subj = self._strip_namespace(self.graph.value(statement, RDF.subject))
                pred = self._strip_namespace(self.graph.value(statement, RDF.predicate))
                obj = self._strip_namespace(self.graph.value(statement, RDF.object))
                qualifiers = {}

                for q_pred, q_obj in self.graph.predicate_objects(statement):
                    if q_pred not in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                        qualifiers[self._strip_namespace(q_pred)] = str(q_obj)

                memory_strings.append(f"[{subj}, {pred}, {obj}, {qualifiers}]")

            return "\n".join(memory_strings)

    def _strip_namespace(self, uri):
        """
        Helper function to strip the namespace and return the last part of a URIRef.

        Args:
            uri (URIRef or Literal): The URIRef to process.

        Returns:
            str: The last part of the URI after the last '/' or '#'.
        """
        if isinstance(uri, URIRef):
            return uri.split("/")[-1].split("#")[-1]
        return str(uri)

    def is_reified_statement_short_term(self, statement) -> bool:
        """
        Check if a given reified statement is a short-term memory by verifying if it
        has a 'currentTime' qualifier.

        Args:
            statement (BNode or URIRef): The reified statement to check.

        Returns:
            bool: True if it's a short-term memory, False otherwise.
        """
        current_time = self.graph.value(
            statement, URIRef("https://humem.ai/ontology/currentTime")
        )
        return current_time is not None

    def _add_reified_statement_to_working_memory_and_increment_recall(
        self,
        subj: URIRef,
        pred: URIRef,
        obj: URIRef,
        working_memory: "Memory",
        specific_statement=None,
    ):
        """
        Helper method to add all reified statements (including qualifiers) of a given triple
        to the working memory and increment the recall count for each reified statement.

        Args:
            subj (URIRef): Subject of the triple.
            pred (URIRef): Predicate of the triple.
            obj (URIRef): Object of the triple.
            working_memory (Memory): The working memory to which the statements and qualifiers are added.
            specific_statement (BNode or URIRef, optional): A specific reified statement to process, if provided.
        """
        # Find all reified statements for this triple
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            s = self.graph.value(statement, RDF.subject)
            p = self.graph.value(statement, RDF.predicate)
            o = self.graph.value(statement, RDF.object)

            if (
                s == subj
                and p == pred
                and o == obj
                and (specific_statement is None or statement == specific_statement)
            ):
                logger.debug(f"Processing reified statement: {statement}")

                # Retrieve the current recalled value
                recalled_value = 0
                for _, _, recalled in self.graph.triples(
                    (statement, URIRef("https://humem.ai/ontology/recalled"), None)
                ):
                    recalled_value = int(recalled)

                # Increment the recalled value in the long-term memory for this reified statement
                new_recalled_value = recalled_value + 1
                self.graph.set(
                    (
                        statement,
                        URIRef("https://humem.ai/ontology/recalled"),
                        Literal(new_recalled_value, datatype=XSD.integer),
                    )
                )
                logger.debug(
                    f"Updated recalled for statement {statement} to {new_recalled_value}"
                )

                # Now, add the updated reified statement to the working memory
                for stmt_p, stmt_o in self.graph.predicate_objects(statement):
                    if stmt_p == URIRef("https://humem.ai/ontology/recalled"):
                        working_memory.graph.add(
                            (
                                statement,
                                stmt_p,
                                Literal(new_recalled_value, datatype=XSD.integer),
                            )
                        )
                        logger.debug(
                            f"Added updated recalled value ({new_recalled_value}) to working memory for statement: {statement}"
                        )
                    else:
                        working_memory.graph.add((statement, stmt_p, stmt_o))
                        logger.debug(
                            f"Added reified statement triple to working memory: ({statement}, {stmt_p}, {stmt_o})"
                        )

    def get_short_term_memories(self) -> "Memory":
        """
        Query the RDF graph to retrieve all short-term memories with a currentTime qualifier
        and include all associated qualifiers (e.g., location, emotion, etc.).

        Returns:
            Memory: A Memory object containing all short-term memories with their qualifiers.
        """
        short_term_memory = Memory(self.verbose_repr)

        # SPARQL query to retrieve all reified statements with a currentTime qualifier, along with other qualifiers
        query = """
        PREFIX humemai: <https://humem.ai/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?statement ?subject ?predicate ?object ?qualifier_pred ?qualifier_obj
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object ;
                    humemai:currentTime ?currentTime .
            OPTIONAL { ?statement ?qualifier_pred ?qualifier_obj }
        }
        """

        logger.debug(
            f"Executing SPARQL query to retrieve short-term memories:\n{query}"
        )
        results = self.graph.query(query)

        # Dictionary to store reified statements and their qualifiers
        statement_dict = {}

        # Iterate through the results and organize the qualifiers for each reified statement
        for row in results:
            subj = row.subject
            pred = row.predicate
            obj = row.object
            statement = row.statement
            qualifier_pred = row.qualifier_pred
            qualifier_obj = row.qualifier_obj

            # Add the main triple to the memory graph if it's not already added
            if statement not in statement_dict:
                statement_dict[statement] = {
                    "triple": (subj, pred, obj),
                    "qualifiers": {},
                }

            # Store the qualifiers, excluding standard reification elements
            if qualifier_pred and qualifier_pred not in (
                RDF.type,
                RDF.subject,
                RDF.predicate,
                RDF.object,
            ):
                statement_dict[statement]["qualifiers"][qualifier_pred] = qualifier_obj

        # Populate the short-term memory object with triples and qualifiers
        for statement, data in statement_dict.items():
            subj, pred, obj = data["triple"]
            qualifiers = data["qualifiers"]

            # Add the main triple to the memory
            short_term_memory.graph.add((subj, pred, obj))

            # Create a reified statement and add all the qualifiers
            reified_statement = BNode()
            short_term_memory.graph.add((reified_statement, RDF.type, RDF.Statement))
            short_term_memory.graph.add((reified_statement, RDF.subject, subj))
            short_term_memory.graph.add((reified_statement, RDF.predicate, pred))
            short_term_memory.graph.add((reified_statement, RDF.object, obj))

            # Add each qualifier to the reified statement
            for qualifier_pred, qualifier_obj in qualifiers.items():
                short_term_memory.graph.add(
                    (reified_statement, qualifier_pred, qualifier_obj)
                )

        return short_term_memory

    def get_long_term_memories(self) -> "Memory":
        """
        Retrieve all long-term memories from the RDF graph.
        Long-term memories are identified by the absence of a 'currentTime' qualifier.

        Returns:
            Memory: A new Memory object containing all long-term memories (episodic and semantic).
        """
        long_term_memory = Memory(self.verbose_repr)

        # SPARQL query to retrieve all reified statements that do not have a currentTime qualifier
        query = """
        PREFIX humemai: <https://humem.ai/ontology/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?statement ?subject ?predicate ?object
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object .
            FILTER NOT EXISTS { ?statement humemai:currentTime ?currentTime }
        }
        """

        logger.debug(f"Executing SPARQL query to retrieve long-term memories:\n{query}")
        results = self.graph.query(query)

        # Add the resulting triples to the new Memory object (long-term memory)
        for row in results:
            subj = row.subject
            pred = row.predicate
            obj = row.object

            # Add the main triple to the long-term memory graph
            long_term_memory.graph.add((subj, pred, obj))

            # Create a reified statement and add it
            reified_statement = BNode()
            long_term_memory.graph.add((reified_statement, RDF.type, RDF.Statement))
            long_term_memory.graph.add((reified_statement, RDF.subject, subj))
            long_term_memory.graph.add((reified_statement, RDF.predicate, pred))
            long_term_memory.graph.add((reified_statement, RDF.object, obj))

            # Now, add all qualifiers (excluding 'currentTime')
            for qualifier_pred, qualifier_obj in self.graph.predicate_objects(
                row.statement
            ):
                if qualifier_pred != humemai.currentTime:
                    long_term_memory.graph.add(
                        (reified_statement, qualifier_pred, qualifier_obj)
                    )

        return long_term_memory

    def load_from_ttl(self, ttl_file: str):
        """
        Load memory data from a Turtle (.ttl) file into the RDF graph.

        Args:
            ttl_file (str): Path to the Turtle file to load.
        """
        if not os.path.exists(ttl_file):
            raise FileNotFoundError(f"Turtle file not found: {ttl_file}")

        logger.info(f"Loading memory from TTL file: {ttl_file}")
        self.graph.parse(ttl_file, format="ttl")
        logger.info(f"Memory loaded from {ttl_file} successfully.")

    def save_to_ttl(self, ttl_file: str):
        """
        Save the current memory graph to a Turtle (.ttl) file.

        Args:
            ttl_file (str): Path to the Turtle file to save.
        """
        logger.info(f"Saving memory to TTL file: {ttl_file}")
        with open(ttl_file, "w") as f:
            f.write(self.graph.serialize(format="ttl"))
        logger.info(f"Memory saved to {ttl_file} successfully.")

    def iterate_memories(self, memory_type=None):
        """
        Iterate over memories in the graph, filtered by memory type (short-term, long-term, episodic, semantic, or all).

        Args:
            memory_type (str, optional): The type of memory to iterate over.
            Valid values: "short_term", "long_term", "episodic", "semantic", or "all".
            - "short_term": Short-term memories (with 'currentTime').
            - "long_term": Long-term memories (without 'currentTime').
            - "episodic": Long-term episodic memories (with 'time').
            - "semantic": Long-term semantic memories (without 'time' and 'currentTime').
            - "all": Iterate over all memories (both short-term and long-term).
            If None, defaults to "all".

        Yields:
            tuple: (subject, predicate, object, qualifiers) for each memory that matches the criteria.
        """
        valid_types = ["all", "short_term", "long_term", "episodic", "semantic"]

        # Default to "all" if no memory_type is provided
        if memory_type is None:
            memory_type = "all"

        assert (
            memory_type in valid_types
        ), f"Invalid memory_type. Valid values: {valid_types}"

        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            subj = self.graph.value(statement, RDF.subject)
            pred = self.graph.value(statement, RDF.predicate)
            obj = self.graph.value(statement, RDF.object)

            # Retrieve qualifiers for the statement
            qualifiers = {}
            for q_pred, q_obj in self.graph.predicate_objects(statement):
                if q_pred not in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                    qualifiers[str(q_pred)] = str(q_obj)

            # Determine the type of memory
            current_time = self.graph.value(statement, humemai.currentTime)
            time = self.graph.value(statement, humemai.time)
            derived_from = self.graph.value(statement, humemai.derivedFrom)

            # Filter based on the memory_type argument
            if memory_type == "short_term" and current_time:
                # Short-term memory has currentTime
                yield (subj, pred, obj, qualifiers)
            elif memory_type == "long_term" and not current_time:
                # Long-term memory does not have currentTime
                yield (subj, pred, obj, qualifiers)
            elif memory_type == "episodic" and not current_time and time:
                # Episodic memory is long-term (no currentTime) and has time
                yield (subj, pred, obj, qualifiers)
            elif (
                memory_type == "semantic"
                and not current_time
                and not time
                and derived_from
            ):
                # Semantic memory is long-term (no currentTime, no time) and has derivedFrom
                yield (subj, pred, obj, qualifiers)
            elif memory_type == "all":
                # All memories, regardless of type
                yield (subj, pred, obj, qualifiers)

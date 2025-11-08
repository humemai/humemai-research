"""Memory Class with RDFLib"""

import collections
import logging
import os
from datetime import datetime

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("humemai.memory")

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class Memory:
    """
    Memory class for managing both short-term and long-term memories.
    Provides methods to add, retrieve, delete, cluster, and manage memories in the RDF graph.
    """

    def __init__(self):
        # Initialize RDF graph for memory storage
        self.graph = Graph()
        self.graph.bind("humemai", humemai)
        self.current_statement_id = 0  # Counter to track the next unique ID

    def add_memory(self, triples: list, qualifiers: dict = {}):
        """

        Add a reified statement to the RDF graph, with the main triple and optional
        qualifiers. You have to make sure that everything is in the right
        URIRef format.

        Args:
            triples (list): A list of triples (subject, predicate, object) to be added.
            qualifiers (dict): A dictionary of qualifiers (e.g., location, currentTime).
        """
        for subj, pred, obj in triples:
            logger.debug(f"Adding triple: ({subj}, {pred}, {obj})")

            # Add the main triple [subject, predicate, object] if it doesn't already
            # exist
            if not (subj, pred, obj) in self.graph:
                self.graph.add((subj, pred, obj))
                logger.debug(f"Main triple added: ({subj}, {pred}, {obj})")
            else:
                logger.debug(f"Main triple already exists: ({subj}, {pred}, {obj})")

            # Create a new reified statement to attach the qualifiers, and assign a
            # unique ID
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
                if not isinstance(key, URIRef):
                    raise ValueError(f"Qualifier key {key} must be a URIRef.")
                if not isinstance(value, (URIRef, Literal)):
                    raise ValueError(
                        f"Qualifier value {value} must be a URIRef or Literal."
                    )
                # assumes that both key and value are in URIRef format
                self.graph.add((statement, key, value))
                logger.debug(f"Added qualifier: ({statement}, {key}, {value})")

    def delete_memory(self, memory_id: Literal):
        """
        Delete a memory (reified statement) by its unique ID, including all associated
        qualifiers.

        Args:
            memory_id (Literal): The unique ID of the memory to be deleted.
        """
        logger.debug(f"Deleting memory with ID: {memory_id}")

        if not isinstance(memory_id, Literal) or memory_id.datatype != XSD.integer:
            raise ValueError(f"memory_id must be a Literal with datatype XSD.integer")

        # Find the reified statement with the given ID
        statement = None
        for stmt in self.graph.subjects(humemai.memoryID, memory_id):
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

    def get_memory_by_id(self, memory_id: Literal) -> dict:
        """
        Retrieve a memory (reified statement) by its unique ID and return its details.

        Args:
            memory_id (Literal): The unique ID of the memory to retrieve.

        Returns:
            dict: A dictionary with the memory details (subject, predicate, object,
            qualifiers).
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
                    qualifiers[q_pred] = q_obj

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

    def add_short_term_memory(self, triples: list, qualifiers: dict = {}):
        """
        Add short-term memories to the RDF graph, enforcing required qualifiers.

        Args:
            triples (list): A list of triples to add.
            qualifiers (dict, optional): Additional qualifiers to add.

        """
        if humemai.currentTime not in qualifiers:
            currentTime = Literal(datetime.now().isoformat(), datatype=XSD.dateTime)
            qualifiers[humemai.currentTime] = currentTime
        else:
            if qualifiers[humemai.currentTime].datatype != XSD.dateTime:
                raise ValueError(
                    f"Invalid currentTime format: {qualifiers[humemai.currentTime]}"
                )

        self.add_memory(triples, qualifiers)

    def add_episodic_memory(
        self,
        triples: list,
        qualifiers: dict = {},
        event_properties: dict = {},
    ):
        """
        Add episodic memories to the RDF graph, enforcing required qualifiers.

        Args:
            triples (list): A list of triples to add.
            qualifiers (dict, optional): Additional qualifiers to add.
                The qualifiers can have the following in URIRef format:
                https://humem.ai/ontology#eventTime: str,
                https://humem.ai/ontology#location: str,
                https://humem.ai/ontology#emotion: str,
                https://humem.ai/ontology#event: str,
            event_properties (dict, optional): Additional properties for the event node.
                The properties should be URIRef format.

        """
        if humemai.currentTime in qualifiers:
            raise ValueError("CurrentTime is not allowed for episodic memories")
        if humemai.knownSince in qualifiers:
            raise ValueError("KnownSince is not allowed for episodic memories")
        if humemai.strength in qualifiers:
            raise ValueError("Strength is not allowed for episodic memories")
        if humemai.derivedFrom in qualifiers:
            raise ValueError("DerivedFrom is not allowed for episodic memories")
        if humemai.eventTime not in qualifiers:
            raise ValueError("Missing required qualifier: eventTime")

        if qualifiers[humemai.eventTime].datatype != XSD.dateTime:
            raise ValueError(
                f"Invalid eventTime format: {qualifiers[humemai.eventTime]}"
            )

        # Add required qualifiers
        qualifiers = {humemai.recalled: Literal(0, datatype=XSD.integer), **qualifiers}
        self.add_memory(triples, qualifiers)

        if humemai.event in qualifiers:

            self.add_event(qualifiers[humemai.event])

            if event_properties:
                self.add_event_properties(qualifiers[humemai.event], event_properties)

    def add_event(self, event: URIRef):
        """
        Create an event node in the RDF graph with custom properties.

        Args:
            event (str): The name of the event.
        """
        # Create the event node (as a node in the graph)
        if (event, None, None) not in self.graph:
            self.graph.add((event, RDF.type, humemai.Event))
            logger.debug(f"Event node created: {event}")

    def add_event_properties(self, event: URIRef, event_properties: dict):
        """
        Add properties to an existing event node in the RDF graph.

        Args:
            event (URIRef): The name of the event.
            event_properties (dict): Additional properties for the event node.
        """
        # Add custom event properties to the event node
        if event_properties:
            for prop, value in event_properties.items():
                self.graph.add(
                    (
                        event,
                        prop,
                        value,
                    )
                )
                logger.debug(f"Added event property: [{event}, {prop}, {value}]")

    def add_semantic_memory(self, triples: list, qualifiers: dict = {}):
        """
        Add semantic memories to the RDF graph, enforcing required qualifiers.

        Args:
            triples (list): A list of triples to add.
            qualifiers (dict, optional): Additional qualifiers to add.
                The qualifiers can have the following
                https://humem.ai/ontology#knownSince: str,
                https://humem.ai/ontology#derivedFrom: str,
                https://humem.ai/ontology#strength: int,


        """
        if humemai.emotion in qualifiers:
            raise ValueError("Emotion is not allowed for semantic memories")
        if humemai.location in qualifiers:
            raise ValueError("Location is not allowed for semantic memories")
        if humemai.event in qualifiers:
            raise ValueError("Event is not allowed for semantic memories")
        if humemai.eventTime in qualifiers:
            raise ValueError("EventTime is not allowed for semantic memories")
        if humemai.currentTime in qualifiers:
            raise ValueError("CurrentTime is not allowed for semantic memories")
        if humemai.knownSince not in qualifiers:
            raise ValueError("Missing required qualifier: knownSince")

        if qualifiers[humemai.knownSince].datatype != XSD.dateTime:
            raise ValueError(
                f"Invalid knownSince format: {qualifiers[humemai.knownSince]}"
            )

        # Add required qualifiers
        qualifiers = {humemai.recalled: Literal(0, datatype=XSD.integer), **qualifiers}
        self.add_memory(triples, qualifiers)

    def get_memories(
        self,
        subject: URIRef = None,
        predicate: URIRef = None,
        object_: URIRef = None,
        qualifiers: dict = {},
        lower_time_bound: Literal = None,
        upper_time_bound: Literal = None,
    ) -> "Memory":
        """
        Retrieve memories with optional filtering based on the qualifiers and triple
        values, including time bounds.

        Args:
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            qualifiers (dict, optional): Additional qualifiers to filter by.
            lower_time_bound (Literal, optional): Lower bound for time filtering (ISO format).
            upper_time_bound (Literal, optional): Upper bound for time filtering (ISO format).

        Returns:
            Memory: A new Memory object containing the filtered memories.
        """

        # SPARQL query to retrieve memories with optional filters
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
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

        # Add qualifier filters
        for key, value in qualifiers.items():
            query += f"?statement {key.n3()} {value.n3()} .\n"

        # Add time filtering logic (for currentTime, eventTime, and knownSince)
        if lower_time_bound and upper_time_bound:
            time_filter = f"""
            OPTIONAL {{ ?statement humemai:currentTime ?currentTime }}
            OPTIONAL {{ ?statement humemai:eventTime ?eventTime }}
            OPTIONAL {{ ?statement humemai:knownSince ?knownSince }}
            FILTER(
                (?currentTime >= {lower_time_bound.n3()} && ?currentTime <= {upper_time_bound.n3()}) ||
                (?eventTime >= {lower_time_bound.n3()} && ?eventTime <= {upper_time_bound.n3()}) ||
                (?knownSince >= {lower_time_bound.n3()} && ?knownSince <= {upper_time_bound.n3()})
            ) .
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
                    statement_dict[statement]["qualifiers"][
                        qualifier_pred
                    ] = qualifier_obj

        # Create a new Memory object to store the filtered results
        filtered_memory = Memory()

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
                    (new_statement, qualifier_pred, qualifier_obj)
                )

        return filtered_memory

    def get_raw_triple_count(self) -> int:
        """
        Count the number of raw triples in the RDF graph.

        Returns:
            int: The count of raw triples.
        """
        return len(self.graph)

    def get_main_triple_count_except_event(self) -> int:
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
        Short-term memories are reified statements that have the 'currentTime'
        qualifier.

        Returns:
            int: The count of short-term memories.
        """
        short_term_count = 0
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            if self.graph.value(statement, humemai.currentTime):
                short_term_count += 1
        return short_term_count

    def get_long_term_episodic_memory_count(self) -> int:
        """
        Count the number of long-term episodic memories in the graph.
        Long-term episodic memories are reified statements that have the 'eventTime'
        qualifier.

        Returns:
            int: The count of long-term episodic memories.
        """
        episodic_count = 0
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            if self.graph.value(statement, humemai.eventTime):
                episodic_count += 1

        return episodic_count

    def get_long_term_semantic_memory_count(self) -> int:
        """
        Count the number of long-term semantic memories in the graph.
        Long-term semantic memories are reified statements that have the 'knownSince'
        qualifier.
        """
        semantic_count = 0
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            if self.graph.value(statement, humemai.knownSince):
                semantic_count += 1

        return semantic_count

    def get_long_term_memory_count(self) -> int:
        """
        Count the number of long-term memories in the graph.
        Long-term memories are reified statements that do NOT have the 'currentTime'
        qualifier.

        Returns:
            int: The count of long-term memories.
        """
        return (
            self.get_long_term_episodic_memory_count()
            + self.get_long_term_semantic_memory_count()
        )

    def get_event_count(self):
        """
        Count the number of Event instances in the RDF graph.

        Returns:
            int: The number of Event instances.
        """
        event_count = 0
        # Query all subjects that are of type humemai.Event
        for event in self.graph.subjects(RDF.type, humemai.Event):
            event_count += 1

        return event_count

    def modify_strength(
        self, filters: dict, increment_by: int = None, multiply_by: float = None
    ):
        """
        Modify the strength of long-term memories by incrementing/decrementing or
        multiplying.

        Args:
            filters (dict): Filters to identify the memory, including subject,
            predicate, object, and any qualifiers.
            increment_by (int, optional): Increment or decrement the strength value by
            this amount.
            multiply_by (float, optional): Multiply the strength value by this factor.
            Rounded to the nearest integer.
        """
        logger.debug(
            f"Modifying strength with filters: {filters}, increment_by: {increment_by}, multiply_by: {multiply_by}"
        )

        subject_filter = filters.get(RDF.subject)

        # SPARQL query to retrieve all reified statements with the same subject,
        # predicate, and object, that have a strength qualifier
        query = f"""
        PREFIX humemai: <https://humem.ai/ontology#>
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
                    URIRef("https://humem.ai/ontology#strength"),
                    Literal(new_strength, datatype=XSD.integer),
                )
            )
            logger.debug(
                f"Updated strength for statement: {statement} to {new_strength}"
            )

    def modify_episodic_event(
        self,
        upper_time_bound: Literal,
        lower_time_bound: Literal,
        new_event: str,
        subject: URIRef = None,
        predicate: URIRef = None,
        object_: URIRef = None,
        qualifiers: dict = {},
    ):
        """
        Modify the event value for episodic memories that fall within a specific time
        range and optional filters.

        Args:
            upper_time_bound (Literal): The upper bound for time filtering (ISO format).
            lower_time_bound (Literal): The lower bound for time filtering (ISO format).
            new_event (str): The new value for the event qualifier.
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            qualifiers (dict, optional): Additional qualifiers to filter by.
        """
        # Start building the SPARQL query to find all reified statements with filters and time bounds
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?subject ?predicate ?object
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object ;
                    humemai:eventTime ?eventTime .
        """

        # Add filters dynamically based on input using .n3() to ensure proper formatting
        if subject is not None:
            query += f"FILTER(?subject = {subject.n3()}) .\n"
        if predicate is not None:
            query += f"FILTER(?predicate = {predicate.n3()}) .\n"
        if object_ is not None:
            query += f"FILTER(?object = {object_.n3()}) .\n"

        # Add filters for any additional qualifiers provided in the dictionary
        for qualifier_pred, qualifier_obj in qualifiers.items():
            query += (
                f"FILTER(?statement {qualifier_pred.n3()} {qualifier_obj.n3()}) .\n"
            )

        # Add time range filter
        query += f"""
        FILTER(?eventTime >= {lower_time_bound.n3()} && ?eventTime <= {upper_time_bound.n3()}) .
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
                    URIRef("https://humem.ai/ontology#event"),
                    Literal(new_event),
                )
            )
            logger.debug(f"Set new event '{new_event}' for statement: {statement}")

            # Create the event node if it doesn't exist
            self.add_event(new_event)

    def increment_recalled(
        self,
        subject: URIRef = None,
        predicate: URIRef = None,
        object_: URIRef = None,
        qualifiers: dict = {},
        lower_time_bound: Literal = None,
        upper_time_bound: Literal = None,
    ):
        """
        Increment the 'recalled' value for memories (episodic or semantic) that match
        the filters.

        Args:
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            qualifiers (dict, optional): Additional qualifiers to filter by.
            lower_time_bound (Literal, optional): Lower bound for time filtering (ISO format).
            upper_time_bound (Literal, optional): Upper bound for time filtering (ISO format).
        """

        # Start building the SPARQL query to find reified statements with filters
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?subject ?predicate ?object ?recalled
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object .
        """

        # Add filters dynamically based on input using .n3() to format correctly
        if subject is not None:
            query += f"FILTER(?subject = {subject.n3()}) .\n"
        if predicate is not None:
            query += f"FILTER(?predicate = {predicate.n3()}) .\n"
        if object_ is not None:
            query += f"FILTER(?object = {object_.n3()}) .\n"

        # Add filters for any additional qualifiers provided in the dictionary
        for qualifier_pred, qualifier_obj in qualifiers.items():
            query += (
                f"FILTER(?statement {qualifier_pred.n3()} {qualifier_obj.n3()}) .\n"
            )

        # Add time filtering logic (for currentTime, eventTime, and knownSince)
        if lower_time_bound and upper_time_bound:
            time_filter = f"""
            OPTIONAL {{ ?statement humemai:currentTime ?currentTime }}
            OPTIONAL {{ ?statement humemai:eventTime ?eventTime }}
            OPTIONAL {{ ?statement humemai:knownSince ?knownSince }}
            FILTER(
                (?currentTime >= {lower_time_bound.n3()} && ?currentTime <= {upper_time_bound.n3()}) ||
                (?eventTime >= {lower_time_bound.n3()} && ?eventTime <= {upper_time_bound.n3()}) ||
                (?knownSince >= {lower_time_bound.n3()} && ?knownSince <= {upper_time_bound.n3()})
            ) .
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
                    URIRef("https://humem.ai/ontology#recalled"),
                    Literal(new_recalled_value, datatype=XSD.integer),
                )
            )

            logger.debug(
                f"Updated recalled for statement {statement} to {new_recalled_value}"
            )

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
        currentTime = self.graph.value(statement, humemai.currentTime)
        return currentTime is not None

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
                    (statement, humemai.recalled, None)
                ):
                    recalled_value = int(recalled)

                # Increment the recalled value in the long-term memory for this reified statement
                new_recalled_value = recalled_value + 1
                self.graph.set(
                    (
                        statement,
                        humemai.recalled,
                        Literal(new_recalled_value, datatype=XSD.integer),
                    )
                )
                logger.debug(
                    f"Updated recalled for statement {statement} to {new_recalled_value}"
                )

                # Now, add the updated reified statement to the working memory
                for stmt_p, stmt_o in self.graph.predicate_objects(statement):
                    if stmt_p == humemai.recalled:
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
        Query the RDF graph to retrieve all short-term memories with a currentTime
        qualifier and include all associated qualifiers (e.g., location, emotion, etc.).

        Returns:
            Memory: A Memory object containing all short-term memories with their qualifiers.
        """
        short_term_memory = Memory()

        # SPARQL query to retrieve all reified statements with a currentTime qualifier, along with other qualifiers
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
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
        Long-term memories are identified by the presence of either 'eventTime' or 'knownSince'
        qualifiers and the absence of a 'currentTime' qualifier.

        Returns:
            Memory: A new Memory object containing all long-term memories (episodic and
            semantic).
        """
        long_term_memory = Memory()

        # SPARQL query to retrieve all reified statements that have either eventTime or knownSince,
        # and do not have a currentTime qualifier
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?statement ?subject ?predicate ?object
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object .
            FILTER NOT EXISTS { ?statement humemai:currentTime ?currentTime }
            {
                FILTER EXISTS { ?statement humemai:eventTime ?eventTime }
            }
            UNION
            {
                FILTER EXISTS { ?statement humemai:knownSince ?knownSince }
            }
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
        Iterate over memories in the graph, filtered by memory type (short-term,
        long-term, episodic, semantic, or all).

        Args:
            memory_type (str, optional): The type of memory to iterate over.
            Valid values: "short_term", "long_term", "episodic", "semantic", or "all".
            - "short_term": Short-term memories (with 'currentTime').
            - "long_term": Long-term memories (with 'eventTime' or 'knownSince', but without 'currentTime').
            - "episodic": Long-term episodic memories (with 'eventTime', but without 'knownSince' and 'currentTime').
            - "semantic": Long-term semantic memories (with 'knownSince', but without 'eventTime' and 'currentTime').
            - "all": Iterate over all memories (both short-term and long-term).
            If None, defaults to "all".

        Yields:
            tuple: (subject, predicate, object, qualifiers) for each memory that matches
            the criteria.
        """
        valid_types = ["all", "short_term", "long_term", "episodic", "semantic"]

        # Default to "all" if no memory_type is provided
        if memory_type is None:
            memory_type = "all"

        if memory_type not in valid_types:
            raise ValueError(f"Invalid memory_type. Valid values: {valid_types}")

        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            subj = self.graph.value(statement, RDF.subject)
            pred = self.graph.value(statement, RDF.predicate)
            obj = self.graph.value(statement, RDF.object)

            # Retrieve qualifiers for the statement
            qualifiers = {}
            for q_pred, q_obj in self.graph.predicate_objects(statement):
                if q_pred not in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                    qualifiers[q_pred] = q_obj

            # Determine the type of memory
            currentTime = self.graph.value(statement, humemai.currentTime)
            eventTime = self.graph.value(statement, humemai.eventTime)
            knownSince = self.graph.value(statement, humemai.knownSince)

            # Filter based on the memory_type argument
            if memory_type == "short_term":
                # Short-term memory has currentTime
                if currentTime:
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "long_term":
                # Long-term memory has either eventTime or knownSince, and no currentTime
                if not currentTime and (eventTime or knownSince):
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "episodic":
                # Episodic memory is long-term with eventTime but no knownSince and no currentTime
                if not currentTime and eventTime and not knownSince:
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "semantic":
                # Semantic memory is long-term with knownSince but no eventTime and no currentTime
                if not currentTime and not eventTime and knownSince:
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "all":
                # All memories, regardless of type
                yield (subj, pred, obj, qualifiers)

    def iterate_events(self):
        """
        Iterate over all event nodes in the RDF graph.

        Yields:
            URIRef: Each event node that is an instance of humemai:Event.
        """
        for event_node in self.graph.subjects(RDF.type, humemai.Event):
            yield event_node

    def print_events(self, debug=False):
        """
        Retrieve all triples where an Event entity is involved either as the subject or the object
        and format them as a readable string.

        Returns:
            str: A formatted string containing all event-related triples.
        """
        event_triples = []

        # Find all triples where the subject is of type Event
        for event_node in self.graph.subjects(RDF.type, humemai.Event):
            # Get all triples where this event node is either subject or object
            for s, p, o in self.graph.triples((event_node, None, None)):
                event_triples.append((s, p, o))
            for s, p, o in self.graph.triples((None, None, event_node)):
                event_triples.append((s, p, o))

        if not event_triples:
            if debug:
                return ""
            else:
                print("")
                return

        event_strings = []

        for subj, pred, obj in event_triples:
            if pred == humemai.event or pred == RDF.type:
                continue
            subj_str = self._strip_namespace(subj)
            pred_str = self._strip_namespace(pred)
            obj_str = self._strip_namespace(obj)
            event_strings.append(f"({subj_str}, {pred_str}, {obj_str})")

        if debug:
            return "\n".join(event_strings)
        else:
            print("\n".join(event_strings))
            return

    def print_all_raw_triples(self, debug=False):
        """
        Print all triples in the graph in a readable format.
        """
        raw_triples_string = []
        for subj, pred, obj in self.graph:
            raw_triples_string.append(f"({subj}, {pred}, {obj})")
        
        if debug:
            return "\n".join(raw_triples_string)
        
        else:
            print("\n".join(raw_triples_string))
            return

    def print_memories(self, debug=False):
        """
        Print all memories in the graph in a readable format.
        """

        memory_strings = []
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            subj = self._strip_namespace(self.graph.value(statement, RDF.subject))
            pred = self._strip_namespace(self.graph.value(statement, RDF.predicate))
            obj = self._strip_namespace(self.graph.value(statement, RDF.object))
            qualifiers = {}

            for q_pred, q_obj in self.graph.predicate_objects(statement):
                if q_pred not in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                    qualifiers[self._strip_namespace(q_pred)] = self._strip_namespace(
                        q_obj
                    )

            memory_strings.append(f"({subj}, {pred}, {obj}, {qualifiers})")
        
        if debug:
            return "\n".join(memory_strings)
        else:
            print("\n".join(memory_strings))
            return

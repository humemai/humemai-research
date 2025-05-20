"""Humemai Class with RDFLib"""

# Enable postponed evaluation of annotations (optional but recommended in Python 3.10)
from __future__ import annotations

import collections
import logging
import os
from datetime import datetime
from typing import Optional, Union

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define custom namespace for humemai ontology
humemai = Namespace("https://humem.ai/ontology#")


class Humemai:
    """
    Humemai class for managing both short-term and long-term memories.
    Provides methods to add, retrieve, delete, cluster, and manage memories in the RDF
    graph.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.graph: Graph = Graph()  # Initialize RDF graph for memory storage
        self.graph.bind("humemai", humemai)
        self.current_statement_id: int = 0  # Counter to track the next unique ID

    def add_memory(
        self,
        triples: list[tuple[URIRef, URIRef, URIRef]],
        qualifiers: dict[URIRef, Union[URIRef, Literal]] = {},
    ) -> None:
        """
        Add a reified statement to the RDF graph, with the main triple and optional
        qualifiers. Ensure that everything is in the correct URIRef format.

        Args:
            triples (list): A list of triples (subject, predicate, object) to be added.
            qualifiers (dict): A dictionary of qualifiers (e.g., location, current_time).
        """
        for subj, pred, obj in triples:
            logger.debug(f"Adding triple: ({subj}, {pred}, {obj})")

            if not (subj, pred, obj) in self.graph:
                self.graph.add((subj, pred, obj))
                logger.debug(f"Main triple added: ({subj}, {pred}, {obj})")
            else:
                logger.debug(f"Main triple already exists: ({subj}, {pred}, {obj})")

            statement: BNode = (
                BNode()
            )  # Blank node to represent the new reified statement
            unique_id: int = self.current_statement_id  # Get the current ID
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

            for key, value in qualifiers.items():
                if not isinstance(key, URIRef):
                    raise ValueError(f"Qualifier key {key} must be a URIRef.")
                if not isinstance(value, (URIRef, Literal)):
                    raise ValueError(
                        f"Qualifier value {value} must be a URIRef or Literal."
                    )
                self.graph.add((statement, key, value))
                logger.debug(f"Added qualifier: ({statement}, {key}, {value})")

    def delete_memory(self, memory_id: Literal) -> None:
        """
        Delete a memory (reified statement) by its unique ID, including all associated
        qualifiers.

        Args:
            memory_id (Literal): The unique ID of the memory to be deleted.
        """
        logger.debug(f"Deleting memory with ID: {memory_id}")

        if not isinstance(memory_id, Literal) or memory_id.datatype != XSD.integer:
            raise ValueError(f"memory_id must be a Literal with datatype XSD.integer")

        statement: Optional[URIRef] = None
        for stmt in self.graph.subjects(humemai.memoryID, memory_id):
            statement = stmt
            break

        if statement is None:
            logger.error(f"No memory found with ID {memory_id}")
            return

        subj = self.graph.value(statement, RDF.subject)
        pred = self.graph.value(statement, RDF.predicate)
        obj = self.graph.value(statement, RDF.object)

        if subj is None or pred is None or obj is None:
            logger.error(
                f"Invalid memory statement {statement}. Cannot find associated triple."
            )
            return

        # Check if there are other reified statements with the same triple
        other_reified_statements = []
        for s in self.graph.subjects(RDF.type, RDF.Statement):
            if s != statement:  # Exclude the current statement
                s_subj = self.graph.value(s, RDF.subject)
                s_pred = self.graph.value(s, RDF.predicate)
                s_obj = self.graph.value(s, RDF.object)
                if s_subj == subj and s_pred == pred and s_obj == obj:
                    other_reified_statements.append(s)

        # Only remove the main triple if no other reified statements exist
        if not other_reified_statements:
            logger.debug(f"Deleting main triple: ({subj}, {pred}, {obj})")
            self.graph.remove((subj, pred, obj))
        else:
            logger.debug(
                f"Other reified statements exist for triple ({subj}, {pred}, {obj}). Not deleting main triple."
            )

        for p, o in list(self.graph.predicate_objects(statement)):
            self.graph.remove((statement, p, o))
            logger.debug(f"Removed qualifier triple: ({statement}, {p}, {o})")

        self.graph.remove((statement, RDF.type, RDF.Statement))
        self.graph.remove((statement, RDF.subject, subj))
        self.graph.remove((statement, RDF.predicate, pred))
        self.graph.remove((statement, RDF.object, obj))

        logger.info(f"Memory with ID {memory_id} deleted successfully.")

    def get_memory_by_id(self, memory_id: Literal) -> Optional[dict]:
        """
        Retrieve a memory (reified statement) by its unique ID and return its details.

        Args:
            memory_id (Literal): The unique ID of the memory to retrieve.

        Returns:
            dict: A dictionary with the memory details (subject, predicate, object,
            qualifiers).
        """
        for stmt in self.graph.subjects(
            humemai.memoryID, Literal(memory_id, datatype=XSD.integer)
        ):
            subj = self.graph.value(stmt, RDF.subject)
            pred = self.graph.value(stmt, RDF.predicate)
            obj = self.graph.value(stmt, RDF.object)
            qualifiers: dict[URIRef, Union[URIRef, Literal]] = {}

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

    def delete_main_triple(
        self, subject: URIRef, predicate: URIRef, object_: URIRef
    ) -> None:
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
                for _, pred_q, obj_q in list(
                    self.graph.triples((statement, None, None))
                ):
                    self.graph.remove((statement, pred_q, obj_q))
                    logger.debug(
                        f"Removed qualifier triple: ({statement}, {pred_q}, {obj_q})"
                    )

    def add_short_term_memory(
        self,
        triples: list[tuple[URIRef, URIRef, URIRef]],
        qualifiers: dict[URIRef, Union[URIRef, Literal]] = {},
    ) -> None:
        """
        Add short-term memories to the RDF graph, enforcing required qualifiers.

        Args:
            triples (list): A list of triples to add.
            qualifiers (dict, optional): Additional qualifiers to add.
        """
        if humemai.current_time not in qualifiers:
            current_time = Literal(
                datetime.now().isoformat(timespec="seconds"), datatype=XSD.dateTime
            )
            qualifiers[humemai.current_time] = current_time
        else:
            if qualifiers[humemai.current_time].datatype != XSD.dateTime:
                raise ValueError(
                    f"Invalid current_time format: {qualifiers[humemai.current_time]}"
                )

        self.add_memory(triples, qualifiers)

    def add_episodic_memory(
        self,
        triples: list[tuple[URIRef, URIRef, URIRef]],
        qualifiers: dict[URIRef, Union[URIRef, Literal]] = {},
    ) -> None:
        """
        Add episodic memories to the RDF graph, enforcing required qualifiers.

        Args:
            triples (list): A list of triples to add.
            qualifiers (dict, optional): Additional qualifiers to add.
                The qualifiers can have the following in URIRef format:
                https://humem.ai/ontology#time_added: str,
                https://humem.ai/ontology#location: str,
                https://humem.ai/ontology#emotion: str,
        """
        forbidden_keys = [
            humemai.current_time,
            humemai.derived_from,
        ]
        for key in forbidden_keys:
            if key in qualifiers:
                raise ValueError(f"{key} is not allowed for episodic memories")

        if humemai.time_added not in qualifiers:
            raise ValueError("Missing required qualifier: time_added")

        if qualifiers[humemai.time_added].datatype != XSD.dateTime:
            raise ValueError(
                f"Invalid time_added format: {qualifiers[humemai.time_added]}"
            )

        qualifiers[humemai.last_accessed] = qualifiers[humemai.time_added]

        # Add required qualifiers
        qualifiers = {
            humemai.num_recalled: Literal(0, datatype=XSD.integer),
            **qualifiers,
        }
        self.add_memory(triples, qualifiers)

    def add_semantic_memory(
        self,
        triples: list[tuple[URIRef, URIRef, URIRef]],
        qualifiers: dict[URIRef, Union[URIRef, Literal]] = {},
    ) -> None:
        """
        Add semantic memories to the RDF graph, enforcing required qualifiers.

        Args:
            triples (list): A list of triples to add.
            qualifiers (dict, optional): Additional qualifiers to add.
                The qualifiers can have the following:
                https://humem.ai/ontology#time_added: str,
                https://humem.ai/ontology#derived_from: str,
        """
        forbidden_keys = [
            humemai.emotion,
            humemai.location,
            humemai.current_time,
        ]
        for key in forbidden_keys:
            if key in qualifiers:
                raise ValueError(f"{key} is not allowed for semantic memories")

        if humemai.time_added not in qualifiers:
            raise ValueError("Missing required qualifier: time_added")

        if qualifiers[humemai.time_added].datatype != XSD.dateTime:
            raise ValueError(
                f"Invalid time_added format: {qualifiers[humemai.time_added]}"
            )

        if humemai.derived_from not in qualifiers:
            raise ValueError("Missing required qualifier: derived_from")

        qualifiers[humemai.last_accessed] = qualifiers[humemai.time_added]

        # Add required qualifiers
        qualifiers = {
            humemai.num_recalled: Literal(0, datatype=XSD.integer),
            **qualifiers,
        }
        self.add_memory(triples, qualifiers)

    def get_memories(
        self,
        subject: Optional[URIRef] = None,
        predicate: Optional[URIRef] = None,
        object_: Optional[URIRef] = None,
        qualifiers: dict[URIRef, Union[URIRef, Literal]] = {},
        lower_time_added_bound: Optional[Literal] = None,
        upper_time_added_bound: Optional[Literal] = None,
    ) -> Humemai:
        """
        Retrieve memories with optional filtering based on the qualifiers and triple
        values, including time bounds.

        Args:
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            qualifiers (dict, optional): Additional qualifiers to filter by.
            lower_time_added_bound (Literal, optional): Lower bound for time filtering (ISO format).
            upper_time_added_bound (Literal, optional): Upper bound for time filtering (ISO format).

        Returns:
            Humemai: A new Humemai object containing the filtered memories.
        """

        # Construct SPARQL query with f-strings for dynamic filters
        query = f"""
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?subject ?predicate ?object
        WHERE {{
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

        # Add time filtering logic (for current_time, time_added)
        if lower_time_added_bound and upper_time_added_bound:
            time_filter = f"""
            OPTIONAL {{ ?statement humemai:current_time ?current_time }}
            OPTIONAL {{ ?statement humemai:time_added ?time_added }}
            FILTER(
                (?current_time >= {lower_time_added_bound.n3()} && ?current_time <= {upper_time_added_bound.n3()}) ||
                (?time_added >= {lower_time_added_bound.n3()} && ?time_added <= {upper_time_added_bound.n3()})
            ) .
            """
            query += time_filter

        # Close the WHERE block
        query += "}"

        logger.debug(f"Executing SPARQL query:\n{query}")

        # Execute the SPARQL query
        results = self.graph.query(query)

        # To store reified statements and their corresponding qualifiers
        statement_dict: dict[URIRef, dict] = {}

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

        # Create a new Humemai object to store the filtered results
        filtered_memory = Humemai()

        # Populate the Humemai object with the main triples and their qualifiers
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

    def get_main_triple_count(self) -> int:
        """
        Count the number of main triples (subject-predicate-object triples) in the
        graph. This does not count reified statements.

        Returns:
            int: The count of main triples.
        """
        main_triple_count = 0
        for s, p, o in self.graph:
            if (s, RDF.type, RDF.Statement) not in self.graph:
                main_triple_count += 1

        return main_triple_count

    def get_memory_count(self) -> int:
        """
        Count the number of reified statements (RDF statements) in the graph.
        This counts the reified statements instead of just the main triples.

        Returns:
            int: The count of reified statements.
        """
        return sum(1 for _ in self.graph.subjects(RDF.type, RDF.Statement))

    def get_short_term_memory_count(self) -> int:
        """
        Count the number of short-term memories in the graph.
        Short-term memories are reified statements that have the 'current_time'
        qualifier.

        Returns:
            int: The count of short-term memories.
        """
        return sum(
            1
            for statement in self.graph.subjects(RDF.type, RDF.Statement)
            if self.graph.value(statement, humemai.current_time) is not None
        )

    def get_long_term_episodic_memory_count(self) -> int:
        """
        Count the number of long-term episodic memories in the graph.
        Long-term episodic memories have time_added qualifier but no derived_from qualifier.

        Returns:
            int: The count of long-term episodic memories.
        """
        return sum(
            1
            for statement in self.graph.subjects(RDF.type, RDF.Statement)
            if self.graph.value(statement, humemai.time_added) is not None
            and self.graph.value(statement, humemai.derived_from) is None
            and self.graph.value(statement, humemai.current_time) is None
        )

    def get_long_term_semantic_memory_count(self) -> int:
        """
        Count the number of long-term semantic memories in the graph.
        Long-term semantic memories have time_added qualifier and derived_from qualifier.

        Returns:
            int: The count of long-term semantic memories.
        """
        return sum(
            1
            for statement in self.graph.subjects(RDF.type, RDF.Statement)
            if self.graph.value(statement, humemai.time_added) is not None
            and self.graph.value(statement, humemai.derived_from) is not None
            and self.graph.value(statement, humemai.current_time) is None
        )

    def get_long_term_memory_count(self) -> int:
        """
        Count the number of long-term memories in the graph.
        Long-term memories are reified statements that have the 'time_added'
        qualifier but not 'current_time' qualifier.

        Returns:
            int: The count of long-term memories.
        """
        return sum(
            1
            for statement in self.graph.subjects(RDF.type, RDF.Statement)
            if self.graph.value(statement, humemai.time_added) is not None
            and self.graph.value(statement, humemai.current_time) is None
        )

    def increment_num_recalled(
        self,
        subject: Optional[URIRef] = None,
        predicate: Optional[URIRef] = None,
        object_: Optional[URIRef] = None,
        qualifiers: dict[URIRef, Union[URIRef, Literal]] = {},
        lower_time_added_bound: Optional[Literal] = None,
        upper_time_added_bound: Optional[Literal] = None,
    ) -> None:
        """
        Increment the 'num_recalled' value for memories (episodic or semantic) that match
        the filters.

        Args:
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            qualifiers (dict, optional): Additional qualifiers to filter by.
            lower_time_added_bound (Literal, optional): Lower bound for time filtering
                (ISO format).
            upper_time_added_bound (Literal, optional): Upper bound for time filtering
                (ISO format).
        """

        # Construct base SPARQL query
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement ?num_recalled
        WHERE {
            ?statement rdf:type rdf:Statement .
        """

        # Add subject/predicate/object constraints directly if provided
        if subject is not None:
            query += f"?statement rdf:subject {subject.n3()} .\n"
        else:
            query += "?statement rdf:subject ?subject .\n"

        if predicate is not None:
            query += f"?statement rdf:predicate {predicate.n3()} .\n"
        else:
            query += "?statement rdf:predicate ?predicate .\n"

        if object_ is not None:
            query += f"?statement rdf:object {object_.n3()} .\n"
        else:
            query += "?statement rdf:object ?object .\n"

        # Add all qualifiers as direct triple patterns
        for qualifier_pred, qualifier_obj in qualifiers.items():
            query += f"?statement {qualifier_pred.n3()} {qualifier_obj.n3()} .\n"

        # Add time filtering with required time_added
        if lower_time_added_bound and upper_time_added_bound:
            query += f"""
            ?statement humemai:time_added ?time_added .
            FILTER(?time_added >= {lower_time_added_bound.n3()} && 
                   ?time_added <= {upper_time_added_bound.n3()})
            """

        # Require num_recalled to exist
        query += """
            ?statement humemai:num_recalled ?num_recalled .
        }
        """

        logger.debug(f"Executing SPARQL query:\n{query}")

        # Execute the SPARQL query to retrieve matching statements
        results = self.graph.query(query)

        # Iterate through the results to increment the num_recalled value
        for row in results:
            statement = row.statement
            current_num_recalled_value = row.num_recalled

            # Increment the num_recalled value by 1
            new_num_recalled_value = int(current_num_recalled_value) + 1

            # Update the num_recalled value in the graph
            self.graph.set(
                (
                    statement,
                    humemai.num_recalled,
                    Literal(new_num_recalled_value, datatype=XSD.integer),
                )
            )

            logger.debug(
                f"Updated num_recalled for statement {statement} to {new_num_recalled_value}"
            )

    def update_last_accessed(
        self,
        subject: Optional[URIRef] = None,
        predicate: Optional[URIRef] = None,
        object_: Optional[URIRef] = None,
        qualifiers: dict[URIRef, Union[URIRef, Literal]] = {},
        new_time: Optional[Literal] = None,
        lower_time_added_bound: Optional[Literal] = None,
        upper_time_added_bound: Optional[Literal] = None,
    ) -> None:
        """
        Update the 'last_accessed' qualifier for all statements that match the given
        filters. This is structurally similar to 'increment_num_recalled' but instead
        sets 'last_accessed' to a new time.

        Args:
            subject (URIRef, optional): Filter by subject URI.
            predicate (URIRef, optional): Filter by predicate URI.
            object_ (URIRef, optional): Filter by object URI.
            qualifiers (dict, optional): Additional qualifiers to filter by.
            new_time (Literal, optional): The new last_accessed time to set. Must be a
                Literal with datatype XSD.dateTime if provided.
            lower_time_added_bound (Literal, optional): Lower bound for time_added
                filtering (ISO format).
            upper_time_added_bound (Literal, optional): Upper bound for time_added
                filtering (ISO format).
        """
        logger.debug(
            f"update_last_accessed called with subject={subject}, predicate={predicate}, "
            f"object_={object_}, qualifiers={qualifiers}, new_time={new_time}, "
            f"lower_time_added_bound={lower_time_added_bound}, upper_time_added_bound={upper_time_added_bound}"
        )

        # If user didn't pass a new_time, there's nothing to set
        if new_time is None:
            raise ValueError("new_time must be provided to update last_accessed.")

        # Validate new_time is an xsd:dateTime literal
        if not (isinstance(new_time, Literal) and new_time.datatype == XSD.dateTime):
            raise ValueError(
                f"new_time must be an rdflib Literal with datatype XSD.dateTime. Got: {new_time}"
            )

        # Construct base SPARQL query
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?statement
        WHERE {
            ?statement rdf:type rdf:Statement .
        """

        # Add subject/predicate/object constraints directly if provided
        if subject is not None:
            query += f"?statement rdf:subject {subject.n3()} .\n"
        else:
            query += "?statement rdf:subject ?subject .\n"

        if predicate is not None:
            query += f"?statement rdf:predicate {predicate.n3()} .\n"
        else:
            query += "?statement rdf:predicate ?predicate .\n"

        if object_ is not None:
            query += f"?statement rdf:object {object_.n3()} .\n"
        else:
            query += "?statement rdf:object ?object .\n"

        # Add all qualifiers as direct triple patterns
        for q_pred, q_obj in qualifiers.items():
            query += f"?statement {q_pred.n3()} {q_obj.n3()} .\n"

        # Add time filtering with required time_added
        if lower_time_added_bound and upper_time_added_bound:
            query += f"""
            ?statement humemai:time_added ?time_added .
            FILTER(?time_added >= {lower_time_added_bound.n3()} && 
                   ?time_added <= {upper_time_added_bound.n3()})
            """

        query += "}"  # close WHERE block

        logger.debug(f"Executing SPARQL query:\n{query}")

        # Execute the query
        results = self.graph.query(query)

        # For each matching statement, set last_accessed to new_time
        for row in results:
            statement = row.statement
            logger.debug(f"Updating last_accessed for statement: {statement}")
            self.graph.set((statement, humemai.last_accessed, new_time))
            logger.debug(f"Set last_accessed={new_time} for statement: {statement}")

    def _strip_namespace(self, uri: Union[URIRef, Literal]) -> str:
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

    def is_reified_statement_short_term(self, statement: URIRef) -> bool:
        """
        Check if a given reified statement is a short-term memory by verifying if it
        has a 'current_time' qualifier.

        Args:
            statement (URIRef): The reified statement to check.

        Returns:
            bool: True if it's a short-term memory, False otherwise.
        """
        current_time = self.graph.value(statement, humemai.current_time)
        return current_time is not None

    def _add_reified_statement_to_working_memory_and_increment_recall(
        self,
        subj: URIRef,
        pred: URIRef,
        obj: URIRef,
        working_memory: Humemai,
        specific_statement: Optional[URIRef] = None,
    ) -> None:
        """
        Helper method to add all reified statements (including qualifiers) of a given triple
        to the working memory and increment the recall count for each reified statement.

        Args:
            subj (URIRef): Subject of the triple.
            pred (URIRef): Predicate of the triple.
            obj (URIRef): Object of the triple.
            working_memory (Humemai): The working memory to which the statements and qualifiers are added.
            specific_statement (URIRef, optional): A specific reified statement to process, if provided.
        """
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

                # Retrieve the current num_recalled value
                num_recalled_value = 0
                for _, _, num_recalled in self.graph.triples(
                    (statement, humemai.num_recalled, None)
                ):
                    num_recalled_value = int(num_recalled)

                # Increment the num_recalled value in the long-term memory for this reified statement
                new_num_recalled_value = num_recalled_value + 1
                self.graph.set(
                    (
                        statement,
                        humemai.num_recalled,
                        Literal(new_num_recalled_value, datatype=XSD.integer),
                    )
                )
                logger.debug(
                    f"Updated num_recalled for statement {statement} to {new_num_recalled_value}"
                )

                # Now, add the updated reified statement to the working memory
                for stmt_p, stmt_o in self.graph.predicate_objects(statement):
                    if stmt_p == humemai.num_recalled:
                        working_memory.graph.add(
                            (
                                statement,
                                stmt_p,
                                Literal(new_num_recalled_value, datatype=XSD.integer),
                            )
                        )
                        logger.debug(
                            f"Added updated num_recalled value ({new_num_recalled_value}) to working memory for statement: {statement}"
                        )
                    else:
                        working_memory.graph.add((statement, stmt_p, stmt_o))
                        logger.debug(
                            f"Added reified statement triple to working memory: ({statement}, {stmt_p}, {stmt_o})"
                        )

    def get_short_term_memories(self) -> Humemai:
        """
        Query the RDF graph to retrieve all short-term memories with a current_time
        qualifier and include all associated qualifiers (e.g., location, emotion, etc.).

        Returns:
            Humemai: A Humemai object containing all short-term memories with their qualifiers.
        """
        short_term_memory = Humemai()

        # SPARQL query to retrieve all reified statements with a current_time qualifier, along with other qualifiers
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?statement ?subject ?predicate ?object ?qualifier_pred ?qualifier_obj
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object ;
                    humemai:current_time ?current_time .
            OPTIONAL { ?statement ?qualifier_pred ?qualifier_obj }
        }
        """

        logger.debug(
            f"Executing SPARQL query to retrieve short-term memories:\n{query}"
        )
        results = self.graph.query(query)

        # Dictionary to store reified statements and their qualifiers
        statement_dict: dict[URIRef, dict] = {}

        # Iterate through the results and organize the qualifiers for each reified statement
        for row in results:
            subj = row.subject
            pred = row.predicate
            obj = row.object
            statement = row.statement
            qualifier_pred = row.qualifier_pred
            qualifier_obj = row.qualifier_obj

            if statement not in statement_dict:
                statement_dict[statement] = {
                    "triple": (subj, pred, obj),
                    "qualifiers": {},
                }

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

    def get_long_term_memories(self) -> Humemai:
        """
        Retrieve all long-term memories from the RDF graph.
        Long-term memories are identified by the presence of 'time_added'
        qualifier and the absence of a 'current_time' qualifier.

        Returns:
            Humemai: A new Humemai object containing all long-term memories (episodic and
            semantic).
        """
        long_term_memory = Humemai()

        # SPARQL query to retrieve all reified statements that have time_added,
        # and do not have a current_time qualifier
        query = """
        PREFIX humemai: <https://humem.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?statement ?subject ?predicate ?object
        WHERE {
            ?statement rdf:type rdf:Statement ;
                    rdf:subject ?subject ;
                    rdf:predicate ?predicate ;
                    rdf:object ?object ;
                    humemai:time_added ?time_added .
            FILTER NOT EXISTS { ?statement humemai:current_time ?current_time }
        }
        """

        logger.debug(f"Executing SPARQL query to retrieve long-term memories:\n{query}")
        results = self.graph.query(query)

        # Add the resulting triples to the new Humemai object (long-term memory)
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

            # Now, add all qualifiers (excluding 'current_time')
            for qualifier_pred, qualifier_obj in self.graph.predicate_objects(
                row.statement
            ):
                if qualifier_pred != humemai.current_time:
                    long_term_memory.graph.add(
                        (reified_statement, qualifier_pred, qualifier_obj)
                    )

        return long_term_memory

    def load_from_ttl(self, ttl_file: str) -> None:
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

    def save_to_ttl(self, ttl_file: str) -> None:
        """
        Save the current memory graph to a Turtle (.ttl) file.

        Args:
            ttl_file (str): Path to the Turtle file to save.
        """
        logger.info(f"Saving memory to TTL file: {ttl_file}")
        with open(ttl_file, "w") as f:
            f.write(self.graph.serialize(format="ttl"))
        logger.info(f"Memory saved to {ttl_file} successfully.")

    def iterate_memories(
        self, memory_type: Optional[str] = None
    ) -> tuple[URIRef, URIRef, URIRef, dict[URIRef, Union[URIRef, Literal]]]:
        """
        Iterate over memories in the graph, filtered by memory type (short-term,
        long-term, episodic, semantic, or all).

        Args:
            memory_type (str, optional): The type of memory to iterate over.
                Valid values: "short_term", "long_term", "episodic", "semantic", or "all".
                - "short_term": Short-term memories (with 'current_time').
                - "long_term": Long-term memories (with 'time_added', but without 'current_time').
                - "episodic": Long-term episodic memories (with 'time_added', but without 'derived_from' and 'current_time').
                - "semantic": Long-term semantic memories (with 'time_added' and 'derived_from', but without 'current_time').
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
            qualifiers: dict[URIRef, Union[URIRef, Literal]] = {}
            for q_pred, q_obj in self.graph.predicate_objects(statement):
                if q_pred not in (RDF.type, RDF.subject, RDF.predicate, RDF.object):
                    qualifiers[q_pred] = q_obj

            # Determine the type of memory
            current_time = self.graph.value(statement, humemai.current_time)
            time_added = self.graph.value(statement, humemai.time_added)
            derived_from = self.graph.value(statement, humemai.derived_from)

            # Filter based on the memory_type argument
            if memory_type == "short_term":
                # Short-term memory has current_time
                if current_time:
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "long_term":
                # Long-term memory has time_added, and no current_time
                if not current_time and time_added:
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "episodic":
                # Episodic memory is long-term with time_added but no derived_from and no current_time
                if not current_time and time_added and not derived_from:
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "semantic":
                # Semantic memory is long-term with time_added and derived_from but no current_time
                if not current_time and time_added and derived_from:
                    yield (subj, pred, obj, qualifiers)

            elif memory_type == "all":
                # All memories, regardless of type
                yield (subj, pred, obj, qualifiers)

    def print_raw_triples(self, debug: bool = False) -> Optional[str]:
        """
        Print all triples in the graph in a readable format.

        Args:
            debug (bool): If True, return the string instead of printing.

        Returns:
            str: A formatted string of all triples.
        """
        raw_triples_string = []
        for subj, pred, obj in self.graph:
            raw_triples_string.append(f"({subj}, {pred}, {obj})")

        if debug:
            return "\n".join(raw_triples_string)
        else:
            print("\n".join(raw_triples_string))
            return

    def print_main_triples(self, debug: bool = False) -> Optional[str]:
        """this counts the number of unique memories in the graph, excluding reified
        statements"""
        raw_triples_string = []

        for subj, pred, obj in self.graph:
            if (subj, RDF.type, RDF.Statement) in self.graph:
                continue
            raw_triples_string.append(f"({subj}, {pred}, {obj})")

        if debug:
            return "\n".join(raw_triples_string)
        else:
            print("\n".join(raw_triples_string))
            return

    def print_memories(self, debug: bool = False) -> Optional[str]:
        """
        Print all memories in the graph in a readable format.

        Args:
            debug (bool): If True, return the string instead of printing.

        Returns:
            str: A formatted string of all memories.
        """

        memory_strings = []
        for statement in self.graph.subjects(RDF.type, RDF.Statement):
            subj = self._strip_namespace(self.graph.value(statement, RDF.subject))
            pred = self._strip_namespace(self.graph.value(statement, RDF.predicate))
            obj = self._strip_namespace(self.graph.value(statement, RDF.object))
            qualifiers: dict[str, str] = {}

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

    def get_working_memory(
        self,
        trigger_node: Optional[URIRef] = None,
        hops: int = 0,
        include_all_long_term: bool = False,
    ) -> Humemai:
        """
        Retrieve working memory based on a trigger node and a specified number of hops.
        It fetches all triples within N hops from the trigger node in the long-term
        memory, including their qualifiers. Also includes all short-term memories.

        If `include_all_long_term` is True, all long-term memories will be included,
        regardless of the BFS traversal or hops.

        For each long-term memory retrieved, the 'num_recalled' qualifier is incremented by 1.

        Args:
            trigger_node (URIRef, optional): The starting node for memory traversal.
            hops (int, optional): The number of hops for BFS traversal (default: 0).
            include_all_long_term (bool, optional): Include all long-term memories
            (default: False).

        Returns:
            Humemai: A new Humemai object containing the working memory (short-term +
            relevant long-term memories).
        """
        working_memory = Humemai()
        processed_statements = set()

        logger.info(
            f"Initializing working memory. Trigger node: {trigger_node}, Hops: {hops}, Include all long-term: {include_all_long_term}"
        )

        # Add short-term memories to working memory
        short_term = self.get_short_term_memories()
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
            for statement in self.graph.subjects(RDF.type, RDF.Statement):
                if not self.is_reified_statement_short_term(statement):
                    subj = self.graph.value(statement, RDF.subject)
                    pred = self.graph.value(statement, RDF.predicate)
                    obj = self.graph.value(statement, RDF.object)

                    if statement not in processed_statements:
                        working_memory.graph.add((subj, pred, obj))
                        self._add_reified_statement_to_working_memory_and_increment_recall(
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
            for p, o in self.graph.predicate_objects(current_node):
                reified_statements = [
                    stmt
                    for stmt in self.graph.subjects(RDF.type, RDF.Statement)
                    if self.graph.value(stmt, RDF.subject) == current_node
                    and self.graph.value(stmt, RDF.predicate) == p
                    and self.graph.value(stmt, RDF.object) == o
                ]

                for statement in reified_statements:
                    if self.is_reified_statement_short_term(statement):
                        continue  # Skip short-term memories

                    if statement not in processed_statements:
                        working_memory.graph.add((current_node, p, o))

                        # Add the reified statement and increment 'num_recalled'
                        self._add_reified_statement_to_working_memory_and_increment_recall(
                            current_node,
                            p,
                            o,
                            working_memory,
                            specific_statement=statement,
                        )

                        processed_statements.add(statement)

                    if isinstance(o, URIRef) and o not in visited:
                        queue.append((o, current_hop + 1))
                        visited.add(o)

            # Explore incoming triples
            for s, p in self.graph.subject_predicates(current_node):
                reified_statements = [
                    stmt
                    for stmt in self.graph.subjects(RDF.type, RDF.Statement)
                    if self.graph.value(stmt, RDF.subject) == s
                    and self.graph.value(stmt, RDF.predicate) == p
                    and self.graph.value(stmt, RDF.object) == current_node
                ]

                for statement in reified_statements:
                    if self.is_reified_statement_short_term(statement):
                        continue  # Skip short-term memories

                    if statement not in processed_statements:
                        working_memory.graph.add((s, p, current_node))

                        # Add the reified statement and increment 'num_recalled'
                        self._add_reified_statement_to_working_memory_and_increment_recall(
                            s,
                            p,
                            current_node,
                            working_memory,
                            specific_statement=statement,
                        )

                        processed_statements.add(statement)

                    if isinstance(s, URIRef) and s not in visited:
                        queue.append((s, current_hop + 1))
                        visited.add(s)

        return working_memory

    def move_short_term_to_episodic(
        self,
        memory_id_to_move: Literal,
        qualifiers: dict[URIRef, URIRef | Literal] = {},
    ) -> None:
        """
        Move the specified short-term memory to long-term episodic memory.

        Args:
            memory_id_to_move (Literal): The memory ID to move from short-term to long-term.
        """
        if memory_id_to_move.datatype != XSD.integer:
            raise ValueError("Memory ID must be an integer.")

        # Iterate through the short-term memories
        for subj, pred, obj, qualifiers_ in self.iterate_memories("short_term"):
            memory_id = qualifiers_.get(humemai.memoryID)

            # Check if the memory ID matches
            if memory_id == memory_id_to_move:
                location = qualifiers_.get(humemai.location)
                current_time = qualifiers_.get(humemai.current_time)

                qualifiers[humemai.time_added] = current_time

                if location:
                    qualifiers[humemai.location] = location

                # Move to long-term episodic memory
                self.add_episodic_memory(
                    triples=[(subj, pred, obj)], qualifiers=qualifiers
                )

                # Remove the short-term memory after moving it to long-term
                self.delete_memory(memory_id)

                logger.debug(
                    f"Moved short-term memory with ID {memory_id_to_move} to episodic long-term memory."
                )
                break

    def move_short_term_to_semantic(
        self,
        memory_id_to_move: Literal,
        qualifiers: dict[URIRef, URIRef | Literal] = {},
    ) -> None:
        """
        Move the specified short-term memory to long-term semantic memory.

        Args:
            memory_id_to_move (Literal): The memory ID to move from short-term to
            long-term.
        """
        if memory_id_to_move.datatype != XSD.integer:
            raise ValueError("Memory ID must be an integer.")

        # Iterate through the short-term memories
        for subj, pred, obj, qualifiers_ in self.iterate_memories("short_term"):
            memory_id = qualifiers_.get(humemai.memoryID)

            # Check if the memory ID matches
            if memory_id == memory_id_to_move:
                current_time = qualifiers_.get(humemai.current_time)

                qualifiers[humemai.time_added] = current_time

                # Ensure derived_from is present for semantic memory
                if humemai.derived_from not in qualifiers:
                    raise ValueError(
                        "Missing required qualifier: derived_from for semantic memory"
                    )

                # Move to long-term semantic memory
                self.add_semantic_memory(
                    triples=[(subj, pred, obj)], qualifiers=qualifiers
                )

                # Remove the short-term memory after moving it to long-term
                self.delete_memory(memory_id)
                logger.debug(
                    f"Moved short-term memory with ID {memory_id_to_move} to semantic long-term memory."
                )
                break

    def move_all_short_term_to_episodic(self) -> None:
        """
        Move all short-term memories to long-term episodic memory.
        """
        for subj, pred, obj, qualifiers in self.iterate_memories("short_term"):
            self.move_short_term_to_episodic(qualifiers.get(humemai.memoryID))

    def move_all_short_term_to_semantic(self) -> None:
        """
        Move all short-term memories to long-term semantic memory.
        """
        for subj, pred, obj, qualifiers in self.iterate_memories("short_term"):
            self.move_short_term_to_semantic(qualifiers.get(humemai.memoryID))

    def clear_short_term_memories(self) -> None:
        """
        Clear all short-term memories from the memory system.
        """
        for subj, pred, obj, qualifiers in self.iterate_memories("short_term"):
            memory_id = qualifiers.get(humemai.memoryID)

            self.delete_memory(memory_id)
            logger.debug(f"Cleared short-term memory with ID {memory_id}.")

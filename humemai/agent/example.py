"""An example agent."""

import random
from datetime import datetime
from dateutil import parser
import spacy
from rdflib import Namespace, URIRef, Literal, XSD

from humemai import MemorySystem

# Define the custom namespace for the ontology
humemai = Namespace("https://humem.ai/ontology#")


class Agent:
    def __init__(self):
        """
        Initialize the Agent with its memory system and NLP model.
        """
        self.memory_system = MemorySystem()
        self.nlp = spacy.load("en_core_web_sm")

    def process_input(self, sentence: str):
        """
        Process the input sentence, extract triples, location, and time, and add them to
        short-term memory. Afterward, check the working memory and decide which memories
        should be moved to long-term memory.

        Args:
            sentence (str): The natural language sentence to process.
        """
        doc = self.nlp(sentence)

        # Extract entities and relations
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print("\nEntities found:", entities)

        triples = []
        location = None
        time = None

        # Function to find the nearest verb to an entity
        def find_nearest_verb(token):
            for ancestor in token.ancestors:
                if ancestor.pos_ == "VERB":
                    return ancestor.lemma_  # Use lemma of the verb
            return None

        # Extract triples (subject, predicate, object)
        for i, token in enumerate(doc):
            if token.ent_type_ in ("PERSON", "ORG", "GPE"):
                nearest_verb = find_nearest_verb(token)
                for j in range(i + 1, len(doc)):
                    next_token = doc[j]
                    if next_token.ent_type_ in ("PERSON", "ORG", "GPE"):
                        if nearest_verb:
                            triples.append(
                                (
                                    URIRef(f"https://example.org/{token.text}"),
                                    URIRef(f"https://example.org/{nearest_verb}"),
                                    URIRef(f"https://example.org/{next_token.text}"),
                                )
                            )
                        break

        # Extract location and time as qualifiers
        for ent in doc.ents:
            if ent.label_ == "GPE":
                location = ent.text
            elif ent.label_ == "DATE":
                try:
                    parsed_date = parser.parse(ent.text)
                    time = parsed_date.isoformat()
                except (ValueError, OverflowError):
                    time = None

        print("Extracted triples:", triples)
        print("Location:", location)
        print("Time:", time)

        # Add the extracted triples, location, and time to short-term memory
        self.add_to_short_term_memory(triples, location, time)

        # After adding to short-term memory, check if we should move any memories to long-term
        self.evaluate_and_move_memories()

    def add_to_short_term_memory(self, triples, location=None, time=None):
        """
        Add triples into short-term memory along with location and time qualifiers using
        MemorySystem.

        Args:
            triples (list): List of triples (subject, predicate, object).
            location (str): Location qualifier.
            time (str): Time qualifier in ISO format.
        """
        for triple in triples:
            subj, pred, obj = triple

            qualifiers = {}

            # Add location and time if available
            if location:
                qualifiers[humemai.location] = Literal(location)
            if time:
                qualifiers[humemai.currentTime] = Literal(time, datatype=XSD.dateTime)

            self.memory_system.memory.add_short_term_memory(
                [(subj, pred, obj)], qualifiers=qualifiers
            )

    def evaluate_and_move_memories(self):
        """
        Evaluate the current working memory (short-term and relevant long-term memories)
        and randomly decide which short-term memories should be moved to long-term memory.
        """
        # Get a random trigger node from the long-term memory if there are any
        long_term_memories = list(
            self.memory_system.memory.iterate_memories("long_term")
        )

        # If there are long-term memories, randomly select a trigger node and hops
        if long_term_memories:
            random_memory = random.choice(long_term_memories)
            trigger_node = random.choice([random_memory[0], random_memory[2]])
            hops = random.randint(1, 3)  # Random hops from 1 to 3
            print(f"\nSelected trigger node: {trigger_node} with {hops} hops")
            include_all_long_term = False
        else:
            # No long-term memories, retrieve all memories
            trigger_node = None
            hops = 0
            include_all_long_term = True
            print("\nNo long-term memories available. Retrieving all memories.")

        # Get the current working memory
        working_memory = self.memory_system.get_working_memory(
            trigger_node=trigger_node,
            hops=hops,
            include_all_long_term=include_all_long_term,
        )

        # Iterate over short-term memories in the working memory
        for subj, pred, obj, qualifiers in working_memory.iterate_memories(
            "short_term"
        ):
            memory_id = qualifiers.get(humemai.memoryID)

            # Check if memory_id exists
            if memory_id is None:
                print(f"Skipping memory with no memoryID: {subj}, {pred}, {obj}")
                continue  # Skip memories without a memoryID

            # Randomly decide the memory type
            memory_type = random.choice(["episodic", "semantic"])

            if memory_type == "episodic":
                # Randomly assign emotion and event
                emotion = random.choice([None, "happy", "excited", "curious"])
                event = random.choice([None, "AI Conference", "Meeting", "Travel"])
                print(
                    f"Moving memory to episodic long-term with emotion: {emotion}, event: {event}"
                )
                self.move_memory_to_long_term(
                    memory_id=int(memory_id),
                    memory_type="episodic",
                    emotion=emotion,
                    event=event,
                )

            elif memory_type == "semantic":
                # Randomly assign strength and derivedFrom
                strength = random.randint(1, 10)  # Strength between 1 and 10
                derivedFrom = random.choice(
                    [None, "Observation", "Conversation", "Research"]
                )
                print(
                    f"Moving memory to semantic long-term with strength: {strength}, derivedFrom: {derivedFrom}"
                )
                self.move_memory_to_long_term(
                    memory_id=int(memory_id),
                    memory_type="semantic",
                    strength=strength,
                    derivedFrom=derivedFrom,
                )

    def move_memory_to_long_term(
        self,
        memory_id,
        memory_type,
        emotion=None,
        strength=None,
        derivedFrom=None,
        event=None,
    ):
        """
        Move the specified short-term memory to long-term memory.
        """
        # Retrieve the memory to modify
        memory = self.memory_system.memory.get_memory_by_id(Literal(memory_id, datatype=XSD.integer))

        # If it's an episodic memory, we need to ensure 'currentTime' is removed and 'eventTime' is added
        if memory_type == "episodic":
            qualifiers = memory["qualifiers"]

            # Remove currentTime, which is not allowed in episodic memories
            if humemai.currentTime in qualifiers:
                del qualifiers[humemai.currentTime]

            # Add the required eventTime qualifier
            qualifiers[humemai.eventTime] = Literal(datetime.now().isoformat(), datatype=XSD.dateTime)

            # Optionally add emotion and event if provided
            if emotion:
                qualifiers[humemai.emotion] = Literal(emotion)
            if event:
                qualifiers[humemai.event] = Literal(event)

            # Move to long-term episodic memory
            self.memory_system.memory.add_episodic_memory(
                [(memory["subject"], memory["predicate"], memory["object"])], qualifiers
            )

        elif memory_type == "semantic":
            qualifiers = memory["qualifiers"]

            # Remove disallowed qualifiers for semantic memories
            disallowed_qualifiers = [humemai.location, humemai.event, humemai.emotion, humemai.currentTime]
            for disallowed in disallowed_qualifiers:
                if disallowed in qualifiers:
                    del qualifiers[disallowed]

            # Add knownSince for semantic memories
            qualifiers[humemai.knownSince] = Literal(datetime.now().isoformat(), datatype=XSD.dateTime)

            # Optionally add strength and derivedFrom
            if strength is not None:
                qualifiers[humemai.strength] = Literal(strength, datatype=XSD.integer)
            if derivedFrom:
                qualifiers[humemai.derivedFrom] = Literal(derivedFrom)

            # Move to long-term semantic memory
            self.memory_system.memory.add_semantic_memory(
                [(memory["subject"], memory["predicate"], memory["object"])], qualifiers
            )

        # Remove the original short-term memory
        self.memory_system.memory.delete_memory(Literal(memory_id, datatype=XSD.integer))

    def print_memory(self):
        """
        Print the current memory system (short-term and long-term memories).
        """
        print("\nMemory System:")
        self.memory_system.memory.print_memories()

    def clear_short_term_memories(self):
        """
        Clear all remaining short-term memories.
        """
        self.memory_system.clear_short_term_memories()

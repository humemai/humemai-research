"""An example agent."""

import random

import spacy
from dateutil import parser
from rdflib import Namespace, URIRef

from humemai import MemorySystem

# Define the custom namespace for the ontology
humemai = Namespace("https://humem.ai/ontology/")


class Agent:
    def __init__(self):
        """
        Initialize the Agent with its memory system and NLP model.
        """
        self.memory_system = MemorySystem(verbose_repr=True)
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
            self.memory_system.memory.add_short_term_memory(
                [(subj, pred, obj)], location, time
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
            trigger_node = random_memory[
                0
            ]  # Use the subject of the memory as the trigger node
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
        working_memory = self.get_working_memory(
            trigger_node=trigger_node,
            hops=hops,
            include_all_long_term=include_all_long_term,
        )

        # Iterate over short-term memories in the working memory
        for subj, pred, obj, qualifiers in working_memory.iterate_memories(
            "short_term"
        ):
            memory_id = qualifiers.get(str(humemai.memoryID))

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
        self.memory_system.move_short_term_to_long_term(
            memory_id, memory_type, emotion, strength, derivedFrom, event
        )

    def clear_short_term_memories(self):
        """
        Clear all remaining short-term memories.
        """
        self.memory_system.clear_short_term_memories()

    def get_working_memory(
        self,
        trigger_node: URIRef = None,
        hops: int = 1,
        include_all_long_term: bool = False,
    ):
        """
        Retrieve the working memory, which includes short-term and relevant long-term
        memories.
        """
        return self.memory_system.get_working_memory(
            trigger_node, hops, include_all_long_term
        )

    def print_memory(self):
        """
        Print the current memory system (short-term and long-term memories).
        """
        print("\nMemory System:")
        print(self.memory_system.memory)

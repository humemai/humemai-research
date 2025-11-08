"""Memory class for HumemAI.

This module provides a Memory class that represents a relationship between two nodes
with an edge connecting them.
"""

from datetime import datetime


class Memory:
    """
    A Memory represents a relationship between two nodes with an edge connecting them.
    """

    def __init__(
        self,
        head_label: str,
        tail_label: str,
        edge_label: str,
        head_properties: dict[str, any] = None,
        tail_properties: dict[str, any] = None,
        edge_properties: dict[str, any] = None,
    ) -> None:
        """
        Initializes a Memory instance.

        Args:
            head_label (str): Label for the head node.
            tail_label (str): Label for the tail node.
            edge_label (str): Label for the edge between the head and tail nodes.
            head_properties (dict[str, any], optional): Properties for the head node.
                Defaults to an empty dict if not provided.
            tail_properties (dict[str, any], optional): Properties for the tail node.
                Defaults to an empty dict if not provided.
            edge_properties (dict[str, any], optional): Properties for the edge.
                Defaults to an empty dict if not provided.
        """
        self.head_label = head_label
        self.tail_label = tail_label
        self.edge_label = edge_label

        # Default properties to empty dictionaries if not provided
        self.head_properties = head_properties if head_properties is not None else {}
        self.tail_properties = tail_properties if tail_properties is not None else {}
        self.edge_properties = edge_properties if edge_properties is not None else {}

    def __repr__(self) -> str:
        """
        Provides a string representation of the Memory instance.

        Returns:
            str: A formatted string representing the Memory instance.
        """
        return (
            f"Memory(\n"
            f"  Head(label='{self.head_label}', properties={self.head_properties}),\n"
            f"  Tail(label='{self.tail_label}', properties={self.tail_properties}),\n"
            f"  Edge(label='{self.edge_label}', properties={self.edge_properties})\n"
            f")"
        )

    def to_dict(self) -> dict[str, dict[str, any]]:
        """
        Converts the Memory instance to a dictionary format.
        Useful for serializing or passing data in a structured format.

        Returns:
            dict[str, dict[str, any]]: A dictionary containing the head, tail, and edge data.
        """
        return {
            "head": {"label": self.head_label, "properties": self.head_properties},
            "tail": {"label": self.tail_label, "properties": self.tail_properties},
            "edge": {"label": self.edge_label, "properties": self.edge_properties},
        }


class ShortMemory(Memory):
    """
    A ShortMemory represents a Memory where the edge properties must include
    `current_time`, if not provided, it will be added with the current time.
    """

    def __init__(
        self,
        head_label: str,
        tail_label: str,
        edge_label: str,
        head_properties: dict[str, any] = None,
        tail_properties: dict[str, any] = None,
        edge_properties: dict[str, any] = None,
    ) -> None:
        """
        Initializes a ShortMemory instance.

        Args:
            head_label (str): Label for the head node.
            tail_label (str): Label for the tail node.
            edge_label (str): Label for the edge between the head and tail nodes.
            head_properties (dict[str, any], optional): Properties for the head node.
                Defaults to an empty dict if not provided.
            tail_properties (dict[str, any], optional): Properties for the tail node.
                Defaults to an empty dict if not provided.
            edge_properties (dict[str, any]): Properties for the edge, including a required
                'current_time' field. Defaults to an empty dict if not provided.
        """
        # Ensure edge_properties includes 'current_time'
        if edge_properties is None:
            edge_properties = {}
        if "current_time" not in edge_properties:
            edge_properties["current_time"] = datetime.now().isoformat(
                timespec="seconds"
            )
        elif not isinstance(edge_properties["current_time"], str):
            raise ValueError(
                "The 'current_time' in edge_properties must be an ISO 8601 string."
            )

        # Initialize the parent Memory class with the modified edge_properties
        super().__init__(
            head_label=head_label,
            tail_label=tail_label,
            edge_label=edge_label,
            head_properties=head_properties,
            tail_properties=tail_properties,
            edge_properties=edge_properties,
        )

    def __repr__(self) -> str:
        """
        Provides a string representation of the ShortMemory instance.

        Returns:
            str: A formatted string representing the ShortMemory instance.
        """
        return (
            f"ShortMemory(\n"
            f"  Head(label='{self.head_label}', properties={self.head_properties}),\n"
            f"  Tail(label='{self.tail_label}', properties={self.tail_properties}),\n"
            f"  Edge(label='{self.edge_label}', properties={self.edge_properties}),\n"
            f")"
        )


class LongMemory(Memory):
    """A LongMemory represents a long-term memory.

    As soon as this memory is created, an edge property `recalled` is added with
    value 0.
    """

    def __init__(
        self,
        head_label: str,
        tail_label: str,
        edge_label: str,
        head_properties: dict[str, any] = None,
        tail_properties: dict[str, any] = None,
        edge_properties: dict[str, any] = None,
    ) -> None:
        """
        Initializes a LongMemory instance.

        Args:
            head_label (str): Label for the head node.
            tail_label (str): Label for the tail node.
            edge_label (str): Label for the edge between the head and tail nodes.
            head_properties (dict[str, any], optional): Properties for the head node.
                Defaults to an empty dict if not provided.
            tail_properties (dict[str, any], optional): Properties for the tail node.
                Defaults to an empty dict if not provided.
            edge_properties (dict[str, any], optional): Properties for the edge.
                Defaults to an empty dict if not provided.
        """
        # Ensure edge_properties includes 'recalled'
        if edge_properties is None:
            edge_properties = {}

        if "num_recalled" not in head_properties:
            head_properties["num_recalled"] = 0

        if "num_recalled" not in tail_properties:
            tail_properties["num_recalled"] = 0

        if "num_recalled" not in edge_properties:
            edge_properties["num_recalled"] = 0

        # Initialize the parent Memory class with the modified edge_properties
        super().__init__(
            head_label=head_label,
            tail_label=tail_label,
            edge_label=edge_label,
            head_properties=head_properties,
            tail_properties=tail_properties,
            edge_properties=edge_properties,
        )

    def __repr__(self) -> str:
        """
        Provides a string representation of the LongMemory instance.

        Returns:
            str: A formatted string representing the LongMemory instance.
        """
        return (
            f"LongMemory(\n"
            f"  Head(label='{self.head_label}', properties={self.head_properties}),\n"
            f"  Tail(label='{self.tail_label}', properties={self.tail_properties}),\n"
            f"  Edge(label='{self.edge_label}', properties={self.edge_properties}),\n"
            f")"
        )


class EpisodicMemory(LongMemory):
    """A subclass of LongMemory that represents episodic memories."""

    def __init__(
        self,
        head_label: str,
        tail_label: str,
        edge_label: str,
        head_properties: dict[str, any] = None,
        tail_properties: dict[str, any] = None,
        edge_properties: dict[str, any] = None,
    ) -> None:
        """
        Initializes an EpisodicMemory instance.

        Args:
            head_label (str): Label for the head node.
            tail_label (str): Label for the tail node.
            edge_label (str): Label for the edge between the head and tail nodes.
            head_properties (dict[str, any], optional): Properties for the head node.
                Defaults to an empty dict if not provided.
            tail_properties (dict[str, any], optional): Properties for the tail node.
                Defaults to an empty dict if not provided.
            edge_properties (dict[str, any], optional): Properties for the edge.
                `event_time` is a required field.
        """
        assert "event_time" in edge_properties, "Edge property 'event_time' is required"

        if not isinstance(edge_properties["event_time"], list):
            raise ValueError(
                "The 'event_time' in edge_properties must be a list of ISO 8601 string."
            )

        # Initialize the parent LongMemory class
        super().__init__(
            head_label=head_label,
            tail_label=tail_label,
            edge_label=edge_label,
            head_properties=head_properties,
            tail_properties=tail_properties,
            edge_properties=edge_properties,
        )

    def __repr__(self) -> str:
        """
        Provides a string representation of the EpisodicMemory instance.

        Returns:
            str: A formatted string representing the EpisodicMemory instance.
        """
        return (
            f"EpisodicMemory(\n"
            f"  Head(label='{self.head_label}', properties={self.head_properties}),\n"
            f"  Tail(label='{self.tail_label}', properties={self.tail_properties}),\n"
            f"  Edge(label='{self.edge_label}', properties={self.edge_properties}),\n"
            f")"
        )


class SemanticMemory(LongMemory):
    """A subclass of LongMemory that represents semantic memories."""

    def __init__(
        self,
        head_label: str,
        tail_label: str,
        edge_label: str,
        head_properties: dict[str, any] = None,
        tail_properties: dict[str, any] = None,
        edge_properties: dict[str, any] = None,
    ) -> None:
        """
        Initializes a SemanticMemory instance.

        Args:
            head_label (str): Label for the head node.
            tail_label (str): Label for the tail node.
            edge_label (str): Label for the edge between the head and tail nodes.
            head_properties (dict[str, any], optional): Properties for the head node.
                Defaults to an empty dict if not provided.
            tail_properties (dict[str, any], optional): Properties for the tail node.
                Defaults to an empty dict if not provided.
            edge_properties (dict[str, any], optional): Properties for the edge.
                `known_since` is a required field.
                `derived_from` is a required field.

        """
        assert (
            "known_since" in edge_properties
        ), "Edge property 'known_since' is required"

        if not isinstance(edge_properties["known_since"], str):
            raise ValueError(
                "The 'known_since' in edge_properties must be an ISO 8601 string."
            )
        assert (
            "derived_from" in edge_properties
        ), "Edge property 'derived_from' is required"

        # Initialize the parent LongMemory class
        super().__init__(
            head_label=head_label,
            tail_label=tail_label,
            edge_label=edge_label,
            head_properties=head_properties,
            tail_properties=tail_properties,
            edge_properties=edge_properties,
        )

    def __repr__(self) -> str:
        """
        Provides a string representation of the SemanticMemory instance.

        Returns:
            str: A formatted string representing the SemanticMemory instance.
        """
        return (
            f"SemanticMemory(\n"
            f"  Head(label='{self.head_label}', properties={self.head_properties}),\n"
            f"  Tail(label='{self.tail_label}', properties={self.tail_properties}),\n"
            f"  Edge(label='{self.edge_label}', properties={self.edge_properties}),\n"
            f")"
        )

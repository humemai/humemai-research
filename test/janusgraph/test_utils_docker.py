"""Test docker containers running"""

import unittest
import os
import docker
import nest_asyncio
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.structure.graph import Graph

from humemai.janusgraph.utils.docker import (
    start_containers,
    stop_containers,
    remove_containers,
)


class TestDockerRunning(unittest.TestCase):
    def setUp(self) -> None:
        start_containers(
            warmup_seconds=30,
        )
        self.graph = Graph()
        self.connection = DriverRemoteConnection(
            "ws://localhost:8182/gremlin",
            "g",
            message_serializer=GraphSONSerializersV3d0(),
        )
        self.g = self.graph.traversal().withRemote(self.connection)

    def test_add_dummy_data(self) -> None:

        # Apply nest_asyncio to allow nested event loops (useful in Jupyter notebooks)
        nest_asyncio.apply()
        self.g.V().drop().iterate()

        try:

            # Add Persons
            alice = (
                self.g.addV("person")
                .property("name", "Alice")
                .property("age", 30)
                .next()
            )
            bob = (
                self.g.addV("person").property("name", "Bob").property("age", 25).next()
            )
            carol = (
                self.g.addV("person")
                .property("name", "Carol")
                .property("age", 27)
                .next()
            )
            dave = (
                self.g.addV("person")
                .property("name", "Dave")
                .property("age", 35)
                .next()
            )

            # Add Organizations
            acme = (
                self.g.addV("organization")
                .property("name", "Acme Corp")
                .property("type", "Company")
                .next()
            )
            globex = (
                self.g.addV("organization")
                .property("name", "Globex Inc")
                .property("type", "Company")
                .next()
            )

            # Add 'knows' relationships
            self.g.V(alice.id).addE("knows").to(bob).property("since", 2015).iterate()
            self.g.V(alice.id).addE("knows").to(carol).property("since", 2018).iterate()
            self.g.V(bob.id).addE("knows").to(dave).property("since", 2020).iterate()
            self.g.V(carol.id).addE("knows").to(dave).property("since", 2019).iterate()

            # Add 'works_at' relationships
            self.g.V(alice.id).addE("works_at").to(acme).property(
                "role", "Engineer"
            ).iterate()
            self.g.V(bob.id).addE("works_at").to(globex).property(
                "role", "Analyst"
            ).iterate()
            self.g.V(carol.id).addE("works_at").to(acme).property(
                "role", "Manager"
            ).iterate()
            self.g.V(dave.id).addE("works_at").to(globex).property(
                "role", "Director"
            ).iterate()

            print("Vertices and edges added successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pass

        try:
            # Query 1: Retrieve all vertices with their properties
            vertices = self.g.V().elementMap().toList()
            print("Vertices:")
            for vertex in vertices:
                print(vertex)

            # Query 2: Retrieve all edges with their properties
            edges = self.g.E().elementMap().toList()
            print("\nEdges:")
            for edge in edges:
                print(edge)

        finally:
            pass

        try:
            # Query 1: Retrieve all vertices with their properties
            vertices = self.g.V().elementMap().toList()
            print("Vertices:")
            for vertex in vertices:
                print(vertex)

            # Query 2: Retrieve all edges with their properties
            edges = self.g.E().elementMap().toList()
            print("\nEdges:")
            for edge in edges:
                print(edge)

        finally:
            pass

    def get_dummy_data(self) -> None:

        try:
            # Query 1: Retrieve all vertices with their properties
            vertices = self.g.V().elementMap().toList()
            print("Vertices:")
            for vertex in vertices:
                print(vertex)

            # Query 2: Retrieve all edges with their properties
            edges = self.g.E().elementMap().toList()
            print("\nEdges:")
            for edge in edges:
                print(edge)

        finally:
            pass

    def tearDown(self) -> None:
        self.connection.close()
        stop_containers(cassandra_container_name="foo", janusgraph_container_name="bar")
        remove_containers(
            cassandra_container_name="foo", janusgraph_container_name="bar"
        )

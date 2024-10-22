"""Test Memory class in jena."""

import unittest
import docker
import requests
from humemai.utils.docker import (
    run_jena_container,
    stop_jena_container,
    start_jena_container,
    create_db_jena,
    remove_db_jena,
    remove_jena_container,
    wait_for_fuseki,
)


class TestMemory(unittest.TestCase):
    def test_flow(self) -> None:
        client = docker.from_env()
        try:
            remove_jena_container(client, "foo")
        except:
            pass

        try:
            run_jena_container(client, "foo")
        except:
            start_jena_container(client, "foo")
        
        wait_for_fuseki()

        try:
            create_db_jena("db")
        except:
            remove_db_jena("db")
            create_db_jena("db")

        stop_jena_container(client, "foo")
        remove_jena_container(client, "foo")

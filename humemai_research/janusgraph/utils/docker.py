"""Utility functions for managing JanusGraph and Cassandra containers."""

import os
import logging
import time
import subprocess
import docker
import nest_asyncio
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.structure.graph import Graph
from gremlin_python.process.graph_traversal import GraphTraversalSource

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def start_containers(
    configs_dir: str = "./configs",
    janusgraph_config: str = "janusgraph.properties",
    gremlin_server_config: str = "gremlin-server.yaml",
    cassandra_data_dir: str = "./cassandra_data",
    cassandra_container_name: str = "cassandra",
    janusgraph_container_name: str = "janusgraph",
    warmup_seconds: int = 10,
) -> None:
    """Start JanusGraph and Cassandra containers.

    Args:
        configs_dir (str): Directory containing JanusGraph and Gremlin Server
        configuration files.
        janusgraph_config (str): JanusGraph configuration file.
        gremlin_server_config (str): Gremlin Server configuration file.
        cassandra_data_dir (str): Directory for Cassandra data persistence.
        cassandra_container_name (str): Name of the Cassandra container.
        janusgraph_container_name (str): Name of the JanusGraph container.
        warmup_seconds (int): Number of seconds to wait for the containers to warm up.

    """
    janusgraph_config = os.path.join(configs_dir, janusgraph_config)
    gremlin_server_config = os.path.join(configs_dir, gremlin_server_config)

    # Ensure the configs and data directories exist
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(cassandra_data_dir, exist_ok=True)

    # Check if janusgraph.properties exists; if not, create it
    if not os.path.isfile(janusgraph_config):
        logger.debug("Creating janusgraph.properties...")
        with open(janusgraph_config, "w") as f:
            f.write(
                """\
storage.backend=cassandra
storage.hostname=cassandra
storage.cassandra.keyspace=janusgraph
"""
            )

    # Check if gremlin-server.yaml exists; if not, create it
    if not os.path.isfile(gremlin_server_config):
        logger.debug("Creating gremlin-server.yaml...")
        with open(gremlin_server_config, "w") as f:
            f.write(
                """\
host: 0.0.0.0
port: 8182
scriptEvaluationTimeout: 30000
channelizer: org.apache.tinkerpop.gremlin.server.channel.WebSocketChannelizer
graphs: {
  graph: conf/janusgraph.properties
}

serializers:
  - { className: org.apache.tinkerpop.gremlin.driver.ser.GraphSONMessageSerializerV3d0,
      config: { 
          ioRegistries: [org.janusgraph.graphdb.tinkerpop.JanusGraphIoRegistry] 
      }
    }
  - { className: org.apache.tinkerpop.gremlin.driver.ser.GryoMessageSerializerV3d0 }

# Optional: Enable SSL if required
# ssl: true
# keyCertChainFile: conf/server.pem
# keyFile: conf/server.key
"""
            )

    # Initialize Docker client
    client = docker.from_env()

    # Ensure the network exists
    network_name = "janusgraph-net"
    try:
        client.networks.get(network_name)
    except docker.errors.NotFound:
        logger.debug(f"Creating network {network_name}...")
        client.networks.create(network_name)

    # Start Cassandra container with volume for data persistence
    try:
        cassandra = client.containers.get(cassandra_container_name)
        if cassandra.status == "exited":
            cassandra.start()
            logger.debug(f"{cassandra_container_name} container started.")
        elif cassandra.status == "running":
            logger.debug(f"{cassandra_container_name} container is already running.")
    except docker.errors.NotFound:
        logger.debug(
            f"{cassandra_container_name} container not found, creating and starting it..."
        )
        client.containers.run(
            "cassandra",
            name=cassandra_container_name,
            ports={"9042/tcp": 9042},
            volumes={
                os.path.abspath(cassandra_data_dir): {
                    "bind": "/var/lib/cassandra/data",
                    "mode": "rw",
                }
            },
            detach=True,
            network=network_name,
        )

    # Start JanusGraph container
    try:
        janusgraph = client.containers.get(janusgraph_container_name)
        if janusgraph.status == "exited":
            janusgraph.start()
            logger.debug(f"{janusgraph_container_name} container started.")
        elif janusgraph.status == "running":
            logger.debug(f"{janusgraph_container_name} container is already running.")
    except docker.errors.NotFound:
        logger.debug(
            f"{janusgraph_container_name} container not found, creating and starting it..."
        )
        client.containers.run(
            "janusgraph/janusgraph",
            name=janusgraph_container_name,
            ports={"8182/tcp": 8182},
            volumes={
                os.path.abspath(janusgraph_config): {
                    "bind": "/opt/janusgraph/conf/janusgraph.properties",
                    "mode": "ro",
                },
                os.path.abspath(gremlin_server_config): {
                    "bind": "/opt/janusgraph/conf/gremlin-server.yaml",
                    "mode": "ro",
                },
            },
            detach=True,
            links={cassandra_container_name: "cassandra"},
            network=network_name,
        )

    # Wait for the containers to warm up
    logger.debug(f"Waiting {warmup_seconds} seconds for the containers to warm up...")
    time.sleep(warmup_seconds)


def stop_containers(
    cassandra_container_name: str = "cassandra",
    janusgraph_container_name: str = "janusgraph",
) -> None:
    """Stop JanusGraph and Cassandra containers.

    Args:
        cassandra_container_name (str): Name of the Cassandra container.
        janusgraph_container_name (str): Name of the JanusGraph container.
    """

    # Initialize Docker client
    client = docker.from_env()

    # Function to stop a container if it exists
    def stop_container(container_name):
        try:
            container = client.containers.get(container_name)
            container.stop()
            logger.debug(f"{container_name} container stopped.")
        except docker.errors.NotFound:
            logger.debug(f"{container_name} container not found, skipping.")

    # Stop Cassandra and JanusGraph containers
    stop_container(cassandra_container_name)
    stop_container(janusgraph_container_name)


def remove_containers(
    cassandra_container_name: str = "cassandra",
    janusgraph_container_name: str = "janusgraph",
) -> None:
    """Remove JanusGraph and Cassandra containers.

    Args:
        cassandra_container_name (str): Name of the Cassandra container.
        janusgraph_container_name (str): Name of the JanusGraph container.
    """

    # Initialize Docker client
    client = docker.from_env()

    def remove_container(container_name):
        try:
            container = client.containers.get(container_name)
            container.remove()
            logger.debug(f"{container_name} container removed.")
        except docker.errors.NotFound:
            logger.debug(f"{container_name} container not found, skipping.")

    # Remove Cassandra and JanusGraph containers
    remove_container(cassandra_container_name)
    remove_container(janusgraph_container_name)


def copy_file_from_docker(
    container_name: str, source_path: str, destination_path: str
) -> None:
    """Copy a file from a Docker container to the host machine using docker cp.

    Args:
        container_name (str): Name or ID of the container.
        source_path (str): Path to the file inside the container.
        destination_path (str): Destination path on the host machine.
    """
    try:
        # Build and run the docker cp command
        command = ["docker", "cp", f"{container_name}:{source_path}", destination_path]
        subprocess.run(command, check=True, capture_output=True, text=True)

        logger.debug(f"File copied successfully to {destination_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during file copy: {e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def copy_file_to_docker(
    container_name: str, source_path: str, destination_path: str
) -> None:
    """Copy a file from the host machine to a Docker container using docker cp.

    Args:
        container_name (str): Name or ID of the container.
        source_path (str): Path to the file on the host machine.
        destination_path (str): Destination path inside the container.
    """
    try:
        # Build and run the docker cp command
        command = ["docker", "cp", source_path, f"{container_name}:{destination_path}"]
        subprocess.run(command, check=True, capture_output=True, text=True)

        logger.debug(f"File copied successfully to {container_name}:{destination_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during file copy: {e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

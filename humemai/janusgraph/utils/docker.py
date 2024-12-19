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


def start_docker_compose(compose_file_path: str, warmup_seconds: int = 30) -> None:
    """
    Starts the Docker Compose services specified in the given compose file.

    Args:
        compose_file_path (str): The path to the docker-compose file.
        warmup_seconds (int): Number of seconds to wait for the containers to warm up.

    Raises:
        Exception: If an error occurs while attempting to start the services.

    Outputs:
        Prints a success message and any output from the docker-compose command if
        successful. Prints an error message if the docker-compose command fails.
    """
    try:
        # Run the docker-compose command to start the services
        result = subprocess.run(
            ["docker-compose", "-f", compose_file_path, "up", "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Output the results
        # if result.returncode == 0:
        time.sleep(warmup_seconds)
        logger.debug("Docker Compose started successfully.")
        logger.debug(result.stdout)
        # else:
        #     logger.error(f"Error running docker-compose: {result.stderr}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e


def stop_docker_compose(compose_file_path: str) -> None:
    """
    Stops the Docker Compose services specified in the given compose file.

    Args:
        compose_file_path (str): The path to the docker-compose file.

    Raises:
        Exception: If an error occurs while attempting to stop the services.

    Outputs:
        Prints a success message and any output from the docker-compose command if
        successful. Prints an error message if the docker-compose command fails.
    """
    try:
        # Run the docker-compose command to stop the services
        result = subprocess.run(
            ["docker-compose", "-f", compose_file_path, "down"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Output the results
        if result.returncode == 0:
            logger.debug("Docker Compose stopped successfully.")
            logger.debug(result.stdout)
        else:
            logger.error(f"Error running docker-compose: {result.stderr}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def remove_docker_compose(compose_file_path: str) -> None:
    """
    Remove the containers listed in the docker-compose file.

    Args:
        compose_file_path (str): The path to the docker-compose file.

    Raises:
        Exception: If an error occurs while attempting to stop the services.

    Outputs:
        Prints a success message and any output from the docker-compose command if
        successful. Prints an error message if the docker-compose command fails.
    """
    try:
        # docker-compose -f humemai/janusgraph/docker-compose-cql-es.yml down --remove-orphans
        result = subprocess.run(
            ["docker-compose", "-f", compose_file_path, "down", "--remove-orphans"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Output the results
        if result.returncode == 0:
            logger.debug("Docker Compose removed successfully.")
            logger.debug(result.stdout)
        else:
            logger.error(f"Error running docker-compose: {result.stderr}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


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

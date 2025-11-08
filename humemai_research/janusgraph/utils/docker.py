"""Utility functions for managing JanusGraph and Cassandra containers."""

import logging
import os
import subprocess
import time

import docker
import nest_asyncio
from gremlin_python.driver.driver_remote_connection import \
    DriverRemoteConnection
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversalSource
from gremlin_python.structure.graph import Graph

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def start_docker_compose(
    compose_file_path: str, project_name: str, warmup_seconds: int = 30
) -> None:
    """
    Starts the Docker Compose services specified in the given compose file.

    Args:
        compose_file_path (str): The path to the docker-compose file.
        project_name (str): The Docker Compose project name.
        warmup_seconds (int): Number of seconds to wait for the containers to warm up.

    Raises:
        Exception: If an error occurs while attempting to start the services.

    Outputs:
        Logs a success message and any output from the docker-compose command if
        successful. Logs an error message if the docker-compose command fails.
    """
    try:
        logger.debug(
            f"Starting Docker Compose with project name '{project_name}' using file '{compose_file_path}'..."
        )
        # Run the docker-compose command to start the services
        result = subprocess.run(
            ["docker-compose", "-p", project_name, "-f", compose_file_path, "up", "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Wait for containers to warm up
        logger.debug(
            f"Waiting for {warmup_seconds} seconds to allow containers to warm up..."
        )
        time.sleep(warmup_seconds)

        # Log success and output
        logger.info("Docker Compose started successfully.")
        logger.debug(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docker-compose up: {e.stderr}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while starting Docker Compose: {e}")
        raise e


def stop_docker_compose(compose_file_path: str, project_name: str) -> None:
    """
    Stops the Docker Compose services specified in the given compose file.

    Args:
        compose_file_path (str): The path to the docker-compose file.
        project_name (str): The Docker Compose project name.

    Raises:
        Exception: If an error occurs while attempting to stop the services.

    Outputs:
        Logs a success message and any output from the docker-compose command if
        successful. Logs an error message if the docker-compose command fails.
    """
    try:
        logger.debug(
            f"Stopping Docker Compose with project name '{project_name}' using file '{compose_file_path}'..."
        )
        # Run the docker-compose command to stop the services
        result = subprocess.run(
            ["docker-compose", "-p", project_name, "-f", compose_file_path, "down"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Log success and output
        logger.info("Docker Compose stopped successfully.")
        logger.debug(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docker-compose down: {e.stderr}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while stopping Docker Compose: {e}")
        raise e


def remove_docker_compose(compose_file_path: str, project_name: str) -> None:
    """
    Removes the Docker Compose services specified in the given compose file along with orphans.

    Args:
        compose_file_path (str): The path to the docker-compose file.
        project_name (str): The Docker Compose project name.

    Raises:
        Exception: If an error occurs while attempting to remove the services.

    Outputs:
        Logs a success message and any output from the docker-compose command if
        successful. Logs an error message if the docker-compose command fails.
    """
    try:
        logger.debug(
            f"Removing Docker Compose with project name '{project_name}' using file '{compose_file_path}'..."
        )
        # Run the docker-compose command to remove the services and orphans
        result = subprocess.run(
            [
                "docker-compose",
                "-p",
                project_name,
                "-f",
                compose_file_path,
                "down",
                "--remove-orphans",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Log success and output
        logger.info("Docker Compose removed successfully.")
        logger.debug(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docker-compose down --remove-orphans: {e.stderr}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while removing Docker Compose: {e}")
        raise e


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

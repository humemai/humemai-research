"""Docker utility functions."""

import time
import docker
import requests
from docker.client import DockerClient

SECONDS_TO_WAIT = 1


# Helper function to wait until Fuseki is ready
def wait_for_fuseki(host: str = "http://localhost:3030", timeout: int = 30) -> None:
    """Wait until Fuseki server is up and ready."""
    for _ in range(timeout):
        try:
            response = requests.get(host)
            if response.status_code == 200:
                print("Fuseki is ready!")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError("Fuseki server did not become ready in time.")


# Function to run a new container
def run_jena_container(client: DockerClient, container_name: str = "jena") -> None:
    """Run a new Jena container.

    Args:
        client (docker.DockerClient): Docker client object.
        container_name (str, optional): Name of the container. Defaults to "jena".
    """
    # Run the container with specific options
    container = client.containers.run(
        "stain/jena-fuseki",  # Image name
        name=container_name,  # Container name
        detach=True,  # Run in detached mode (like -d)
        tty=True,  # Allocate a pseudo-TTY (like -it)
        ports={"3030/tcp": 3030},  # Port mapping (like -p 3030:3030)
        environment={"ADMIN_PASSWORD": "admin"},  # Environment variable
        stdin_open=True,  # Keep STDIN open (for interactive mode)
    )
    time.sleep(SECONDS_TO_WAIT)  # Wait for the container to start
    print(f"Container '{container.name}' started successfully!")


def stop_jena_container(client: DockerClient, container_name: str = "jena") -> None:
    """Stop the Jena container.

    Args:
        client (docker.DockerClient): Docker client object.
        container_name (str, optional): Name of the container. Defaults to "jena".
    """
    container = client.containers.get(container_name)
    if container.status == "running":
        container.stop()
        print(f"Container '{container_name}' stopped successfully!")
    else:
        print(f"Container '{container_name}' is not running.")
    time.sleep(SECONDS_TO_WAIT)  # Wait for the container to stop


def start_jena_container(client: DockerClient, container_name: str = "jena") -> None:
    """Start the existing Jena container.

    Args:
        client (docker.DockerClient): Docker client object.
        container_name (str, optional): Name of the container. Defaults to "jena".

    """
    container = client.containers.get(container_name)
    if container.status == "exited":
        container.start()
        print(f"Container '{container_name}' started successfully!")
    else:
        print(f"Container '{container_name}' is already running.")
    time.sleep(SECONDS_TO_WAIT)  # Wait for the container to start


def remove_jena_container(
    client: DockerClient, container_name: str = "jena", force: bool = True
) -> None:
    """Remove the Jena container.

    Args:
        client (docker.DockerClient): Docker client object.
        container_name (str, optional): Name of the container. Defaults to "jena".
        force (bool, optional): Whether to forcefully remove the container even if it is
        running. Defaults to False.
    """
    container = client.containers.get(container_name)
    if container.status == "running":
        if force:
            container.remove(force=True)
            print(f"Container '{container_name}' removed forcefully!")
        else:
            print(
                f"Container '{container_name}' is running. Stop it first or use force=True."
            )
    else:
        container.remove()
        print(f"Container '{container_name}' removed successfully!")
    time.sleep(SECONDS_TO_WAIT)  # Wait for the container to be removed


def create_db_jena(db_name: str = "db") -> None:
    """Create a new database in Jena Fuseki.

    Args:
        db_name (str, optional): Name of the database. Defaults to "db".

    """
    # URL for the Fuseki Admin API to create a new db
    fuseki_admin_url = "http://localhost:3030/$/datasets"

    # Data for the new db
    data = {
        "dbName": db_name,  # Name of the new dataset
        "dbType": "tdb2",  # Dataset type ('mem' for in-memory, 'tdb2' for persistent)
    }

    # Send the POST request to create the dataset with authentication
    response = requests.post(fuseki_admin_url, data=data, auth=("admin", "admin"))

    # Check if the dataset was created successfully
    if response.status_code == 200:
        print("Dataset created successfully!")
    else:
        raise Exception(
            f"Failed to create dataset: {response.status_code} - {response.text}"
        )

    time.sleep(SECONDS_TO_WAIT)  # Wait for the dataset to be created


def remove_db_jena(db_name: str = "db") -> None:
    """Remove a database in Jena Fuseki.

    Args:
        db_name (str, optional): Name of the database. Defaults to "db".

    """
    # URL for the Fuseki Admin API to remove a dataset
    fuseki_admin_url = f"http://localhost:3030/$/datasets/{db_name}"

    # Send the DELETE request to remove the dataset with authentication
    response = requests.delete(fuseki_admin_url, auth=("admin", "admin"))

    # Check if the dataset was removed successfully
    if response.status_code == 200:
        print("Dataset removed successfully!")
    else:
        raise Exception(
            f"Failed to remove dataset: {response.status_code} - {response.text}"
        )
    time.sleep(SECONDS_TO_WAIT)  # Wait for the dataset to be removed

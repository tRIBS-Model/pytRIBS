import docker
import os
import subprocess
import platform
import time

class tRIBSDocker:
    """
    A class to manage the execution of the tRIBS model in a Docker container.

    This class provides functionality to set up and run the tRIBS model within a Docker container. It allows for specifying
    the Docker image, volume path, execution mode, and number of processes. The class handles the creation and management
    of Docker containers for executing the tRIBS model.

    :param volume_path: Path to the directory that will be mounted as a volume inside the Docker container.
    :type volume_path: str
    :param input_file: Path to the input file used by the tRIBS model.
    :type input_file: str
    :param execution_mode: Mode of execution for the tRIBS model. Can be 'serial' or 'parallel'. Default is 'serial'.
    :type execution_mode: str, optional
    :param num_processes: Number of processes to use for parallel execution. Required if execution_mode is 'parallel'.
    :type num_processes: int, optional
    """

    def __init__(self, volume_path, input_file, execution_mode='serial', num_processes=None):
        self.image_name = "tribs/tribs:latest"
        self.volume_path = volume_path
        self.execution_mode = execution_mode
        self.num_processes = num_processes
        self.input_file = input_file
        self.client = None
        self.container = None

    def start_docker_desktop(self, attempts=0):
        """Ensure Docker is running, and if not, start it or prompt for installation."""
        max_attempts = 5
        system = platform.system()

        try:
            self.client = docker.from_env()
            self.client.ping()
            print("Docker is running.")
            return True
        except docker.errors.DockerException:
            if attempts >= max_attempts:
                print(f"Failed to start Docker after {max_attempts} attempts.")
                self.prompt_docker_installation(system)
                return False

            print(f"Docker is not running. Attempt {attempts + 1}/{max_attempts} to start Docker...")

            if system == 'Windows':
                subprocess.run(["powershell", "-Command", "Start-Process", "Docker Desktop"])
            elif system == 'Darwin':  # macOS
                subprocess.run(["open", "-a", "Docker"])
            elif system == 'Linux':
                subprocess.run(["systemctl", "start", "docker"])
            else:
                print(f"Unsupported system: {system}")
                return False

            time.sleep(15)
            return self.start_docker_desktop(attempts + 1)
    @staticmethod
    def prompt_docker_installation(system):
        print("Failed to start Docker. Please start Docker manually or install it if not available.")
        if system in ['Windows', 'Darwin']:
            print("Download and install Docker Desktop from https://www.docker.com/products/docker-desktop")
        elif system == 'Linux':
            print("Install Docker using your package manager, e.g., 'sudo apt install docker.io' for Ubuntu.")

    def initialize_docker_client(self):
        """Initialize the Docker client."""
        try:
            self.client = docker.from_env()
            print("Connected to Docker daemon successfully.")
        except Exception as e:
            print(f"Failed to connect to Docker daemon: {e}")
            raise

    def pull_image(self):
        """Pull the Docker image."""
        try:
            print(f"Pulling the Docker image {self.image_name}...")
            self.client.images.pull(self.image_name)
            print("Image pulled successfully.")
        except Exception as e:
            print(f"Error pulling Docker image: {e}")
            raise

    def run_container(self):
        """Run the Docker container with the specified volume."""
        if os.name == 'nt':
            self.volume_path = self.volume_path.replace('\\', '/')

        try:
            print("Running the Docker container...")
            self.container = self.client.containers.run(
                self.image_name,
                entrypoint="/bin/bash",
                volumes={self.volume_path: {'bind': '/tribs/shared', 'mode': 'rw'}},
                tty=True,
                stdin_open=True,
                detach=True
            )
            print("Container started successfully.")
        except Exception as e:
            print(f"Error running Docker container: {e}")
            raise

    def execute_command_in_container(self, command):
        """Execute a command in the running Docker container."""
        try:
            print(f"Executing command in the container: {command}")
            # Use the shell to run commands
            exec_command = f"bash -c 'cd /tribs/shared && {command}'"
            exit_code, output = self.container.exec_run(exec_command, tty=True, stream=True)
            print("Command executed. Output:")
            for line in output:
                print(line.decode('utf-8').strip())

            if exit_code == 0:
                print("Command ran successfully.")
            else:
                print(f"Command failed with exit code {exit_code}.")
        except Exception as e:
            print(f"Error running commands in the container: {e}")

    def cleanup_container(self):
        """Stop and remove the Docker container."""
        try:
            print("Stopping and removing the container...")
            self.container.stop()
            self.container.remove()
            print("Container stopped and removed.")
        except Exception as e:
            print(f"Error stopping or removing the container: {e}")

    def run_serial(self):
        """Run the serial version of tRIBS."""
        command = f"/tribs/bin/tRIBS {self.input_file}"
        self.execute_command_in_container(command)

    def run_parallel(self):
        """Run the parallel version of tRIBS with MPI."""
        if self.num_processes is None or self.num_processes < 1:
            print("Error: The number of processes must be 1 or more for parallel execution.")
            return
        command = f"mpirun -np {self.num_processes} /tribs/bin/tRIBSpar {self.input_file}"
        self.execute_command_in_container(command)

    def execute(self):
        """Execute the chosen version of tRIBS based on the mode."""
        if self.execution_mode == 'serial':
            print("Running the serial version of tRIBS.")
            self.run_serial()
        elif self.execution_mode == 'parallel':
            print("Running the parallel version of tRIBS.")
            self.run_parallel()
        else:
            print("Invalid execution mode. Please specify 'serial' or 'parallel'.")



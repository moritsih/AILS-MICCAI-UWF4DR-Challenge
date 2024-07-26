import subprocess
import sys
import os

def install_requirements(requirements_file='requirements.txt'):
    try:
        # Determine the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the requirements file relative to the script's directory
        requirements_path = os.path.join(script_dir, '..', requirements_file)

        if not os.path.isfile(requirements_path):
            current_directory = os.getcwd()
            raise Exception(f"The file {requirements_file} was not found in directory {current_directory}.")

        with open(requirements_path, 'r') as file:
            packages = file.readlines()

        for package in packages:
            package = package.strip()
            if package:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Successfully installed {package}")
    except FileNotFoundError:
        current_directory = os.getcwd()
        raise Exception(f"The file {requirements_file} was not found in directory {current_directory}.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"An error occurred while installing packages: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    install_requirements()

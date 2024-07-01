import subprocess
import sys

def install_requirements(requirements_file='requirements.txt'):
    try:
        with open(requirements_file, 'r') as file:
            packages = file.readlines()
        
        for package in packages:
            package = package.strip()
            if package:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Successfully installed {package}")
    except FileNotFoundError:
        raise Exception(f"The file {requirements_file} was not found.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"An error occurred while installing packages: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    install_requirements()

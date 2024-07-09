import os
import shutil
import zipfile
import importlib.util
import typer
from datetime import datetime


class SubmissionBuilder:
    """
    Build a submission zip file for the CodaLab challenge.

    The submission includes the following files in a zip archive:
    - model.py (MUST): contains a class named "model". The class must have implementations of "init", "load", and "predict" functions.
    - metadata (MUST): indicates the submission is in a code submission format - do not remove this file.
    - model weights (Optional): the model weights file can be in any format as long as it is compatible with the model and the permitted Python packages.

    Usage: `python submission_builder.py`
        Looks up all folders in the "models" directory and prompts to choose one to package it into a submission zip file ready to upload to codalab (including metadata file).
        Additionally verifies the model file for mandatory methods.
    """

    def __init__(self, models_dir: str = ".", output_dir: str = "../submissions"):
        """
        Initialize the SubmissionBuilder.

        Args:
        models_dir (str): Path to the directory containing model folders. Defaults to "models".
        output_dir (str): Path to the directory where the submission zip file will be saved. Defaults to "submission".
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.app = typer.Typer()

        self.app.command()(self.package_submission)

    def list_model_folders(self):
        """
        List all subdirectories in the models directory.

        Returns:
        List[str]: List of subdirectory names.
        """
        return [f.name for f in os.scandir(self.models_dir) if f.is_dir()]

    @staticmethod
    def verify_model_class(model_file: str):
        """
        Verify that the model class contains the required methods.

        Args:
        model_file (str): Path to the model.py file.

        Raises:
        ValueError: If the model class does not contain the required methods.
        """
        # Load the model module
        spec = importlib.util.spec_from_file_location("model", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Check for the model class
        if not hasattr(model_module, "model"):
            raise ValueError("The model file must contain a class named 'model'.")

        model_class = model_module.model

        # Check for required methods
        required_methods = ["init", "load", "predict"]
        for method in required_methods:
            if not hasattr(model_class, method):
                raise ValueError(f"The model class must contain a '{method}' method.")

    def create_metadata_file(self, metadata_path: str):
        """
        Create the metadata file if it does not exist.

        Args:
        metadata_path (str): Path to the metadata file.
        """
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w') as f:
                f.write('Indicates the submission is in a code submission format - do not remove this file.')

    def create_submission_zip(self, model_folder: str, auto_include_weights: bool = True):
        """
        Create a submission zip file for the CodaLab challenge.

        Args:
        model_folder (str): Path to the model folder.
        auto_include_weights (bool): Automatically lookup and include the .pth weights file if True. Defaults to True.
        """
        model_file = os.path.join(model_folder, "model.py")
        weights_file = None

        if auto_include_weights:
            for file in os.listdir(model_folder):
                if file.endswith(".pth"):
                    weights_file = os.path.join(model_folder, file)
                    break

        # Verify the model class
        self.verify_model_class(model_file)

        temp_dir = "temp_submission"
        os.makedirs(temp_dir, exist_ok=True)

        shutil.copy(model_file, os.path.join(temp_dir, "model.py"))

        metadata_file = os.path.join(temp_dir, "metadata")
        self.create_metadata_file(metadata_file)

        # Copy the weights file if provided
        if weights_file:
            shutil.copy(weights_file, os.path.join(temp_dir, os.path.basename(weights_file)))

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a zip file with the folder name and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_zip = os.path.join(self.output_dir, f"{os.path.basename(model_folder)}_{timestamp}.zip")
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

        typer.echo(f"Submission zip file '{output_zip}' created successfully.")

    def package_submission(self, auto_include_weights: bool = True):
        """
        Package the model, metadata, and weights into a submission zip file.

        Args:
        auto_include_weights (bool, optional): Automatically lookup and include the .pth weights file. Defaults to True.
        """
        # list folders in model directory
        model_folders = self.list_model_folders()
        if not model_folders:
            typer.echo("No model folders found in the specified directory.")
            raise typer.Exit()

        # prompt the user to choose one
        typer.echo("Choose a model folder:")
        for idx, folder in enumerate(model_folders, start=1):
            typer.echo(f"{idx}. {folder}")
        folder_index = typer.prompt("Enter the number of the model folder", type=int) - 1

        if folder_index < 0 or folder_index >= len(model_folders):
            typer.echo("Invalid choice.")
            raise typer.Exit()

        model_folder = model_folders[folder_index]

        # verify and create the submission zip
        self.create_submission_zip(os.path.join(self.models_dir, model_folder), auto_include_weights)

    def run(self):
        """
        Run the Typer app.
        """
        self.app()


if __name__ == "__main__":
    builder = SubmissionBuilder()
    builder.run()

import os
import shutil
import zipfile
import importlib.util
import typer
from datetime import datetime

app = typer.Typer()

class SubmissionBuilder:
    """
    Build a submission zip file for the CodaLab challenge.

    The submission includes the following files in a zip archive:
    - model.py (MUST): contains a class named "model". The class must have implementations of "init", "load", and "predict" functions.
    - metadata (MUST): indicates the submission is in a code submission format - do not remove this file.
    - model weights (Optional): the model weights file can be in any format as long as it is compatible with the model and the permitted Python packages.

    Usage: `python build_submission.py`
        Looks up all folders in the "models" directory and prompts to choose one to package it into a submission zip file ready to upload to codalab (including metadata file).
        Additionally verifies the model file for mandatory methods.
    """

    def __init__(self, models_dir: str = "../models", output_dir: str = "../submissions"):
        """
        Initialize the SubmissionBuilder.

        Args:
        models_dir (str): Path to the directory containing model folders. Defaults to "models".
        output_dir (str): Path to the directory where the submission zip file will be saved. Defaults to "submission".
        """
        self.models_dir = models_dir
        self.output_dir = output_dir

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

    def copy_or_create_metadata_file(self, model_dir: str, temp_dir: str):
        """
        Check for a metadata file in the model folder. If not present, create an empty one in the temp directory.

        Args:
        model_folder (str): Path to the model folder.
        temp_dir (str): Path to the temporary directory where the metadata file will be created if not found.
        """
        metadata_file_path = os.path.join(model_dir, "metadata")
        temp_metadata_path = os.path.join(temp_dir, "metadata")

        # Check if metadata file exists in the model folder
        if not os.path.exists(metadata_file_path):
            # If not, create an empty metadata file in the temp directory
            with open(temp_metadata_path, 'w') as f:
                f.write('Indicates the submission is in a code submission format - do not remove this file.')
        else:
            # If exists, copy the existing metadata file to the temp directory
            shutil.copy(metadata_file_path, temp_metadata_path)

    def create_submission_zip(self, model_dir: str, include_pth_files: bool = True):
        """
        Create a submission zip file for the CodaLab challenge.

        Args:
        model_folder (str): Path to the model folder.
        auto_include_weights (bool): Automatically lookup and include the .pth weights files if True. Defaults to True.
        """
        model_file = os.path.join(model_dir, "model.py")
        weights_files = []

        # Verify the model class
        self.verify_model_class(model_file)

        temp_dir = "temp_submission"
        os.makedirs(temp_dir, exist_ok=True)

        shutil.copy(model_file, os.path.join(temp_dir, "model.py"))

        self.copy_or_create_metadata_file(model_dir, temp_dir)

        # copy weights files
        if include_pth_files:
            for file in os.listdir(model_dir):
                if file.endswith(".pth"):
                    weights_file = os.path.join(model_dir, file)
                    shutil.copy(weights_file, os.path.join(temp_dir, os.path.basename(weights_file)))

        # create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a zip file with the folder name and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_zip = os.path.join(self.output_dir, f"{os.path.basename(model_dir)}_{timestamp}.zip")
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            print(f"Create submission zip file '{output_zip}'...")
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    print(f"-- add file [{file}]")
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

        # clean up temporary directory
        shutil.rmtree(temp_dir)

        typer.echo(f"Submission zip file '{output_zip}' created successfully.")
        typer.echo(f"Please dont forget to upload the submission zip file to gDrive [https://drive.google.com/drive/folders/1K9Ojqr-6IWIPC7TZuNQt3qhyIN2ExKPY?usp=sharing]")

@app.command()
def build_submission(auto_include_weights: bool = True, models_dir: str = "../models", output_dir: str = "../submissions"):
    """
    Package the model, metadata, and weights into a submission zip file.

    Args:
    auto_include_weights (bool, optional): Automatically lookup and include the .pth weights files. Defaults to True.
    """
    builder = SubmissionBuilder(models_dir, output_dir)
    # list folders in model directory
    model_folders = builder.list_model_folders()
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
    builder.create_submission_zip(os.path.join(builder.models_dir, model_folder), auto_include_weights)


def run():
    app()


if __name__ == "__main__":
    run()

import os
import shutil
import zipfile
import importlib.util
import typer
from datetime import datetime
import numpy as np

app = typer.Typer()

class SubmissionBuilder:
    """
    Build a submission zip file for the CodaLab challenge.

    The submission includes the following files in a zip archive:
    - model.py (MUST): contains a class named "model". The class must have implementations of "init", "load", and "predict" functions.
    - metadata (MUST): indicates the submission is in a code submission format - do not remove this file.
    - model weights (Optional): the model weights file can be in any format as long as it is compatible with the model and the permitted Python packages.
    """

    CHECK_POINT_FILE_PATH_PLACEHOLDER = "#checkpoint_file_path#"

    def __init__(self, model_file: str, checkpoint_file: str = None, label: str = None, output_dir: str = "../submissions"):
        """
        Initialize the SubmissionBuilder.

        Args:
        model_file (str): Path to the model.py file.
        checkpoint_file (str): Path to the checkpoint file. Defaults to None.
        label (str): Optional label for the zip file. Defaults to None.
        output_dir (str): Path to the directory where the submission zip file will be saved. Defaults to "submission".

        NOTE: The checkpoint file path in the model can be specified as "#checkpoint_file_path#" - this will be replaced
        then with the actual checkpoint file path in the copied model file.
        """
        self.model_file = model_file
        self.checkpoint_file = checkpoint_file
        self.label = label
        self.output_dir = output_dir

    @staticmethod
    def verify_model_class(model_file: str):
        """
        Verify that the model class contains the required methods.

        Args:
        model_file (str): Path to the model.py file.

        Raises:
        ValueError: If the model class does not contain the required methods.
        """
        spec = importlib.util.spec_from_file_location("model", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        if not hasattr(model_module, "model"):
            raise ValueError("The model file must contain a class named 'model'.")

        model_class = model_module.model
        required_methods = ["init", "load", "predict"]
        for method in required_methods:
            if not hasattr(model_class, method):
                raise ValueError(f"The model class must contain a '{method}' method.")

    def create_metadata_file(self, temp_dir: str):
        """
        Create an empty metadata file in the temp directory.

        Args:
        temp_dir (str): Path to the temporary directory where the metadata file will be created.
        """
        temp_metadata_path = os.path.join(temp_dir, "metadata")
        with open(temp_metadata_path, 'w') as f:
            f.write('Indicates the submission is in a code submission format - do not remove this file.')

    @staticmethod
    def create_random_image(height, width, channels):
        """
        Create a random image with the given dimensions.

        Args:
        height (int): Height of the image.
        width (int): Width of the image.
        channels (int): Number of channels in the image.

        Returns:
        ndarray: Randomly generated image.
        """
        return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

    def test_model_with_random_image(self, model_class, dir_path):
        """
        Test the model by running a prediction on a random image.

        Args:
        model_module: The model module loaded from the model file.
        model_class: The model class.

        Raises:
        ValueError: If the model fails to make a prediction.
        """
        model_instance = model_class()
        if self.checkpoint_file:
            model_instance.load(dir_path)  # Load the model without the actual directory path

        random_image = self.create_random_image(800, 1016, 3)

        try:
            prediction = model_instance.predict(random_image)
            typer.echo(f"Prediction on random image: {prediction}")
        except Exception as e:
            raise ValueError(f"Model prediction failed: {e}")

    def create_submission_zip(self):
        """
        Create a submission zip file for the CodaLab challenge.
        """
        self.verify_model_class(self.model_file)

        temp_dir = "temp_submission"
        os.makedirs(temp_dir, exist_ok=True)

        self.create_metadata_file(temp_dir)

        checkpoint_file_basename = ''# Copy the checkpoint file
        if self.checkpoint_file:
            checkpoint_file_basename = os.path.basename(self.checkpoint_file)
            shutil.copy(self.checkpoint_file, os.path.join(temp_dir, checkpoint_file_basename))

        # Copy and modify the model file
        with open(self.model_file, 'r') as temp_dir_file:
            model_content = temp_dir_file.read()

        if checkpoint_file_basename:  # Replace the placeholder with the actual checkpoint file path
            model_content = model_content.replace(self.CHECK_POINT_FILE_PATH_PLACEHOLDER, checkpoint_file_basename)

        target_model_file_path = os.path.join(temp_dir, "model.py")
        with open(target_model_file_path, 'w') as target_model_file:
            target_model_file.write(model_content)

        typer.echo("Testing instantiating the model and predict based on a random image...")
        spec = importlib.util.spec_from_file_location("model", target_model_file_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        self.test_model_with_random_image(model_module.model, temp_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        zip_label = self.label if self.label else f'Submission_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        output_zip = os.path.join(self.output_dir, f"{zip_label}.zip")

        typer.echo(f"Creating submission zip file '{output_zip}'")

        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for temp_dir_file in files:
                    typer.echo(f"-- Adding file {temp_dir_file}")
                    zipf.write(os.path.join(root, temp_dir_file), os.path.relpath(os.path.join(root, temp_dir_file), temp_dir))

        shutil.rmtree(temp_dir)
        typer.echo(f"Submission zip file '{output_zip}' created successfully.")

@app.command()
def build_submission(models_dir: str = "../models", output_dir: str = "../submissions"):
    """
    Package the model, metadata, and weights into a submission zip file.

    Args:
    models_dir (str): Path to the directory containing model folders. Defaults to "models".
    output_dir (str): Path to the directory where the submission zip file will be saved. Defaults to "submission".
    """

    #  create preamble with typer echo
    typer.echo("#######################################################")
    typer.echo("# Build Codalab submission zip file interactively")
    typer.echo("#######################################################")


    model_folders = [f.name for f in os.scandir(models_dir) if f.is_dir()]
    if not model_folders:
        typer.echo("No model folders found in the specified directory.")
        raise typer.Exit()

    typer.echo("Choose a model folder:")
    for idx, folder in enumerate(model_folders, start=1):
        typer.echo(f"{idx}. {folder}")
    folder_index = typer.prompt("Enter the number of the model folder", type=int) - 1

    if folder_index < 0 or folder_index >= len(model_folders):
        typer.echo("Invalid choice.")
        raise typer.Exit()

    model_folder = model_folders[folder_index]
    model_dir = os.path.join(models_dir, model_folder)

    model_file = os.path.join(model_dir, "model.py")
    if not os.path.exists(model_file):
        typer.echo("model.py not found in the selected model folder.")
        raise typer.Exit()

    typer.echo("#######################################################")

    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if checkpoint_files:
        typer.echo("Choose a checkpoint file:")
        for idx, checkpoint in enumerate(checkpoint_files, start=1):
            typer.echo(f"{idx}. {checkpoint}")
        checkpoint_index = typer.prompt("Enter the number of the checkpoint file", type=int) - 1

        if checkpoint_index < 0 or checkpoint_index >= len(checkpoint_files):
            typer.echo("Invalid choice.")
            raise typer.Exit()

        checkpoint_file = os.path.join(model_dir, checkpoint_files[checkpoint_index])
    else:
        checkpoint_file = None

    typer.echo("#######################################################")

    default_label = model_folder
    # try to build label based on checkpoint file name starting at "_2024" until *.pth using sub-string - if not found use current date and time
    start_index = checkpoint_file.find("_2024")
    end_index = checkpoint_file.find(".pth")
    if start_index != -1 and end_index != -1:
        default_label += f'{checkpoint_file[start_index:end_index]}'
    else:
        default_label += f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    label = typer.prompt("Optionally enter a different label or accept the default (press enter):", default=default_label)
    typer.echo(f'label: {label}')
    typer.echo("#######################################################")
    typer.echo("Building submission...")
    builder = SubmissionBuilder(model_file, checkpoint_file, label, output_dir)
    builder.create_submission_zip()

if __name__ == "__main__":
    app()

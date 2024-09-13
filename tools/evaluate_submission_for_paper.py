import json
import os
import time
from typing import List
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

from models.best_final_submissions.task_1.model import model as Task1Model
from models.best_final_submissions.task_2.model import model as Task2Model
from models.best_final_submissions.task_3.model import model as Task3Model
from ails_miccai_uwf4dr_challenge.dataset import ChallengeTaskType, DatasetBuilder, CustomDataset, DatasetOriginationType
from ails_miccai_uwf4dr_challenge.dataset_strategy import CombinedDatasetStrategy, Task1Strategy, Task2Strategy, Task3Strategy
from tools.submission_evaluation.confidence_visualizer import ConfidenceVisualizer
from tools.submission_evaluation.inference_result import InferenceResult

class ModelEvaluator:
    def __init__(self, task, model, model_path, val_data):
        """
        Initialize the evaluator with the model and dataset builder.

        :param task: current task
        :param model: Model to be evaluated.
        :param model_path: Path to the directory where the model is stored.
        :param val_data: validation data to evaluate the model.
        """
        self.task = task
        self.model = model
        self.model_path = model_path
        self.val_data = val_data

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        checkpoint_path = os.path.abspath(self.model_path)
        assert os.path.isdir(checkpoint_path), f"Model checkpoint not found at {checkpoint_path}"
        self.model.load(checkpoint_path) 

    def find_last_conv_layer(self, model):
        """
        Find the last convolutional layer in the model.
        
        :param model: The PyTorch model.
        :return: The name of the last convolutional layer.
        """
        last_conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_layer = name
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model.")
        return last_conv_layer

    def save_results(self, results: List[InferenceResult], file_path: str):
        """
        Save the inference results to a JSON file.

        :param results: List of InferenceResult objects.
        :param file_path: The file path to save the results.
        """
        with open(file_path, 'w') as f:
            json.dump([result.to_dict() for result in results], f)

    def load_results(self, file_path: str) -> List[InferenceResult]:
        """
        Load the inference results from a JSON file.

        :param file_path: The file path to load the results from.
        :return: List of InferenceResult objects.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [InferenceResult.from_dict(item) for item in data]

    def evaluate_model(self, save_path: str):
        """
        Evaluate the model on the validation set and compute average inference time.

        :param save_path: The path where the results will be saved.
        """
        # Load model
        self.load_model()
        
        # Create a DataLoader for the validation dataset
        val_dataset = CustomDataset(self.val_data, transform=None, load_like_challenge_analyzers=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

        results: List[InferenceResult] = []  # Initialize list of InferenceResults

        # First loop: Inference with timing (no gradients)
        for images, labels, image_paths in tqdm(val_loader, desc="Evaluating - Inference"):
            # Extract the single image from the batch and move to device
            image = images[0].to(self.model.device)  # Remove batch dimension
            image_path = image_paths[0]
            labels = labels.to(self.model.device)  # No need to remove batch dimension for labels
            image_dims = image.shape[0:2]  # Extract image dimensions (height, width) - channel is third the way we load it

            # Measure the inference time
            with torch.no_grad():  # Disable gradients for inference time measurement
                start_time = time.time()
                output = self.model.predict(image.numpy())  # Get model prediction for the single image
                end_time = time.time()

            inference_time = end_time - start_time

            predicted_label = 1 if output > 0.5 else 0
                    
            results.append(InferenceResult(output, image_path, image_dims, labels.item(), predicted_label, inference_time, None))
            self.save_results(results, save_path)

        self.save_results(results, save_path)

        cam_extractor = GradCAM(self.model.model, self.find_last_conv_layer(self.model.model)) 

        # Second loop: Grad-CAM computation with gradients enabled
        for i, (images, labels, image_paths) in enumerate(tqdm(val_loader, desc="Evaluating - Grad-CAM")):
            image = images[0].to(self.model.device)  # Remove batch dimension

            # Re-compute prediction with gradients enabled
            output_with_grads = self.model.predict(image.numpy(), with_grads=True)

            activation_map = cam_extractor(0, output_with_grads)
            grayscale_cam = activation_map[0].cpu().numpy()

            results[i].activation_map = grayscale_cam
            self.save_results(results, save_path)
        
        self.save_results(results, save_path)

    def visualize_results(self, results: List[InferenceResult]):
        """
        Visualize the results including confidence sorting, confusion matrix, and Grad-CAM.

        :param results: List of InferenceResult objects.
        """
        # Visualization and metrics calculation
        visualizer = ConfidenceVisualizer(display_original=True, display_grad_cam=False, display_combined=True)
        visualizer.concat_and_display_image(self.task, results, labels=[0])
        visualizer.concat_and_display_image(self.task, results, labels=[1])

        # Calculate average inference time
        avg_inference_time = sum(result.inference_time for result in results) / len(results)
        print(f"Average Inference Time: {avg_inference_time:.6f} seconds per image")

        # Calculate metrics
        auc_score = roc_auc_score([r.true_label for r in results], [r.predicted_label for r in results])
        accuracy = accuracy_score([r.true_label for r in results], [r.predicted_label for r in results])

        print(f"ROC AUC Score: {auc_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Compute confusion matrix
        cm = confusion_matrix([r.true_label for r in results], [r.predicted_label for r in results])
        print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    
    #ATTENTION, read this carefully: 
    # if you just want to do the visualization, you can run this script as is - it will load the results from the json files and visualize them
    # if you want to run the inference again, you need to execute the data downloads - the models are downloaded into the models/best_final_submissions folder
    #then you need to edit model.py files: add a param: with_grads=True to the predict method, and implement this accordingly (basically an if / else around the model forward call to return the gradients or not, I am sure you will be able to manage
    # - also, in the with grads = true case, return the output unprocessed)
    
    for task in [ChallengeTaskType.TASK1, ChallengeTaskType.TASK2, ChallengeTaskType.TASK3]:
        # Define the paths and strategies for dataset creation
        model_path_prefix = f"models/best_final_submissions/"
    
        # Use CombinedDatasetStrategy and Task1Strategy to evaluate on the combined dataset for Task 1
        dataset_strategy = CombinedDatasetStrategy()
        
        if task == ChallengeTaskType.TASK1:
            task_strategy = Task1Strategy()
            model = Task1Model()
            model_path = os.path.join(model_path_prefix, "task_1")
        elif task == ChallengeTaskType.TASK2:
            task_strategy = Task2Strategy()
            model = Task2Model()
            model_path = os.path.join(model_path_prefix, "task_2")
        elif task == ChallengeTaskType.TASK3:
            task_strategy = Task3Strategy()
            model = Task3Model()
            model_path = os.path.join(model_path_prefix, "task_3")
        else:
            raise ValueError("Unknown task_strategy : "+task_strategy)
        
        # Build the dataset
        origination = DatasetOriginationType.VALIDATION
        
        dataset_builder = DatasetBuilder(
            dataset=origination,
            task=task,
            split_ratio=0.0,
        )
        
        # Get the validation dataset
        train_data, val_data = dataset_builder.get_train_val()
        
        print(f"Validation data: {len(val_data)} samples - will be evaluated")
        print(f"Training data: {len(train_data) if train_data is not None else 0} samples - will not be evaluated")

        # Initialize the evaluator with the model and dataset
        evaluator = ModelEvaluator(task, model, model_path, val_data)
        
        results_file_name = f"tools/submission_evaluation/results_{origination.name}_{task.value}.json"

        if os.path.exists(results_file_name):
            results: List[InferenceResult] = evaluator.load_results(results_file_name)
            evaluator.save_results(results, results_file_name)
        else:
            evaluator.evaluate_model(results_file_name)
            results: List[InferenceResult] = evaluator.load_results(results_file_name)
        
        evaluator.visualize_results(results)
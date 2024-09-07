import os
import time  # Import time module to measure inference time
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.best_final_submissions.task_1.model import model as Task1Model
from models.best_final_submissions.task_2.model import model as Task2Model
from models.best_final_submissions.task_3.model import model as Task3Model
from ails_miccai_uwf4dr_challenge.dataset import ChallengeTaskType, DatasetBuilder, CustomDataset, DatasetOriginationType
from ails_miccai_uwf4dr_challenge.dataset_strategy import CombinedDatasetStrategy, Task1Strategy, Task2Strategy, Task3Strategy

class ModelEvaluator:
    def __init__(self, model, model_path, dataset_builder):
        """
        Initialize the evaluator with the model and dataset builder.

        :param model: Model to be evaluated.
        :param model_path: Path to the directory where the model is stored.
        :param dataset_builder: DatasetBuilder object to build the dataset.
        """
        self.model = model
        self.model_path = model_path
        self.dataset_builder = dataset_builder

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        self.model.load(self.model_path)

    def evaluate(self):
        """
        Evaluate the model on the validation set, compute average inference time, and generate a confusion matrix.
        """
        # Load model
        self.load_model()

        # Get the validation dataset
        train_data, val_data = self.dataset_builder.get_train_val()
        
        print(f"Loaded model from {self.model_path}")
        print(f"Validation data: {len(val_data)} samples - will be evaluated")
        print(f"Training data: {len(train_data)} samples - will not be evaluated")
        
        # Create a DataLoader for the validation dataset
        val_dataset = CustomDataset(val_data, transform=None, load_like_challenge_analyzers=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

        all_labels = []
        all_preds = []
        total_inference_time = 0  # Initialize total inference time

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                # Extract the single image from the batch and move to device
                image = images[0].to(self.model.device).numpy()  # Remove batch dimension and convert to NumPy
                labels = labels.to(self.model.device)  # No need to remove batch dimension for labels

                # Measure the inference time
                start_time = time.time()
                output = self.model.predict(image)  # Get model prediction for the single image
                end_time = time.time()
                
                # Calculate the inference time for this sample
                inference_time = end_time - start_time
                total_inference_time += inference_time  # Accumulate total inference time

                # Store predictions and labels
                all_preds.append(1 if output > 0.5 else 0)  # Convert output to binary classification
                all_labels.extend(labels.cpu().numpy())  # `labels` is already compatible

        # Calculate average inference time
        avg_inference_time = total_inference_time / len(val_data)
        print(f"Average Inference Time: {avg_inference_time:.6f} seconds per image")

        # Calculate metrics
        auc_score = roc_auc_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)

        print(f"ROC AUC Score: {auc_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Confusion Matrix:\n{cm}")

        # Plot confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

if __name__ == "__main__":
    # Define the paths and strategies for dataset creation
    model_path = "models/best_final_submissions/task_1"  # Adjust the path as needed

    # Use CombinedDatasetStrategy and Task1Strategy to evaluate on the combined dataset for Task 1
    dataset_strategy = CombinedDatasetStrategy()
    task_strategy = Task1Strategy()

    # Build the dataset
    dataset_builder = DatasetBuilder(
        dataset=DatasetOriginationType.ORIGINAL,  # Use the enum value
        task=ChallengeTaskType.TASK1,  # Task type enum
        split_ratio=0.8
    )

    # Initialize the evaluator with the model and dataset
    evaluator = ModelEvaluator(Task1Model(), model_path, dataset_builder)

    # Evaluate the model
    evaluator.evaluate()
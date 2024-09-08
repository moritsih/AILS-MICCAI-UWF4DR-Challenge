import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from typing import List, Optional, Tuple
import numpy as np
import json

class InferenceResult:
    def __init__(self, 
                 output: float, 
                 image_path: str, 
                 image_dims: Tuple[int, int], 
                 true_label: float, 
                 predicted_label: int, 
                 activation_map: Optional[np.ndarray] = None,
                 inference_time: Optional[float] = None):
        """
        Initialize the InferenceResult object.

        :param output: The output confidence score of the model.
        :param image_path: The file path of the image.
        :param image_dims: The dimensions of the image (height, width).
        :param true_label: The true label of the image.
        :param predicted_label: The predicted label by the model.
        :param activation_map: The Grad-CAM activation map.
        :param inference_time: The time taken for inference.
        """
        self.output: float = output
        self.image_path: str = image_path
        self.image_dims: Tuple[int, int] = image_dims
        self.true_label: float = true_label
        self.predicted_label: int = predicted_label
        self.activation_map: Optional[np.ndarray] = activation_map
        self.inference_time: Optional[float] = inference_time
        
    @property
    def confidence(self) -> float:
        """
        Dynamically calculate the confidence score as the absolute difference 
        between the output and the true label.
        """
        return abs(self.output - self.true_label)

    def to_dict(self):
        """
        Convert the InferenceResult object to a dictionary.
        """
        return {
            'output': self.output,
            'image_path': self.image_path,
            'image_dims': self.image_dims,
            'true_label': self.true_label,
            'predicted_label': self.predicted_label,
            'activation_map': self.activation_map.tolist() if self.activation_map is not None else None,
            'inference_time': self.inference_time
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create an InferenceResult object from a dictionary.
        """
        activation_map = np.array(data['activation_map']) if data['activation_map'] is not None else None
        return cls(data['output'], data['image_path'], tuple(data['image_dims']), data['true_label'], data['predicted_label'], activation_map, data.get('inference_time'))
    
class ConfidenceVisualizer:
    def __init__(self, image_size=(100, 100), images_per_row=10):
        """
        Initialize the ConfidenceVisualizer with desired image size and number of images per row.

        :param image_size: Tuple indicating the size of each image (width, height).
        :param images_per_row: Number of images to display per row.
        """
        self.image_size = image_size
        self.images_per_row = images_per_row

    def sort_by_confidence(self, results):
        """
        Sort images by classification confidence.

        :param results: List of InferenceResult objects.
        :return: Sorted list of InferenceResult objects.
        """
        return sorted(results, key=lambda x: x.confidence, reverse=True)  # Sort by confidence

    def get_confidence_color(self, confidence, max_confidence=1.0):
        """
        Get the color representing confidence.

        :param confidence: The confidence score (from 0 to max_confidence).
        :param max_confidence: The maximum possible confidence score.
        :return: A tuple representing the RGB color.
        """
        normalized_confidence = confidence / max_confidence
        red = int(normalized_confidence * 255)
        green = int((1 - normalized_confidence) * 255)
        return (red, green, 0)  # RGB color

    def create_concatenated_image(self, results, labels=None):
        """
        Create a large image displaying images sorted by classification performance, optionally filtered by labels.

        :param results: List of InferenceResult objects.
        :param labels: Optional list of labels to filter the results. If None, display all results.
        :return: PIL Image object representing the large concatenated image.
        """
        # Filter results by the specified labels if provided
        if labels is not None:
            results = [result for result in results if result.true_label in labels or result.predicted_label in labels]

        # Sort the filtered results by confidence
        sorted_results = self.sort_by_confidence(results)

        num_images = len(sorted_results)
        rows = (num_images // self.images_per_row) + 1

        # Create an empty canvas to display the images
        canvas = Image.new('RGB', (self.image_size[0] * self.images_per_row, self.image_size[1] * rows + 10 * rows))

        for idx, result in enumerate(sorted_results):
            # Load image from the path
            img = cv2.imread(result.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.resize(self.image_size)

            # Convert Grad-CAM to a heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * result.activation_map), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = Image.fromarray(heatmap).resize(self.image_size)

            # Overlay heatmap on the original image
            combined_img = Image.blend(pil_img, heatmap, alpha=0.5)

            # Create a bar to represent confidence
            bar_height = 10
            bar = Image.new('RGB', (self.image_size[0], bar_height), self.get_confidence_color(result.confidence))

            # Draw the true and predicted labels on the bar
            draw = ImageDraw.Draw(bar)
            font = ImageFont.load_default()
            text = f'True: {result.true_label}, Pred: {result.predicted_label}'
            text_width, text_height = draw.textsize(text, font=font)
            draw.text(((self.image_size[0] - text_width) / 2, (bar_height - text_height) / 2), text, fill="white", font=font)

            # Create an image with the picture and the bar below it
            img_with_bar = Image.new('RGB', (self.image_size[0], self.image_size[1] + bar_height))
            img_with_bar.paste(combined_img, (0, 0))
            img_with_bar.paste(bar, (0, self.image_size[1]))

            # Paste the combined image onto the canvas
            x_offset = (idx % self.images_per_row) * self.image_size[0]
            y_offset = (idx // self.images_per_row) * (self.image_size[1] + bar_height)
            canvas.paste(img_with_bar, (x_offset, y_offset))

        return canvas

    def concat_and_display_image(self, results, labels=None):
        """
        Display a concatenated image for the specified labels.

        :param results: List of InferenceResult objects.
        :param labels: List of labels to display (e.g., [0], [1], [0, 1]). If None, display all labels.
        """
        image = self.create_concatenated_image(results, labels)
        label_text = ', '.join(map(str, labels)) if labels else 'All'
        self.display_image(image, title=f'Images Sorted by Confidence for Label(s): {label_text}')

    def display_image(self, image, title='Images Sorted by Classification Confidence'):
        """
        Display the large image using matplotlib.

        :param image: PIL Image object to display.
        :param title: Title of the image.
        """
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        plt.axis('off')
        plt.title(title)
        plt.show()
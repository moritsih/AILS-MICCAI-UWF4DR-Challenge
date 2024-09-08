import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Optional
import json

from tools.submission_evaluation.inference_result import InferenceResult

class ConfidenceVisualizer:
    def __init__(self, image_size=(100, 100), images_per_row=5, best_images = 5, worst_images = 10):
        """
        Initialize the ConfidenceVisualizer with desired image size and number of images per row.

        :param image_size: Tuple indicating the size of each image (width, height).
        :param images_per_row: Number of images to display per row.
        :param best_images: Number of best confidence images to display.
        :param worst_images: Number of worst confidence images to display.
        """
        self.image_size = image_size
        self.images_per_row = images_per_row
        self.best_images = best_images
        self.worst_images = worst_images

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
        if labels is not None:
            results = [result for result in results if result.true_label in labels or result.predicted_label in labels]

        sorted_results = self.sort_by_confidence(results)
        
        sorted_results : List[InferenceResult] = sorted_results[:self.worst_images] + sorted_results[-self.best_images:]
        
        num_images = len(sorted_results)
        rows = (num_images // self.images_per_row) + 1

        # Create an empty canvas to display the images
        canvas = Image.new('RGB', (self.image_size[0] * self.images_per_row, self.image_size[1] * rows + 10 * rows))

        for idx, result in enumerate(sorted_results):
            # Load image from the path
            img = cv2.imread(result.image_path)
            assert img is not None, f"Image could not be loaded from path: {result.image_path}"

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.resize(self.image_size)

            # Convert activation map to a heatmap
            activation_map = result.activation_map
            assert isinstance(activation_map, np.ndarray), "Activation map must be a numpy array."
            assert activation_map.ndim == 2, f"Activation map must be 2D, but got shape {activation_map.shape}."

            # Ensure activation map is in 8-bit format
            if activation_map.dtype != np.uint8:
                activation_map = np.uint8(255 * activation_map)

            heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

            # Convert heatmap to PIL Image
            heatmap_pil = Image.fromarray(heatmap)
            heatmap_pil = heatmap_pil.resize(self.image_size)

            # Combine original image and heatmap
            combined_img = Image.blend(pil_img, heatmap_pil, alpha=0.5)

            # Create a bar to represent confidence
            bar_height = int(self.image_size[1]/10)
            bar_color = self.get_confidence_color(result.confidence)
            bar = Image.new('RGB', (self.image_size[0], bar_height), bar_color)

            # Draw the true and predicted labels on the bar
            draw = ImageDraw.Draw(bar)
            font = ImageFont.load_default()
            text = f'Pred: {result.output:.2f}, True: {result.true_label:.0f}'
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
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
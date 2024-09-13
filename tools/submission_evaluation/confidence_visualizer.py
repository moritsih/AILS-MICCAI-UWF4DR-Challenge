import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List

from tools.submission_evaluation.inference_result import InferenceResult
from tools.submission_evaluation.inference_result_selector import InferenceResultSelector
class ConfidenceVisualizer:
    def __init__(self, image_size=(500, 500), images_per_row=5, best_images=5, worst_images=5, display_combined=True, display_grad_cam=False, display_original=False, display_activation_map_as_text=False):
        """
        Initialize the ConfidenceVisualizer with desired image size and number of images per row.

        :param image_size: Tuple indicating the size of each image (width, height).
        :param images_per_row: Number of images to display per row.
        :param best_images: Number of best confidence images to display.
        :param worst_images: Number of worst confidence images to display.
        """
        self.image_size = image_size
        self.image_width = int(image_size[0])
        self.image_height = int(image_size[1])
        self.bar_height = int(self.image_height / 10)
        self.images_per_row = images_per_row
        self.best_images = best_images
        self.worst_images = worst_images
        self.display_combined = display_combined
        self.display_grad_cam = display_grad_cam
        self.display_original = display_original
        self.display_activation_map_as_text = display_activation_map_as_text

    def get_confidence_color(self, confidence, max_confidence=1.0):
        """
        Get the color representing confidence.

        :param confidence: The confidence score (from 0 to max_confidence).
        :param max_confidence: The maximum possible confidence score.
        :return: A tuple representing the RGB color.
        """
        normalized_confidence = confidence / max_confidence
        red = int((1 - normalized_confidence) * 255)
        green = int(normalized_confidence * 255)
        return (red, green, 0)

    def create_bar(self, confidence, bar_text):
        # Create a bar to represent confidence
        bar_color = self.get_confidence_color(confidence)
        bar = Image.new('RGB', (self.image_width, self.bar_height), bar_color)

        # Draw the true and predicted labels on the bar
        draw = ImageDraw.Draw(bar)
        font = self.get_available_font(int(self.bar_height * 0.8))
        text_width, text_height = draw.textbbox((0, 0), bar_text, font=font)[2:4]
        draw.text(((self.image_width - text_width) / 2, (self.bar_height - text_height) / 2), bar_text, fill="white", font=font)

        return bar

    def apply_blue_to_red_colormap(self, activation_map):
        # Normalize the activation map to range [0, 1] (it is already in range but for safety)
        normalized_map = activation_map / np.max(activation_map)

        colormap = plt.get_cmap('coolwarm')  # 'coolwarm' goes from blue to red
        colored_map = colormap(normalized_map)

        # Convert to RGB format and scale to [0, 255]
        colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)

        return colored_map

    def get_available_font(self, font_size):
        """
        Returns the default font with the specified size.
        """
        return ImageFont.load_default(font_size)

    def add_value_overlay(self, image, activation_map):
        """
        Draws a grid of rounded values (0-9) on top of the image.

        :param image: The PIL image to draw on.
        :param activation_map: The original activation map.
        :return: PIL Image with overlaid text.
        """
        # Round the activation map values to the nearest 0.1 and scale to integers 0-9
        rounded_values = np.round(activation_map / np.max(activation_map) * 9).astype(int)

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Choose a font size based on image size (you may need to adjust)
        font_size = max(image.size) // 30
        font = self.get_available_font(font_size)

        # Define grid cell size
        rows, cols = activation_map.shape
        cell_width = image.width // cols
        cell_height = image.height // rows

        # Overlay text for each grid cell
        for i in range(rows):
            for j in range(cols):
                text = str(rounded_values[i, j])
                # Calculate position to center the text
                x = j * cell_width + cell_width // 2
                y = i * cell_height + cell_height // 2
                draw.text((x, y), text, fill='white', font=font, anchor='mm')  # Draw text at the center

        return image

    def create_concatenated_image(self, results: List[InferenceResult], labels=None):
        """
        Create a large image displaying images sorted by classification performance, optionally filtered by labels.

        :param results: List of InferenceResult objects.
        :param labels: Optional list of labels to filter the results. If None, display all results.
        :return: PIL Image object representing the large concatenated image.
        """
        if labels is not None:
            results = [result for result in results if result.true_label in labels]

        selector = InferenceResultSelector(best_images=self.best_images, worst_images=self.worst_images)
        sorted_results = selector.select_best_and_worst(results)

        num_images = len(sorted_results)

        if num_images == 0:
            raise ValueError("No images found for the specified params.")

        rows = (num_images // self.images_per_row)
        bar_height = int(self.image_height / 10)

        how_many_images_per_row = (1 if self.display_combined else 0) + (1 if self.display_grad_cam else 0) + (1 if self.display_original else 0)
        row_height = self.image_height * how_many_images_per_row + bar_height

        canvas = Image.new('RGB', (self.image_width * self.images_per_row, row_height * rows))

        for idx, result in enumerate(sorted_results):
            # Load image from the path
            img = cv2.imread(result.image_path)
            assert img is not None, f"Image could not be loaded from path: {result.image_path}"

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.resize(self.image_size)

            activation_map = result.activation_map
            assert isinstance(activation_map, np.ndarray), "Activation map must be a numpy array."
            assert activation_map.ndim == 2, f"Activation map must be 2D, but got shape {activation_map.shape}."

            # Normalize the activation map to the range [0, 1]
            activation_map = activation_map / np.max(activation_map) if np.max(activation_map) != 0 else activation_map

            heatmap = self.apply_blue_to_red_colormap(activation_map)

            heatmap_pil = Image.fromarray(heatmap)
            heatmap_pil = heatmap_pil.resize(self.image_size)

            if self.display_activation_map_as_text:
                heatmap_pil = self.add_value_overlay(heatmap_pil, activation_map)

            # Combine original image and heatmap
            combined_img = Image.blend(pil_img, heatmap_pil, alpha=0.5)

            bar = self.create_bar(result.confidence, f'Pred: {result.output:.2f}, True: {result.true_label:.0f}')

            # Create an image with the picture and the bar below it
            img_with_bar = Image.new('RGB', (self.image_width, row_height))
            current_height = 0
            if self.display_combined:
                img_with_bar.paste(combined_img, (0, current_height))
                current_height += self.image_height
            if self.display_grad_cam:
                img_with_bar.paste(heatmap_pil, (0, current_height))
                current_height += self.image_height
            if self.display_original:
                img_with_bar.paste(pil_img, (0, current_height))
                current_height += self.image_height
            img_with_bar.paste(bar, (0, row_height - bar_height))

            # Paste the combined image onto the canvas
            x_offset = (idx % self.images_per_row) * self.image_width
            y_offset = (idx // self.images_per_row) * row_height
            canvas.paste(img_with_bar, (x_offset, y_offset))

        return canvas

    def concat_and_display_image(self, task, results, labels=None):
        """
        Display a concatenated image for the specified labels.

        :param results: List of InferenceResult objects.
        :param labels: List of labels to display (e.g., [0], [1], [0, 1]). If None, display all labels.
        """
        image = self.create_concatenated_image(results, labels)
        label_text = ', '.join(map(str, labels)) if labels else 'All'
        self.display_image(image, title=f'{task.value}: Images Sorted by Confidence for Label(s): {label_text}')

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
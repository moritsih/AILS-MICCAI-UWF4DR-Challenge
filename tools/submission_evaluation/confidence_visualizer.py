import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class ConfidenceResult:
    """
    Class to hold the evaluation result of each image.
    """
    def __init__(self, output, image_path, true_label, predicted_label):
        """
        Initialize the ConfidenceResult.

        :param output: output of the model
        :param image_path: The file path to the image.
        :param true_label: The true label of the image.
        :param predicted_label: The predicted label of the image.
        
        and we calculate: confidence: The confidence score (e.g., absolute difference between prediction and true label).
        """
        self.output = output
        self.confidence = abs(output - true_label)
        self.image_path = image_path
        self.true_label = true_label
        self.predicted_label = predicted_label

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

        :param results: List of ConfidenceResult objects.
        :return: Sorted list of ConfidenceResult objects.
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
        Create a large image displaying images sorted by classification performance for specified labels.

        :param results: List of ConfidenceResult objects.
        :param labels: List of labels to include (e.g., [0], [1], [0, 1]). If None, include all labels.
        :return: PIL Image object representing the large concatenated image.
        """
        # Filter results by the specified labels
        if labels is not None:
            filtered_results = [result for result in results if result.true_label in labels]
        else:
            filtered_results = results  # No filtering if labels are None

        # Sort the filtered results by confidence
        sorted_results = self.sort_by_confidence(filtered_results)
        num_images = len(sorted_results)
        rows = (num_images // self.images_per_row) + 1

        # Create an empty canvas to display the images
        canvas = Image.new('RGB', (self.image_size[0] * self.images_per_row, self.image_size[1] * rows + 20 * rows))

        for idx, result in enumerate(sorted_results):
            img = cv2.imread(result.image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.resize(self.image_size)

            # Create a bar to represent confidence
            bar_height = 20
            bar = Image.new('RGB', (self.image_size[0], bar_height), self.get_confidence_color(result.confidence))

            draw = ImageDraw.Draw(bar)
            text = f"True: {result.true_label:.0f} / Pred: {result.output:.2f}"
            font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            text_x = (self.image_size[0] - text_width) // 2
            text_y = (bar_height - text_height) // 2
            draw.text((text_x, text_y), text, fill="white", font=font)

            img_with_bar = Image.new('RGB', (self.image_size[0], self.image_size[1] + bar_height))
            img_with_bar.paste(pil_img, (0, 0))
            img_with_bar.paste(bar, (0, self.image_size[1]))

            x_offset = (idx % self.images_per_row) * self.image_size[0]
            y_offset = (idx // self.images_per_row) * (self.image_size[1] + bar_height)
            canvas.paste(img_with_bar, (x_offset, y_offset))

        return canvas

    def concat_and_display_image(self, results, labels=None):
        """
        Display a concatenated image for the specified labels.

        :param results: List of ConfidenceResult objects.
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
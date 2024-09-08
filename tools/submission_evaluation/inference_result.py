import numpy as np


from typing import Tuple


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
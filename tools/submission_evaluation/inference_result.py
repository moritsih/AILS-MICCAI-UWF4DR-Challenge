import numpy as np
from typing import Optional, Tuple

class InferenceResult:
    def __init__(self,
                 output: float,
                 image_path: str,
                 image_dims: Tuple[int, int],
                 true_label: float,
                 predicted_label: int,
                 inference_time: Optional[float] = None,
                 activation_map: Optional[np.ndarray] = None,
            ):
        """
        Initialize the InferenceResult object.

        :param output: The output confidence score of the model.
        :param image_path: The file path of the image.
        :param image_dims: The dimensions of the image (height, width).
        :param true_label: The true label of the image.
        :param predicted_label: The predicted label by the model.
        :param inference_time: The time taken for inference.
        :param activation_map: The Grad-CAM activation map.
        """
        # Assertions for type checking
        assert isinstance(output, float), "Output should be a float."
        assert isinstance(image_path, str), "Image path should be a string."
        assert isinstance(image_dims, tuple) and len(image_dims) == 2, "Image dims should be a tuple of two integers."
        assert isinstance(true_label, float), "True label should be a float."
        assert isinstance(predicted_label, int), "Predicted label should be an integer."
        if inference_time is not None:
            assert isinstance(inference_time, float), "Inference time should be a float if provided."
        if activation_map is not None:
            assert isinstance(activation_map, np.ndarray), "Activation map should be a numpy array if provided, is of type : " + str(type(activation_map))
            assert activation_map.ndim == 2, f"Activation map must be 2D, but got shape {activation_map.shape}."

        self.output: float = output
        self.image_path: str = image_path
        self.image_dims: Tuple[int, int] = image_dims
        self.true_label: float = true_label
        self.predicted_label: int = predicted_label
        self.inference_time: Optional[float] = inference_time
        self.activation_map: Optional[np.ndarray] = activation_map

    @property
    def confidence(self) -> float:
        """
        Dynamically calculate the confidence score as the absolute difference 
        between the output and the true label.
        """
        return abs(self.output - self.true_label)

    def to_dict(self) -> dict:
        """
        Convert the InferenceResult object to a dictionary.
        """
        return {
            'output': self.output,
            'image_path': self.image_path,
            'image_dims': self.image_dims,
            'true_label': self.true_label,
            'predicted_label': self.predicted_label,
            'inference_time': self.inference_time,
            'activation_map': self.activation_map.tolist() if self.activation_map is not None else None
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an InferenceResult object from a dictionary.
        """
        try:
            activation_map = np.array(data['activation_map']) if data['activation_map'] is not None else None
            # Squeeze activation map to remove singleton dimension
            if activation_map is not None and activation_map.ndim == 3 and activation_map.shape[0] == 1:
                activation_map = np.squeeze(activation_map, axis=0)  # Remove the singleton dimension

            # Validate the activation map after processing
            if activation_map is not None:
                assert activation_map.ndim == 2, f"Activation map must be 2D, but got shape {activation_map.shape}."

            return cls(data['output'], data['image_path'], tuple(data['image_dims']),
                       data['true_label'], data['predicted_label'], data.get('inference_time'), activation_map)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Error in creating InferenceResult from dict: {e}")
        
import math
from typing import List
import numpy as np

from tools.submission_evaluation.inference_result import InferenceResult

class InferenceResultSelector:
    def __init__(self, best_images=5, worst_images=0):
        self.best_images = best_images
        self.worst_images = worst_images

    def sort_by_confidence(self, results: List[InferenceResult]):
        # Sort in descending order since higher confidence is better
        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def select_best_and_worst(self, results: List[InferenceResult]):
        sorted_results = self.sort_by_confidence(results)
        
        mid = len(sorted_results) // 2  # Define mid only once

        selected_results = []
        # Select best images (highest confidences)
        if self.best_images != 0:
            if self.best_images > 0:
                best_images_list = sorted_results[:self.best_images]
            elif self.best_images < 0:
                best_images_list = sorted_results[:mid]
            selected_results.extend(best_images_list)

        # Select worst images (lowest confidences)
        if self.worst_images != 0:
            if self.worst_images > 0:
                worst_images_list = sorted_results[-self.worst_images:]
            elif self.worst_images < 0:
                worst_images_list = sorted_results[mid:]
            selected_results.extend(worst_images_list)

        return selected_results

# Testing InferenceResultSelector
if __name__ == "__main__":
    dims = (10, 10)  # Placeholder image dimensions

    # Desired confidence values for testing (sorted in descending order)
    desired_confidences = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    results = []

    true_label = 1.0  # Fixing true_label to 1.0

    for idx, confidence in enumerate(desired_confidences):
        image_path = f'image_{idx}.jpg'  # Placeholder path
        activation_map = np.zeros((10, 10))  # Placeholder activation map

        # Calculate output so that confidence = 1 - abs(output - true_label) = desired_confidence
        # Rearranged, we get: output = true_label - (1 - desired_confidence)

        output = true_label - (1 - confidence)
        predicted_label = int(round(output))  # For testing purposes

        result = InferenceResult(
            output=output,
            image_path=image_path,
            image_dims=dims,
            true_label=true_label,
            predicted_label=predicted_label,
            activation_map=activation_map
        )
        results.append(result)

    # Test case 1: Select top 3 best and top 2 worst images
    selector = InferenceResultSelector(best_images=3, worst_images=2)
    selected_results = selector.select_best_and_worst(results)
    # Expected confidences: best images first, then worst images
    expected_confidences = [0.95, 0.85, 0.75, 0.15, 0.05]
    selected_confidences = [res.confidence for res in selected_results]
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(selected_confidences, expected_confidences)), \
        f"Test 1 Failed: Expected confidences {expected_confidences}, but got {selected_confidences}"
    print("Test 1 Passed: Correctly selected best 3 and worst 2 images.")

    # Test case 2: Select half best and half worst images (best_images=-1, worst_images=-1)
    selector = InferenceResultSelector(best_images=-1, worst_images=-1)
    selected_results = selector.select_best_and_worst(results)
    # Expected confidences: best half, then worst half
    expected_confidences = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    selected_confidences = [res.confidence for res in selected_results]
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(selected_confidences, expected_confidences)), \
        f"Test 2 Failed: Expected confidences {expected_confidences}, but got {selected_confidences}"
    print("Test 2 Passed: Correctly selected half best and half worst images.")

    # Test case 3: Select only best 4 images
    selector = InferenceResultSelector(best_images=4, worst_images=0)
    selected_results = selector.select_best_and_worst(results)
    expected_confidences = [0.95, 0.85, 0.75, 0.65]
    selected_confidences = [res.confidence for res in selected_results]
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(selected_confidences, expected_confidences)), \
        f"Test 3 Failed: Expected confidences {expected_confidences}, but got {selected_confidences}"
    print("Test 3 Passed: Correctly selected best 4 images.")

    # Test case 4: Select only worst 5 images
    selector = InferenceResultSelector(best_images=0, worst_images=5)
    selected_results = selector.select_best_and_worst(results)
    expected_confidences = [0.45, 0.35, 0.25, 0.15, 0.05]
    selected_confidences = [res.confidence for res in selected_results]
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(selected_confidences, expected_confidences)), \
        f"Test 4 Failed: Expected confidences {expected_confidences}, but got {selected_confidences}"
    print("Test 4 Passed: Correctly selected worst 5 images.")

    # Test case 5: Select all images when best_images and worst_images are negative
    selector = InferenceResultSelector(best_images=-1, worst_images=-1)
    selected_results = selector.select_best_and_worst(results)
    expected_confidences = desired_confidences  # All images selected in order
    selected_confidences = [res.confidence for res in selected_results]
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(selected_confidences, expected_confidences)), \
        f"Test 5 Failed: Expected confidences {desired_confidences}, but got {selected_confidences}"
    print("Test 5 Passed: Correctly selected all images.")

    # Test case 6: Edge case where no images are selected
    selector = InferenceResultSelector(best_images=0, worst_images=0)
    selected_results = selector.select_best_and_worst(results)
    assert len(selected_results) == 0, \
        f"Test 6 Failed: Expected 0 images, but got {len(selected_results)}"
    print("Test 6 Passed: Correctly handled case with no images selected.")

    # All tests passed
    print("\nAll tests passed successfully.")
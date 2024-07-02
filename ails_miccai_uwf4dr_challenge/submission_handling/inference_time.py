import os
import sys
import cv2
import numpy as np
import time


if __name__=="__main__":
    input_dir = 'path_to_test_images'
    submission_program_dir = 'path_to_submission_dir'
    sys.path.append(submission_program_dir)

    # Creating model
    from model import model
    network = model()

    # Loading model weights
    # arch：'SqueezeNet', 'MobileNetV3Small', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    network.load(submission_program_dir, arch='resnet34')

    # Start predicting
    image_names = os.listdir(input_dir)
    num_images = len(image_names)
    total_time = 0.0
    num_runs = 100
    for _ in range(num_runs):
        for one_image_name in image_names:
            img = cv2.imread(os.path.join(input_dir, one_image_name), 1)
            start_time = time.time()
            prob = network.predict(img)
            end_time = time.time()
            total_time += end_time - start_time
    average_time = total_time / (num_runs * num_images)
    print(np.around(average_time * 1000, 1))

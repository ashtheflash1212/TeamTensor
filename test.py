# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Paths to your dataset folders
train_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/images'
train_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/train/labels'

test_image_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/images'
test_label_folder = 'c2a-dataset/C2A_Dataset/new_dataset3/test/labels'

# Function to visualize YOLO annotations on images
def visualize_yolo_annotations(image_path, label_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Get image dimensions
    height, width, _ = image.shape
    
    # Open and read the annotation file
    with open(label_path, 'r') as file:
        for line in file.readlines():
            # Parse YOLO format annotation (class, x_center, y_center, width, height)
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())

            # Convert normalized coordinates to pixel values
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Calculate top-left and bottom-right corners of the bounding box
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw the bounding box on the image
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage

sample_image = os.path.join(train_image_folder, 'collapsed_building_image0001_0.png')
sample_label = os.path.join(train_label_folder, 'collapsed_building_image0001_0.txt')

visualize_yolo_annotations(sample_image, sample_label)

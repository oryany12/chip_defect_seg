import cv2
import numpy as np
import random

# Load an image (replace with your own image path)
image = cv2.imread('data/defective_examples/case1_reference_image.tif')

# Choose a random angle from the list of desired angles
angles = [0, 90, 180, 270]
random_angle = random.choice(angles)

# Get the image dimensions (height, width)
height, width = image.shape[:2]

# Get the rotation matrix for the chosen angle
rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), random_angle, 1)

# Apply the rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

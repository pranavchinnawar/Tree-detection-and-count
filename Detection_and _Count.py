import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'image1.png'  # Path to the input image
image = cv2.imread(image_path)
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define green color range for tree detection
lower_green = np.array([30, 30, 30])  # Lower HSV range
upper_green = np.array([90, 255, 255])  # Upper HSV range

# Create a mask for green regions
mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply Gaussian blur to reduce noise (smaller blur for finer details)
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Morphological operations to merge fragmented regions
kernel = np.ones((5, 5), np.uint8)  # Smaller kernel size to ensure no merging
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Distance Transform to prepare for segmentation
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
_, dist_transform_binary = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, cv2.THRESH_BINARY)
dist_transform_binary = dist_transform_binary.astype(np.uint8)

# Use a small kernel to detect small dots (break larger spots into small dots)
dot_kernel = np.ones((5, 5), np.uint8)
dist_transform_binary = cv2.morphologyEx(dist_transform_binary, cv2.MORPH_CLOSE, dot_kernel)

# Find contours (the individual dots)
contours, _ = cv2.findContours(dist_transform_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count individual dots and draw small circles for visualization
num_dots = 0
for contour in contours:
    if cv2.contourArea(contour) > 5:  # Ignore very small contours (noise)
        num_dots += 1
        # Draw a small circle at each detected dot's centroid
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw a small circle (dot) at each centroid
        cv2.circle(original, center, 2, (0, 255, 0), -1)  # Small green dot with radius 2

# Save the processed image with dots
output_with_dots = "/mnt/data/original_with_dots.jpg"
cv2.imwrite(output_with_dots, original)

# Display the result
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Trees: {num_dots}')
plt.axis('off')
plt.show()

# Output the number of detected dots
print(f"Number of detected dots: {num_dots}")

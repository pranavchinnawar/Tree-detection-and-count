import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'image1.png'  # Path to the input image
image = cv2.imread(image_path)
original = image.copy()

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define green color range for tree detection
lower_green = np.array([30, 30, 30])  # Lower HSV range
upper_green = np.array([90, 255, 255])  # Upper HSV range

# Create a mask for green regions
mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply Gaussian blur to reduce noise
mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Morphological operations to merge fragmented regions
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Distance Transform to prepare for segmentation
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
_, dist_transform_binary = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, cv2.THRESH_BINARY)
dist_transform_binary = dist_transform_binary.astype(np.uint8)

# Find contours
contours, _ = cv2.findContours(dist_transform_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count individual trees and record their centroids
MIN_TREE_SIZE = 50  # Minimum area threshold for a tree
tree_centroids = []
for contour in contours:
    if cv2.contourArea(contour) > MIN_TREE_SIZE:
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            tree_centroids.append((cx, cy))

# Divide the image into equal grid cells
height, width = original.shape[:2]
grid_size = (100, 100)  # Size of each cell (adjust as needed)
cell_width, cell_height = grid_size

# Create an overlay for transparent shading
overlay = original.copy()

# Process each grid cell
for y in range(0, height, cell_height):
    for x in range(0, width, cell_width):
        # Define the cell's boundaries
        x_end = min(x + cell_width, width)
        y_end = min(y + cell_height, height)
        
        # Count trees in this cell
        tree_count = sum(x <= cx < x_end and y <= cy < y_end for cx, cy in tree_centroids)
        
        # Determine the color based on the number of trees
        if tree_count == 0:
            color = (0, 0, 255)  # Red for 0 trees
        elif tree_count == 1:
            color = (0, 255, 255)  # Yellow for 1 tree
        else:
            color = (0, 255, 0)  # Green for ≥ 2 trees
        
        # Add a semi-transparent overlay
        cv2.rectangle(overlay, (x, y), (x_end, y_end), color, -1)

        # Simulate a dashed black border
        dash_length = 10  # Length of each dash
        for i in range(x, x_end, 2 * dash_length):
            cv2.line(original, (i, y), (min(i + dash_length, x_end), y), (0, 0, 0), 1)  # Top border
            cv2.line(original, (i, y_end), (min(i + dash_length, x_end), y_end), (0, 0, 0), 1)  # Bottom border
        for i in range(y, y_end, 2 * dash_length):
            cv2.line(original, (x, i), (x, min(i + dash_length, y_end)), (0, 0, 0), 1)  # Left border
            cv2.line(original, (x_end, i), (x_end, min(i + dash_length, y_end)), (0, 0, 0), 1)  # Right border

# Blend the overlay with the original image
alpha = 0.185 # Transparency factor
cv2.addWeighted(overlay, alpha, original, 1 - alpha, 0, original)

# Save and display the processed image
output_path = "/mnt/data/processed_with_grid_overlay.jpg"
cv2.imwrite(output_path, original)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Classified Grid with Semi-Transparent Shading')
plt.axis('off')
plt.show()

print(f"Processed image saved at: {output_path}")





# 1. `cv2.imread`: Loads the input image from a file.

# 2. `cv2.cvtColor`: Converts the color space of the image (e.g., from BGR to HSV).

# 3. `cv2.inRange`: Creates a binary mask by identifying pixels within a specified HSV range.

# 4. `cv2.GaussianBlur`: Applies Gaussian blur to smooth the mask and reduce noise.

# 5. `cv2.morphologyEx`: Performs morphological operations (e.g., closing) to clean up fragmented regions in the mask.

# 6. `cv2.distanceTransform`: Computes the distance of each pixel to the nearest zero pixel (useful for segmentation).

# 7. `cv2.threshold`: Converts a grayscale image to binary using a threshold value.

# 8. `cv2.findContours`: Detects contours in a binary image, representing object boundaries.

# 9. `cv2.moments`: Calculates spatial moments of a contour, used to find centroids.

# 10. `cv2.rectangle`: Draws rectangles (grid cells or overlays) on the image.

# 11. `cv2.line`: Draws dashed lines for grid cell borders.

# 12. `cv2.addWeighted`: Combines the original image and a transparent overlay.

# 13. `plt.imshow`: Displays the processed image in a plot.

# 14. `cv2.imwrite`: Saves the final processed image to a file.
import cv2
import numpy as np

# Create a 44x44 fully transparent image (Perfect @2x size for macOS 22pt menubar icon)
img = np.zeros((44, 44, 4), dtype=np.uint8)

color = (0, 0, 0, 255)

# Draw sleek modern camera body
cv2.rectangle(img, (4, 12), (40, 36), color, -1)
# Draw top viewfinder bump
cv2.rectangle(img, (14, 6), (30, 12), color, -1)

# Create a perfectly smooth circular cutout for the lens by manipulating the numpy array
y, x = np.ogrid[:44, :44]
dist_from_center = np.sqrt((x - 22)**2 + (y - 24)**2)

# Erase the lens ring (transparent)
img[dist_from_center <= 9] = (0, 0, 0, 0)
# Draw the inner lens dot
img[dist_from_center <= 4] = color

# Save it to disk!
cv2.imwrite("menubar_icon.png", img)

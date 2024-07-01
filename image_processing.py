import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self, input_image, binary_threshold):
        self.input_image = input_image
        self.binary_threshold = binary_threshold

    def process_image(self):
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(self.input_image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold
        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY_INV)

        return image, binary

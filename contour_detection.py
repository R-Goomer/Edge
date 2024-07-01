import cv2
import numpy as np

class ContourDetector:
    def __init__(self, image, binary, epsilon):
        self.image = image
        self.binary = binary
        self.epsilon = epsilon

    def find_and_draw_contours(self):
        # Find contours
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        min_contour_area = 500  # Adjust this value based on your needs
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Create an empty mask
        mask = np.zeros_like(self.binary)

        # Draw the filtered contours on the mask
        for contour in filtered_contours:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Use the mask to segment the bottles
        segmented = cv2.bitwise_and(self.image, self.image, mask=mask)

        # Approximate contours to smooth them
        approx_contours = []
        for contour in filtered_contours:
            epsilon_val = self.epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon_val, True)
            approx_contours.append(approx)

        # Draw the approximated contours on the original image
        output = self.image.copy()
        for approx in approx_contours:
            cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)

        return segmented, output

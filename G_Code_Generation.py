# Define a separate class for generating G Code from contours
class GCodeGenerator:
    def __init__(self, filename='output.txt'):
        self.filename = filename
        self.z_height = 5.0  # Example initial Z-axis height

    def generate_from_contours(self, contours):
        with open(self.filename, 'w') as f:
            f.write("%\n")  # Start of program marker
            f.write("G21\n")  # Set units to millimeters (assuming metric)
            f.write("G17\n")  # Select XY plane
            f.write("G90\n")  # Absolute positioning

            for contour in contours:
                # Move to the initial Z height
                f.write(f"G1 Z{self.z_height}\n")
                f.write(f"G1 X{contour[0][0][0]} Y{contour[0][0][1]}\n")  # Move to the start of the contour
                f.write(f"G1 Z0\n")

                for point in contour:
                    x, y = point.ravel()
                    f.write(f"G1 X{x} Y{y}\n")  # Move to the start of the contour

                # Close the contour by returning to the first point
                first_point = contour[0].ravel()
                f.write(f"G1 X{first_point[0]} Y{first_point[1]}\n")

            f.write("M2\n")  # End of program marker
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from image_processing import ImageProcessor
from contour_detection import ContourDetector
import cv2

# Streamlit UI
st.title('Bottle Segmentation App')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader('Original Image')
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Image', use_column_width=True)

    # Sidebar for parameters
    st.sidebar.subheader('Parameters')
    binary_threshold = st.sidebar.slider('Binary Threshold', 150, 255, 240, 2)
    epsilon = st.sidebar.slider('Epsilon', 0.001, 0.05, 0.002, 0.001, format="%.3f")
    brush_size = st.sidebar.slider('Brush Size', 1, 50, 10, 1)

    # Remove and Include selection buttons
    selection_mode = st.sidebar.radio('Selection Mode', ['Remove', 'Include'], index=0)

    # Process image to get the initial binary image
    image_processor = ImageProcessor(input_image, binary_threshold)
    image, binary = image_processor.process_image()

    # Display the binary image
    st.subheader('Binary Image')
    st.image(binary, caption='Binary Image', use_column_width=True)

    # Add canvas for editing binary image
    st.subheader('Edit Binary Image')
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=brush_size,
        stroke_color="black",
        background_color="white",
        background_image=Image.fromarray(binary),
        update_streamlit=True,
        height=binary.shape[0],
        width=binary.shape[1],
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert the canvas RGBA image to a NumPy array
        canvas_image = canvas_result.image_data.astype(np.uint8)

        # Extract the alpha channel as mask
        alpha_channel = canvas_image[:, :, 3]
        brush_strokes = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)[1]

        # Depending on selection mode, apply/remove brush strokes
        if selection_mode == 'Remove':
            # Invert brush strokes (areas to be removed)
            brush_strokes_inv = cv2.bitwise_not(brush_strokes)
            edited_binary = cv2.bitwise_and(binary, brush_strokes_inv)
        elif selection_mode == 'Include':
            edited_binary = cv2.bitwise_or(binary, brush_strokes)

        # Find and draw contours on the edited binary image
        contour_detector = ContourDetector(image, edited_binary, epsilon)
        segmented, output = contour_detector.find_and_draw_contours()

        # Display output
        st.subheader('Segmented Bottles')
        st.image(segmented, caption='Segmented Bottles', use_column_width=True)

        st.subheader('Bottle Boundaries (Approximated)')
        st.image(output, caption='Bottle Boundaries', use_column_width=True)

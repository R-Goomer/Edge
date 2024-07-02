import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from image_processing import ImageProcessor
from contour_detection import ContourDetector
from G_Code_Generation import GCodeGenerator
import cv2

# Initialize or reset masks and operations log
if 'remove_mask' not in st.session_state:
    st.session_state.remove_mask = None
if 'include_mask' not in st.session_state:
    st.session_state.include_mask = None
if 'operations' not in st.session_state:
    st.session_state.operations = []

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
    epsilon = st.sidebar.slider('Epsilon', 0.001, 0.01, 0.002, 0.001, format="%.3f")
    brush_size = st.sidebar.slider('Brush Size', 1, 50, 10, 1)

    # Remove and Include selection buttons
    selection_mode = st.sidebar.radio('Selection Mode', ['Remove', 'Include'], index=0)

    # Process image to get the initial binary image
    image_processor = ImageProcessor(input_image, binary_threshold)
    image, binary = image_processor.process_image()

    # Initialize masks if they are None
    if st.session_state.remove_mask is None:
        st.session_state.remove_mask = np.zeros_like(binary, dtype=np.uint8)
    if st.session_state.include_mask is None:
        st.session_state.include_mask = np.zeros_like(binary, dtype=np.uint8)

    # Apply operations in the order they were performed
    current_binary = binary.copy()
    for operation in st.session_state.operations:
        if operation['mode'] == 'Remove':
            mask = cv2.bitwise_not(operation['mask'])
            current_binary = cv2.bitwise_and(current_binary, mask)
        elif operation['mode'] == 'Include':
            current_binary = cv2.bitwise_or(current_binary, operation['mask'])

    # # Display the binary image
    # st.subheader('Binary Image')
    # st.image(current_binary, caption='Binary Image', use_column_width=True)

    # Add canvas for editing binary image
    st.subheader('Edit Binary Image')
    brush_color = "black" if selection_mode == 'Remove' else "white"

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=brush_size,
        stroke_color=brush_color,
        background_image=Image.fromarray(current_binary),
        update_streamlit=True,
        height=current_binary.shape[0],
        width=current_binary.shape[1],
        drawing_mode="freedraw",
        key="canvas_" + selection_mode,
    )

    if canvas_result.image_data is not None:
        # Convert the canvas RGBA image to a NumPy array
        canvas_image = canvas_result.image_data.astype(np.uint8)

        # Extract the alpha channel as mask
        alpha_channel = canvas_image[:, :, 3]
        brush_strokes = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)[1]

        # Update the corresponding mask and log the operation
        if selection_mode == 'Remove':
            st.session_state.remove_mask = cv2.bitwise_or(st.session_state.remove_mask, brush_strokes)
            st.session_state.operations.append({'mode': 'Remove', 'mask': brush_strokes})
        elif selection_mode == 'Include':
            st.session_state.include_mask = cv2.bitwise_or(st.session_state.include_mask, brush_strokes)
            st.session_state.operations.append({'mode': 'Include', 'mask': brush_strokes})

    # Add a button to process contours
    if st.button('Process Contours'):
        # Apply operations in the order they were performed
        final_binary = binary.copy()
        for operation in st.session_state.operations:
            if operation['mode'] == 'Remove':
                mask = cv2.bitwise_not(operation['mask'])
                final_binary = cv2.bitwise_and(final_binary, mask)
            elif operation['mode'] == 'Include':
                final_binary = cv2.bitwise_or(final_binary, operation['mask'])

        # Find and draw contours on the edited binary image
        contour_detector = ContourDetector(image, final_binary, epsilon)
        segmented, output, contours = contour_detector.find_and_draw_contours()
        GCodeGenerator().generate_from_contours(contours)

        # Display output
        st.subheader('Segmented Bottles')
        st.image(segmented, caption='Segmented Bottles', use_column_width=True)

        st.subheader('Bottle Boundaries (Approximated)')
        st.image(output, caption='Bottle Boundaries', use_column_width=True)

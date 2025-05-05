import streamlit as st
import tensorflow as tf
from PIL import Image
from Unet import UNet
import keras
import cv2
import numpy as np
import subprocess
import os

# initialise session state variables for conditional rendering 
if 'mask_generated' not in st.session_state: # to check if mask is generated
    st.session_state.mask_generated = False 
if 'outlined_image' not in st.session_state: # to check if outlined mask image is generated
    st.session_state.outlined_image = None 
if 'mask_image' not in st.session_state: # to check if mask image is generated
    st.session_state.mask_image = None
if 'area_result' not in st.session_state: # to check if area result is generated
    st.session_state.area_result = None
if 'pred_mask' not in st.session_state: # to check if prediction mask is generated
    st.session_state.pred_mask = None

def download_file_from_google_drive(destination):
    """
        Download the model file from Google Drive.
    """
    # Access the file ID from Streamlit secrets
    file_id = st.secrets["model_file_id"]

    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # use subprocess to run gdown command
    subprocess.run(["gdown", "--id", file_id, "-O", destination], check=True)

@st.cache_resource # Decorator to cache non-data objects
def load_model(model_path: str) -> keras.Model:
    """
        Load the trained UNet model.
    """
    model = tf.keras.models.load_model(model_path, custom_objects={'UNet': UNet})
    return model

@st.cache_resource
def get_model(destination):
    """
        Download model once and cache it.
    """
    if not os.path.exists(destination):
        download_file_from_google_drive(destination)
    return load_model(destination)

def prep_img(uploaded_file) -> tf.Tensor:
    """
        Read and prepare the image as a tensor.
    """
    bytes_data = uploaded_file.read() 
    img = tf.image.decode_image(bytes_data, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    original_shape = tf.shape(img)

    return img, original_shape
    
def preprocess_image(uploaded_file) -> tf.Tensor:
    """
        Preprocess the input image before predicting.
    """

    # read image
    img, original_shape = prep_img(uploaded_file)

    # preprocess image
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0

    # add batch dimension
    img = tf.expand_dims(img, axis=0)

    return img, original_shape

def generate_mask(model: keras.Model, uploaded_file) -> tf.Tensor:
    """
        Generate mask for the input image using the pre-trained UNet model.
    """

    # preprocess image
    img, original_shape = preprocess_image(uploaded_file)

    # predict mask
    pred = model.predict(img)
    pred_mask = tf.cast(tf.greater(pred, 0.5), tf.float32)

    # resize mask to original shape
    pred_mask = tf.image.resize(pred_mask, (original_shape[0], original_shape[1]))
    # remove batch dimension
    pred_mask = tf.squeeze(pred_mask)
    # un-normalise mask
    pred_mask = tf.cast(pred_mask, tf.uint8) * 255

    return pred_mask

def outline_mask(pred_mask: tf.Tensor, uploaded_file) -> np.ndarray:
    """
        Outline the predicted mask.
    """
    # get contours
    contours, _ = cv2.findContours(pred_mask.numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours on original image
    img, _ = prep_img(uploaded_file)
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)

    # draw contours on the original wound image
    img = cv2.drawContours(
        img,
        contours,
        -1,  # draw all contours
        (0, 255, 0),  # green color
        1,  # thickness
        cv2.LINE_AA,  # line type
    )
    # convert back to RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# ----------------------------------------------------------------------------------------------------- #
# Define the functions for button actions
def show_mask(model, uploaded_file):
    """
        Generate and store the mask.
    """
    with st.spinner("Generating mask..."):
        # generate mask
        pred_mask = generate_mask(model, uploaded_file)
        
        # Store the mask in session state
        st.session_state.pred_mask = pred_mask
        st.session_state.mask_image = pred_mask.numpy()
        
        # change mask generated session state variable to True
        st.session_state.mask_generated = True

def show_outline(model, uploaded_file):
    """
        Outline the mask on the original image and store it.
    """
    with st.spinner("Outlining mask..."):
        # generate mask if not already generated
        if st.session_state.pred_mask is None:
            pred_mask = generate_mask(model, uploaded_file)
            st.session_state.pred_mask = pred_mask
        else:
            pred_mask = st.session_state.pred_mask
        
        # reset file pointer before passing to outline_mask
        uploaded_file.seek(0)
        pred_mask_outlined = outline_mask(pred_mask, uploaded_file)
        
        # Store the outlined image in session state
        st.session_state.outlined_image = pred_mask_outlined
        
        # change mask generated session state variable to True
        st.session_state.mask_generated = True

# calculating area

# ----------------------------------------------------------------------------------------------------- #

def run_app() -> None:
    """
        Run the Streamlit app.
    """

    st.set_page_config(
        page_title="Generate Mask",
        page_icon= "⚙️", 
    )
    # title
    st.write("## Image Segmentation & Area Calculation ⚙️")

    # upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], )

    # download model from Google Drive
    # Determine the absolute path to the directory containing the script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # go back two directories to reach the root directory (assessment3)
    BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))

    # Construct the absolute path to the destination file
    destination = os.path.join(BASE_DIR, "Checkpoints", "unet_best_model.keras")
    # download model if it doesn't exist and load it
    model = get_model(destination)

    if uploaded_file is not None:
        # Always display the original image first
        img = Image.open(uploaded_file).convert("RGB")
        uploaded_file.seek(0)  # Reset file pointer
        st.image(img, caption='Uploaded Image.', use_container_width=True)

        # display buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Generate Mask", on_click=show_mask, args=(model, uploaded_file))
        with col2:
            st.button("Show Mask on Image", on_click=show_outline, args=(model, uploaded_file))
        # show area calc button after btn1 or btn2 are clicked
        with col3:
            if st.session_state.mask_generated:
                st.button("Calculate Area", on_click=calc_area, args=(model, uploaded_file))

        # Display results based on session state
        if st.session_state.mask_image is not None:
            st.image(st.session_state.mask_image, caption='Generated Mask', use_container_width=True)
        
        if st.session_state.outlined_image is not None:
            st.image(st.session_state.outlined_image, caption='Outlined Mask', use_container_width=True)
        
        if st.session_state.area_result is not None:
            st.success(f"Area of the wound: {st.session_state.area_result}", icon="✅")

# call in main
if __name__ == "__main__":
    run_app()
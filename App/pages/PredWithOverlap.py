import streamlit as st
import tensorflow as tf
from PIL import Image
from Unet import UNet
import keras
import cv2
import numpy as np

# load the model
def load_model(model_path: str) -> keras.Model:
    """
        Load the pre-trained UNet model.
    """
    model = tf.keras.models.load_model(model_path, custom_objects={'UNet': UNet})
    return model

# preprocess image

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

# generate mask
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

# outline the mask
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

# # calculate area
# def calc_area() -> float:

# run app
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

    # load model
    model = load_model('../Checkpoints/unet_best_model.keras')

    if uploaded_file is not None:
        # read image
        img = Image.open(uploaded_file).convert("RGB")

        # reset file pointer before passing to generate_mask
        uploaded_file.seek(0)

        st.image(img, caption='Uploaded Image.', use_container_width=True)

        # run prediction when button is pressed
        if st.button("Generate Mask"):
            with st.spinner("Generating mask..."):
                
                # generate mask
                pred_mask = generate_mask(model, uploaded_file)

                # outline mask
                # reset file pointer before passing to generate_mask
                uploaded_file.seek(0)
                pred_mask_outlined = outline_mask(pred_mask, uploaded_file)

                # # display mask
                # st.image(pred_mask.numpy(), caption='Predicted Mask.', use_container_width=True)

                # display outlined mask
                st.image(pred_mask_outlined, caption='Outlined Mask', use_container_width=True)

# call in main
if __name__ == "__main__":
    run_app()
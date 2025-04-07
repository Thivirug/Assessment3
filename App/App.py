import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from Assessment3.App.Unet import UNet
import keras

# load the model
def load_model(model_path: str) -> keras.Model:
    """
        Load the pre-trained UNet model.
    """
    model = tf.keras.models.load_model(model_path, custom_objects={'UNet': UNet})
    return model

# preprocess image
def preprocess_image(uploaded_file) -> tf.Tensor:
    """
        Preprocess the input image before predicting.
    """

    # read image
    # img = tf.io.read_file(img_path)
    bytes_data = uploaded_file.read() 
    img = tf.image.decode_image(bytes_data, channels=3, expand_animations=False)
    img = tf.ensure_shape(img, [None, None, 3])
    original_shape = tf.shape(img)

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


# run app
def run_app() -> None:
    """
        Run the Streamlit app.
    """

    # title
    st.title("Image Segmentation")

    # upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # load model
    model = load_model('Checkpoints/unet_best_model.keras')

    if uploaded_file is not None:
        # read image
        img = Image.open(uploaded_file).convert("RGB")

        # Reset file pointer before passing to generate_mask
        uploaded_file.seek(0)

        st.image(img, caption='Uploaded Image.', use_container_width=True)

        # run prediction when button is pressed
        if st.button("Generate Mask"):
            with st.spinner("Generating mask..."):
                
                # generate mask
                pred_mask = generate_mask(model, uploaded_file)

                # display mask
                st.image(pred_mask.numpy(), caption='Predicted Mask.', use_container_width=True)

# call in main
if __name__ == "__main__":
    run_app()
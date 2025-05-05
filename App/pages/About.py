import streamlit as st
import os

def model_info() -> None:
    """
        Display the model information.
    """
    st.subheader("Model Information", divider=True)
    st.markdown("""
        The model used is a U-Net architecture, which is a convolutional neural network (CNN) designed for image segmentation tasks. 
        It consists of an encoder-decoder structure with skip connections, allowing it to capture both local and global features effectively.
    """)

def dataset_info() -> None:
    """
        Display the dataset information.
    """
    st.subheader("Dataset Information", divider=True)
    st.markdown("""
        The dataset used is from the Foot Ulcer Segmentation Challenge:
        * 810 training images + masks  
        * 200 validation images + masks  
        * 200 testing images (no masks)  

        We performed data augmentation with Albumentations, increasing the dataset to 4860 training samples. Each image is paired with a binary mask used for segmentation.
    """)

def run_app() -> None:
    # Set up the page configuration
    st.set_page_config(
        page_title="About",
        page_icon="👀"
    )

    # Title 
    st.header("About", divider=True)
    
    # Determine the absolute path to the directory containing the script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # go back two directories to reach the root directory (assessment3)
    BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))

    # Construct the absolute path to the destination file
    img_dir = os.path.join(BASE_DIR, "App", "Assets", "unet.jpg")
    st.image(img_dir, caption="The U-Net", use_container_width=True)

    # Model information
    model_info()

    # Divider
    st.markdown("---")

    # Dataset information
    dataset_info()
    
if __name__ == "__main__":
    run_app()

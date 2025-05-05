import streamlit as st
import os

def run_app() -> None:
    # Set up the page configuration
    st.set_page_config(
        page_title="About",
        page_icon="👀"
    )

    # Title 
    st.header("About", divider=True)

    # About the App 
    
    # Determine the absolute path to the directory containing the script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # go back two directories to reach the root directory (assessment3)
    BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))

    # Construct the absolute path to the destination file
    img_dir = os.path.join(BASE_DIR, "App", "Assets", "unet.jpg")
    st.image(img_dir, caption="The U-Net", use_container_width=True)
    
    st.subheader("More detailed description of MedAI's usage", divider=True)
    st.markdown("""
        Our application uses deep learning to segment foot ulcers from clinical images. This allows medical professionals to quickly and accurately assess the size and severity of wounds.  
        
        The goal is to provide an easy-to-use, real-time tool that automates wound segmentation and helps calculate surface area using a reference marker.
    """)

    # Divider
    st.markdown("---")

    # Dataset Used Header 
    st.header("Dataset used")

    # Second Row: Dataset Details 
    st.subheader("Description of our dataset")
    st.markdown("""
        The dataset used is from the Foot Ulcer Segmentation Challenge:
        - 810 training images + masks  
        - 200 validation images + masks  
        - 200 testing images (no masks)  

        We performed data augmentation with Albumentations, increasing the dataset to 4860 training samples. Each image is paired with a binary mask used for segmentation.
    """)


if __name__ == "__main__":
    run_app()

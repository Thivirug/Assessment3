import streamlit as st
import os

def model_info() -> None:
    """
        Display the model information.
    """
    st.subheader("Model Information", divider=True)
    st.markdown("""
        * The model used is a U-Net architecture, which is a convolutional neural network (CNN) designed for image segmentation tasks. 
        * It consists of an encoder-decoder structure with skip connections, allowing it to capture both local and global features effectively.
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

def application_info() -> None:
    """
        Display the application information.
    """
    st.subheader("Application", divider=True)
    st.markdown("""
        * This tool assists healthcare professionals in identifying and segmenting wounds from clinical images. 
        * It can be used to monitor wound healing and guide treatment planning.
    """)

def limitations() -> None:
    """
        Display the limitations of the application.
    """
    st.subheader("Limitations", divider=True)
    st.markdown("""
        * May not generalise well to some wound types or low-quality images.
        * Area calculation may not be scientifically accurate due to the simple approach used. 
        * The model is not a substitute for professional medical advice or diagnosis.
        * It is essential to consult a healthcare professional for accurate assessment and treatment.
    """)

def future_improvements() -> None:
    """
        Display the future improvements for the application.
    """
    st.subheader("Future Improvements", divider=True)
    st.markdown("""
        * We plan to extend the model's capabilities to generalise better to different wound types.
        * We aim to improve the accuracy of area calculation by implementing more advanced techniques.
        * We will explore the use of additional data sources and advanced models to enhance performance.
    """)

def acknowledgements() -> None:
    """
        Display the acknowledgements.
    """
    st.subheader("Acknowledgements", divider=True)
    st.markdown("""
        * We would like to thank our subject coordinator, Dr. Nabin and tutor Sudharshan for their guidance and support throughout this project.
        * We acknowledge the use of the Foot Ulcer Segmentation Challenge dataset for training and evaluating our model. : https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge
        * The U-Net architecture was inspired by the original paper by Ronneberger et al. (2015). : https://arxiv.org/abs/1505.04597
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

    # Dataset information
    dataset_info()

    # Application information
    application_info()

    # Limitations
    limitations()

    # Future improvements
    future_improvements()

    # Acknowledgements
    acknowledgements()
    
if __name__ == "__main__":
    run_app()

import streamlit as st

st.set_page_config(
    page_title="MedAI",
    page_icon="🩺",
)

st.write("# Greetings! 👋")

st.markdown(
    """
    ### Welcome to MedAI, your go-to platform for medical image segmentation! 🩺
    This web application is designed to help you segment medical images using a UNet model.
    You can upload your own images and visualize the segmentation results in real-time.
    Additionally, it outputs the area of the wounded region in square centimeters (cm²) for your convenience.

    **👈 Navigate using the sidebar** 
    * About -> Learn more about the app, how the model was trained, and future plans.
    * Contact -> Get in touch with us for any inquiries or feedback.
    * Result -> Upload your own images for segmentation and view the results.
"""
)
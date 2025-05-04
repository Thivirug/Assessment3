import streamlit as st

def run() -> None:
    """
        Run the Streamlit app's entry point.
    """
    # Set up the page configuration
    st.set_page_config(
        page_title="MedAI",
        page_icon="🩺",
    )

    # Set the title and header
    st.write("# Greetings! 👋")

    # Add a subheader
    st.subheader("MedAI: Wound Image Segmentation", divider=True)
    # Add an image
    st.image("../App/Assets/logo.png", width=300)
    # Add a horizontal line
    st.markdown(
        """
        ---
        """
    )
    # Add a description
    st.markdown(
        """
        ### Welcome to MedAI, your go-to platform for wound image segmentation! 🩺
        This web application is designed to help you segment wound images using a UNet model.
        You can upload your own images and visualise the segmentation results in real-time.
        Additionally, it outputs the area of the wounded region in square centimeters (cm²) for your convenience.

        **👈 Navigate using the sidebar** 
        * About -> Learn more about the app, how the model was trained, and future plans.
        * Contact -> Get in touch with us for any inquiries or feedback.
        * Predictions -> Upload your own images for segmentation and view the predicted results.
    """
    )


if __name__ == "__main__":
    run()
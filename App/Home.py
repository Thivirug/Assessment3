import streamlit as st

def page_info() -> None:
    """
        Display the page information.
    """

    col1, col2, col3 = st.columns([0.5, 0.2, 3])
    with col1:
        st.page_link("pages/About.py", label="About", icon="👀")
    with col2:
        st.write("➡️")
    with col3:
        st.write("Learn more about the app, how the model was trained, and future plans.")

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
    st.image("App/Assets/logo.png", width=300)
    
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

        ### How to Use:
        1. Upload your wound image in JPG, JPEG, or PNG format.
        2. The model will process the image and generate a segmented mask.
        3. The segmented area will be displayed along with the original image.

        ---

        **👈 Navigate to Pages**
    """
    )

    # display the page information
    page_info()

if __name__ == "__main__":
    run()
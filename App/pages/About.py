import streamlit as st

def run_app() -> None:
    # Set up the page configuration
    st.set_page_config(
        page_title="About",
        page_icon="👀"
    )

    # Title 
    st.title("About us")

    # First Row: About the App 
    col1, col2 = st.columns(2)
    with col1:
        st.image("../Assets/unet.jpg", caption="Another picture related to MedAI", use_container_width=True)
    with col2:
        st.subheader("More detailed description of MedAI's usage")
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

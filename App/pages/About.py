import streamlit as st

# Set up the page configuration
st.set_page_config(
    page_title="About",
    page_icon="👀"
)

def run_app() -> None:

    # Title 
    st.title("About us")

    # First Row: About the App 
    col1, col2 = st.columns(2)
    with col1:
        st.image("path_to_medai_image.png", caption="Another picture related to MedAI", use_column_width=True)
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
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Description of our dataset")
        st.markdown("""
            The dataset used is from the Foot Ulcer Segmentation Challenge:
            - 810 training images + masks  
            - 200 validation images + masks  
            - 200 testing images (no masks)  

            We performed data augmentation with Albumentations, increasing the dataset to 4860 training samples. Each image is paired with a binary mask used for segmentation.
        """)

    with col4:
        st.image("path_to_sample_image.png", caption="Pictures of samples from our dataset", use_column_width=True)


if __name__ == "__main__":
    run_app()

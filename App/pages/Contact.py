import streamlit as st

def run_app() -> None:
    # Set up the page configuration
    st.set_page_config(
        page_title="Contact Us",
        page_icon="✉️"
    )

    st.header("Contact us", divider=True)

    st.markdown("""
        If you have questions about the project, would like to report an issue, or are interested in collaboration, feel free to contact us via the emails below.
    """)

    st.markdown("""
    ### 🧑‍💻 Team Members & Roles
    - **Thiviru Gunawardena (14542791)** – Data Collection, Model Training, & Model Deployment
    - **Jachym Zamouril (25639802)** – Frontend Development & Documentation
    - **Carlos Daroy (24752370)** – Model Evaluation, Area Calculation, & Documentation  
    """)

    st.markdown("### Thiviru Gunawardena – 14542791 – [thiviru.gunawardena@student.uts.edu.au](mailto:thiviru.gunawardena@student.uts.edu.au)")
    st.markdown("### Jachym Zamouril – 25639802 – [Jachym.Zamouril@student.uts.edu.au](mailto:Jachym.Zamouril@student.uts.edu.au)")
    st.markdown("### Carlos Daroy – 24752370 – [Carlos.Daroy@student.uts.edu.au](mailto:Carlos.Daroy@student.uts.edu.au)")

    st.markdown("""
    ### 📁 Additional Links
    * [Project GitHub Repository](https://github.com/Thivirug/Assessment3)
    * [Video Pitch](To be added)
    """)
if __name__ == "__main__":
    run_app()

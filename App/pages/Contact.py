import streamlit as st

def run_app() -> None:
    # Set up the page configuration
    st.set_page_config(
        page_title="Contact Us",
        page_icon="✉️"
    )

    st.title("Contact us")

    st.markdown("### Thiviru Gunawardena – 14542791 – [thirivu.gunawardena@student.uts.edu.au](mailto:thirivu.gunawardena@student.uts.edu.au)")
    st.markdown("### Jachym Zamouril – 25639802 – [Jachym.Zamouril@student.uts.edu.au](mailto:Jachym.Zamouril@student.uts.edu.au)")
    st.markdown("### Carlos Daroy – 24752370 – [Carlos.Daroy@student.uts.edu.au](mailto:Carlos.Daroy@student.uts.edu.au)")

if __name__ == "__main__":
    run_app()

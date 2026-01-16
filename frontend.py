import streamlit as st
from PIL import Image


def main() -> None:
    """Main function of the Streamlit frontend for Image-to-LaTeX translation."""
    st.set_page_config(page_title="Image-to-LaTeX", page_icon="📐", layout="wide")

    # Header
    st.markdown("# Image-to-LaTeX Translator")
    st.markdown("Convert images of mathematical equations to LaTeX code", help="Upload an equation image")

    # Main content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📸 Upload Image")
        uploaded_file = st.file_uploader(
            "Drop your equation image here", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)  # type: ignore[call-arg]

    with col2:
        st.markdown("### 📝 LaTeX Output")
        if uploaded_file is not None:
            st.info("Backend connection coming soon...")
            st.markdown("**Predicted LaTeX:**")
            latex_code = r"\frac{x}{y} + \sqrt{z}"
            st.code(latex_code, language="latex")
            st.button("Copy LaTeX")
        else:
            st.warning("Upload an image to get started")

    st.divider()

    # Info section
    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
        1. **Upload** an image of a mathematical equation
        2. **Process** using our ML model
        3. **Get** LaTeX code output
        4. **Copy** and use in your documents

        **Supported formats:** JPG, JPEG, PNG
        """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <small>Made with using Streamlit</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

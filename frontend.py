import streamlit as st
from PIL import Image


def main() -> None:
    """Main function of the Streamlit frontend for Image-to-LaTeX translation."""
    st.set_page_config(page_title="Image-to-LaTeX", layout="wide")

    st.title("Image-to-LaTeX Translator")
    st.markdown("Convert images of mathematical equations to LaTeX code")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Upload an equation image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("LaTeX Output")
        if uploaded_file is not None:
            st.info("🔄 Backend connection coming soon...")
            st.markdown("**Predicted LaTeX:**")
            st.code(r"\frac{x}{y} + \sqrt{z}", language="latex")
        else:
            st.info("👈 Upload an image to get started")

    st.divider()

    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Upload** an image of a mathematical equation
        2. The model translates it to **LaTeX** code
        3. Copy the LaTeX and use it in your documents

        Supported formats: JPG, JPEG, PNG
        """)


if __name__ == "__main__":
    main()
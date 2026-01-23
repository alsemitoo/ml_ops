import os

import requests  # type: ignore
import streamlit as st
from PIL import Image

# Get API URL from environment variable (set in Cloud Run) or use default for local testing
API_URL = os.getenv("API_URL", "http://localhost:8080")


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
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 📝 LaTeX Output")
        if uploaded_file is not None:
            with st.spinner("🔄 Processing image..."):
                try:
                    # Prepare the file for API request
                    uploaded_file.seek(0)  # Reset file pointer
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

                    # Make API request
                    response = requests.post(f"{API_URL}/predict/", files=files, timeout=120)

                    if response.status_code == 200:
                        result = response.json()
                        latex_code = result.get("prediction", "")

                        st.success("✅ Prediction complete!")
                        st.markdown("**Predicted LaTeX:**")
                        st.code(latex_code, language="latex")

                        if st.button("📋 Copy LaTeX"):
                            st.toast("Copied to clipboard!")
                    else:
                        st.error(f"❌ API error: {response.status_code}")
                        st.json(response.json())

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_URL}")
                    st.info("Make sure the API is running and accessible.")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Try a smaller image.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
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

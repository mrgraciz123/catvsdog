# -*- coding: utf-8 -*-
"""A polished Streamlit app for classifying cat and dog images."""

import os
import tempfile

import cv2
import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model.pkl")
IMG_SIZE = 64


def apply_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #f5f9ff 0%, #eef4ff 100%);
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 900px;
            }
            /* Ensure all standard labels and paragraph text are dark for readability */
            label, .stMarkdown p, .stText p, [data-testid="stWidgetLabel"] p {
                color: #1e293b !important;
                font-weight: 500;
            }
            .hero-card {
                background: linear-gradient(135deg, #007aff 0%, #34c759 100%);
                border-radius: 22px;
                padding: 1.1rem 1.2rem;
                color: white !important;
                box-shadow: 0 10px 24px rgba(0, 122, 255, 0.2);
                margin-bottom: 1rem;
            }
            .hero-card h1 {
                margin: 0;
                font-size: 1.8rem;
                color: white !important;
            }
            .hero-card p {
                margin: 0.3rem 0 0;
                opacity: 0.95;
                color: white !important;
            }
            .info-card {
                background: white;
                color: #334155 !important;
                border-radius: 18px;
                padding: 0.9rem 1rem;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.15);
                margin-bottom: 1rem;
            }
            .info-card strong {
                color: #0f172a !important;
            }
            .info-card code {
                color: #007aff !important;
                background-color: #f1f5f9 !important;
                padding: 0.1rem 0.3rem !important;
                border-radius: 4px !important;
            }
            .result-card {
                background: #f7fbff;
                color: #1e293b !important;
                border-left: 5px solid #007aff;
                border-radius: 16px;
                padding: 0.9rem 1rem;
                margin-top: 0.8rem;
            }
            .result-card h3 {
                color: #0f172a !important;
                margin-top: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model(model_file=None):
    if model_file is None:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError("The model file was not found. Please upload a .pkl model file.")
        return joblib.load(DEFAULT_MODEL_PATH)

    if hasattr(model_file, "getvalue"):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_file.write(model_file.getvalue())
            temp_path = temp_file.name

        try:
            return joblib.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    model_path = model_file if isinstance(model_file, str) else str(model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError("The model file was not found. Please upload a .pkl model file.")
    return joblib.load(model_path)


def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    resized = resized.flatten()
    return image, resized


def main():
    st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout="centered")
    apply_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h1>🐱 Cat vs Dog Image Classifier</h1>
            <p>Upload a photo and get a quick prediction in a clean, modern experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <strong>Upload an image</strong><br>
            Supported formats: JPG, JPEG, and PNG.<br>
            Also upload your trained <code>.pkl</code> model file if you are not using the default one.
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_upload = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
    model_upload = st.file_uploader("Upload Model (.pkl)", type=["pkl"])

    if image_upload is not None:
        try:
            image, resized = preprocess_image(image_upload)
            model = load_model(model_upload)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=300)

        prediction = model.predict([resized])[0]
        probability = model.predict_proba([resized])[0]

        label = "CAT" if prediction == 0 else "DOG"
        emoji = "🐱" if prediction == 0 else "🐶"
        confidence = probability[prediction] * 100

        st.markdown(
            f"""
            <div class="result-card">
                <h3>{emoji} {label}</h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(confidence / 100.0, 1.0))
        st.write(f"Cat Probability: {probability[0] * 100:.2f}%")
        st.write(f"Dog Probability: {probability[1] * 100:.2f}%")
    else:
        st.info("Upload an image to begin classification.")


if __name__ == "__main__":
    main()
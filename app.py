# -*- coding: utf-8 -*-
"""A polished Streamlit app for classifying cat and dog images."""

import os
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
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
            
            .stApp {
                background: radial-gradient(circle at top center, #0f172a 0%, #020617 100%);
                font-family: 'Plus Jakarta Sans', sans-serif;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 4rem;
                max-width: 800px;
            }
            
            /* Global text settings */
            label, .stMarkdown p, .stText p, [data-testid="stWidgetLabel"] p {
                color: #cbd5e1 !important;
                font-weight: 500;
            }
            
            /* Custom Hero Card */
            .hero-card {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 2rem;
                text-align: center;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                margin-bottom: 1.5rem;
                backdrop-filter: blur(12px);
            }
            .hero-card h1 {
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #a78bfa 0%, #818cf8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0 0 0.5rem 0;
            }
            .hero-card p {
                color: #94a3b8 !important;
                font-size: 1rem;
                font-weight: 400;
                margin: 0;
            }
            
            /* Custom Info Card */
            .info-card {
                background: rgba(30, 41, 59, 0.4);
                color: #cbd5e1 !important;
                border-radius: 16px;
                padding: 1.2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.05);
                margin-bottom: 2rem;
                backdrop-filter: blur(8px);
                font-size: 0.95rem;
            }
            .info-card strong {
                color: #f8fafc !important;
                font-size: 1.05rem;
            }
            .info-card code {
                color: #a78bfa !important;
                background-color: rgba(167, 139, 250, 0.1) !important;
                padding: 0.2rem 0.4rem !important;
                border-radius: 6px !important;
                font-weight: 600;
            }
            
            /* File Uploader styling */
            [data-testid="stFileUploader"] {
                background: rgba(30, 41, 59, 0.3);
                border: 1px dashed rgba(255, 255, 255, 0.15) !important;
                border-radius: 18px !important;
                padding: 1rem;
                transition: all 0.3s ease;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #818cf8 !important;
                background: rgba(30, 41, 59, 0.5);
            }
            
            /* Result Card */
            .result-container {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
                animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            }
            
            .result-card {
                background: rgba(30, 41, 59, 0.6);
                border-radius: 24px;
                padding: 2rem;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                text-align: center;
                backdrop-filter: blur(16px);
            }
            
            .result-card.cat {
                border-top: 6px solid #60a5fa;
                box-shadow: 0 10px 40px rgba(96, 165, 250, 0.15);
            }
            .result-card.dog {
                border-top: 6px solid #a78bfa;
                box-shadow: 0 10px 40px rgba(167, 139, 250, 0.15);
            }
            
            .result-label {
                font-size: 2.5rem;
                font-weight: 800;
                margin: 0.5rem 0;
            }
            .result-label.cat {
                color: #60a5fa !important;
                text-shadow: 0 0 15px rgba(96, 165, 250, 0.3);
            }
            .result-label.dog {
                color: #a78bfa !important;
                text-shadow: 0 0 15px rgba(167, 139, 250, 0.3);
            }
            
            .confidence-meter {
                font-size: 1.2rem;
                color: #94a3b8 !important;
                margin-bottom: 1.5rem;
            }
            .confidence-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #f8fafc !important;
            }
            
            /* Custom animated progress bar container */
            .meter-container {
                background: rgba(15, 23, 42, 0.6);
                border-radius: 12px;
                height: 12px;
                width: 100%;
                overflow: hidden;
                margin: 1rem 0 2rem 0;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            .meter-bar {
                height: 100%;
                border-radius: 12px;
                transition: width 1s ease-out;
            }
            .meter-bar.cat {
                background: linear-gradient(90deg, #3b82f6, #60a5fa);
                box-shadow: 0 0 10px rgba(96, 165, 250, 0.5);
            }
            .meter-bar.dog {
                background: linear-gradient(90deg, #8b5cf6, #a78bfa);
                box-shadow: 0 0 10px rgba(167, 139, 250, 0.5);
            }
            
            /* Probability Breakdown Grid */
            .prob-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-top: 1rem;
            }
            .prob-card {
                background: rgba(15, 23, 42, 0.4);
                border-radius: 16px;
                padding: 1.2rem;
                border: 1px solid rgba(255, 255, 255, 0.05);
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.3rem;
            }
            .prob-card.active.cat {
                border: 1px solid rgba(96, 165, 250, 0.3);
                background: rgba(96, 165, 250, 0.05);
            }
            .prob-card.active.dog {
                border: 1px solid rgba(167, 139, 250, 0.3);
                background: rgba(167, 139, 250, 0.05);
            }
            .prob-title {
                font-size: 0.9rem;
                color: #94a3b8 !important;
                font-weight: 600;
            }
            .prob-value {
                font-size: 1.6rem;
                font-weight: 700;
            }
            .prob-value.cat {
                color: #60a5fa !important;
            }
            .prob-value.dog {
                color: #a78bfa !important;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(15px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError("The built-in model file (cat_dog_model.pkl) was not found in the workspace.")
    return joblib.load(DEFAULT_MODEL_PATH)


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
            <p>Upload a pet photo and get an instant prediction using our machine learning model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <strong>How it works:</strong><br>
            1. Select a JPEG or PNG image of a cat or dog.<br>
            2. The app uses the pre-trained model <code>cat_dog_model.pkl</code> to run the classification in real-time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_upload = st.file_uploader("Choose a photo of a Cat or Dog", type=["jpg", "jpeg", "png"])

    if image_upload is not None:
        try:
            image, resized = preprocess_image(image_upload)
            model = load_model()
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return

        st.write("")  # Spacing
        
        # Display the layout
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("<p style='font-weight: 600; margin-bottom: 0.8rem; color: #f8fafc;'>Uploaded Image</p>", unsafe_allow_html=True)
            # Display image with rounded corners
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        prediction = model.predict([resized])[0]
        probability = model.predict_proba([resized])[0]

        label = "CAT" if prediction == 0 else "DOG"
        class_name = "cat" if prediction == 0 else "dog"
        emoji = "🐱" if prediction == 0 else "🐶"
        confidence = probability[prediction] * 100

        cat_prob = probability[0] * 100
        dog_prob = probability[1] * 100

        cat_active = "active" if prediction == 0 else ""
        dog_active = "active" if prediction == 1 else ""

        with col2:
            st.markdown(
                f"""
<div class="result-container">
    <div class="result-card {class_name}">
        <div style="font-size: 0.95rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Analysis Result</div>
        <div class="result-label {class_name}">{emoji} {label}</div>
        <div class="confidence-meter">
            Confidence: <span class="confidence-value">{confidence:.1f}%</span>
        </div>
        <div class="meter-container">
            <div class="meter-bar {class_name}" style="width: {confidence:.1f}%;"></div>
        </div>
        <div class="prob-grid">
            <div class="prob-card {cat_active} cat">
                <div class="prob-title">🐱 Cat Prob</div>
                <div class="prob-value cat">{cat_prob:.1f}%</div>
            </div>
            <div class="prob-card {dog_active} dog">
                <div class="prob-title">🐶 Dog Prob</div>
                <div class="prob-value dog">{dog_prob:.1f}%</div>
            </div>
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 1rem; color: #64748b; font-style: italic;">
                Upload a pet photo above to begin the classification.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
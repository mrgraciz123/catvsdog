# -*- coding: utf-8 -*-
"""Train and run a simple cat-vs-dog classifier with Streamlit."""

import os

import cv2
import joblib
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Sample")
CLASSES = ["cat", "dog"]
IMG_SIZE = 64
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model.pkl")


def apply_ios_style():
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 900px;
            }
            .hero-card {
                background: linear-gradient(135deg, #007aff 0%, #34c759 100%);
                border-radius: 24px;
                padding: 1.2rem 1.3rem;
                color: white;
                box-shadow: 0 12px 30px rgba(0, 122, 255, 0.2);
                margin-bottom: 1rem;
            }
            .hero-card h1 {
                margin: 0;
                font-size: 1.8rem;
            }
            .hero-card p {
                margin: 0.3rem 0 0;
                opacity: 0.95;
            }
            .info-card {
                background: white;
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.15);
                margin-bottom: 1rem;
            }
            .result-card {
                background: #f7fbff;
                border-left: 5px solid #007aff;
                border-radius: 16px;
                padding: 0.9rem 1rem;
                margin-top: 0.8rem;
            }
            div[data-testid="stFileUploader"] > label {
                font-weight: 600;
                color: #1f2937;
            }
            button[kind="primary"] {
                border-radius: 999px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_images():
    images = []
    labels = []

    for label, folder_name in enumerate(CLASSES):
        folder_path = os.path.join(DATASET_PATH, folder_name)

        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).flatten()
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


def train_model():
    X, y = load_images()
    print("Training Images", len(X))

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Model Saved Successfully!")

    return model


def main():
    st.set_page_config(
        page_title="Cat vs Dog Classifier",
        page_icon="🐶",
        layout="centered",
    )

    apply_ios_style()

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_model()

    st.markdown(
        """
        <div class="hero-card">
            <h1>🐱 Cat vs Dog Classifier</h1>
            <p>Snap a photo and let the app identify it in a clean, modern experience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <strong>Upload a photo</strong><br>
            Supported formats: JPG, JPEG, and PNG.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Unable to read the uploaded image.")
            return

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, channels="BGR", width=280)
        with col2:
            resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).flatten()
            prediction = model.predict([resized])[0]
            probability = model.predict_proba([resized])[0]

            if prediction == 0:
                label = "Cat"
                emoji = "🐱"
            else:
                label = "Dog"
                emoji = "🐶"

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>{emoji} {label}</h3>
                    <p><strong>Confidence:</strong> {probability[prediction] * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(probability[prediction], 1.0))
            st.write(f"Cat Probability: {probability[0] * 100:.2f}%")
            st.write(f"Dog Probability: {probability[1] * 100:.2f}%")
    else:
        st.info("Upload an image to begin classification.")


if __name__ == "__main__":
    main()


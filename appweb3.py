import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from face_recognition import preprocessing
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

# Cache model download and loading
@st.cache_resource
def load_model():
    try:
        # Download the model from Hugging Face Hub
        repo_id = "Yashas2477/SE2_og"  # Replace with your Hugging Face repo
        filename = "face_recogniser.pkl"  # Replace with your model filename

        st.info("Downloading model from Hugging Face...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir="model_cache")
        st.success("Model downloaded successfully!")

        # Load the model
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

face_recogniser = load_model()
preprocess = preprocessing.ExifOrientationNormalize()

# Define the video transformer class
class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to RGB
        img = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Preprocess and predict faces
        pil_img = preprocess(pil_img).convert('RGB')
        faces = face_recogniser(pil_img)

        # Annotate the frame
        for face in faces:
            bb = face.bb._asdict()
            top_left = (int(bb['left']), int(bb['top']))
            bottom_right = (int(bb['right']), int(bb['bottom']))
            label = face.top_prediction.label
            confidence = face.top_prediction.confidence

            # Draw bounding box and label
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                img, f"{label} ({confidence:.2f})", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Streamlit app
st.title("Live Face Recognition")
st.write("This app performs face recognition on live webcam feed using a pre-trained model.")

# Start the webcam with webrtc_streamer
webrtc_streamer(key="face-recognition", video_transformer_factory=FaceRecognitionTransformer)

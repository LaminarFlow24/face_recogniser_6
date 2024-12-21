import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from PIL import Image
from face_recognition import preprocessing
from huggingface_hub import hf_hub_download
import joblib

# Define the Hugging Face repository details
REPO_ID = "Yashas2477/SE2_og"
FILENAME = "face_recogniser.pkl"

# Cache the model download
@st.cache_data
def download_model_from_huggingface():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir="model_cache")
        return model_path
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        raise

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model_path = download_model_from_huggingface()
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

# Load the cached model
face_recogniser = load_model()
preprocess = preprocessing.ExifOrientationNormalize()

# Streamlit app
st.title("Live Face Recognition")
st.write("This app performs face recognition using webcam feed.")

# Video processing class for Streamlit-webrtc
class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        
        # Convert frame to PIL Image for preprocessing
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_img = preprocess(pil_img)
        pil_img = pil_img.convert('RGB')

        # Predict faces
        faces = face_recogniser(pil_img)

        # Annotate the frame with bounding boxes and labels
        for face in faces:
            bb = face.bb._asdict()
            top_left = (int(bb['left']), int(bb['top']))
            bottom_right = (int(bb['right']), int(bb['bottom']))
            label = face.top_prediction.label
            confidence = face.top_prediction.confidence

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} ({confidence:.2f})", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        return frame

# Streamlit-webrtc integration
webrtc_streamer(key="face-recognition", video_transformer_factory=FaceRecognitionTransformer)

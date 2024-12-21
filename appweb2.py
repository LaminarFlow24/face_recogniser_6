import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from face_recognition import preprocessing
from huggingface_hub import hf_hub_download

# Define the Hugging Face repository details
REPO_ID = "Yashas2477/SE2_og"  # Replace with your Hugging Face repository
FILENAME = "face_recogniser.pkl"  # Replace with your model filename

# Cache the model download
@st.cache_data
def download_model_from_huggingface():
    st.info("Downloading model from Hugging Face...")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir="model_cache")
        st.success("Model downloaded successfully!")
        return model_path
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        raise

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model_path = download_model_from_huggingface()
        st.write(f"Model loaded from: {model_path}")
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

# Load the cached model
face_recogniser = load_model()
preprocess = preprocessing.ExifOrientationNormalize()

# Streamlit app
st.title("Live Face Recognition")
st.write("This app performs face recognition on live webcam feed.")

# Helper function to process and predict faces
def process_frame(frame):
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

# Start the webcam feed
run = st.checkbox('Start Webcam', key="start_webcam")
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("Unable to access the webcam. Make sure it is connected and try again.")
    else:
        st.info("Uncheck the 'Start Webcam' box to stop the webcam.")

    while st.session_state["start_webcam"]:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from webcam.")
            break

        # Process the frame for face recognition
        annotated_frame = process_frame(frame)

        # Display the annotated frame in Streamlit
        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

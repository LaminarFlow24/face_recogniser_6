import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from face_recognition import preprocessing
from huggingface_hub import hf_hub_download

# Define the Hugging Face repository details
REPO_ID = "Yashas2477/SE2_og"  # Replace with your Hugging Face repository
FILENAME = "face_recogniser_out_80.pkl"  # Replace with your model filename

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
st.write("This app performs face recognition on webcam images.")

# Helper function to process and predict faces
def process_frame(pil_img):
    pil_img = preprocess(pil_img)
    pil_img = pil_img.convert('RGB')

    # Predict faces
    faces = face_recogniser(pil_img)

    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Annotate the frame with bounding boxes and labels
    for face in faces:
        bb = face.bb._asdict()
        top_left = (int(bb['left']), int(bb['top']))
        bottom_right = (int(bb['right']), int(bb['bottom']))
        label = face.top_prediction.label
        confidence = face.top_prediction.confidence

        # Draw bounding box and label on the frame
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        cv2.putText(
            frame, f"{label} ({confidence:.2f})", (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    return frame

# Capture image using Streamlit's webcam input
image_data = st.camera_input("Take a photo for face recognition")

if image_data:
    # Convert the uploaded image to a PIL Image
    pil_image = Image.open(image_data)

    # Process the frame for face recognition
    annotated_frame = process_frame(pil_image)

    # Display the annotated frame in Streamlit
    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont
from face_recognition import preprocessing
from huggingface_hub import hf_hub_download

# Define the Hugging Face repository details
REPO_ID = "Yashas2477/SE2_og"  # Replace with your Hugging Face repository
FILENAME = "face_recogniser_out_75.pkl"  # Replace with your model filename

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
def process_image(pil_img):
    pil_img = preprocess(pil_img)
    pil_img = pil_img.convert('RGB')

    # Predict faces
    faces = face_recogniser(pil_img)

    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(pil_img)
    for face in faces:
        bb = face.bb._asdict()
        top_left = (int(bb['left']), int(bb['top']))
        bottom_right = (int(bb['right']), int(bb['bottom']))
        label = face.top_prediction.label
        confidence = face.top_prediction.confidence

        # Define colors and draw bounding box
        color = "green" if label != "Unknown" else "red"  # Green for known, red for unknown
        draw.rectangle([top_left, bottom_right], outline=color, width=2)

        # Draw label and confidence
        text = f"{label} ({confidence:.2f})"
        draw.text((top_left[0], top_left[1] - 10), text, fill=color)

    return pil_img

# Capture image using Streamlit's webcam input
image_data = st.camera_input("Take a photo for face recognition")

if image_data:
    # Convert the uploaded image to a PIL Image
    pil_image = Image.open(image_data)

    # Process the image for face recognition
    annotated_image = process_image(pil_image)

    # Display the annotated image in Streamlit
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
    

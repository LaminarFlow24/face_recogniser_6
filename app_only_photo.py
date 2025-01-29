import streamlit as st
import joblib
import requests
from PIL import Image, ImageDraw
from face_recognition import preprocessing
from huggingface_hub import hf_hub_download
from datetime import datetime
import pytz  # Library for timezone handling

# Define Hugging Face model details
REPO_ID = "Yashas2477/SE2_og"
FILENAME = "face_recogniser_100f_50e_final.pkl"

# Node.js server URL
NODE_SERVER_URL = "https://face-attendance-server.vercel.app/api/store-face-data"

# Define IST timezone
IST = pytz.timezone("Asia/Kolkata")

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

face_recogniser = load_model()
preprocess = preprocessing.ExifOrientationNormalize()

st.title("Live Face Recognition")

# Store processed labels globally to avoid duplicates across frames
if 'seen_labels' not in st.session_state:
    st.session_state.seen_labels = set()

def process_image(pil_img):
    pil_img = preprocess(pil_img)
    pil_img = pil_img.convert('RGB')

    faces = face_recogniser(pil_img)
    unique_faces = []

    draw = ImageDraw.Draw(pil_img)

    for face in faces:
        label = face.top_prediction.label
        confidence = face.top_prediction.confidence

        # Skip "Unknown" labels and duplicate labels within the session
        if label == "Unknown" or label in st.session_state.seen_labels:
            continue

        st.session_state.seen_labels.add(label)

        bb = face.bb._asdict()
        top_left = (int(bb['left']), int(bb['top']))
        bottom_right = (int(bb['right']), int(bb['bottom']))

        color = "green"
        draw.rectangle([top_left, bottom_right], outline=color, width=2)
        text = f"{label} ({confidence:.2f})"
        draw.text((top_left[0], top_left[1] - 10), text, fill=color)

        unique_faces.append({
            "label": label,
            "confidence": confidence,
            "timestamp": datetime.now(tz=IST).isoformat()  # Save timestamp in IST
        })

    return pil_img, unique_faces

image_data = st.camera_input("Take a photo for face recognition")

if image_data:
    pil_image = Image.open(image_data)
    annotated_image, output_details = process_image(pil_image)

    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    st.write("*Face Recognition Output:*")
    for detail in output_details:
        st.write(f"Label: {detail['label']}, Confidence: {detail['confidence']:.2f}")
    print(output_details)
    if output_details:
        # Send data to Node.js server
        response = requests.post(NODE_SERVER_URL, json={"faces": output_details})

        if response.status_code == 200:
            st.success("Data successfully stored in the database!")
        else:
            st.error("Failed to store data.")
    else:
        st.warning("No valid face data to store.")

st.title("Retrieve Face Recognition Data")

# Initialize session state for start_time and end_time if not already set
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now(IST).time()  # Default to current IST time
if 'end_time' not in st.session_state:
    st.session_state.end_time = datetime.now(IST).time()  # Default to current IST time

# Time input fields that update session state
start_time = st.time_input(
    "Start Time", 
    value=st.session_state.start_time, 
    key="start_time_input"
)
end_time = st.time_input(
    "End Time", 
    value=st.session_state.end_time, 
    key="end_time_input"
)

# Update session state on time input changes
st.session_state.start_time = start_time
st.session_state.end_time = end_time

# Add button to fetch data
if st.button("Get Data"):
    query_params = {
        "date": st.date_input("Select Date").strftime("%Y-%m-%d"),
        "start_time": st.session_state.start_time.strftime("%H:%M:%S"),
        "end_time": st.session_state.end_time.strftime("%H:%M:%S")
    }

    response = requests.get("https://face-attendance-server.vercel.app/api/get-face-data", params=query_params)


    if response.status_code == 200:
        data = response.json()
        st.write("Total Count: ", len(data))
        st.write("Retrieved Data:")

        for d in data:
            # Convert server timestamps to IST
            timestamp = datetime.fromisoformat(d['timestamp']).astimezone(IST).strftime("%H:%M:%S")
            st.write(f"{d['label']}     {timestamp}")
    else:
        st.error("Failed to fetch data.")

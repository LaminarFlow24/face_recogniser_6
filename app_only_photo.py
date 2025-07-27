import streamlit as st
import joblib
import requests
from PIL import Image, ImageDraw
from face_recognition import preprocessing
from huggingface_hub import hf_hub_download
from datetime import datetime

# Define Hugging Face model details
REPO_ID = "Yashas2477/SE2_og"
FILENAME = "face_recogniser_100f_50e_final.pkl"

# Node.js server URL
NODE_SERVER_URL = "https://face-attendance-server.vercel.app/api/store-face-data"

@st.cache_data
def download_model_from_huggingface():
    """Downloads the model from Hugging Face Hub."""
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
    """Loads the downloaded model into memory."""
    try:
        model_path = download_model_from_huggingface()
        st.write(f"Model loaded from: {model_path}")
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

# Load model and preprocessing tools
face_recogniser = load_model()
preprocess = preprocessing.ExifOrientationNormalize()

st.title("Face Recognition Attendance")

# Store processed labels globally to avoid duplicates across frames
if 'seen_labels' not in st.session_state:
    st.session_state.seen_labels = set()

# --- START OF MODIFIED FUNCTION ---
def process_image(pil_img):
    """
    Processes an image to detect all faces, draws boxes around all of them,
    but only logs new, recognized faces.
    """
    pil_img = preprocess(pil_img)
    pil_img = pil_img.convert('RGB')

    faces = face_recogniser(pil_img)
    faces_to_log = []
    draw = ImageDraw.Draw(pil_img)

    for face in faces:
        label = face.top_prediction.label
        confidence = face.top_prediction.confidence
        bb = face.bb._asdict()
        top_left = (int(bb['left']), int(bb['top']))
        bottom_right = (int(bb['right']), int(bb['bottom']))

        # Check if the face is new and recognized
        is_new_and_known = label != "Unknown" and label not in st.session_state.seen_labels

        # Determine box color based on status
        if is_new_and_known:
            color = "lime"  # Green for new faces to be logged
        elif label == "Unknown":
            color = "red"    # Red for unknown faces
        else:
            color = "dodgerblue"   # Blue for recognized but already logged faces

        # Draw bounding box and label for ALL faces
        draw.rectangle([top_left, bottom_right], outline=color, width=3)
        text = f"{label} ({confidence:.2f})"
        
        # Adjust text position if it's too close to the top edge
        text_y = top_left[1] - 15 if top_left[1] - 15 > 0 else top_left[1] + 5
        draw.text((top_left[0], text_y), text, fill=color)

        # Only add new, recognized faces to the list for logging
        if is_new_and_known:
            st.session_state.seen_labels.add(label)
            faces_to_log.append({
                "label": label,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })

    return pil_img, faces_to_log
# --- END OF MODIFIED FUNCTION ---


# Add a radio button to choose the input method
input_option = st.radio(
    "Choose your input method:",
    ('Camera', 'Upload Image')
)

image_data = None
if input_option == 'Camera':
    image_data = st.camera_input("Take a photo for face recognition")
elif input_option == 'Upload Image':
    image_data = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if image_data:
    pil_image = Image.open(image_data)
    annotated_image, output_details = process_image(pil_image)

    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    if output_details:
        st.write("*New Faces Logged:*")
        for detail in output_details:
            st.write(f"**Label:** {detail['label']}, **Confidence:** {detail['confidence']:.2f}")

        # Send data to Node.js server
        try:
            response = requests.post(NODE_SERVER_URL, json={"faces": output_details})
            response.raise_for_status()
            st.success("Data successfully stored in the database! ✅")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to store data: {e}")
    else:
        st.warning("No new or known faces were detected to log.")

# --- Data Retrieval Section ---
st.divider()
st.title("Retrieve Face Recognition Data")

col1, col2, col3 = st.columns(3)
with col1:
    date = st.date_input("Select Date")
with col2:
    start_time = st.time_input("Start Time")
with col3:
    end_time = st.time_input("End Time")

if st.button("Get Data"):
    query_params = {
        "date": date.strftime("%Y-%m-%d"),
        "start_time": start_time.strftime("%H:%M:%S"),
        "end_time": end_time.strftime("%H:%M:%S")
    }

    try:
        response = requests.get("https://face-attendance-server.vercel.app/api/get-face-data", params=query_params)
        response.raise_for_status()
        data = response.json()

        st.write(f"**Total Records Found:** {len(data)}")
        if data:
            st.write("**Retrieved Data:**")
            for d in data:
                timestamp = datetime.fromisoformat(d['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"**Label:** {d['label']}, **Timestamp:** {timestamp}")
        else:
            st.info("No data found for the selected time range.")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data: {e}")

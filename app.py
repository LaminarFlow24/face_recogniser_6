import io
import joblib
from PIL import Image
import streamlit as st
from face_recognition import preprocessing

# Load model and preprocessing
face_recogniser = joblib.load('model/face_recogniser.pkl')
preprocess = preprocessing.ExifOrientationNormalize()

# Streamlit app
st.title("Face Recognition App")

# Sidebar inputs
include_predictions = st.sidebar.checkbox("Include All Predictions", value=False)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file:
    try:
        # Load image
        img = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = preprocess(img)
        img = img.convert('RGB')  # Ensure image is in RGB format

        # Perform face recognition
        faces = face_recogniser(img)

        # Display results
        if faces:
            st.subheader("Detected Faces")
            for i, face in enumerate(faces, start=1):
                st.write(f"### Face {i}")
                
                # Bounding box
                bb = face.bb._asdict()
                st.write("Bounding Box:", bb)

                # Top prediction
                top_pred = face.top_prediction._asdict()
                st.write("Top Prediction:", top_pred)

                # All predictions (optional)
                if include_predictions:
                    all_preds = [p._asdict() for p in face.all_predictions]
                    st.write("All Predictions:", all_preds)

        else:
            st.warning("No faces detected in the image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image file to get started.")

import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Live Face Recognition", layout="centered")
st.title("📷 Real-Time Face Recognition")

# Upload a known face
known_image_file = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png"])
camera_start = st.button("Start Camera") if known_image_file else None

if known_image_file:
    # Load and encode known face
    known_image = face_recognition.load_image_file(known_image_file)
    try:
        known_encoding = face_recognition.face_encodings(known_image)[0]
    except IndexError:
        st.error("No face detected in the uploaded image.")
        st.stop()

    st.image(known_image, caption="Known Face", use_column_width=True)

    if camera_start:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera.")
                break

            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            match_text = "No Match"
            box_color = (0, 0, 255)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([known_encoding], face_encoding)
                face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

                if matches[0]:
                    match_text = f"Match ({face_distance:.2f})"
                    box_color = (0, 255, 0)
                else:
                    match_text = f"No Match ({face_distance:.2f})"

            # Draw box and label
            for (top, right, bottom, left) in face_locations:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.putText(frame, match_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
# Instructions for the user
st.title("Face Detection App")
st.write("""
         ## Instructions:
         1. Upload an image file (JPG or PNG).
         2. The app will detect faces and highlight them in the image.
         3. You can download the image with detected faces.
         """)

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert the image back to the PIL format
    result_image = Image.fromarray(image)

    # Display the image
    st.image(result_image, caption="Processed Image", use_column_width=True)

    # Provide an option to download the image with detected faces
    st.markdown(get_image_download_link(result_image), unsafe_allow_html=True)

def get_image_download_link(img):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    import base64
    from io import BytesIO

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="detected_faces.jpg">Download Image with Detected Faces</a>'
    return href
import base64
from io import BytesIO
from PIL import Image

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Load an image from a file
image_path = r'C:\Users\HP\Downloads\facerecogn\istockphoto-1368965646-2048x2048.jpg'
image = Image.open(image_path)

# Convert the image to a numpy array
image_array = np.array(image)

# Now you can use the image_array variable
result_image = Image.fromarray(image_array)

# Display the image
st.image(result_image, caption="Processed Image", use_column_width=True)

# Provide an option to download the image with detected faces
st.markdown(get_image_download_link(result_image, "detected_faces.jpg", "Download Image with Detected Faces"), unsafe_allow_html=True)

  # Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    # Convert the image back to the PIL format
result_image = Image.fromarray(image)

    # Display the image
st.image(result_image, caption="Processed Image", use_column_width=True)

    # Provide an option to download the image with detected faces
st.markdown(get_image_download_link(result_image, "detected_faces.jpg", "Download Image with Detected Faces"), unsafe_allow_html=True)

    # Provide an option to download the image with detected faces
st.markdown(get_image_download_link(result_image, "detected_faces.jpg", "Download Image with Detected Faces"), unsafe_allow_html=True)
# Detect faces with adjustable scaleFactor and minNeighbors







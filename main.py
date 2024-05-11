import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)
class_names = {0: "Bird Drop",
               1: "Clean",
               2: "Dusty",
               3: "Electric Damage",
               4: "Physical Damage",
               5: "Snow Covered"}

remedies = {
    "Bird Drop": "Clean the panel using a mild detergent solution and rinse it with water. Inspect the panel for any damage caused by bird droppings.",
    "Clean": "No action required. The solar panel is clean.",
    "Dusty": "Clean the panel using a soft cloth or sponge dampened with water. Do not use abrasive materials to prevent scratches.",
    "Snow Covered": "Remove snow from the panel gently using a soft brush or warm water. Do not use sharp tools to avoid damaging the panel surface.",
    "Electric Damage": "Contact a certified electrician to inspect and repair any electrical damage on the solar panel system. Do not attempt to fix electrical issues yourself.",
    "Physical Damage": "Inspect the panel for physical damage such as cracks or dents. If damage is found, contact the manufacturer for repair or replacement.",
}

def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape((1, 224, 224, 3))

st.title("Solar Obstacle Identification using Image Processing and GoogLeNet")

option = st.radio("Select Input Option:", ("Select Image", "Open Camera"))

if option == "Select Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Selected Image", use_column_width=True)
        
        st.write("Predicting...")
        data = preprocess_image(image)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        st.success(f"Prediction: **{class_name}** (Confidence: {confidence_score:.2f})")
        
        remedy = remedies.get(class_name, "No remedy information available")
        st.write("Remedy:", remedy)

else:
    st.write("Camera functionality is not implemented yet.")

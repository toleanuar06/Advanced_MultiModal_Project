import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import gdown

# --- Негізгі параметрлер ---
IMG_SIZE = 256
MODEL_FILENAME = 'advanced_multitask_model.keras'
PROJECT_PATH = '/content/drive/MyDrive/Advanced_MultiModal_Project' # Дұрыс жолды көрсету
MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, MODEL_FILENAME)

# --- GOOGLE DRIVE FILE ID ---
# Осы жерге 2-қадамда сақталған модельдің FILE ID-ін қою керек
# Оны Google Drive-тан қолмен алып, осында қойыңыз
GDRIVE_FILE_ID = '1fLHVcMHc24Gl7suvg2F7sN8I1Ng-TfR1' # <<<--- ОСЫНЫ ӨЗГЕРТУ КЕРЕК

# --- Модельді жүктеу функциясы ---
@st.cache_resource
def load_keras_model(file_id, output_path):
    if not os.path.exists(output_path):
        st.info(f"Модель Google Drive-тан жүктелуде...")
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output_path, quiet=False)
            st.success("Модель сәтті жүктелді!")
        except Exception as e:
            st.error(f"Google Drive-тан жүктеу қатесі: {e}")
            return None
    try:
        model = tf.keras.models.load_model(output_path)
        st.success("Модель жадыға сәтті жүктелді.")
        return model
    except Exception as e:
        st.error(f"Модель файлын оқу қатесі: {e}")
        return None

# --- Негізгі Streamlit қосымшасы ---
st.set_page_config(layout="wide")
st.title("🛰️ Жетілдірілген Топырақ Құнарлылығын Болжау Жүйесі")
st.write("Суретті жүктеңіз және сол жердің сандық көрсеткіштерін енгізіңіз.")

# Модельді жүктеу
model = load_keras_model(GDRIVE_FILE_ID, MODEL_FILENAME)

if model is None:
    st.warning("Модельді жүктеу мүмкін болмады. Файл ID-ін немесе бөлісу рұқсаттарын тексеріңіз.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Сандық деректерді енгізіңіз")
        # Пайдаланушыдан сандық деректерді сұрау (слайдер арқылы)
        soil_moisture = st.slider("Топырақ ылғалдылығы (0-1)", 0.0, 1.0, 0.5)
        soil_salinity = st.slider("Топырақ тұздылығы (0-1)", 0.0, 1.0, 0.2)
        urban_density = st.slider("Қала тығыздығы (0-1)", 0.0, 1.0, 0.1)
        agri_density = st.slider("Егістік тығыздығы (0-1)", 0.0, 1.0, 0.6)
        
        st.subheader("2. Спутниктік суретті жүктеңіз")
        uploaded_file = st.file_uploader("Сурет (.jpg, .png)", type=["jpg", "png"])

    with col2:
        st.subheader("3. Болжам Нәтижесі")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Жүктелген сурет', use_column_width=True)
            
            # Суретті модельге дайындау
            img_array = img_to_array(image)
            img_resized = tf.image.resize(img_array, [IMG_SIZE, IMG_SIZE])
            img_normalized = img_resized / 255.0
            image_input = tf.expand_dims(img_normalized, axis=0) # (1, 256, 256, 3)

            # Сандық деректерді модельге дайындау
            tabular_input = np.array([[
                soil_moisture, 
                soil_salinity, 
                urban_density, 
                agri_density
            ]], dtype=np.float32) # (1, 4)
            
            # Болжам жасау
            with st.spinner('Болжам жасалуда...'):
                prediction = model.predict({'image_input': image_input, 'tabular_input': tabular_input})
                fertility = prediction[0][0]
            
            st.success(f"Болжалды құнарлылық коэффициенті: {fertility:.3f}")
            st.progress(fertility)
            
            if fertility > 0.8:
                st.markdown("### Бағалау: 🟢 Өте құнарлы жер.")
            elif fertility > 0.6:
                st.markdown("### Бағалау: 🟡 Жақсы, құнарлы жер.")
            else:
                st.markdown("### Бағалау: 🔴 Нашар, құнарсыз жер.")
        else:
            st.info("Нәтижені көру үшін сандық деректерді толтырып, суретті жүктеңіз.")

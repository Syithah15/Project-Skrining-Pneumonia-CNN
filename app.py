import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deteksi Pneumonia CNN", page_icon="🫁", layout="centered")
st.title("🫁 Sistem Deteksi Pneumonia (CNN From Scratch)")
st.write("Unggah citra X-Ray dada. AI akan mengklasifikasikan apakah Normal atau Pneumonia.")

# Load Model (pastikan nama file sesuai)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_pneumonia_from_scratch.h5')

model = load_model()

uploaded_file = st.file_uploader("Pilih gambar X-Ray (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)
        
    with col2:
        st.write("### Hasil Analisis AI:")
        if st.button("Mulai Analisis 🔍"):
            with st.spinner("AI sedang menganalisis gambar..."):
                # Preprocessing (Sama seperti di Colab)
                # 1. Ubah ke Grayscale
                img = Image.open(uploaded_file).convert('L')
                # 2. Resize ke 150x150
                img = img.resize((150, 150))
                # 3. Ubah ke array (JANGAN dibagi 255, karena di model kita sudah ada layer Rescaling)
                img_array = np.array(img)
                # 4. Tambah dimensi batch dan channel
                img_array = np.expand_dims(img_array, axis=0) # (1, 150, 150)
                img_array = np.expand_dims(img_array, axis=-1) # (1, 150, 150, 1)
                
                # Prediksi
                prediction = model.predict(img_array)
                prob = prediction[0][0]

                # Logika Threshold (0.5) - SUDAH DIPERBAIKI
                if prob > 0.5:
                    hasil = "PNEUMONIA"  # <--- INI DIBALIK
                    warna = "red"
                    prob_persen = prob * 100
                else:
                    hasil = "NORMAL"     # <--- INI DIBALIK
                    warna = "green"
                    prob_persen = (1 - prob) * 100
                
                st.markdown(f"<h3 style='color: {warna};'>Terdeteksi: {hasil}</h3>", unsafe_allow_html=True)
                st.write(f"Tingkat Keyakinan Model: **{prob_persen:.2f}%**")
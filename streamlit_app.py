import streamlit as st
import numpy as np
import joblib
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
import base64

print("=== STREAMLIT APP STARTED ===")
# Mapping label Inggris ke Indonesia (edit sesuai kebutuhan)
label_mapping = {
    "Apple___Apple_scab": "Apel - Kudis Apel",
    "Apple___Black_rot": "Apel - Busuk Hitam",
    "Apple___Cedar_apple_rust": "Apel - Karat Apel Cedar",
    "Apple___healthy": "Apel - Sehat",
    "Blueberry___healthy": "Blueberry - Sehat",
    "Cherry___Powdery_mildew": "Ceri - Embun Tepung",
    "Cherry___healthy": "Ceri - Sehat",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Jagung - Bercak Daun Cercospora",
    "Corn___Common_rust": "Jagung - Karat Umum",
    "Corn___Northern_Leaf_Blight": "Jagung - Hawar Daun Utara",
    "Corn___healthy": "Jagung - Sehat",
    "Grape___Black_rot": "Anggur - Busuk Hitam",
    "Grape___Esca_(Black_Measles)": "Anggur - Esca (Campak Hitam)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Anggur - Hawar Daun (Bercak Daun Isariopsis)",
    "Grape___healthy": "Anggur - Sehat",
    "Orange___Haunglongbing_(Citrus_greening)": "Jeruk - Haunglongbing (Citrus Greening)",
    "Peach___Bacterial_spot": "Persik - Bercak Bakteri",
    "Peach___healthy": "Persik - Sehat",
    "Pepper__bell___Bacterial_spot": "Paprika - Bercak Bakteri",
    "Pepper__bell___healthy": "Paprika - Sehat",
    "Potato___Early_blight": "Kentang - Hawar Dini",
    "Potato___Late_blight": "Kentang - Hawar Lambat",
    "Potato___healthy": "Kentang - Sehat",
    "Raspberry___healthy": "Raspberry - Sehat",
    "Soybean___healthy": "Kedelai - Sehat",
    "Squash___Powdery_mildew": "Labu - Embun Tepung",
    "Strawberry___Leaf_scorch": "Stroberi - Luka Daun",
    "Strawberry___healthy": "Stroberi - Sehat",
    "Tomato_Bacterial_spot": "Tomat - Bercak Bakteri",
    "Tomato_Early_blight": "Tomat - Hawar Dini",
    "Tomato_Late_blight": "Tomat - Hawar Lambat",
    "Tomato_Leaf_Mold": "Tomat - Jamur Daun",
    "Tomato_Septoria_leaf_spot": "Tomat - Bercak Daun Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomat - Tungau Berbintik Dua",
    "Tomato__Target_Spot": "Tomat - Bercak Target",
    "Tomato__Tomato_mosaic_virus": "Tomat - Virus Mosaic",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomat - Virus Daun Kuning Keriting",
    "Tomato_healthy": "Tomat - Sehat",
}

import os
import subprocess

# Download model otomatis jika belum ada
MODEL_PATH = 'klasifikasi_penyakit_daunnn/models/random_forest_model.pkl'
MODEL_DIR = os.path.dirname(MODEL_PATH)
MODEL_GDRIVE_ID = '1TiBzISDtQR4_vuyPr7hgSA0Iwh3wHnkH'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
# Download & load model dari GitHub Release (model hasil compress, path sejajar dengan klasifikasi_deploy)
import os
import joblib
import requests
import streamlit as st

MODEL_PATH = './models/random_forest_model_compressed.pkl'
MODEL_URL = 'https://github.com/HilmiNurpadilah/ai8deploy/releases/download/v1.0/random_forest_model_compressed.pkl'

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with st.spinner('Downloading model...'):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    st.success("Model downloaded!")

# Cek ukuran file model
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH)/1024/1024
    st.write(f"Model size: {size_mb:.2f} MB")
    if size_mb < 1:
        st.error("Model file too small, kemungkinan gagal download!")
        st.stop()
else:
    st.error("Model file not found!")
    st.stop()

# Load model dengan error handling
with st.spinner('Loading model...'):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded!")
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909763.png", width=80)
st.sidebar.title('Tentang Aplikasi')
st.sidebar.info('''\
Aplikasi ini menggunakan Random Forest untuk mendeteksi penyakit daun tanaman dari gambar.\n\n**Langkah Penggunaan:**\n1. Upload gambar daun tanaman.\n2. Tunggu hasil prediksi muncul.\n3. Lihat hasil prediksi di bawah gambar.\n\n**Tips:**\n- Gunakan gambar daun yang jelas dan fokus.\n- Format gambar: JPG/JPEG/PNG\n''')
st.sidebar.markdown('---')
st.sidebar.markdown('**Dibuat oleh:** Hilmi | Praktikum AI')

# Header
st.markdown("""
    <div style='display: flex; align-items: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2909/2909763.png' width='60' style='margin-right: 16px;'>
        <div>
            <h2 style='margin-bottom:0;'>Klasifikasi Penyakit Daun Tanaman</h2>
            <small>Deteksi otomatis penyakit daun menggunakan Random Forest</small>
        </div>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Main upload & prediksi
st.markdown("### Upload Gambar Daun Tanaman")
col1, col2 = st.columns([2,1])
with col1:
    uploaded_file = st.file_uploader('Pilih gambar daun...', type=['jpg', 'jpeg', 'png'])
with col2:
    st.markdown('''<div style='font-size:1.1em; color:#555;'>\nGambar yang diupload akan diproses otomatis.\n</div>''', unsafe_allow_html=True)

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Tampilkan preview
    st.image(img, caption='Gambar yang diupload', use_column_width=True, channels='BGR')
    # Preprocessing
    img_prep = cv2.resize(img, (128,128))
    img_prep = img_prep.astype(np.float32) / 255.0
    # Ekstraksi fitur HOG
    fitur = hog(rgb2gray(img_prep), pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
    fitur = fitur.reshape(1, -1)
    # Prediksi
    pred = model.predict(fitur)[0]
    pred_id = label_mapping.get(pred, pred)
    # Badge warna hasil prediksi
    st.markdown(f"""
        <div style='margin-top:16px; padding:16px; border-radius:12px; background:linear-gradient(90deg,#d0f0c0,#e2ffe2); text-align:center;'>
            <span style='font-size:1.3em; font-weight:bold; color:#388e3c;'>Prediksi Penyakit: <span style='color:#b71c1c'>{pred_id}</span></span>
        </div>
    """, unsafe_allow_html=True)
    st.success('Prediksi selesai! Jika ingin mencoba gambar lain, upload ulang.')
else:
    st.info('Silakan upload gambar daun tanaman untuk mulai prediksi.')

# Footer
st.markdown('---')
st.markdown('<center><small>© 2025 Hilmi | Praktikum AI | Streamlit App</small></center>', unsafe_allow_html=True)

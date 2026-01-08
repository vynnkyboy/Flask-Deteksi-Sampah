import streamlit as st
import numpy as np
import sqlite3
from PIL import Image
import tensorflow as tf

# ================= CONFIG =================
st.set_page_config(
    page_title="Deteksi Sampah Berbasis CNN",
    page_icon="🗑️",
    layout="centered"
)

IMG_SIZE = 224
DB_NAME = "prediksi.db"

# ================= STYLE =================
st.markdown("""
<style>
.main { padding-top: 1.5rem; }
.stButton>button {
    width: 100%;
    border-radius: 10px;
    font-size: 16px;
    padding: 10px;
}
.metric-container {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD TFLITE =================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_sampah.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(label, confidence):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO prediksi (label, confidence) VALUES (?, ?)",
        (label, confidence)
    )
    conn.commit()
    conn.close()

init_db()

# ================= HEADER =================
st.title("🗑️ Deteksi Sampah Menggunakan CNN")
st.caption("Aplikasi klasifikasi sampah organik & anorganik berbasis Deep Learning")

st.divider()

# ================= INPUT =================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar Sampah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Gambar Input", use_container_width=True)

    img = image_pil.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Prediksi"):
        with st.spinner("Menganalisis gambar..."):
            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])

            # Aman untuk sigmoid / softmax
            if preds.shape[-1] == 1:
                score = preds[0][0]
                label = "Sampah Organik" if score > 0.5 else "Sampah Anorganik"
                confidence = score if score > 0.5 else 1 - score
            else:
                idx = np.argmax(preds[0])
                confidence = preds[0][idx]
                label = "Sampah Organik" if idx == 1 else "Sampah Anorganik"

            confidence = round(confidence * 100, 2)
            log_prediction(label, confidence)

            st.success(f"**{label}**")
            st.metric("Confidence", f"{confidence}%")

# ================= DASHBOARD =================
st.divider()
st.subheader("📊 Statistik Prediksi")

conn = sqlite3.connect(DB_NAME)
c = conn.cursor()
c.execute("SELECT label, COUNT(*) FROM prediksi GROUP BY label")
data = c.fetchall()
conn.close()

if data:
    labels = [d[0] for d in data]
    counts = [d[1] for d in data]
    st.bar_chart(dict(zip(labels, counts)))
else:
    st.info("Belum ada data prediksi")

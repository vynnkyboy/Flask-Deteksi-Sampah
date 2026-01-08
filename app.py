import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
from PIL import Image

# ================= CONFIG =================
st.set_page_config(
    page_title="Deteksi Sampah CNN",
    page_icon="🗑️",
    layout="centered"
)

IMG_SIZE = 224
DB_NAME = "prediksi.db"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_mobilenet_sampah.h5")

model = load_model()

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

# ================= GRAD-CAM =================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        # === FIX UTAMA DI SINI ===
        if preds.shape[-1] == 1:
            loss = preds[:, 0]          # sigmoid
        else:
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]  # softmax

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-10

    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# ================= UI =================
st.title("🗑️ Deteksi Sampah Menggunakan CNN")
st.write("Upload gambar sampah untuk diklasifikasikan")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Gambar Input", use_container_width=True)

    img = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    img_array = np.expand_dims(img, axis=0)

    if st.button("🔍 Prediksi"):
        with st.spinner("Menganalisis gambar..."):
            pred = model.predict(img_array)[0][0]
            label = "Sampah Organik" if pred > 0.5 else "Sampah Anorganik"
            confidence = pred if pred > 0.5 else 1 - pred
            confidence = round(confidence * 100, 2)

            log_prediction(label, confidence)

            st.success(f"**{label}**")
            st.metric("Confidence", f"{confidence}%")

            # Grad-CAM
            heatmap = make_gradcam_heatmap(img_array, model)
            cam = overlay_heatmap((img * 255).astype("uint8"), heatmap)

            st.image(cam, caption="Grad-CAM Visualization", use_container_width=True)

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

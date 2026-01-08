from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import cv2
import sqlite3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = tf.keras.models.load_model("model_mobilenet_sampah.h5")
IMG_SIZE = 224

# ================= DATABASE =================
def log_prediction(label, confidence):
    conn = sqlite3.connect('prediksi.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            confidence REAL
        )
    ''')
    c.execute("INSERT INTO prediksi (label, confidence) VALUES (?, ?)",
              (label, confidence))
    conn.commit()
    conn.close()

# ================= GRAD-CAM =================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path="static/heatmap.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    output = heatmap * alpha + img
    cv2.imwrite(cam_path, output)

# ================= ROUTES =================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        confidence = pred if pred > 0.5 else 1 - pred
        label = "Sampah Organik" if pred > 0.5 else "Sampah Anorganik"

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, "Conv_1")
        save_gradcam(filepath, heatmap)

        # Log ke database
        log_prediction(label, round(confidence * 100, 2))

        return render_template(
            'result.html',
            label=label,
            confidence=round(confidence * 100, 2),
            image_path=filepath,
            heatmap_path="static/heatmap.jpg"
        )

    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('prediksi.db')
    c = conn.cursor()
    c.execute("SELECT label, COUNT(*) FROM prediksi GROUP BY label")
    data = c.fetchall()
    conn.close()

    labels = [d[0] for d in data]
    counts = [d[1] for d in data]

    return render_template('dashboard.html', labels=labels, counts=counts)

if __name__ == '__main__':
    app.run(debug=True)

import tensorflow as tf

model = tf.keras.models.load_model("model_mobilenet_sampah.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model_sampah.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model TFLite berhasil dibuat")

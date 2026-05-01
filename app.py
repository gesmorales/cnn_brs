from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="brs_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [
    "Low Resilience",
    "Normal Resilience",
    "High Resilience"
]

@app.route("/")
def home():
    return "TFLite API Running"

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        features = np.array(
            data["features"],
            dtype=np.float32
        )

        features = features.reshape(1, 8, 1)

        interpreter.set_tensor(
            input_details[0]['index'],
            features
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]['index']
        )

        predicted_class = int(np.argmax(prediction))

        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": labels[predicted_class],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

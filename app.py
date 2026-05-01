from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model("brs_cnn_with_demographics.keras")

labels = [
    "Low Resilience",
    "Normal Resilience",
    "High Resilience"
]

@app.route("/")
def home():
    return "CNN API Running Successfully"

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        print("Received:", data)

        features = np.array(data["features"], dtype=float)

        print("Before reshape:", features.shape)

        # reshape for CNN
        features = features.reshape(1, 8, 1)

        print("After reshape:", features.shape)

        prediction = model.predict(features)

        print("Prediction:", prediction)

        predicted_class = int(np.argmax(prediction))

        result = labels[predicted_class]

        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

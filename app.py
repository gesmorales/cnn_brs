from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("brs_cnn_with_demographics.keras")

# Labels
labels = [
    "Low Resilience",
    "Normal Resilience",
    "High Resilience"
]

@app.route("/")
def home():
    return "BRS CNN API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Example input
    features = np.array(data["features"])

    # Reshape for CNN
    features = features.reshape(1, len(features), 1)

    prediction = model.predict(features)

    predicted_class = int(np.argmax(prediction))

    result = labels[predicted_class]

    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # recevoir un JSON
    df = pd.DataFrame([data])  # transformer en DataFrame
    prediction = model.predict(df)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    # Prédiction de la classe
    pred = model.predict(df)[0]
    # Prédiction de la probabilité pour la classe 1
    prob = model.predict_proba(df)[0][1]  # [0] pour la première ligne, [1] pour la classe 1
    return jsonify({"prediction": int(pred), "probability": float(prob)})

if __name__ == '__main__':
    app.run(debug=True)
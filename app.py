from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)
CORS(app)  # Permite peticiones desde tu frontend

# Guarda el modelo en el sistema de archivos de Render (persistente)
MODEL_PATH = "model.pkl"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "¡Entrena el modelo primero!"}), 400
    
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(input_df)
    return jsonify({"prediction": int(prediction[0])})

@app.route('/train', methods=['POST'])
def train():
    csv_file = request.files['file']
    data = pd.read_csv(csv_file)
    
    # Preprocesamiento (igual que antes)
    data['GoalDifference'] = data['FTHG'] - data['FTAG']
    data['HomeWinRate'] = data.groupby('HomeTeam')['FTR'].apply(lambda x: (x == 'H').rolling(5, min_periods=1).mean())
    data['HomeTeamID'] = data['HomeTeam'].astype('category').cat.codes
    data['AwayTeamID'] = data['AwayTeam'].astype('category').cat.codes
    data['Result'] = data['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    features = ['HomeTeamID', 'AwayTeamID', 'GoalDifference', 'HomeWinRate', 'HS', 'AS', 'HC', 'AC']
    target = 'Result'
    X = data[features]
    y = data[target]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    return jsonify({"message": "Modelo entrenado con éxito!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

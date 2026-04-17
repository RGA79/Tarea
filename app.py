import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = os.getenv("MODEL_PATH", "models/california_housing_model.pkl")

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
metrics  = artifact["metrics"]

REQUIRED_NUMERIC = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup","Latitude", "Longitude", "Habitaciones_x_Occupantes", "Ingresos_x_Ubicacion"]

# ── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "service":   "Predictor de Precio de Viviendas en California",
        "version":   "1.0.0",
        "endpoints": ["/","/predict", "/health","/features","/form"],
    })

@app.route("/features")
def features():
    return jsonify({
        "features": REQUIRED_NUMERIC
    })

@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "train_metrics": metrics,
    })

@app.route("/form")
def form():
    return '''
    <html>
    <head><title>Formulario de Predicción</title></head>
    <form method="POST" action="/predict" enctype="application/json">
        <input type="number" name="MedInc" placeholder="MedInc" step="0.01" value="8.3252" required><br>
        <input type="number" name="HouseAge" placeholder="HouseAge" step="1" value="41" required><br>
        <input type="number" name="AveRooms" placeholder="AveRooms" step="0.01" value="5.3809" required><br>
        <input type="number" name="AveBedrms" placeholder="AveBedrms" step="0.01" value="1.0926" required><br>
        <input type="number" name="Population" placeholder="Population" step="1" value="1296.0" required><br>
        <input type="number" name="AveOccup" placeholder="AveOccup" step="0.01" value="2.5649" required><br>
        <input type="number" name="Latitude" placeholder="Latitude" step="0.0001" value="33.88" required><br>
        <input type="number" name="Longitude" placeholder="Longitude" step="0.0001" value="-118.24" required><br>
        <input type="number" name="Habitaciones_x_Occupantes" placeholder="Habitaciones_x_Occupantes" step="0.0001" value="1" required><br>
        <input type="number" name="Ingresos_x_Ubicacion" placeholder="Ingresos_x_Ubicacion" step="0.0001" value="150000" required><br>
        <button type="submit">Predecir Precio</button>
    </form>
    </html>
    '''

@app.route("/predict", methods=["POST"])
def predict():    
    if request.is_json:    
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Body debe ser JSON válido"}), 400
    else:
        data = request.form.to_dict()
        
    try:
        features  = pd.DataFrame([data])
        log_price = pipeline.predict(features)[0]
        price     = float(np.expm1(log_price))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "predicted_value": round(price, 2),
        "note": "Estimación basada en dataset California Housing (20,640 registros).",
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Método HTTP no permitido en este endpoint"}), 405

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
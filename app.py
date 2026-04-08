"""
AquaWatch - Water Quality Dashboard
Flask: ML prediction via /api/predict
Firebase handled in HTML using JS SDK
Production ready for Render deployment
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ─────────────────────────────────────────────
# EMBEDDED DATASET
# ─────────────────────────────────────────────
EMBEDDED_DATA = """tds_ppm,turbidity_ntu,label
45,0.3,Excellent
80,0.5,Excellent
120,0.8,Excellent
200,1.2,Good
250,1.5,Good
300,2.0,Good
310,2.5,Good
350,3.0,Good
400,3.5,Good
400,4.0,Good
420,5.0,Fair
500,5.0,Fair
550,6.0,Fair
600,7.0,Fair
650,8.0,Fair
700,9.0,Fair
720,9.5,Fair
750,10.0,Poor
800,12.0,Poor
850,13.5,Poor
900,15.0,Poor
950,18.0,Poor
1000,20.0,Poor
1100,25.0,Poor
1200,30.0,Poor
50,0.4,Excellent
90,0.6,Excellent
150,1.0,Excellent
180,1.1,Good
270,1.8,Good
380,3.2,Good
450,4.5,Fair
530,5.8,Fair
620,7.5,Fair
680,8.8,Fair
780,11.0,Poor
880,14.0,Poor
960,19.0,Poor
1050,22.0,Poor
60,0.35,Excellent
110,0.75,Excellent
230,1.4,Good
340,2.8,Good
480,4.8,Fair
560,6.5,Fair
710,9.2,Fair
820,12.5,Poor
920,16.0,Poor
70,0.45,Excellent
100,0.7,Excellent
190,1.3,Good
320,2.6,Good
460,4.3,Fair
590,6.8,Fair
730,9.8,Poor
860,13.0,Poor"""

MODEL_PATH   = "water_quality_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
def train_model():
    print("[ML] Training Random Forest model...")
    DATASET_PATH = "water_quality_dataset.csv"

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"[ML] Loaded {DATASET_PATH} ({len(df)} rows)")
    else:
        df = pd.read_csv(StringIO(EMBEDDED_DATA))
        print(f"[ML] Using embedded dataset ({len(df)} rows)")

    X     = df[['tds_ppm', 'turbidity_ntu']]
    y     = df['label']
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[ML] Accuracy: {acc*100:.1f}% | Classes: {list(le.classes_)}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le,    ENCODER_PATH)
    print("[ML] Model saved.")
    return model, le, round(acc * 100, 1)

# ─────────────────────────────────────────────
# LOAD OR TRAIN AT STARTUP
# ─────────────────────────────────────────────
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    print("[ML] Loading saved model...")
    model     = joblib.load(MODEL_PATH)
    le        = joblib.load(ENCODER_PATH)
    MODEL_ACC = "N/A"
else:
    model, le, MODEL_ACC = train_model()

SAFE_LABELS = {"Excellent", "Good"}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('dashboard.html', accuracy=MODEL_ACC)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    POST { "tds": 245.3, "turbidity": 3.2 }
    GET  ?tds=245.3&turbidity=3.2  (fallback for simple clients)
    """
    try:
        # Accept both JSON body and query params
        if request.is_json:
            body      = request.get_json(force=True)
            tds       = float(body['tds'])
            turbidity = float(body['turbidity'])
        else:
            tds       = float(request.args.get('tds', 0))
            turbidity = float(request.args.get('turbidity', 0))

        inp   = pd.DataFrame([[tds, turbidity]],
                             columns=['tds_ppm', 'turbidity_ntu'])
        enc   = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]
        label = le.inverse_transform([enc])[0]
        probs = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le.classes_, proba)
        }
        safe       = label in SAFE_LABELS
        confidence = round(float(max(proba)) * 100, 1)

        print(f"[Predict] TDS={tds} Turb={turbidity} → {label} ({confidence}%) safe={safe}")

        return jsonify({
            "quality":       label,
            "safe":          safe,
            "confidence":    confidence,
            "probabilities": probs
        })

    except Exception as e:
        print(f"[Predict] ERROR: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model": "loaded"})


@app.route('/api/retrain')
def api_retrain():
    global model, le, MODEL_ACC
    model, le, MODEL_ACC = train_model()
    return jsonify({"status": "ok", "accuracy": MODEL_ACC})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n  AquaWatch Dashboard → http://localhost:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
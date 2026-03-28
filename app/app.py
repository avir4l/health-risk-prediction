import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import pytesseract
import re
import logging

from utils import (
    HEALTH_RANGES,
    CHEST_PAIN_TYPES,
    validate_all_inputs,
    assess_health_status,
    get_risk_level,
    generate_recommendations,
    validate_value,
)

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========== TESSERACT PATH (update for your OS) ==========
# Replace the pytesseract import block with this:
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
# ========== FLASK APP ==========
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

BASE_DIR = Path(__file__).resolve().parent.parent
# Add this safety check:
if not (BASE_DIR / 'model').exists():
    BASE_DIR = Path(__file__).resolve().parent

# ========== LOAD MODELS ==========
try:
    diabetes_model   = joblib.load(BASE_DIR / "model" / "diabetes_stacking_model.pkl")
    diabetes_scaler  = joblib.load(BASE_DIR / "model" / "diabetes_advanced_scaler.pkl")
    diabetes_features = joblib.load(BASE_DIR / "model" / "diabetes_feature_names.pkl")
    
    heart_model      = joblib.load(BASE_DIR / "model" / "heart_stacking_model.pkl")
    heart_scaler     = joblib.load(BASE_DIR / "model" / "heart_advanced_scaler.pkl")
    heart_features   = joblib.load(BASE_DIR / "model" / "heart_feature_names.pkl")
    logger.info("✓ All advanced models loaded successfully")
except Exception as e:
    logger.error(f"✗ Model loading failed: {e}. Are advanced models trained?")
    raise

# ========== DATASET MEANS FOR MISSING HEART FEATURES ==========
# These are approximate means from the Cleveland Heart Disease dataset.
# Using means instead of 0 avoids biasing predictions for unknown values.
HEART_FEATURE_MEANS = {
    "fasting_bs":  0,      # default: glucose-derived below
    "rest_ecg":    0,      # 0 = normal (most common)
    "oldpeak":     1.0,    # mean ≈ 1.04
    "slope":       1,      # 1 = flat (most common)
    "ca":          0,      # 0 major vessels coloured (most common)
    "thal":        2,      # 2 = normal (encoded value)
}

# ========== OCR EXTRACTION ==========
def extract_from_report(image):
    if not OCR_AVAILABLE:
        return {}
    """
    Extract health metrics from a medical report image using OCR.
    Multiple regex patterns are tried per field to handle varied report formats.
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        text = pytesseract.image_to_string(image, config="--psm 6").lower()
        logger.info(f"OCR text length: {len(text)} chars")

        extracted = {}

        patterns = {
            "glucose": [
                r"glucose[:\s-]*([\d.]+)",
                r"blood\s*sugar[:\s-]*([\d.]+)",
                r"fasting\s*glucose[:\s-]*([\d.]+)",
                r"fbs[:\s-]*([\d.]+)",
                r"random\s*glucose[:\s-]*([\d.]+)",
            ],
            "chol": [
                r"total\s*cholesterol[:\s-]*([\d.]+)",
                r"cholesterol[:\s-]*([\d.]+)",
                r"\bchol\b[:\s-]*([\d.]+)",
                r"\btc\b[:\s-]*([\d.]+)",
            ],
            "bp": [
                r"systolic[:\s-]*([\d.]+)",
                r"blood\s*pressure[:\s-]*([\d.]+)",
                r"(\d{2,3})\s*/\s*\d{2,3}\s*mmhg",
                r"\bbp\b[:\s-]*([\d.]+)/",
            ],
            "bmi": [
                r"body\s*mass\s*index[:\s-]*([\d.]+)",
                r"\bbmi\b[:\s-]*([\d.]+)",
            ],
            "age": [
                r"\bage\b[:\s-]*(\d+)",
                r"(\d+)\s*y/?o\b",
                r"years\s*old[:\s-]*(\d+)",
            ],
            "heart_rate": [
                r"heart\s*rate[:\s-]*([\d.]+)",
                r"\bpulse\b[:\s-]*([\d.]+)",
                r"\bhr\b[:\s-]*([\d.]+)",
                r"(\d{2,3})\s*bpm",
            ],
        }

        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text)
                if match:
                    value = float(match.group(1))
                    if validate_value(key, value):
                        extracted[key] = value
                        logger.info(f"  OCR → {key}: {value}")
                        break

        return extracted

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return {}


# ========== MAIN ROUTE ==========
@app.route("/", methods=["GET", "POST"])
def index():
    extracted       = {}
    missing         = []
    results         = {}
    metrics         = {}
    assessments     = {}
    recommendations = []
    input_errors    = []

    if request.method == "POST":
        try:
            # ------ STEP 1: OCR (if image uploaded) ------
            if "report" in request.files and request.files["report"].filename:
                try:
                    image = Image.open(request.files["report"])
                    extracted = extract_from_report(image)
                    logger.info(f"OCR extracted: {extracted}")
                except Exception as e:
                    logger.error(f"Image error: {e}")
                    extracted = {"error": "Failed to process image"}

            # ------ STEP 2: COLLECT FORM VALUES ------
            def get_val(name, cast=float):
                """Return typed form value, falling back to OCR, then None."""
                v = request.form.get(name, "").strip()
                if v:
                    try:
                        return cast(v)
                    except ValueError:
                        return None
                return extracted.get(name)

            age          = get_val("age")
            sex          = get_val("sex", int)          # 0 = female, 1 = male
            glucose      = get_val("glucose")
            bp           = get_val("bp")
            chol         = get_val("chol")

            # BMI: prefer direct entry; fall back to height/weight calculation
            bmi = get_val("bmi")
            if bmi is None:
                height_cm = get_val("height_cm")
                weight_kg = get_val("weight_kg")
                if height_cm and weight_kg and height_cm > 0:
                    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

            # Optional cardiac fields
            heart_rate    = get_val("heart_rate")
            max_heart_rate = get_val("max_heart_rate")
            chest_pain    = get_val("chest_pain", int)
            exercise_angina = get_val("exercise_angina", int)

            # Pregnancies — only relevant for female users
            pregnancies = 0
            if sex == 0:                                # female
                pregnancies = get_val("pregnancies", int) or 0

            # ------ STEP 3: BUILD METRICS DICT ------
            metrics = {
                "age":            age,
                "glucose":        glucose,
                "bp":             bp,
                "bmi":            bmi,
                "chol":           chol,
                "heart_rate":     heart_rate,
                "max_heart_rate": max_heart_rate,
                "chest_pain":     chest_pain,
                "exercise_angina": exercise_angina,
            }

            # ------ STEP 4: VALIDATE RANGES ------
            numeric_metrics = {k: v for k, v in metrics.items() if v is not None}
            input_errors = validate_all_inputs(numeric_metrics)

            # ------ STEP 5: CHECK REQUIRED FIELDS ------
            required = {
                "age":     age,
                "glucose": glucose,
                "bp":      bp,
                "bmi":     bmi,
                "chol":    chol,
            }
            missing = [k.upper() for k, v in required.items() if v is None]

            # ------ STEP 6: PER-METRIC HEALTH ASSESSMENTS ------
            for key, value in metrics.items():
                if value is not None and key in HEALTH_RANGES:
                    assessments[key] = {
                        "value":     value,
                        "status":    assess_health_status(key, value),
                        "unit":      HEALTH_RANGES[key].get("unit", ""),
                        "relevance": HEALTH_RANGES[key].get("relevance", ""),
                    }

            if chest_pain is not None:
                assessments["chest_pain"] = {
                    "value":     CHEST_PAIN_TYPES.get(str(chest_pain), "Unknown"),
                    "status":    "Symptom Present" if chest_pain in [0, 1] else "Asymptomatic",
                    "unit":      "",
                    "relevance": "Symptom-based cardiac screening",
                }

            if exercise_angina is not None:
                assessments["exercise_angina"] = {
                    "value":     "Yes" if exercise_angina == 1 else "No",
                    "status":    "Warning Sign" if exercise_angina == 1 else "Normal",
                    "unit":      "",
                    "relevance": "Oxygen deprivation indicator",
                }

            # ------ STEP 7: PREDICTIONS ------
            if not missing and not input_errors:

                # --- Diabetes ---
                pregnancies_val = pregnancies if pregnancies is not None else 0
                glucose_val = glucose if glucose is not None else 120.9
                bp_val = bp if bp is not None else 69.1
                bmi_val = bmi if bmi is not None else 31.99
                age_val = age if age is not None else 33.2
                
                d_dict = {
                    'pregnancies': [pregnancies_val],
                    'glucose': [glucose_val],
                    'bloodpressure': [bp_val],
                    'skinthickness': [0],
                    'insulin': [0],
                    'bmi': [bmi_val],
                    'diabetespedigreefunction': [0.5],
                    'age': [age_val]
                }
                d_df = pd.DataFrame(d_dict)
                
                # Apply same feature engineering as advanced_preprocessing
                d_df['glucose_bmi_interaction'] = d_df['glucose'] * d_df['bmi']
                d_df['age_glucose_risk'] = d_df['age'] * d_df['glucose'] / 1000
                d_df['bmi_bp_interaction'] = d_df['bmi'] * d_df['bloodpressure']
                d_df['insulin_resistance_proxy'] = d_df['glucose'] / (d_df['insulin'] + 1)
                d_df['glucose_age_ratio'] = d_df['glucose'] / (d_df['age'] + 1)
                d_df['metabolic_syndrome_score'] = (
                    (d_df['glucose'] > 100).astype(int) +
                    (d_df['bmi'] > 25).astype(int) +
                    (d_df['bloodpressure'] > 130).astype(int) +
                    (d_df['insulin'] > 166).astype(int)
                )
                d_df['age_group'] = pd.cut(d_df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
                d_df['bmi_category'] = pd.cut(d_df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
                d_df['glucose_category'] = pd.cut(d_df['glucose'], bins=[0, 100, 126, 600], labels=[0, 1, 2]).astype(int)
                d_df['glucose_squared'] = d_df['glucose'] ** 2
                d_df['bmi_squared'] = d_df['bmi'] ** 2
                
                d_input = d_df[diabetes_features]
                d_scaled = diabetes_scaler.transform(d_input)
                d_prob   = diabetes_model.predict_proba(d_scaled)[0][1]
                d_risk   = d_prob * 100

                # --- Heart disease ---
                sex_val = sex if sex is not None else 1 
                h_age = age if age is not None else 54.4
                cp_val = chest_pain if chest_pain is not None else 3
                h_bp = bp if bp is not None else 131.6
                h_chol = chol if chol is not None else 246.6
                fasting_bs = 1 if (glucose and glucose > 120) else 0
                h_thalach = max_heart_rate if max_heart_rate else 149.6
                exang_val = exercise_angina if exercise_angina is not None else 0

                h_dict = {
                    'age': [h_age],
                    'sex': [sex_val],
                    'cp': [cp_val],
                    'trestbps': [h_bp],
                    'chol': [h_chol],
                    'fbs': [fasting_bs],
                    'restecg': [HEART_FEATURE_MEANS["rest_ecg"]],
                    'thalach': [h_thalach],
                    'exang': [exang_val],
                    'oldpeak': [HEART_FEATURE_MEANS["oldpeak"]],
                    'slope': [HEART_FEATURE_MEANS["slope"]],
                    'ca': [HEART_FEATURE_MEANS["ca"]],
                    'thal': [HEART_FEATURE_MEANS["thal"]]
                }
                h_df = pd.DataFrame(h_dict)
                
                # Apply same feature engineering
                h_df['age_bp_risk'] = h_df['age'] * h_df['trestbps'] / 1000
                h_df['heart_rate_reserve'] = (220 - h_df['age']) - h_df['thalach']
                h_df['st_severity'] = h_df['oldpeak'] * h_df['slope']
                h_df['age_chol_interaction'] = h_df['age'] * h_df['chol'] / 1000
                h_df['chol_age_ratio'] = h_df['chol'] / (h_df['age'] + 1)
                h_df['hr_percent_max'] = h_df['thalach'] / (220 - h_df['age'] + 1) * 100
                h_df['cardiac_risk_score'] = (
                    (h_df['age'] > 55).astype(int) +
                    (h_df['trestbps'] > 130).astype(int) +
                    (h_df['chol'] > 240).astype(int) +
                    (h_df['fbs'] == 1).astype(int) +
                    (h_df['thalach'] < 120).astype(int) +
                    (h_df['exang'] == 1).astype(int) +
                    (h_df['oldpeak'] > 1.5).astype(int) +
                    (h_df['ca'] > 0).astype(int)
                )
                h_df['symptom_severity'] = (
                    (h_df['cp'].isin([1, 2])).astype(int) * 2 +
                    (h_df['exang'] == 1).astype(int) * 2 +
                    (h_df['oldpeak'] > 2).astype(int)
                )
                h_df['age_group'] = pd.cut(h_df['age'], bins=[0, 40, 55, 65, 100], labels=[0, 1, 2, 3]).astype(int)
                h_df['bp_category'] = pd.cut(h_df['trestbps'], bins=[0, 120, 130, 140, 300], labels=[0, 1, 2, 3]).astype(int)
                h_df['oldpeak_squared'] = h_df['oldpeak'] ** 2
                h_df['age_squared'] = h_df['age'] ** 2

                h_input = h_df[heart_features]
                h_scaled = heart_scaler.transform(h_input)
                h_prob = heart_model.predict_proba(h_scaled)[0][1]
                h_risk = h_prob * 100

                results = {
                    "diabetes": {
                        "probability": round(d_risk, 2),
                        "risk":        get_risk_level(d_risk),
                    },
                    "heart": {
                        "probability": round(h_risk, 2),
                        "risk":        get_risk_level(h_risk),
                    },
                }

                recommendations = generate_recommendations(d_risk, h_risk, metrics)
                logger.info(
                    f"Prediction complete — Diabetes: {d_risk:.1f}%  Heart: {h_risk:.1f}%"
                )

        except Exception as e:
            logger.error(f"Processing error: {e}")
            results = {"error": str(e)}

    return render_template(
        "index.html",
        extracted=extracted,
        missing=missing,
        results=results,
        metrics=metrics,
        assessments=assessments,
        recommendations=recommendations,
        chest_pain_types=CHEST_PAIN_TYPES,
        input_errors=input_errors,
    )


# ========== ERROR HANDLERS ==========
@app.errorhandler(413)
def too_large(e):
    return "File too large. Maximum allowed size is 16 MB.", 413

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal error: {e}")
    return "Internal server error — please try again.", 500


# ========== RUN ==========
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
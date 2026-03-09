# ========== utils.py ==========
# Helper functions for health assessment, validation, and recommendations.
# Separated from app.py to keep the main route file clean.

# ========== HEALTH METRICS REFERENCE RANGES ==========
HEALTH_RANGES = {
    "glucose": {
        "normal": (70, 100),
        "prediabetic": (100, 126),
        "diabetic": (126, 300),
        "unit": "mg/dL",
        "relevance": "Primary diabetes indicator"
    },
    "bp": {
        "normal": (90, 120),
        "elevated": (120, 130),
        "stage1": (130, 140),
        "stage2": (140, 300),
        "unit": "mmHg",
        "relevance": "Hypertension risk factor"
    },
    "bmi": {
        "underweight": (0, 18.5),
        "normal": (18.5, 25),
        "overweight": (25, 30),
        "obese": (30, 50),
        "unit": "kg/m²",
        "relevance": "Obesity-heart correlation"
    },
    "chol": {
        "desirable": (0, 200),
        "borderline": (200, 240),
        "high": (240, 400),
        "unit": "mg/dL",
        "relevance": "Atherosclerosis indicator"
    },
    "heart_rate": {
        "low": (0, 60),
        "normal": (60, 100),
        "elevated": (100, 120),
        "high": (120, 200),
        "unit": "bpm",
        "relevance": "Cardiovascular fitness indicator"
    },
    "age": {
        "young": (0, 40),
        "middle": (40, 60),
        "senior": (60, 150),
        "unit": "years",
        "relevance": "Strongest non-modifiable risk factor"
    }
}

# Chest pain types mapping (Cleveland dataset encoding)
CHEST_PAIN_TYPES = {
    "0": "Typical Angina",
    "1": "Atypical Angina",
    "2": "Non-anginal Pain",
    "3": "Asymptomatic"
}

# ========== VALUE VALIDATION ==========
VALID_RANGES = {
    "age":           (1,   120),
    "glucose":       (20,  600),
    "bp":            (50,  250),
    "bmi":           (10,  60),
    "chol":          (50,  500),
    "heart_rate":    (30,  220),
    "max_heart_rate":(50,  250),
    "height_cm":     (50,  250),
    "weight_kg":     (10,  300),
    "pregnancies":   (0,   20),
}

def validate_value(key, value):
    """Return True if value is within a medically plausible range."""
    if key in VALID_RANGES:
        lo, hi = VALID_RANGES[key]
        return lo <= value <= hi
    return True


def validate_all_inputs(metrics: dict) -> list:
    """
    Validate each metric in the dict.
    Returns a list of error strings (empty list = all good).
    """
    errors = []
    for key, value in metrics.items():
        if value is not None and not validate_value(key, value):
            lo, hi = VALID_RANGES.get(key, (None, None))
            errors.append(
                f"{key.upper()} value {value} is out of the expected range ({lo}–{hi})."
            )
    return errors


# ========== HEALTH STATUS ASSESSMENT ==========
def assess_health_status(key, value):
    """Return a plain-English status label for a given metric + value."""
    if key not in HEALTH_RANGES:
        return "Unknown"

    ranges = HEALTH_RANGES[key]

    if key == "glucose":
        if value < ranges["normal"][1]:
            return "Normal"
        elif value < ranges["prediabetic"][1]:
            return "Prediabetic"
        else:
            return "Diabetic Range"

    elif key == "bp":
        if value < ranges["elevated"][0]:
            return "Normal"
        elif value < ranges["elevated"][1]:
            return "Elevated"
        elif value < ranges["stage1"][1]:
            return "Stage 1 Hypertension"
        else:
            return "Stage 2 Hypertension"

    elif key == "bmi":
        if value < ranges["normal"][0]:
            return "Underweight"
        elif value < ranges["overweight"][0]:
            return "Normal"
        elif value < ranges["obese"][0]:
            return "Overweight"
        else:
            return "Obese"

    elif key == "chol":
        if value < ranges["borderline"][0]:
            return "Desirable"
        elif value < ranges["high"][0]:
            return "Borderline High"
        else:
            return "High"

    elif key == "heart_rate":
        if value < ranges["normal"][0]:
            return "Athletic (Low)"
        elif value < ranges["elevated"][0]:
            return "Normal"
        elif value < ranges["high"][0]:
            return "Elevated"
        else:
            return "High"

    elif key == "age":
        if value < ranges["middle"][0]:
            return "Young Adult"
        elif value < ranges["senior"][0]:
            return "Middle Age"
        else:
            return "Senior"

    return "Unknown"


# ========== RISK LEVEL ==========
def get_risk_level(probability):
    """
    Return a dict with risk label, CSS colour class, and icon.
    Also flags low-confidence results (40–60 % boundary zone).
    """
    if 40 <= probability <= 60:
        confidence = "low"
    else:
        confidence = "high"

    if probability < 20:
        level = {"level": "Low",      "color": "success", "icon": "✓"}
    elif probability < 50:
        level = {"level": "Moderate", "color": "warning", "icon": "⚠"}
    else:
        level = {"level": "High",     "color": "danger",  "icon": "⚠"}

    level["confidence"] = confidence
    return level


# ========== RECOMMENDATIONS ENGINE ==========
def generate_recommendations(diabetes_risk, heart_risk, metrics):
    """Generate prioritised, personalised health recommendations."""
    recommendations = []

    age          = metrics.get("age",            0) or 0
    glucose      = metrics.get("glucose",        0) or 0
    bmi          = metrics.get("bmi",            0) or 0
    bp           = metrics.get("bp",             0) or 0
    chol         = metrics.get("chol",           0) or 0
    heart_rate   = metrics.get("heart_rate",     0) or 0
    chest_pain   = metrics.get("chest_pain")
    ex_angina    = metrics.get("exercise_angina")

    # --- Urgent: symptoms that need immediate attention ---
    if chest_pain is not None and int(chest_pain) in [0, 1]:
        recommendations.append({
            "category": "Urgent",
            "icon": "🚨",
            "text": (
                "Chest pain reported — seek immediate medical evaluation "
                "to rule out a cardiac event."
            ),
            "priority": "urgent"
        })

    if ex_angina is not None and int(ex_angina) == 1:
        recommendations.append({
            "category": "Urgent",
            "icon": "⚠️",
            "text": (
                "Exercise-induced chest pain detected. A cardiac stress test "
                "is strongly recommended — consult a cardiologist."
            ),
            "priority": "urgent"
        })

    # --- Age-based screening ---
    if age > 45:
        recommendations.append({
            "category": "Screening",
            "icon": "🏥",
            "text": (
                "Annual comprehensive health screening is recommended "
                "for your age group (45+)."
            ),
            "priority": "high"
        })

    # --- Diabetes-specific ---
    if diabetes_risk > 30:
        if glucose > 100:
            recommendations.append({
                "category": "Diet",
                "icon": "🥗",
                "text": (
                    "Elevated glucose detected. Adopt a low-glycaemic diet: "
                    "whole grains, legumes, leafy vegetables, and lean protein."
                ),
                "priority": "high"
            })
        if bmi > 25:
            recommendations.append({
                "category": "Weight Management",
                "icon": "⚖️",
                "text": (
                    "A 5–10 % reduction in body weight can substantially "
                    "lower diabetes risk. Aim for a modest caloric deficit "
                    "combined with regular exercise."
                ),
                "priority": "high"
            })

    # --- Cardiovascular-specific ---
    if heart_risk > 30:
        if bp > 130:
            recommendations.append({
                "category": "Blood Pressure",
                "icon": "💊",
                "text": (
                    "Blood pressure is elevated. Reduce sodium intake to "
                    "<2 300 mg/day, follow the DASH diet, and monitor BP daily."
                ),
                "priority": "high"
            })
        if chol > 200:
            recommendations.append({
                "category": "Cholesterol",
                "icon": "🐟",
                "text": (
                    "Cholesterol is above desirable levels. Increase omega-3 "
                    "fatty acids, reduce saturated fats, and discuss statin "
                    "therapy with your doctor."
                ),
                "priority": "high"
            })
        if heart_rate > 100:
            recommendations.append({
                "category": "Cardiovascular Fitness",
                "icon": "🏃",
                "text": (
                    "Resting heart rate is elevated. 150 min/week of moderate "
                    "aerobic exercise (brisk walking, cycling) will improve "
                    "cardiovascular fitness over time."
                ),
                "priority": "medium"
            })

    # --- BMI-based ---
    if bmi > 30:
        recommendations.append({
            "category": "Lifestyle",
            "icon": "🚶",
            "text": (
                "Obesity significantly elevates risk for multiple conditions. "
                "Consider consulting a nutritionist for a personalised plan."
            ),
            "priority": "high"
        })
    elif bmi < 18.5 and bmi > 0:
        recommendations.append({
            "category": "Nutrition",
            "icon": "🍽️",
            "text": (
                "Low BMI detected. Ensure adequate caloric and nutrient intake; "
                "consider speaking with a dietitian."
            ),
            "priority": "medium"
        })

    # --- All-clear ---
    if diabetes_risk < 30 and heart_risk < 30:
        recommendations.append({
            "category": "Prevention",
            "icon": "✨",
            "text": (
                "Your current profile shows low risk. Maintain a balanced diet, "
                "regular exercise, good sleep, and annual check-ups."
            ),
            "priority": "low"
        })
        recommendations.append({
            "category": "Monitoring",
            "icon": "📊",
            "text": (
                "Continue annual health check-ups to track trends "
                "and catch any changes early."
            ),
            "priority": "low"
        })

    # Sort urgent → high → medium → low
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    recommendations.sort(
        key=lambda x: priority_order.get(x.get("priority", "low"), 3)
    )

    return recommendations
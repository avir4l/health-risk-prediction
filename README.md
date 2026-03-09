# Health Risk Prediction

A machine learning web application that predicts the risk of Type 2 Diabetes and Cardiovascular Disease based on patient health metrics. Built as a minor project using Python, Flask, and scikit-learn.

**Live demo:** [health-risk-prediction.vercel.app](https://health-risk-prediction-px8y.vercel.app/)

---

## What it does

You enter basic health data — age, glucose, blood pressure, BMI, cholesterol — and the app runs two trained ML models to give you a risk percentage for diabetes and heart disease, along with a breakdown of your individual metrics and personalised recommendations.

There's also an OCR feature that lets you upload a photo of a medical report and it tries to pull the values out automatically, so you don't have to type everything in manually.

---

## Tech stack

- **Backend** — Python, Flask
- **ML** — scikit-learn (Logistic Regression, Random Forest)
- **Data** — Pima Indians Diabetes Dataset, Cleveland Heart Disease Dataset
- **OCR** — Tesseract via pytesseract
- **Frontend** — HTML, CSS, vanilla JS
- **Deployment** — Vercel

---

## Models

| Model | Dataset | Algorithm | ROC-AUC |
|---|---|---|---|
| Diabetes | Pima Indians (768 records) | Logistic Regression | ~0.83 |
| Heart Disease | Cleveland (303 records) | Random Forest | ~0.89 |

Both datasets had missing/zero values for certain fields which were replaced with column means before training — standard practice for clinical datasets where zeros are physiologically invalid (e.g. glucose = 0 makes no sense).

---

## Project structure

```
health-risk-prediction/
├── app/
│   ├── static/          # CSS
│   ├── templates/       # HTML
│   ├── app.py           # Flask routes and prediction logic
│   └── utils.py         # Health assessment and recommendation engine
├── data/
│   ├── diabetes.csv
│   └── heart.csv
├── model/
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   └── heart_model.pkl
├── notebooks/           # Training and evaluation scripts
├── requirements.txt
└── vercel.json
```

---

## Running locally

```bash
git clone https://github.com/avir4l/health-risk-prediction.git
cd health-risk-prediction

pip install -r requirements.txt

cd app
python app.py
```

Then open `http://localhost:5000` in your browser.

You'll also need Tesseract installed if you want the OCR feature to work:
- Windows: [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- Mac: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

---

## Limitations

This is an academic project, not a medical tool. The models are trained on relatively small, older datasets and several features are defaulted to population means when not provided by the user. Predictions should not be used to make any real health decisions.

---

## Acknowledgements

- Pima Indians Diabetes Dataset — UCI Machine Learning Repository
- Cleveland Heart Disease Dataset — UCI Machine Learning Repository

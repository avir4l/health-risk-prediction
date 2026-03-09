import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Locate project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
data = pd.read_csv(BASE_DIR / 'data' / 'diabetes.csv')

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Diabetes Model Accuracy:", accuracy)

# Save model and scaler
joblib.dump(model, BASE_DIR / 'model' / 'diabetes_model.pkl')
joblib.dump(scaler, BASE_DIR / 'model' / 'diabetes_scaler.pkl')

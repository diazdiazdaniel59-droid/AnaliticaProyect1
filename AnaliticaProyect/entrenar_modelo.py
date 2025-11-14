import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from imblearn.over_sampling import SMOTE


os.makedirs("models", exist_ok=True)

data = pd.read_excel("uploads/DEMALE-HSJM_2025_data.xlsx")

variables = [
    "ALT (SGPT)", "AST (SGOT)", "total_bilirubin", "direct_bilirubin",
    "hemoglobin", "hematocrit", "age", "urea", "creatinine"
]
objetivo = "diagnosis"

data = data[variables + [objetivo]].dropna()
X = data[variables]
y = data[objetivo]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# === Balanceo con SMOTE ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

modelo = LogisticRegression(max_iter=800, random_state=42)
modelo.fit(X_resampled, y_resampled)

accuracy = accuracy_score(y_test, modelo.predict(X_test_scaled))
print(f"Precisión del modelo de Regresión Logística: {accuracy:.2f}")

with open("models/modelo_logistica.pkl", "wb") as f:
    pickle.dump(modelo, f)
with open("models/escalador.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modelo y escalador guardados en /models/")
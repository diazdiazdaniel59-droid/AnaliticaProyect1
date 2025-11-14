import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rna = MLPClassifier(hidden_layer_sizes=(15, 8), max_iter=900, random_state=42)
rna.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, rna.predict(X_test_scaled))
print(f"Precisión del modelo RNA: {accuracy:.2f}")

with open("models/modelo_rna.pkl", "wb") as f:
    pickle.dump(rna, f)
with open("models/escalador_rna.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ RNA y escalador guardados en /models/")
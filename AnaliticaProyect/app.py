from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64

app = Flask(__name__)
app.secret_key = "clave_secreta_analitica"

# === Cargar modelos ===
models = {
    "rna": {
        "modelo": pickle.load(open("models/modelo_rna.pkl", "rb")),
        "scaler": pickle.load(open("models/escalador_rna.pkl", "rb"))
    },
    "logistica": {
        "modelo": pickle.load(open("models/modelo_logistica.pkl", "rb")),
        "scaler": pickle.load(open("models/escalador.pkl", "rb"))
    }
}

# === Variables seleccionadas (9 más relevantes) ===
variables = [
    "ALT (SGPT)", "AST (SGOT)", "total_bilirubin", "direct_bilirubin",
    "hemoglobin", "hematocrit", "age", "urea", "creatinine"
]

nombres_variables = {
    "ALT (SGPT)": "ALT (SGPT)",
    "AST (SGOT)": "AST (SGOT)",
    "total_bilirubin": "Bilirrubina Total",
    "direct_bilirubin": "Bilirrubina Directa",
    "hemoglobin": "Hemoglobina",
    "hematocrit": "Hematocrito",
    "age": "Edad",
    "urea": "Urea",
    "creatinine": "Creatinina"
}

# === Cargar dataset base para rangos ===
data = pd.read_excel("uploads/DEMALE-HSJM_2025_data.xlsx")
rangos = {}

# Variables que deben mostrarse como enteras
enteras = ["age", "hematocrit", "hemoglobin", "AST (SGOT)", "ALT (SGPT)"]

for v in variables:
    min_val = float(data[v].min())
    max_val = float(data[v].max())
    if v in enteras:
        min_val = int(round(min_val))
        max_val = int(round(max_val))
        step_val = 1
    else:
        step_val = round((max_val - min_val) / 100, 2)
    rangos[v] = {"min": min_val, "max": max_val, "step": step_val}

# === RUTAS ===

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/individual', methods=['GET', 'POST'])
def individual():
    resultado = None
    color_resultado = None
    if request.method == 'POST':
        try:
            modelo_sel = request.form.get("modelo")
            if modelo_sel not in models:
                flash("Modelo no válido.", "error")
                return redirect(url_for('individual'))

            # Capturar valores del formulario
            datos = [float(request.form.get(v)) for v in variables]
            df = pd.DataFrame([datos], columns=variables)

            scaler = models[modelo_sel]["scaler"]
            modelo = models[modelo_sel]["modelo"]

            pred = modelo.predict(scaler.transform(df))[0]

            if pred == 1:
                resultado = "⚠️ Resultado Positivo para enfermedad"
                color_resultado = "danger"
            else:
                resultado = "✅ Resultado Negativo para enfermedad"
                color_resultado = "success"

        except Exception as e:
            flash(f"Error: {str(e)}", "error")

    return render_template(
        'individual.html',
        columnas=variables,
        nombres=nombres_variables,
        rangos=rangos,
        resultado=resultado,
        color_resultado=color_resultado
    )


@app.route('/lotes', methods=['GET', 'POST'])
def lotes():
    tabla, metricas, cm = None, None, None
    model_name = None
    if request.method == 'POST':
        try:
            archivo = request.files.get("dataset")
            modelo_sel = request.form.get("modelo")

            if not archivo:
                flash("Debes subir un archivo primero.", "error")
                return redirect(url_for('lotes'))

            # === Leer archivo ===
            if archivo.filename.endswith(".csv"):
                df = pd.read_csv(archivo)
            elif archivo.filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(archivo)
            else:
                flash("Formato no soportado. Sube un archivo .csv o .xlsx", "error")
                return redirect(url_for('lotes'))

            df = df.dropna(subset=variables)
            X = df[variables]
            y_true = df["diagnosis"] if "diagnosis" in df.columns else None

            scaler = models[modelo_sel]["scaler"]
            modelo = models[modelo_sel]["modelo"]
            model_name = "Red Neuronal" if modelo_sel == "rna" else "Regresión Logística"

            y_pred = modelo.predict(scaler.transform(X))
            df["Predicción"] = ["Positivo" if p == 1 else "Negativo" for p in y_pred]

            # === Métricas ===
            if y_true is not None:
                acc = round(accuracy_score(y_true, y_pred) * 100, 2)
                prec = round(precision_score(y_true, y_pred, average='weighted') * 100, 2)
                rec = round(recall_score(y_true, y_pred, average='weighted') * 100, 2)
                f1 = round(f1_score(y_true, y_pred, average='weighted') * 100, 2)
                metricas = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

                # === Matriz de confusión ===
                cmatrix = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
                ax.set_xlabel("Predicción")
                ax.set_ylabel("Valor Real")
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                cm = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)

            # === Guardar resultados ===
            os.makedirs("uploads", exist_ok=True)
            salida = os.path.join("uploads", "resultados_prediccion.xlsx")
            df.to_excel(salida, index=False)

            tabla = df.head(30).to_html(classes="tabla-scroll", index=False)
            flash("✅ Predicción por lotes completada correctamente.", "success")

        except Exception as e:
            flash(f"Error durante la predicción: {str(e)}", "error")

    return render_template("lotes.html", tabla=tabla, metricas=metricas, cm=cm, model_name=model_name)


@app.route('/descargar_resultados')
def descargar_resultados():
    ruta = os.path.join("uploads", "resultados_prediccion.xlsx")
    if not os.path.exists(ruta):
        flash("No hay resultados para descargar.", "error")
        return redirect(url_for('lotes'))
    return send_file(ruta, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
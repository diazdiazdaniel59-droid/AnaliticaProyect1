import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuración de página ===
st.set_page_config(
    page_title="Proyecto Analítica",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Cargar modelos ===
@st.cache_resource
def load_models():
    """Cargar modelos entrenados"""
    # Determinar el directorio base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    # Verificar que existe el directorio
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Directorio de modelos no encontrado: {models_dir}. Directorio actual: {os.getcwd()}")
    
    modelo_rna_path = os.path.join(models_dir, "modelo_rna.pkl")
    escalador_rna_path = os.path.join(models_dir, "escalador_rna.pkl")
    modelo_logistica_path = os.path.join(models_dir, "modelo_logistica.pkl")
    escalador_path = os.path.join(models_dir, "escalador.pkl")
    
    # Verificar que existan los archivos
    for path, name in [(modelo_rna_path, "modelo_rna.pkl"), 
                       (escalador_rna_path, "escalador_rna.pkl"),
                       (modelo_logistica_path, "modelo_logistica.pkl"),
                       (escalador_path, "escalador.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {name} en {path}")
    
    with open(modelo_rna_path, "rb") as f:
        modelo_rna = pickle.load(f)
    with open(escalador_rna_path, "rb") as f:
        scaler_rna = pickle.load(f)
    with open(modelo_logistica_path, "rb") as f:
        modelo_logistica = pickle.load(f)
    with open(escalador_path, "rb") as f:
        scaler_logistica = pickle.load(f)
    
    models = {
        "rna": {
            "modelo": modelo_rna,
            "scaler": scaler_rna
        },
        "logistica": {
            "modelo": modelo_logistica,
            "scaler": scaler_logistica
        }
    }
    return models

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
@st.cache_data
def load_ranges():
    """Cargar rangos de valores del dataset"""
    try:
        # Determinar el directorio base
        base_dir = os.path.dirname(os.path.abspath(__file__))
        uploads_dir = os.path.join(base_dir, "uploads")
        data_path = os.path.join(uploads_dir, "DEMALE-HSJM_2025_data.xlsx")
        
        if not os.path.exists(data_path):
            st.warning(f"Archivo de datos no encontrado: {data_path}. Usando rangos por defecto.")
            # Retornar rangos por defecto si no se encuentra el archivo
            return {
                "ALT (SGPT)": {"min": 0, "max": 500, "step": 1.0},
                "AST (SGOT)": {"min": 0, "max": 500, "step": 1.0},
                "total_bilirubin": {"min": 0.0, "max": 50.0, "step": 0.5},
                "direct_bilirubin": {"min": 0.0, "max": 20.0, "step": 0.2},
                "hemoglobin": {"min": 0, "max": 20, "step": 1.0},
                "hematocrit": {"min": 0, "max": 60, "step": 1.0},
                "age": {"min": 0, "max": 100, "step": 1.0},
                "urea": {"min": 0.0, "max": 100.0, "step": 1.0},
                "creatinine": {"min": 0.0, "max": 10.0, "step": 0.1}
            }
        
        data = pd.read_excel(data_path)
        rangos = {}
        enteras = ["age", "hematocrit", "hemoglobin", "AST (SGOT)", "ALT (SGPT)"]
        
        for v in variables:
            min_val = float(data[v].min())
            max_val = float(data[v].max())
            if v in enteras:
                min_val = int(round(min_val))
                max_val = int(round(max_val))
                step_val = 1.0
            else:
                step_val = round((max_val - min_val) / 100, 2)
            rangos[v] = {"min": min_val, "max": max_val, "step": step_val}
        return rangos
    except Exception as e:
        st.warning(f"Error al cargar rangos: {str(e)}. Usando rangos por defecto.")
        # Retornar rangos por defecto
        return {
            "ALT (SGPT)": {"min": 0, "max": 500, "step": 1.0},
            "AST (SGOT)": {"min": 0, "max": 500, "step": 1.0},
            "total_bilirubin": {"min": 0.0, "max": 50.0, "step": 0.5},
            "direct_bilirubin": {"min": 0.0, "max": 20.0, "step": 0.2},
            "hemoglobin": {"min": 0, "max": 20, "step": 1.0},
            "hematocrit": {"min": 0, "max": 60, "step": 1.0},
            "age": {"min": 0, "max": 100, "step": 1.0},
            "urea": {"min": 0.0, "max": 100.0, "step": 1.0},
            "creatinine": {"min": 0.0, "max": 10.0, "step": 0.1}
        }

# === Cargar modelos ===
try:
    models = load_models()
    rangos = load_ranges()
except Exception as e:
    st.error(f"Error al cargar modelos: {str(e)}")
    st.stop()

# === Interfaz principal ===
st.title("🧠 Proyecto Analítica")
st.markdown("### Sistema de diagnóstico asistido por inteligencia artificial")

# === Sidebar para navegación ===
pagina = st.sidebar.selectbox(
    "Selecciona una opción:",
    ["Inicio", "Predicción Individual", "Predicción por Lotes"]
)

# === Página de Inicio ===
if pagina == "Inicio":
    st.markdown("""
    ### Bienvenido al Sistema de Predicción Médica
    
    Este sistema utiliza inteligencia artificial para ayudar en el diagnóstico médico
    basado en análisis de variables clínicas.
    
    **Características:**
    - ✅ Predicción individual de casos
    - 📊 Predicción por lotes con análisis estadístico
    - 🧠 Dos modelos de IA: Red Neuronal y Regresión Logística
    - 📈 Visualización de métricas y matrices de confusión
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Predicción Individual**\n\nPredice un caso específico ingresando los valores manualmente.")
    
    with col2:
        st.info("**Predicción por Lotes**\n\nSube un archivo Excel o CSV para procesar múltiples casos.")
    
    with col3:
        st.info("**Modelos Disponibles**\n\n- Red Neuronal Artificial\n- Regresión Logística")

# === Página de Predicción Individual ===
elif pagina == "Predicción Individual":
    st.header("📋 Predicción Individual")
    
    modelo_sel = st.selectbox(
        "Selecciona el modelo:",
        ["rna", "logistica"],
        format_func=lambda x: "Red Neuronal" if x == "rna" else "Regresión Logística"
    )
    
    st.markdown("---")
    
    # Crear formulario con columnas
    col1, col2, col3 = st.columns(3)
    
    valores = {}
    with col1:
        for i, var in enumerate(variables[:3]):
            if var in rangos:
                valores[var] = st.number_input(
                    nombres_variables[var],
                    min_value=float(rangos[var]["min"]),
                    max_value=float(rangos[var]["max"]),
                    step=float(rangos[var]["step"]),
                    value=float(rangos[var]["min"]),
                    key=f"input_{var}"
                )
    
    with col2:
        for i, var in enumerate(variables[3:6]):
            if var in rangos:
                valores[var] = st.number_input(
                    nombres_variables[var],
                    min_value=float(rangos[var]["min"]),
                    max_value=float(rangos[var]["max"]),
                    step=float(rangos[var]["step"]),
                    value=float(rangos[var]["min"]),
                    key=f"input_{var}"
                )
    
    with col3:
        for i, var in enumerate(variables[6:]):
            if var in rangos:
                valores[var] = st.number_input(
                    nombres_variables[var],
                    min_value=float(rangos[var]["min"]),
                    max_value=float(rangos[var]["max"]),
                    step=float(rangos[var]["step"]),
                    value=float(rangos[var]["min"]),
                    key=f"input_{var}"
                )
    
    st.markdown("---")
    
    if st.button("🔮 Predecir", type="primary", use_container_width=True):
        try:
            # Preparar datos
            datos = [valores[v] for v in variables]
            df = pd.DataFrame([datos], columns=variables)
            
            # Obtener modelo y scaler
            scaler = models[modelo_sel]["scaler"]
            modelo = models[modelo_sel]["modelo"]
            
            # Predecir
            pred = modelo.predict(scaler.transform(df))[0]
            
            # Mostrar resultado
            st.markdown("---")
            if pred == 1:
                st.error("⚠️ **Resultado Positivo para enfermedad**")
            else:
                st.success("✅ **Resultado Negativo para enfermedad**")
                
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")

# === Página de Predicción por Lotes ===
elif pagina == "Predicción por Lotes":
    st.header("📊 Predicción por Lotes")
    
    modelo_sel = st.selectbox(
        "Selecciona el modelo:",
        ["logistica", "rna"],
        format_func=lambda x: "Regresión Logística" if x == "logistica" else "Red Neuronal"
    )
    
    st.markdown("---")
    
    archivo = st.file_uploader(
        "Sube tu archivo Excel o CSV",
        type=["xlsx", "xls", "csv"],
        help="El archivo debe contener las columnas requeridas"
    )
    
    if archivo is not None:
        try:
            # Leer archivo
            if archivo.name.endswith(".csv"):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)
            
            st.success(f"✅ Archivo cargado: {archivo.name} ({len(df)} registros)")
            
            # Verificar columnas requeridas
            columnas_faltantes = [v for v in variables if v not in df.columns]
            if columnas_faltantes:
                st.error(f"❌ Columnas faltantes: {', '.join(columnas_faltantes)}")
            else:
                if st.button("🔮 Procesar Predicciones", type="primary", use_container_width=True):
                    # Limpiar datos
                    df_clean = df.dropna(subset=variables).copy()
                    X = df_clean[variables]
                    
                    # Obtener modelo y scaler
                    scaler = models[modelo_sel]["scaler"]
                    modelo = models[modelo_sel]["modelo"]
                    
                    # Predecir
                    y_pred = modelo.predict(scaler.transform(X))
                    df_clean["Predicción"] = ["Positivo" if p == 1 else "Negativo" for p in y_pred]
                    
                    # Mostrar resultados
                    st.markdown("### 📈 Resultados")
                    st.dataframe(df_clean.head(30), use_container_width=True)
                    
                    # Métricas si existe columna de diagnóstico real
                    if "diagnosis" in df_clean.columns:
                        y_true = df_clean["diagnosis"]
                        
                        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
                        prec = round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
                        rec = round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
                        f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{acc}%")
                        col2.metric("Precision", f"{prec}%")
                        col3.metric("Recall", f"{rec}%")
                        col4.metric("F1-Score", f"{f1}%")
                        
                        # Matriz de confusión
                        st.markdown("### 📊 Matriz de Confusión")
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
                        ax.set_xlabel("Predicción")
                        ax.set_ylabel("Valor Real")
                        ax.set_title("Matriz de Confusión")
                        st.pyplot(fig)
                    
                    # Descargar resultados
                    os.makedirs("uploads", exist_ok=True)
                    salida = os.path.join("uploads", "resultados_prediccion.xlsx")
                    df_clean.to_excel(salida, index=False)
                    
                    with open(salida, "rb") as f:
                        st.download_button(
                            label="📥 Descargar Resultados",
                            data=f.read(),
                            file_name="resultados_prediccion.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")

# 🧠 Proyecto Analítica - Streamlit

Aplicación web para predicción médica asistida por inteligencia artificial, desarrollada con Streamlit.

## 📋 Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`

## 🚀 Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## ▶️ Ejecución Local

Para ejecutar la aplicación localmente:

```bash
streamlit run app_streamlit.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📦 Despliegue en Streamlit Cloud

1. Sube tu repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio de GitHub
4. Configura:
   - **Main file path**: `app_streamlit.py`
   - **Python version**: 3.8 o superior

## 📁 Estructura del Proyecto

```
AnaliticaProyect/
├── app_streamlit.py          # Aplicación principal Streamlit
├── app.py                     # Aplicación Flask (alternativa)
├── requirements.txt           # Dependencias del proyecto
├── models/                    # Modelos entrenados
│   ├── modelo_rna.pkl
│   ├── modelo_logistica.pkl
│   ├── escalador_rna.pkl
│   └── escalador.pkl
├── uploads/                   # Archivos de datos
│   └── DEMALE-HSJM_2025_data.xlsx
└── .streamlit/
    └── config.toml           # Configuración de Streamlit
```

## 🔧 Características

- **Predicción Individual**: Ingresa valores manualmente para un caso específico
- **Predicción por Lotes**: Sube un archivo Excel o CSV para procesar múltiples casos
- **Modelos de IA**: 
  - Red Neuronal Artificial
  - Regresión Logística
- **Análisis Estadístico**: Métricas de rendimiento y matriz de confusión

## 📝 Notas

- Asegúrate de que los archivos de modelos estén presentes en la carpeta `models/`
- El archivo de datos debe estar en `uploads/DEMALE-HSJM_2025_data.xlsx` para cargar los rangos de valores


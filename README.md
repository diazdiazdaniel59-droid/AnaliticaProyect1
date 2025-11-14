# Proyecto Analítica

Proyecto de análisis de datos y predicción utilizando modelos de Machine Learning.

## Descripción

Este proyecto incluye:
- Análisis de variables
- Entrenamiento de modelos de Machine Learning
- Red Neuronal Artificial (RNA)
- Regresión Logística
- Aplicación web con Streamlit

## 🚀 Despliegue en Streamlit Cloud

Para desplegar en Streamlit Cloud:

1. Asegúrate de que todos los archivos estén en tu repositorio de GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Configura:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.8 o superior
5. Haz clic en "Deploy"

**Archivos necesarios en el repositorio:**
- `streamlit_app.py` (aplicación principal)
- `requirements.txt` (dependencias)
- `AnaliticaProyect/models/*.pkl` (modelos entrenados)
- `AnaliticaProyect/uploads/DEMALE-HSJM_2025_data.xlsx` (datos para rangos)

## 📋 Instalación Local

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## ▶️ Ejecución Local

Para ejecutar la aplicación localmente con Streamlit:

```bash
streamlit run streamlit_app.py
```

O si estás en la carpeta AnaliticaProyect:

```bash
streamlit run ../streamlit_app.py
```

Para ejecutar la aplicación Flask (alternativa):

```bash
cd AnaliticaProyect
python app.py
```

## Estructura del Proyecto

```
.
├── streamlit_app.py          # Aplicación principal Streamlit (RAÍZ)
├── requirements.txt          # Dependencias del proyecto (RAÍZ)
├── AnaliticaProyect/
│   ├── app.py               # Aplicación Flask (alternativa)
│   ├── app_streamlit.py     # Versión Streamlit en subdirectorio
│   ├── analisis_variables.py
│   ├── entrenar_modelo.py
│   ├── entrenar_rna.py
│   ├── models/              # Modelos entrenados
│   │   ├── modelo_rna.pkl
│   │   ├── modelo_logistica.pkl
│   │   ├── escalador_rna.pkl
│   │   └── escalador.pkl
│   ├── uploads/             # Archivos de datos
│   │   └── DEMALE-HSJM_2025_data.xlsx
│   ├── templates/           # Plantillas HTML (solo Flask)
│   └── static/              # Archivos estáticos (solo Flask)
```

## 🔧 Características

- **Predicción Individual**: Ingresa valores manualmente para un caso específico
- **Predicción por Lotes**: Sube un archivo Excel o CSV para procesar múltiples casos
- **Modelos de IA**: 
  - Red Neuronal Artificial
  - Regresión Logística
- **Análisis Estadístico**: Métricas de rendimiento y matriz de confusión

## 📝 Notas Importantes

- Para Streamlit Cloud, el archivo principal debe ser `streamlit_app.py` en la raíz
- Los modelos (.pkl) deben estar en `AnaliticaProyect/models/`
- El archivo de datos debe estar en `AnaliticaProyect/uploads/` para cargar los rangos de valores

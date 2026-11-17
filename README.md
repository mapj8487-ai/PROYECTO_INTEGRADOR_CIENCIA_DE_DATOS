# Centro de Inteligencia de Demanda de Bicicletas

Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial.
Universidad de La Salle - Maestría en Inteligencia Artificial.

## Dashboard

Aplicación interactiva desarrollada en Streamlit.

### Módulos

1. Exploración interactiva de datos.
2. Indicadores clave de rendimiento.
3. Predicción de demanda mediante Random Forest optimizado.
4. Segmentación K-Means y visualización PCA.

## Ejecución local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run dashboard/app.py
```

## Datos

Bike Sharing Dataset, registros horarios de 2011 y 2012.

## Observaciones

El modelo predictivo fue entrenado previamente.
Las predicciones representan patrones históricos y no demanda actual.

El archivo de PCA contiene 12.165 observaciones de entrenamiento.

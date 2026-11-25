# Segmentación de perfiles metabólicos con K-Means

Proyecto integrador de minería de datos usando el dataset `Diabetes_Mexico.csv`. El objetivo es agrupar adultos en México según cinco variables clínicas (`edad`, `imc`, `glu_suero`, `insulina` y `trig`) con el algoritmo no supervisado K-Means y mostrar los resultados en un dashboard hecho con Streamlit.

## Herramientas

- Python 3
- Pandas y NumPy
- Scikit-learn (`StandardScaler`, `KMeans`, `PCA`, métricas de evaluación)
- Matplotlib y Seaborn
- Streamlit

## Estructura del repositorio

```
Diabetes_KMeans/
├── data/
│   ├── raw/                  # Dataset original (Diabetes_Mexico.csv)
│   └── processed/            # Datos limpios, escalados y con clúster asignado
├── src/
│   ├── 1_exploracion.py            # Revisión inicial y gráficas exploratorias
│   ├── 2_limpieza_preparacion.py   # Limpieza, tratamiento de atípicos y StandardScaler
│   └── 3_kmeans_analisis.py        # Evaluación de K (2 a 8), modelo final (K=4) y tablas
├── dashboard/
│   └── app.py                # Dashboard interactivo de 2 páginas en Streamlit
├── outputs/
│   ├── figures/              # Gráficas generadas y capturas del dashboard
│   └── tables/               # Tablas de métricas y centroides en CSV
├── docs/                     # Documentación del proyecto (PDF)
├── requirements.txt
└── README.md
```

## Cómo ejecutar el proyecto

1. Instalar las librerías necesarias:

```bash
pip install -r requirements.txt
```

2. Correr los scripts en orden desde la carpeta principal del proyecto:

```bash
python src/1_exploracion.py
python src/2_limpieza_preparacion.py
python src/3_kmeans_analisis.py
```

Al terminar se generan los archivos procesados en `data/processed/`, las tablas en `outputs/tables/` y las gráficas en `outputs/figures/`.

3. Abrir el dashboard en el navegador:

```bash
streamlit run dashboard/app.py
```

El dashboard cuenta con dos páginas seleccionables desde la barra lateral: **Análisis General** y **Segmentación K-Means**, además de filtros por sexo, rango de edad, perfil de clúster y ciudad.

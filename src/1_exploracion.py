"""
Parte 2: Obtención y Exploración del Dataset (EDA)
Proyecto: Segmentación de Perfiles Metabólicos de Adultos en México mediante K-Means
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Rutas portables relativas a la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Diabetes_Mexico.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")

print("PARTE 2: OBTENCIÓN Y EXPLORACIÓN DEL DATASET")

# 1. Importar dataset con read_csv()
df = pd.read_csv(RAW_DATA_PATH)
print(f"\nFuente de datos: {RAW_DATA_PATH.name} (Encuesta poblacional de salud en México - ENSANUT)")

# 2. Primeros 10 registros
print("\n--- Primeros 10 registros ---")
print(df.head(10).to_string())

# 3. Dimensiones, columnas y tipos de datos
print("\n--- Dimensiones, columnas y tipos de datos ---")
print(f"Dimensiones: {df.shape[0]} registros x {df.shape[1]} columnas")
print("Columnas:", list(df.columns))
print("\nInformación detallada de columnas y tipos:")
df.info()

# 4. Estadísticas descriptivas (columnas numéricas originales)
print("\n--- Estadísticas descriptivas (variables numéricas iniciales) ---")
desc_num = df.describe().round(2)
print(desc_num.to_string())
desc_num.to_csv(TABLES_DIR / "01_eda_estadisticas_iniciales.csv")

# 5. Valores nulos y registros duplicados
print("\n--- Valores nulos por columna ---")
nulos = df.isnull().sum()
nulos_pct = (df.isnull().mean() * 100).round(2)
resumen_nulos = pd.DataFrame({"Nulos": nulos, "Porcentaje (%)": nulos_pct})
print(resumen_nulos[resumen_nulos["Nulos"] > 0].sort_values("Nulos", ascending=False).to_string())
resumen_nulos.to_csv(TABLES_DIR / "01_eda_resumen_nulos.csv")

duplicados = df.duplicated().sum()
print(f"\nRegistros completamente duplicados: {duplicados}")

# Vista temporal con conversión de comas solo para graficar correctamente en el EDA
df_plot = df.copy()
for col in ["Peso", "Estatura", "ac_urico", "albu", "col_hdl", "creat", "insulina"]:
    df_plot[col] = pd.to_numeric(df_plot[col].astype(str).str.replace(",", "."), errors="coerce")
df_plot = df_plot[df_plot["edad"] > 0]
df_plot["imc_temp"] = (df_plot["Peso"] / ((df_plot["Estatura"] / 100) ** 2)).fillna(df_plot["imc"])

# Grafica exploratoria 1: Valores nulos por columna
nulos_positivos = resumen_nulos[resumen_nulos["Nulos"] > 0].sort_values("Porcentaje (%)", ascending=True)
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(nulos_positivos.index, nulos_positivos["Porcentaje (%)"], color="#4C72B0")
ax.set_title("Porcentaje de Valores Nulos por Columna (Dataset Original)", fontsize=13, fontweight="bold")
ax.set_xlabel("Porcentaje de registros nulos (%)")
ax.set_ylabel("Variable")
for bar in bars:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_eda_valores_nulos.png", dpi=150)
plt.close()

# Grafica exploratoria 2: Distribucion de variables metabolicas clave
vars_dist = [
    ("edad", "Edad (años)"),
    ("imc_temp", "IMC (kg/m²)"),
    ("glu_suero", "Glucosa en suero (mg/dL)"),
    ("trig", "Triglicéridos (mg/dL)"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (col, titulo) in zip(axes.flat, vars_dist):
    serie = df_plot[col].dropna()
    # Limitar eje visual al percentil 99 para apreciar la distribución principal
    limite = serie.quantile(0.99)
    sns.histplot(serie[serie <= limite], kde=True, bins=30, ax=ax, color="#2A9D8F")
    ax.set_title(f"Distribución de {titulo}", fontsize=11, fontweight="bold")
    ax.set_xlabel(titulo)
    ax.set_ylabel("Frecuencia")
fig.suptitle("Exploración de Variables Metabólicas Principales (hasta P99 visual)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_eda_distribucion_variables.png", dpi=150)
plt.close()

# Grafica exploratoria 3: Mapa de calor de correlaciones
cols_corr = ["edad", "imc_temp", "glu_suero", "insulina", "trig", "colest", "col_hdl", "col_ldl", "ac_urico"]
corr_matrix = df_plot[cols_corr].rename(columns={"imc_temp": "imc"}).corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Matriz de Correlación de Variables Clínicas y Metabólicas", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_eda_correlacion.png", dpi=150)
plt.close()

# Seleccion y justificacion de variables para K-Means
print("\n--- Variables seleccionadas para K-Means (5 variables) ---")
justificacion = {
    "edad": "Factor demográfico clave; el riesgo metabólico y glucémico cambia significativamente con la edad.",
    "imc": "Indicador antropométrico estándar de sobrepeso y obesidad asociado a resistencia a la insulina.",
    "glu_suero": "Biomarcador directo del metabolismo de la glucosa (normal, prediabetes y diabetes).",
    "insulina": "Permite diferenciar resistencia a la insulina (hiperinsulinemia) de deficiencia insulínica.",
    "trig": "Indicador lipídico fundamental del síndrome metabólico y riesgo cardiovascular.",
}
for var, razon in justificacion.items():
    print(f"  * {var}: {razon}")

print(f"\n[OK] Gráficas exploratorias guardadas en: {FIGURES_DIR}")

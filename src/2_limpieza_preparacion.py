"""
Parte 3: Limpieza, Tratamiento de Outliers y Preparación de Datos
Proyecto: Segmentación de Perfiles Metabólicos de Adultos en México mediante K-Means
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Diabetes_Mexico.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_limpio.csv"
SCALED_DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_escalado.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

sns.set_theme(style="whitegrid")

FEATURES_KMEANS = ["edad", "imc", "glu_suero", "insulina", "trig"]

print("PARTE 3: LIMPIEZA Y PREPARACIÓN DE DATOS")

# 1. Cargar datos crudos
df = pd.read_csv(RAW_DATA_PATH)
registros_iniciales = len(df)
print(f"Registros iniciales: {registros_iniciales} | Columnas iniciales: {df.shape[1]}")

# 2. Eliminar registros duplicados
duplicados = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicados eliminados: {duplicados}")

# 3. Eliminar columnas administrativas de diseño muestral sin valor clínico
cols_irrelevantes = ["muestra_suero", "ponde_venosa", "ponde_hemo", "estrato", "est_sel"]
df = df.drop(columns=cols_irrelevantes)
print(f"Columnas eliminadas ({len(cols_irrelevantes)}): {cols_irrelevantes}")

# 4. Corregir inconsistencias de formato decimal (coma -> punto) y convertir a numérico
cols_coma = ["Peso", "Estatura", "ac_urico", "albu", "col_hdl", "creat", "insulina"]
for col in cols_coma:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("Tipos de datos corregidos a float64 en:", cols_coma)

# 5. Eliminar inconsistencias de edad (nulas o <= 0, incluyendo valor corrupto -7975)
n_edad_nula = df["edad"].isna().sum()
n_edad_invalida = (df["edad"] <= 0).sum()
df = df.dropna(subset=["edad"])
df = df[df["edad"] > 0].copy()
print(f"Edades nulas eliminadas: {n_edad_nula} | Edades <= 0 eliminadas: {n_edad_invalida}")

# 6. Recalcular IMC a partir de Peso y Estatura conservando el IMC previo cuando falten ambos
imc_calculado = (df["Peso"] / ((df["Estatura"] / 100) ** 2)).round(2)
df["imc"] = imc_calculado.fillna(df["imc"]).round(2)

# 7. Estandarizar texto en variables categóricas
df["sexo"] = df["sexo"].astype(str).str.strip().str.capitalize()
df["Ciudad"] = df["Ciudad"].astype(str).str.strip().str.title()

# 8. Crear variables derivadas útiles para análisis y filtros del dashboard
df["rango_edad"] = pd.cut(
    df["edad"],
    bins=[19, 29, 44, 59, 120],
    labels=["20-29", "30-44", "45-59", "60+"],
)
df["categoria_glucosa"] = pd.cut(
    df["glu_suero"],
    bins=[0, 100, 126, 2000],
    labels=["Normal", "Prediabetes", "Diabetes"],
    right=False,
)
# Codificación binaria de sexo para referencia numérica
df["sexo_cod"] = df["sexo"].map({"Mujer": 0, "Hombre": 1})

# 9. Tratamiento de valores nulos en las 5 variables del modelo K-Means
# Se eliminan registros que carecen de alguna variable esencial (principalmente IMC)
# para evitar imputaciones artificiales masivas que distorsionen las distancias euclidianas.
antes_dropna = len(df)
df_modelo = df.dropna(subset=FEATURES_KMEANS).copy()
eliminados_nulos = antes_dropna - len(df_modelo)
print(f"Registros antes de filtrar nulos en variables K-Means: {antes_dropna}")
print(f"Registros descartados por falta de variables clave: {eliminados_nulos}")
print(f"Registros completos para modelado: {len(df_modelo)}")

# 10. Identificación de valores atípicos (método IQR) y tratamiento por Winsorización (P1-P99)
print("\n--- Análisis de valores atípicos (IQR) y Winsorización (P1-P99) ---")
df_antes_outliers = df_modelo[FEATURES_KMEANS].copy()
resumen_outliers = []

for col in FEATURES_KMEANS:
    q1 = df_modelo[col].quantile(0.25)
    q3 = df_modelo[col].quantile(0.75)
    iqr = q3 - q1
    lim_inf_iqr = q1 - 1.5 * iqr
    lim_sup_iqr = q3 + 1.5 * iqr
    n_out_iqr = int(((df_modelo[col] < lim_inf_iqr) | (df_modelo[col] > lim_sup_iqr)).sum())

    p01 = df_modelo[col].quantile(0.01)
    p99 = df_modelo[col].quantile(0.99)
    max_orig = df_modelo[col].max()
    min_orig = df_modelo[col].min()

    # Guardar valores originales en copia para referencia y aplicar winsorización en las columnas de trabajo
    df_modelo[col] = df_modelo[col].clip(lower=p01, upper=p99).round(2)

    resumen_outliers.append({
        "Variable": col,
        "Min Original": round(min_orig, 2),
        "Max Original": round(max_orig, 2),
        "Atípicos (IQR)": n_out_iqr,
        "Límite P1": round(p01, 2),
        "Límite P99": round(p99, 2),
    })

df_outliers = pd.DataFrame(resumen_outliers)
print(df_outliers.to_string(index=False))
df_outliers.to_csv(TABLES_DIR / "02_resumen_outliers.csv", index=False)

print(
    "\nJustificación del tratamiento de atípicos: En datos clínicos, valores elevados de "
    "glucosa, insulina o triglicéridos corresponden a pacientes reales con diabetes o "
    "dislipidemia severa. Eliminar todos los atípicos por IQR borraría al grupo de mayor "
    "riesgo; por ello se aplica winsorización (P1-P99) para contener valores extremos "
    "(como glucosa=1116 o triglicéridos=2834) sin perder registros ni el perfil clínico."
)

# Gráfica comparativa antes vs después de winsorización
fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
for idx, col in enumerate(FEATURES_KMEANS):
    sns.boxplot(y=df_antes_outliers[col], ax=axes[0, idx], color="#E76F51")
    axes[0, idx].set_title(f"{col} (Antes)", fontsize=10, fontweight="bold")
    axes[0, idx].set_ylabel("")

    sns.boxplot(y=df_modelo[col], ax=axes[1, idx], color="#2A9D8F")
    axes[1, idx].set_title(f"{col} (Después P1-P99)", fontsize=10, fontweight="bold")
    axes[1, idx].set_ylabel("")

fig.suptitle("Tratamiento de Valores Atípicos: Antes vs Después de Winsorización (P1-P99)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_outliers_antes_despues.png", dpi=150)
plt.close()

# 11. Escalamiento con StandardScaler (z = (x - media) / desv_estandar)
scaler = StandardScaler()
X_escalado = scaler.fit_transform(df_modelo[FEATURES_KMEANS])
df_escalado = pd.DataFrame(X_escalado, columns=FEATURES_KMEANS, index=df_modelo.index)

print("\n--- Comparación antes y después de StandardScaler ---")
comp_escala = pd.DataFrame({
    "Media Original": df_modelo[FEATURES_KMEANS].mean().round(2),
    "Std Original": df_modelo[FEATURES_KMEANS].std().round(2),
    "Media Escalada": df_escalado.mean().round(4),
    "Std Escalada": df_escalado.std().round(4),
})
print(comp_escala.to_string())
comp_escala.to_csv(TABLES_DIR / "02_comparacion_escalamiento.csv")

# 12. Guardar datasets preparados
df_modelo.to_csv(CLEAN_DATA_PATH, index=False)
df_escalado.to_csv(SCALED_DATA_PATH, index=False)

print(f"\n[OK] Dataset limpio guardado en: {CLEAN_DATA_PATH} ({df_modelo.shape[0]} filas x {df_modelo.shape[1]} cols)")
print(f"[OK] Dataset escalado guardado en: {SCALED_DATA_PATH}")

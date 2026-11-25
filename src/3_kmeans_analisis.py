"""
Partes 4 y 5: Implementación de K-Means, Comparación de Configuraciones e Interpretación
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_limpio.csv"
CLUSTERS_DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_clusters.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

sns.set_theme(style="whitegrid")

FEATURES = ["edad", "imc", "glu_suero", "insulina", "trig"]
LABELS_ES = {
    "edad": "Edad (años)",
    "imc": "IMC (kg/m²)",
    "glu_suero": "Glucosa (mg/dL)",
    "insulina": "Insulina (µU/mL)",
    "trig": "Triglicéridos (mg/dL)",
}

print("PARTE 4: IMPLEMENTACIÓN DEL ALGORITMO K-MEANS")

# 1. Cargar datos preparados y re-aplicar StandardScaler para poder revertir centroides
df = pd.read_csv(CLEAN_DATA_PATH)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

# 2. Evaluar al menos 5 valores diferentes de K (evaluamos 7: K = 2 a 8)
valores_k = list(range(2, 9))
metricas_k = []

for k in valores_k:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    etiquetas = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, etiquetas)
    db = davies_bouldin_score(X_scaled, etiquetas)
    metricas_k.append({
        "K": k,
        "Inercia (WCSS)": round(km.inertia_, 2),
        "Silhouette Score": round(sil, 4),
        "Davies-Bouldin": round(db, 4),
    })

df_metricas_k = pd.DataFrame(metricas_k)
print("\n--- Evaluación de K = 2 a 8 ---")
print(df_metricas_k.to_string(index=False))
df_metricas_k.to_csv(TABLES_DIR / "03_evaluacion_k.csv", index=False)

# Gráfica 5: Método del Codo + Silhouette Score
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(df_metricas_k["K"], df_metricas_k["Inercia (WCSS)"], marker="o", linewidth=2.5, color="#264653")
axes[0].axvline(4, color="#E76F51", linestyle="--", label="K seleccionado = 4")
axes[0].set_title("Método del Codo (Inercia / WCSS)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Número de clústeres (K)")
axes[0].set_ylabel("Inercia (WCSS)")
axes[0].set_xticks(valores_k)
axes[0].legend()

axes[1].plot(df_metricas_k["K"], df_metricas_k["Silhouette Score"], marker="s", linewidth=2.5, color="#2A9D8F")
axes[1].axvline(4, color="#E76F51", linestyle="--", label="K seleccionado = 4")
axes[1].set_title("Coeficiente de Silueta (Silhouette Score)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Número de clústeres (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(valores_k)
axes[1].legend()

fig.suptitle("Evaluación del Número Óptimo de Clústeres (K = 2 a 8)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_metodo_codo_silhouette.png", dpi=150)
plt.close()

# 3. Entrenamiento del modelo final (Configuración A: K = 4 con StandardScaler)
K_OPTIMO = 4
kmeans_final = KMeans(n_clusters=K_OPTIMO, init="k-means++", n_init=10, random_state=42)
df["cluster_raw"] = kmeans_final.fit_predict(X_scaled)

# Reordenar etiquetas de clúster (0, 1, 2, 3) por orden lógico de riesgo/severidad clínica
# para garantizar nombres consistentes e intuitivos:
# Ordenamos según perfiles reales:
#   - Menor edad y glucosa normal -> Clúster 0: Adultos Jóvenes con Perfil Saludable
#   - Mayor edad y glucosa normal-alta -> Clúster 1: Adultos Mayores en Control Estable
#   - Mayor IMC e insulina -> Clúster 2: Obesidad con Resistencia a la Insulina
#   - Mayor glucosa -> Clúster 3: Hiperglucemia Severa y Dislipidemia
centroides_orig_raw = pd.DataFrame(
    scaler.inverse_transform(kmeans_final.cluster_centers_),
    columns=FEATURES,
)

id_hiper = int(centroides_orig_raw["glu_suero"].idxmax())
restantes = [i for i in range(K_OPTIMO) if i != id_hiper]
id_obesidad = int(centroides_orig_raw.loc[restantes, "insulina"].idxmax())
restantes = [i for i in restantes if i != id_obesidad]
id_jovenes = int(centroides_orig_raw.loc[restantes, "edad"].idxmin())
id_mayores = [i for i in restantes if i != id_jovenes][0]

mapa_reorden = {
    id_jovenes: 0,
    id_mayores: 1,
    id_obesidad: 2,
    id_hiper: 3,
}
nombres_clusters = {
    0: "Adultos Jóvenes con Perfil Saludable",
    1: "Adultos Mayores en Control Estable",
    2: "Obesidad con Resistencia a la Insulina",
    3: "Hiperglucemia Severa y Dislipidemia",
}

df["cluster"] = df["cluster_raw"].map(mapa_reorden)
df["perfil_cluster"] = df["cluster"].map(nombres_clusters)
df = df.drop(columns=["cluster_raw"])

# 4. Obtener centroides en escala original y en escala estandarizada (ordenados 0..3)
orden_inverso = [id_jovenes, id_mayores, id_obesidad, id_hiper]
centroides_escalados = pd.DataFrame(
    kmeans_final.cluster_centers_[orden_inverso],
    columns=FEATURES,
).round(3)
centroides_escalados.insert(0, "Clúster", [0, 1, 2, 3])
centroides_escalados.insert(1, "Perfil", [nombres_clusters[i] for i in range(4)])

centroides_originales = pd.DataFrame(
    scaler.inverse_transform(kmeans_final.cluster_centers_[orden_inverso]),
    columns=FEATURES,
).round(2)
conteos = df["cluster"].value_counts().sort_index()
porcentajes = (conteos / len(df) * 100).round(2)
centroides_originales.insert(0, "Clúster", [0, 1, 2, 3])
centroides_originales.insert(1, "Perfil", [nombres_clusters[i] for i in range(4)])
centroides_originales["Registros"] = conteos.values
centroides_originales["Porcentaje (%)"] = porcentajes.values

print("\n--- Centroides en Escala Original (Modelo Final K=4) ---")
print(centroides_originales.to_string(index=False))
centroides_originales.to_csv(TABLES_DIR / "04_centroides_originales.csv", index=False)
centroides_escalados.to_csv(TABLES_DIR / "04_centroides_escalados.csv", index=False)

# 5. PCA a 2 componentes para visualizar los agrupamientos en 2D
pca = PCA(n_components=2, random_state=42)
coords_pca = pca.fit_transform(X_scaled)
df["pca_1"] = coords_pca[:, 0].round(4)
df["pca_2"] = coords_pca[:, 1].round(4)
var_exp = pca.explained_variance_ratio_ * 100

centroides_pca = pca.transform(kmeans_final.cluster_centers_[orden_inverso])

# Gráfica 6: Dispersión de clústeres (PCA 2D + Glucosa vs IMC con centroides)
paleta = {0: "#2A9D8F", 1: "#457B9D", 2: "#F4A261", 3: "#E63946"}
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for cid in range(4):
    sub = df[df["cluster"] == cid]
    axes[0].scatter(
        sub["pca_1"],
        sub["pca_2"],
        s=32,
        alpha=0.65,
        color=paleta[cid],
        label=f"C{cid}: {nombres_clusters[cid]} (n={len(sub)})",
    )
axes[0].scatter(
    centroides_pca[:, 0],
    centroides_pca[:, 1],
    marker="*",
    s=320,
    c="black",
    edgecolors="white",
    linewidths=1.2,
    zorder=10,
    label="Centroides",
)
axes[0].set_title(
    f"Agrupamientos K-Means en Plano PCA ({var_exp[0]:.1f}% + {var_exp[1]:.1f}% var.)",
    fontsize=11.5,
    fontweight="bold",
)
axes[0].set_xlabel(f"Componente Principal 1 ({var_exp[0]:.1f}%)")
axes[0].set_ylabel(f"Componente Principal 2 ({var_exp[1]:.1f}%)")
axes[0].legend(fontsize=8.5, loc="upper right")

for cid in range(4):
    sub = df[df["cluster"] == cid]
    axes[1].scatter(
        sub["glu_suero"],
        sub["imc"],
        s=32,
        alpha=0.65,
        color=paleta[cid],
        label=f"C{cid}: {nombres_clusters[cid]}",
    )
axes[1].scatter(
    centroides_originales["glu_suero"],
    centroides_originales["imc"],
    marker="*",
    s=320,
    c="black",
    edgecolors="white",
    linewidths=1.2,
    zorder=10,
    label="Centroides",
)
axes[1].set_title("Agrupamientos K-Means: Glucosa en Suero vs IMC", fontsize=11.5, fontweight="bold")
axes[1].set_xlabel("Glucosa en suero (mg/dL)")
axes[1].set_ylabel("IMC (kg/m²)")
axes[1].legend(fontsize=8.5, loc="upper right")

fig.suptitle("Visualización de Clústeres y Centroides (Modelo Final K=4)", fontsize=13.5, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_clusters_dispersion.png", dpi=150)
plt.close()

# Gráfica 7: Perfil comparativo de centroides (mapa de calor normalizado + distribución por clúster)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})

# Panel izquierdo: Distribución de registros por clúster
etiquetas_cortas = [
    "C0\nJóvenes\nSaludables",
    "C1\nMayores\nEstables",
    "C2\nObesidad /\nInsulina",
    "C3\nHipergluc.\nSevera",
]
bars = axes[0].bar(etiquetas_cortas, conteos.values, color=[paleta[i] for i in range(4)])
axes[0].tick_params(axis="x", labelsize=9)
axes[0].set_title("Distribución de Pacientes por Clúster", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Cantidad de pacientes")
for bar, pct in zip(bars, porcentajes.values):
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, h + 15, f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_ylim(0, max(conteos.values) * 1.18)

# Panel derecho: Heatmap de centroides estandarizados (Z-score) con valores originales anotados
matriz_z = centroides_escalados[FEATURES].copy()
matriz_z.index = [f"C{i}: {nombres_clusters[i]}" for i in range(4)]
matriz_z.columns = [LABELS_ES[c] for c in FEATURES]

anotaciones = centroides_originales[FEATURES].copy()
anotaciones_str = anotaciones.apply(lambda col: col.map(lambda v: f"{v:.1f}"))

sns.heatmap(
    matriz_z,
    annot=anotaciones_str.values,
    fmt="",
    cmap="RdYlGn_r",
    center=0,
    linewidths=0.8,
    cbar_kws={"label": "Desviaciones estándar respecto a la media (Z-score)"},
    ax=axes[1],
)
axes[1].set_title("Perfil de Centroides por Clúster (valores en escala real)", fontsize=12, fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_perfil_centroides_distribucion.png", dpi=150)
plt.close()

# Comparacion de dos configuraciones del modelo (Parte 4)
print("\n--- Comparación de Dos Configuraciones del Modelo ---")
# Configuración A: K=4 con StandardScaler (modelo final)
sil_a = silhouette_score(X_scaled, df["cluster"])
db_a = davies_bouldin_score(X_scaled, df["cluster"])
inercia_a = kmeans_final.inertia_

# Configuración B: K=3 con StandardScaler (más simple, mayor silueta, menor granularidad)
kmeans_b = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
etiquetas_b = kmeans_b.fit_predict(X_scaled)
sil_b = silhouette_score(X_scaled, etiquetas_b)
db_b = davies_bouldin_score(X_scaled, etiquetas_b)
inercia_b = kmeans_b.inertia_

# Configuración C (referencia de control): K=4 SIN StandardScaler (para demostrar el impacto del escalamiento)
kmeans_sin_escala = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
etiquetas_sin_escala = kmeans_sin_escala.fit_predict(df[FEATURES])
# Evaluamos calidad sobre los datos escalados para comparación justa de separación multivariable
sil_sin_escala_en_escalado = silhouette_score(X_scaled, etiquetas_sin_escala)

comparacion_configs = pd.DataFrame([
    {
        "Configuración": "Config. A (Final): K=4 + StandardScaler",
        "K": 4,
        "Escalamiento": "Sí (StandardScaler)",
        "Inercia (WCSS)": round(inercia_a, 2),
        "Silhouette Score": round(sil_a, 4),
        "Davies-Bouldin": round(db_a, 4),
        "Tamaño mín. / máx. clúster": f"{conteos.min()} / {conteos.max()}",
    },
    {
        "Configuración": "Config. B (Alternativa): K=3 + StandardScaler",
        "K": 3,
        "Escalamiento": "Sí (StandardScaler)",
        "Inercia (WCSS)": round(inercia_b, 2),
        "Silhouette Score": round(sil_b, 4),
        "Davies-Bouldin": round(db_b, 4),
        "Tamaño mín. / máx. clúster": f"{pd.Series(etiquetas_b).value_counts().min()} / {pd.Series(etiquetas_b).value_counts().max()}",
    },
    {
        "Configuración": "Control: K=4 SIN escalamiento",
        "K": 4,
        "Escalamiento": "No",
        "Inercia (WCSS)": round(kmeans_sin_escala.inertia_, 2),
        "Silhouette Score": round(sil_sin_escala_en_escalado, 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_scaled, etiquetas_sin_escala), 4),
        "Tamaño mín. / máx. clúster": f"{pd.Series(etiquetas_sin_escala).value_counts().min()} / {pd.Series(etiquetas_sin_escala).value_counts().max()}",
    },
])
print(comparacion_configs.to_string(index=False))
comparacion_configs.to_csv(TABLES_DIR / "05_comparacion_configuraciones.csv", index=False)

# Gráfica 8: Comparación visual Configuración A (K=4) vs Configuración B (K=3) en PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for cid in range(4):
    sub = df[df["cluster"] == cid]
    axes[0].scatter(sub["pca_1"], sub["pca_2"], s=26, alpha=0.65, color=paleta[cid], label=f"C{cid} (n={len(sub)})")
axes[0].set_title(f"Config. A (Final): K=4 + StandardScaler\nInercia={inercia_a:.0f} | Silhouette={sil_a:.3f}", fontsize=11, fontweight="bold")
axes[0].set_xlabel("PCA 1")
axes[0].set_ylabel("PCA 2")
axes[0].legend()

paleta_b = {0: "#2A9D8F", 1: "#F4A261", 2: "#E63946"}
for cid in range(3):
    mask = etiquetas_b == cid
    axes[1].scatter(df.loc[mask, "pca_1"], df.loc[mask, "pca_2"], s=26, alpha=0.65, color=paleta_b[cid], label=f"Grupo {cid} (n={mask.sum()})")
axes[1].set_title(f"Config. B (Alternativa): K=3 + StandardScaler\nInercia={inercia_b:.0f} | Silhouette={sil_b:.3f}", fontsize=11, fontweight="bold")
axes[1].set_xlabel("PCA 1")
axes[1].set_ylabel("PCA 2")
axes[1].legend()

fig.suptitle("Comparación de Configuraciones del Modelo K-Means", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_comparacion_configuraciones.png", dpi=150)
plt.close()

# Parte 5: Estadisticas descriptivas detalladas por cluster
print("\nPARTE 5: ANÁLISIS E INTERPRETACIÓN POR CLÚSTER")

stats_por_cluster = (
    df.groupby(["cluster", "perfil_cluster"])[FEATURES]
    .agg(["mean", "median", "std", "min", "max"])
    .round(2)
)
stats_por_cluster.to_csv(TABLES_DIR / "06_estadisticas_detalladas_clusters.csv")

# Tabla comparativa resumida (medias + conteo + % mujeres + % diabetes por glucosa)
resumen_comp = df.groupby(["cluster", "perfil_cluster"]).agg(
    Pacientes=("edad", "count"),
    Edad_media=("edad", "mean"),
    IMC_medio=("imc", "mean"),
    Glucosa_media=("glu_suero", "mean"),
    Insulina_media=("insulina", "mean"),
    Trigliceridos_medio=("trig", "mean"),
    Pct_Mujeres=("sexo", lambda s: (s == "Mujer").mean() * 100),
    Pct_Glucosa_Diabetes=("categoria_glucosa", lambda s: (s == "Diabetes").mean() * 100),
).round(2).reset_index()
resumen_comp["Pct_Total"] = (resumen_comp["Pacientes"] / len(df) * 100).round(2)
print("\n--- Tabla Comparativa de Clústeres ---")
print(resumen_comp.to_string(index=False))
resumen_comp.to_csv(TABLES_DIR / "07_tabla_comparativa_clusters.csv", index=False)

# Guardar dataset final procesado con los clústeres asignados (requisito Parte 8)
df.to_csv(CLUSTERS_DATA_PATH, index=False)
print(f"\n[OK] Dataset procesado con clústeres guardado en: {CLUSTERS_DATA_PATH}")
print(f"[OK] Todas las tablas guardadas en: {TABLES_DIR}")
print(f"[OK] Todas las gráficas guardadas en: {FIGURES_DIR}")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Segmentación Metabólica K-Means | México",
    page_icon="🩺",
    layout="wide",
)

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CLUSTERS_PATH = BASE_DIR / "data" / "processed" / "diabetes_clusters.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

FEATURES = ["edad", "imc", "glu_suero", "insulina", "trig"]
LABELS_ES = {
    "edad": "Edad (años)",
    "imc": "IMC (kg/m²)",
    "glu_suero": "Glucosa en suero (mg/dL)",
    "insulina": "Insulina (µU/mL)",
    "trig": "Triglicéridos (mg/dL)",
}
PALETA = {
    0: "#2A9D8F",
    1: "#457B9D",
    2: "#F4A261",
    3: "#E63946",
}
NOMBRES_CLUSTERS = {
    0: "Adultos Jóvenes con Perfil Saludable",
    1: "Adultos Mayores en Control Estable",
    2: "Obesidad con Resistencia a la Insulina",
    3: "Hiperglucemia Severa y Dislipidemia",
}
DESCRIPCIONES_CLUSTERS = {
    0: (
        "Grupo más numeroso y joven (~30 años en promedio). Presenta el menor IMC promedio (~27.2), "
        "glucosa normal (~84 mg/dL), insulina en rango saludable (~8.3) y triglicéridos bajos (~118 mg/dL). "
        "Representa la población base de menor riesgo metabólico actual."
    ),
    1: (
        "Adultos de mayor edad (~50 años en promedio) que mantienen glucosa en rango normal-alto (~95 mg/dL), "
        "insulina controlada (~8.8) y triglicéridos cerca del límite superior normal (~147 mg/dL). "
        "Grupo prioritario para monitoreo preventivo por factor etario."
    ),
    2: (
        "Perfil clásico de resistencia a la insulina y síndrome metabólico (~39 años): exhibe el IMC más elevado "
        "(~36.3, obesidad grado II), hiperinsulinemia marcada (~28.5 µU/mL), glucosa en zona de prediabetes "
        "(~104 mg/dL) y triglicéridos elevados (~210 mg/dL)."
    ),
    3: (
        "Grupo de mayor severidad clínica (~48 años): presenta hiperglucemia severa en rango diabético "
        "(~259 mg/dL) y los triglicéridos más altos (~250 mg/dL), acompañados de insulina moderada (~11.4), "
        "indicativo de descontrol glucémico avanzado."
    ),
}


@st.cache_data
def cargar_datos():
    df_c = pd.read_csv(DATA_CLUSTERS_PATH)
    df_eval = pd.read_csv(TABLES_DIR / "03_evaluacion_k.csv")
    df_cent = pd.read_csv(TABLES_DIR / "04_centroides_originales.csv")
    df_comp = pd.read_csv(TABLES_DIR / "05_comparacion_configuraciones.csv")
    return df_c, df_eval, df_cent, df_comp


df, df_eval_k, df_centroides, df_comparacion = cargar_datos()


st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Selecciona una página:",
    ["Página 1: Análisis General", "Página 2: Segmentación K-Means"],
)

st.sidebar.divider()
st.sidebar.header("Filtros Interactivos")

# Filtro 1: Sexo
sexos_disp = sorted(df["sexo"].dropna().unique().tolist())
sexos_sel = st.sidebar.multiselect("1. Sexo", sexos_disp, default=sexos_disp)

# Filtro 2: Rango de edad
orden_edades = ["20-29", "30-44", "45-59", "60+"]
edades_disp = [e for e in orden_edades if e in df["rango_edad"].dropna().unique()]
edades_sel = st.sidebar.multiselect("2. Rango de edad", edades_disp, default=edades_disp)

# Filtro 3: Perfil de clúster
perfiles_disp = [NOMBRES_CLUSTERS[i] for i in range(4)]
perfiles_sel = st.sidebar.multiselect("3. Perfil de clúster (K-Means)", perfiles_disp, default=perfiles_disp)

# Filtro adicional opcional: Ciudad
ciudades_disp = sorted(df["Ciudad"].dropna().unique().tolist())
ciudades_sel = st.sidebar.multiselect("Ciudad / Entidad (opcional)", ciudades_disp)

# Aplicar filtros
datos = df[
    df["sexo"].isin(sexos_sel)
    & df["rango_edad"].isin(edades_sel)
    & df["perfil_cluster"].isin(perfiles_sel)
].copy()
if ciudades_sel:
    datos = datos[datos["Ciudad"].isin(ciudades_sel)]

if datos.empty:
    st.warning("No hay registros que coincidan con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

# Encabezado común
st.title("Segmentación de Perfiles Metabólicos en México mediante K-Means")


if pagina == "Página 1: Análisis General":
    st.header("Página 1: Análisis General del Dataset")

    # 4 Indicadores KPI relevantes
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pacientes analizados", f"{len(datos):,}")
    k2.metric("Glucosa promedio", f"{datos['glu_suero'].mean():.1f} mg/dL")
    k3.metric("IMC promedio", f"{datos['imc'].mean():.1f} kg/m²")
    k4.metric("Triglicéridos promedio", f"{datos['trig'].mean():.1f} mg/dL")

    st.divider()

    # Visualizaciones generales del dataset
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribución por Categoría de Glucosa")
        orden_glu = ["Normal", "Prediabetes", "Diabetes"]
        conteo_glu = datos["categoria_glucosa"].value_counts().reindex(orden_glu, fill_value=0)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        colores_glu = ["#2A9D8F", "#F4A261", "#E63946"]
        bars = ax.bar(conteo_glu.index, conteo_glu.values, color=colores_glu)
        for bar in bars:
            h = bar.get_height()
            pct = h / len(datos) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(conteo_glu.values) * 0.02, f"{int(h)} ({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Número de pacientes")
        ax.set_ylim(0, max(conteo_glu.values) * 1.18 if max(conteo_glu.values) > 0 else 10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Distribución de Variable Clínica Seleccionada")
        var_hist = st.selectbox(
            "Elige una variable para inspeccionar su distribución:",
            FEATURES,
            format_func=lambda v: LABELS_ES[v],
        )
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.histplot(datos[var_hist], kde=True, bins=25, color="#457B9D", ax=ax)
        ax.axvline(datos[var_hist].mean(), color="#E63946", linestyle="--", label=f"Media: {datos[var_hist].mean():.1f}")
        ax.axvline(datos[var_hist].median(), color="#2A9D8F", linestyle=":", label=f"Mediana: {datos[var_hist].median():.1f}")
        ax.set_xlabel(LABELS_ES[var_hist])
        ax.set_ylabel("Frecuencia")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Glucosa Promedio por Rango de Edad y Sexo")
        pivot_edad = datos.pivot_table(
            index="rango_edad", columns="sexo", values="glu_suero", aggfunc="mean", observed=False
        ).reindex(orden_edades)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        pivot_edad.plot(kind="bar", ax=ax, color=["#457B9D", "#E76F51"])
        ax.set_ylabel("Glucosa promedio (mg/dL)")
        ax.set_xlabel("Rango de edad")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Sexo")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        st.subheader("Matriz de Correlación (Variables del Modelo)")
        corr = datos[FEATURES].rename(columns=LABELS_ES).corr()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Estadísticas Descriptivas Generales (Datos Filtrados)")
    desc_general = datos[FEATURES].rename(columns=LABELS_ES).describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
    desc_general.columns = ["Conteo", "Media", "Desv. Est.", "Mínimo", "Mediana", "Máximo"]
    st.dataframe(desc_general, use_container_width=True)

    with st.expander("Vista previa de registros filtrados (primeros 25)"):
        cols_vista = ["sexo", "edad", "rango_edad", "Ciudad", "imc", "glu_suero", "insulina", "trig", "categoria_glucosa", "perfil_cluster"]
        st.dataframe(datos[cols_vista].head(25), use_container_width=True)


else:
    st.header("Página 2: Segmentación mediante K-Means (K = 4)")

    # 4 Indicadores KPI relevantes de segmentación
    sil_k4 = df_eval_k.loc[df_eval_k["K"] == 4, "Silhouette Score"].values[0]
    pct_riesgo_metab = datos["cluster"].isin([2, 3]).mean() * 100
    pct_hiper = (datos["cluster"] == 3).mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pacientes segmentados", f"{len(datos):,}")
    m2.metric("Clústeres óptimos (K)", "4 perfiles")
    m3.metric("Silhouette Score (modelo)", f"{sil_k4:.4f}")
    m4.metric("Perfiles de riesgo (C2 + C3)", f"{pct_riesgo_metab:.1f} %", delta=f"{pct_hiper:.1f}% severo (C3)", delta_color="inverse")

    st.divider()

    # Fila 1: Distribución de registros por clúster + Dispersión de agrupamientos
    g1, g2 = st.columns([1, 1.4])

    with g1:
        st.subheader("1. Distribución de Registros por Clúster")
        conteo_c = datos["cluster"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        etiquetas_x = [f"C{i}" for i in range(4)]
        bars = ax.bar(etiquetas_x, conteo_c.values, color=[PALETA[i] for i in range(4)])
        for bar in bars:
            h = bar.get_height()
            pct = (h / len(datos) * 100) if len(datos) > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(conteo_c.values) * 0.02, f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Número de pacientes")
        ax.set_xlabel("Clúster")
        ax.set_ylim(0, max(conteo_c.values) * 1.22 if max(conteo_c.values) > 0 else 10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with g2:
        st.subheader("2. Gráfica de Dispersión de Agrupamientos")
        modo_disp = st.radio(
            "Ejes de visualización:",
            ["Glucosa vs IMC", "Plano PCA (2D)", "Insulina vs Triglicéridos", "Edad vs Glucosa"],
            horizontal=True,
        )
        pares = {
            "Glucosa vs IMC": ("glu_suero", "imc", "Glucosa en suero (mg/dL)", "IMC (kg/m²)"),
            "Plano PCA (2D)": ("pca_1", "pca_2", "Componente Principal 1", "Componente Principal 2"),
            "Insulina vs Triglicéridos": ("insulina", "trig", "Insulina (µU/mL)", "Triglicéridos (mg/dL)"),
            "Edad vs Glucosa": ("edad", "glu_suero", "Edad (años)", "Glucosa en suero (mg/dL)"),
        }
        col_x, col_y, lbl_x, lbl_y = pares[modo_disp]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for cid in range(4):
            sub = datos[datos["cluster"] == cid]
            if not sub.empty:
                ax.scatter(
                    sub[col_x],
                    sub[col_y],
                    color=PALETA[cid],
                    alpha=0.65,
                    s=30,
                    label=f"C{cid}: {NOMBRES_CLUSTERS[cid]}",
                )
        # Dibujar centroides si los ejes están en la tabla de centroides o calcular medias PCA
        if col_x in FEATURES and col_y in FEATURES:
            cents_plot = df_centroides[df_centroides["Clúster"].isin(datos["cluster"].unique())]
            ax.scatter(
                cents_plot[col_x],
                cents_plot[col_y],
                marker="*",
                s=260,
                c="black",
                edgecolors="white",
                linewidths=1.2,
                zorder=10,
                label="Centroides",
            )
        else:
            cents_pca = df.groupby("cluster")[["pca_1", "pca_2"]].mean().loc[sorted(datos["cluster"].unique())]
            ax.scatter(
                cents_pca["pca_1"],
                cents_pca["pca_2"],
                marker="*",
                s=260,
                c="black",
                edgecolors="white",
                linewidths=1.2,
                zorder=10,
                label="Centroides",
            )
        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Fila 2: Dos gráficas adicionales para comparar características de los grupos
    st.subheader("Comparación de Características entre Clústeres")
    comp1, comp2 = st.columns(2)

    with comp1:
        st.markdown("**Gráfica adicional 1: Perfil de Centroides (Z-score con medias reales)**")
        medias_c = datos.groupby("cluster")[FEATURES].mean()
        medias_z = (medias_c - df[FEATURES].mean()) / df[FEATURES].std()
        medias_z.index = [f"C{i}" for i in medias_z.index]
        medias_z.columns = [LABELS_ES[c].split(" (")[0] for c in FEATURES]
        anot = medias_c.apply(lambda col: col.map(lambda v: f"{v:.1f}"))

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        sns.heatmap(
            medias_z,
            annot=anot.values,
            fmt="",
            cmap="RdYlGn_r",
            center=0,
            linewidths=0.7,
            cbar_kws={"label": "Z-score"},
            ax=ax,
        )
        ax.set_ylabel("Clúster")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with comp2:
        st.markdown("**Gráfica adicional 2: Comparación de Distribución por Variable (Boxplot)**")
        var_box = st.selectbox(
            "Variable a comparar entre clústeres:",
            FEATURES,
            index=2,
            format_func=lambda v: LABELS_ES[v],
        )
        fig, ax = plt.subplots(figsize=(6.8, 3.85))
        orden_c = sorted(datos["cluster"].unique())
        sns.boxplot(
            data=datos,
            x="cluster",
            y=var_box,
            order=orden_c,
            hue="cluster",
            palette={c: PALETA[c] for c in orden_c},
            legend=False,
            ax=ax,
        )
        ax.set_xticklabels([f"C{c}" for c in orden_c])
        ax.set_xlabel("Clúster")
        ax.set_ylabel(LABELS_ES[var_box])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Tabla con estadísticas descriptivas de los clústeres
    st.subheader("Tabla de Estadísticas Descriptivas por Clúster")
    tabla_stats = datos.groupby(["cluster", "perfil_cluster"]).agg(
        Registros=("edad", "count"),
        Edad_media=("edad", "mean"),
        IMC_medio=("imc", "mean"),
        Glucosa_media=("glu_suero", "mean"),
        Glucosa_mediana=("glu_suero", "median"),
        Insulina_media=("insulina", "mean"),
        Trig_medio=("trig", "mean"),
        Pct_Diabetes_Glu=("categoria_glucosa", lambda s: (s == "Diabetes").mean() * 100),
    ).round(2).reset_index()
    tabla_stats.columns = [
        "Clúster", "Perfil Descriptivo", "Registros", "Edad Media",
        "IMC Medio", "Glucosa Media", "Glucosa Mediana", "Insulina Media",
        "Triglicéridos Medio", "% Glucosa >= 126",
    ]
    st.dataframe(tabla_stats, use_container_width=True, hide_index=True)

    with st.expander("Ver evaluación de K=2 a 8 y comparación de configuraciones del modelo"):
        c_ev1, c_ev2 = st.columns(2)
        with c_ev1:
            st.markdown("**Evaluación de K (Método del Codo y Silhouette)**")
            st.dataframe(df_eval_k, use_container_width=True, hide_index=True)
        with c_ev2:
            st.markdown("**Comparación de Configuraciones (K=4 vs K=3 vs Sin escalar)**")
            st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

    st.divider()

    # Sección: Descripción de los grupos identificados
    st.subheader("Descripción de los Grupos Identificados")
    d_cols = st.columns(2)
    for idx in range(4):
        with d_cols[idx % 2]:
            row_c = df_centroides[df_centroides["Clúster"] == idx].iloc[0]
            st.markdown(
                f"""
                <div style="border-left: 5px solid {PALETA[idx]}; padding: 10px 14px; margin-bottom: 12px; background-color: rgba(128,128,128,0.08); border-radius: 6px;">
                    <h4 style="margin:0 0 6px 0;">Clúster {idx}: {NOMBRES_CLUSTERS[idx]}</h4>
                    <p style="margin:0 0 6px 0; font-size:0.9rem;">
                        <strong>{int(row_c['Registros'])} pacientes ({row_c['Porcentaje (%)']:.1f}%)</strong> |
                        Edad: {row_c['edad']:.1f} | IMC: {row_c['imc']:.1f} |
                        Glucosa: {row_c['glu_suero']:.1f} | Insulina: {row_c['insulina']:.1f} | Trig: {row_c['trig']:.1f}
                    </p>
                    <p style="margin:0; font-size:0.92rem;">{DESCRIPCIONES_CLUSTERS[idx]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Sección: Principales hallazgos y recomendaciones
    st.subheader("Principales Hallazgos y Recomendaciones")
    h_col, r_col = st.columns(2)

    with h_col:
        st.markdown("### Hallazgos Relevantes")
        st.markdown(
            """
            1. **Diferenciación entre resistencia a la insulina y diabetes avanzada:** K-Means separó de forma natural al **Clúster 2** (obesidad grado II con IMC 36.3 e hiperinsulinemia de 28.5, pero glucosa en 104 mg/dL) del **Clúster 3** (hiperglucemia severa de 258.7 mg/dL con insulina en 11.4), reflejando dos fases clínicas distintas del deterioro metabólico.
            2. **El 23.7% de la población analizada ya se encuentra en perfiles de riesgo metabólico (C2 y C3):** 333 personas (18.2%) muestran síndrome metabólico con resistencia a la insulina y 100 personas (5.5%) presentan descontrol glucémico severo.
            3. **La edad segmenta claramente a la población aparentemente estable:** el Clúster 0 (~30 años, glucosa 84.4) y el Clúster 1 (~50 años, glucosa 94.6 y triglicéridos 146.8) muestran cómo los biomarcadores se acercan progresivamente a los umbrales de riesgo conforme avanza la edad.
            4. **Impacto crítico del escalamiento:** sin `StandardScaler`, la variable `trig` (desviación de ~137) domina las distancias euclidianas y oculta diferencias de `imc` e `insulina`.
            """
        )

    with r_col:
        st.markdown("### Recomendaciones Basadas en los Resultados")
        st.markdown(
            """
            1. **Intervención clínica prioritaria para el Clúster 3 (5.5%):** canalizar a atención médica inmediata, confirmación diagnóstica de Hb1ac y control farmacológico de glucosa y triglicéridos para prevenir complicaciones cardiovasculares y renales.
            2. **Programas intensivos de reducción de peso para el Clúster 2 (18.2%):** implementar seguimiento nutricional y actividad física enfocada en revertir la resistencia a la insulina antes de que progrese a diabetes franca.
            3. **Tamizaje periódico anual para el Clúster 1 (36.9%):** establecer chequeos anuales de glucosa y perfil lipídico en adultos mayores de 45–50 años cuyos valores aún están en rango normal-alto.
            4. **Prevención primaria en el Clúster 0 (39.4%):** mantener campañas educativas de alimentación y ejercicio en adultos jóvenes para evitar el incremento gradual del IMC (ya en rango de sobrepeso ligero con 27.2).
            """
        )

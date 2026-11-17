def mostrar_grafico_pca():
    st.subheader("Distribución interactiva de los clústeres (PCA)")

    ruta_pca = BASE / "resultados" / "observaciones_clusters_pca.csv"

    if ruta_pca.exists():
        datos_pca = pd.read_csv(ruta_pca)
        datos_pca["Cluster"] = datos_pca["Cluster"].astype(str)

        figura_pca = px.scatter(
            datos_pca,
            x="Componente_1",
            y="Componente_2",
            color="Cluster",
            hover_data=["Demanda"],
            opacity=0.55,
            title="Observaciones agrupadas mediante K-Means"
        )

        figura_pca.update_traces(marker=dict(size=4))
        st.plotly_chart(figura_pca, use_container_width=True)

        st.caption(
            "12.165 observaciones originales. "
            "Varianza explicada por PCA: 57,18 %."
        )
    else:
        st.warning("No se encontró el archivo PCA.")


from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

st.set_page_config(
    page_title="Inteligencia de Demanda de Bicicletas",
    page_icon="🚲",
    layout="wide"
)

BASE = Path(__file__).resolve().parent.parent

@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / "data" / "hour_original_uci.csv")
    df["fecha"] = pd.to_datetime(df["dteday"])
    return df

@st.cache_resource
def cargar_modelo():
    return joblib.load(
        BASE / "modelos" / "random_forest_optimizado.joblib"
    )

@st.cache_data
def cargar_resultados():
    ruta = BASE / "resultados"
    distribucion = pd.read_csv(
        ruta / "distribucion_clusters.csv"
    )
    demanda = pd.read_csv(
        ruta / "demanda_por_cluster.csv"
    )
    centroides = pd.read_csv(
        ruta / "centroides_kmeans.csv"
    )
    metricas = pd.read_csv(
        ruta / "comparacion_regresion.csv"
    )
    return distribucion, demanda, centroides, metricas

datos = cargar_datos()
distribucion, demanda_cluster, centroides, metricas = cargar_resultados()

st.title("🚲 Centro de Inteligencia de Demanda de Bicicletas")
st.caption(
    "Universidad de La Salle | Maestría en Inteligencia Artificial | "
    "Proyecto Integrador — Fase 4"
)

st.sidebar.header("Navegación")
modulo = st.sidebar.radio(
    "Selecciona un módulo",
    [
        "Exploración interactiva",
        "Indicadores KPI",
        "Predicción en vivo",
        "Segmentación y clustering"
    ]
)

# ============================================================
# MODULO 1 - EXPLORACION
# ============================================================

if modulo == "Exploración interactiva":

    st.header("1. Exploración interactiva de datos")

    minimo = datos["fecha"].min().date()
    maximo = datos["fecha"].max().date()

    fechas = st.date_input(
        "Selecciona el rango de fechas",
        value=(minimo, maximo),
        min_value=minimo,
        max_value=maximo
    )

    estaciones = st.multiselect(
        "Estaciones",
        options=sorted(datos["season"].unique()),
        default=sorted(datos["season"].unique())
    )

    climas = st.multiselect(
        "Condiciones meteorológicas",
        options=sorted(datos["weathersit"].unique()),
        default=sorted(datos["weathersit"].unique())
    )

    demanda_min, demanda_max = int(datos["cnt"].min()), int(datos["cnt"].max())

    rango_demanda = st.slider(
        "Rango de demanda horaria",
        min_value=demanda_min,
        max_value=demanda_max,
        value=(demanda_min, demanda_max)
    )

    filtrados = datos[
        datos["season"].isin(estaciones)
        & datos["weathersit"].isin(climas)
        & datos["cnt"].between(*rango_demanda)
    ].copy()

    if len(fechas) == 2:
        filtrados = filtrados[
            filtrados["fecha"].dt.date.between(
                fechas[0], fechas[1]
            )
        ]

    if filtrados.empty:
        st.warning("No hay registros para los filtros seleccionados.")
    else:
        a, b, c = st.columns(3)
        a.metric("Registros", f"{len(filtrados):,}")
        b.metric("Demanda promedio", f"{filtrados['cnt'].mean():,.1f}")
        c.metric("Demanda total", f"{filtrados['cnt'].sum():,}")

        serie = filtrados.groupby("fecha", as_index=False)["cnt"].sum()

        st.plotly_chart(
            px.line(
                serie,
                x="fecha",
                y="cnt",
                title="Evolución diaria de la demanda",
                labels={"cnt": "Alquileres", "fecha": "Fecha"}
            ),
            use_container_width=True
        )

        horas = filtrados.groupby("hr", as_index=False)["cnt"].mean()

        st.plotly_chart(
            px.bar(
                horas,
                x="hr",
                y="cnt",
                title="Demanda promedio por hora",
                labels={"hr": "Hora", "cnt": "Demanda promedio"}
            ),
            use_container_width=True
        )

        hora_pico = int(horas.loc[horas["cnt"].idxmax(), "hr"])

        st.info(
            f"Interpretación: en el conjunto filtrado, la hora "
            f"con mayor demanda promedio es las {hora_pico:02d}:00."
        )

        st.dataframe(filtrados.head(100), use_container_width=True)

# ============================================================
# MODULO 2 - KPI
# ============================================================

elif modulo == "Indicadores KPI":

    st.header("2. Indicadores clave de rendimiento")

    a, b, c, d = st.columns(4)

    a.metric("Observaciones", f"{len(datos):,}")
    b.metric("Demanda promedio", f"{datos['cnt'].mean():,.2f}")
    c.metric("Demanda máxima", f"{datos['cnt'].max():,}")
    d.metric("Total de alquileres", f"{datos['cnt'].sum():,}")

    st.subheader("Rendimiento de modelos de regresión")
    st.dataframe(metricas, use_container_width=True)

    st.subheader("Patrones temporales")

    por_mes = datos.groupby("mnth", as_index=False)["cnt"].mean()

    st.plotly_chart(
        px.bar(
            por_mes,
            x="mnth",
            y="cnt",
            title="Demanda horaria promedio por mes"
        ),
        use_container_width=True
    )

    st.info(
        "Los indicadores describen el comportamiento histórico "
        "del conjunto de datos. La tabla presenta los resultados "
        "de evaluación obtenidos durante la Actividad 6."
    )

# ============================================================
# MODULO 3 - PREDICCION EN VIVO
# ============================================================

elif modulo == "Predicción en vivo":

    st.header("3. Predicción de demanda en vivo")
    st.write(
        "Introduce las condiciones de una hora para obtener "
        "una estimación mediante el Random Forest optimizado."
    )

    with st.form("formulario_prediccion"):

        c1, c2, c3 = st.columns(3)

        with c1:
            mes = st.selectbox("Mes", list(range(1, 13)), index=5)
            dia = st.slider("Día del mes", 1, 31, 15)
            hora = st.slider("Hora", 0, 23, 8)
            estacion = st.selectbox(
                "Estación (1 invierno, 2 primavera, 3 verano, 4 otoño)",
                [1, 2, 3, 4],
                index=2
            )

        with c2:
            dia_semana = st.selectbox(
                "Día de la semana (0 domingo, 6 sábado)",
                list(range(7)),
                index=1
            )
            clima = st.selectbox(
                "Condición meteorológica (1 a 4)",
                [1, 2, 3, 4]
            )
            festivo = st.checkbox("Día festivo")
            anio = st.selectbox(
                "Año del conjunto histórico",
                [2011, 2012],
                index=1
            )

        with c3:
            temperatura = st.slider(
                "Temperatura normalizada", 0.0, 1.0, 0.5, 0.01
            )
            sensacion = st.slider(
                "Sensación térmica normalizada", 0.0, 1.0, 0.5, 0.01
            )
            humedad = st.slider(
                "Humedad normalizada", 0.0, 1.0, 0.6, 0.01
            )
            viento = st.slider(
                "Velocidad del viento normalizada", 0.0, 1.0, 0.2, 0.01
            )

        ejecutar = st.form_submit_button(
            "Calcular predicción",
            type="primary"
        )

    if ejecutar:

        # Variables derivadas según el esquema del modelo
        franja = (
            "Madrugada" if hora < 6 else
            "Manana" if hora < 12 else
            "Tarde" if hora < 18 else
            "Noche"
        )

        trimestre = (mes - 1) // 3 + 1
        fin_semana = int(dia_semana in [0, 6])
        laboral = int(not festivo and not fin_semana)

        entrada = pd.DataFrame([{
            "season": estacion,
            "mnth": mes,
            "hr": hora,
            "weekday": dia_semana,
            "weathersit": clima,
            "franja_horaria": franja,
            "trimestre": trimestre,
            "temp": temperatura,
            "atemp": sensacion,
            "hum": humedad,
            "windspeed": viento,
            "hora_seno": np.sin(2 * np.pi * hora / 24),
            "hora_coseno": np.cos(2 * np.pi * hora / 24),
            "dia_mes": dia,
            "yr": anio - 2011,
            "holiday": int(festivo),
            "workingday": laboral,
            "fin_semana": fin_semana
        }])

        modelo = cargar_modelo()

        columnas = list(modelo.feature_names_in_)
        entrada = entrada[columnas]

        try:
            prediccion = float(modelo.predict(entrada)[0])
            prediccion = max(0, prediccion)

            st.success("Predicción calculada correctamente")
            st.metric(
                "Demanda horaria estimada",
                f"{prediccion:,.0f} bicicletas"
            )

            st.caption(
                "Estimación basada en patrones históricos de 2011–2012. "
                "No representa una predicción de demanda actual."
            )

        except Exception as error:
            st.error(f"No fue posible calcular la predicción: {error}")

# ============================================================
# MODULO 4 - CLUSTERING
# ============================================================

elif modulo == "Segmentación y clustering":

    st.header("4. Segmentación mediante K-Means")

    st.write(
        "Análisis de los cuatro segmentos identificados "
        "en la Actividad 5."
    )

    st.plotly_chart(
        px.bar(
            demanda_cluster,
            x="Cluster",
            y="Registros",
            color="Cluster",
            title="Distribución de observaciones por clúster"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            demanda_cluster,
            x="Demanda_promedio",
            y="Desviacion_estandar",
            size="Registros",
            color="Cluster",
            hover_data=[
                "Demanda_mediana",
                "Demanda_minima",
                "Demanda_maxima"
            ],
            title="Perfil de los segmentos: demanda y variabilidad",
            labels={
                "Demanda_promedio": "Demanda promedio",
                "Desviacion_estandar": "Desviación estándar"
            }
        ),
        use_container_width=True
    )

    st.subheader("Perfiles numéricos promedio")
    st.dataframe(
        demanda_cluster.drop(
            columns=["Unnamed: 0"],
            errors="ignore"
        ),
        use_container_width=True
    )

    st.subheader("Centroides en el espacio transformado")
    st.dataframe(
        centroides,
        use_container_width=True
    )

    mayor = demanda_cluster.loc[
        demanda_cluster["Demanda_promedio"].idxmax()
    ]
    menor = demanda_cluster.loc[
        demanda_cluster["Demanda_promedio"].idxmin()
    ]

    st.info(
        f"El clúster {int(mayor['Cluster'])} presenta la mayor "
        f"demanda promedio ({mayor['Demanda_promedio']:.1f}), "
        f"mientras que el clúster {int(menor['Cluster'])} "
        f"presenta la menor ({menor['Demanda_promedio']:.1f})."
    )

    st.caption(
        "Los resultados de clustering corresponden a "
        "12.165 observaciones de la Actividad 5, "
        "no a la totalidad de los 17.379 registros."
    )
    mostrar_grafico_pca()

st.divider()
st.caption(
    "Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial "
    "| Universidad de La Salle | Fase 4 — Actividad 7"
)


# GRAFICO PCA INTERACTIVO - ACTIVIDAD 7

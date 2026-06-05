"""
dashboard_evaluacion.py — Dashboard Streamlit para métricas DeepEval
Muestra resultados del reporte_evaluacion.json y permite correr la evaluación
"""

import json
import os
import subprocess
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard de Evaluación — Asistente Mecatrónica",
    page_icon="🤖",
    layout="wide",
)

REPORTE_PATH = Path("reporte_evaluacion.json")

METRIC_LABELS = {
    "AnswerRelevancyMetric":    "Answer Relevancy",
    "FaithfulnessMetric":       "Faithfulness",
    "ContextualPrecisionMetric":"Contextual Precision",
    "ContextualRecallMetric":   "Contextual Recall",
}

METRIC_COLORS = {
    "Answer Relevancy":    "#00b4d8",
    "Faithfulness":        "#ff6b6b",
    "Contextual Precision":"#06d6a0",
    "Contextual Recall":   "#ffd166",
}

UMBRAL = 0.5


# ─────────────────────────────────────────────
#  FUNCIONES
# ─────────────────────────────────────────────
def cargar_reporte():
    if not REPORTE_PATH.exists():
        return None
    with open(REPORTE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extraer_promedios(reporte):
    promedios = {}
    for key, val in reporte.items():
        if key.startswith("promedio_"):
            nombre_raw = key.replace("promedio_", "")
            nombre = METRIC_LABELS.get(nombre_raw, nombre_raw)
            promedios[nombre] = val
    return promedios


def extraer_por_caso(reporte):
    filas = []
    for caso in reporte.get("resultados", []):
        fila = {"Pregunta": caso["input"][:60] + "..."}
        for nombre_raw, datos in caso.get("metricas", {}).items():
            nombre = METRIC_LABELS.get(nombre_raw, nombre_raw)
            fila[nombre] = datos.get("score", None)
        filas.append(fila)
    return pd.DataFrame(filas)


def gauge_chart(nombre, score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": nombre, "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "#1e1e2e",
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 50],  "color": "#2a1a1a"},
                {"range": [50, 100], "color": "#1a2a1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": UMBRAL * 100,
            },
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0e1117",
        font={"color": "white"},
    )
    return fig


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#00ff88; font-family:monospace;'>
        🤖 Dashboard de Evaluación
    </h1>
    <p style='text-align:center; color:#888; font-family:monospace;'>
        Sistema Asistente Inteligente — Mecatrónica/Electrónica
    </p>
    <hr style='border-color:#00ff8833;'>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BOTÓN CORRER EVALUACIÓN
# ─────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    correr = st.button("▶ Correr Evaluación", type="primary", use_container_width=True)

if correr:
    with st.status("⏳ Ejecutando evaluación con Ollama...", expanded=True) as status:
        st.write("Iniciando pipeline multi-agente...")
        st.write("Esto puede tardar 15-30 minutos. No cierres esta ventana.")

        try:
            proceso = subprocess.Popen(
                ["python", "evaluate.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            log_area = st.empty()
            logs = []

            for linea in proceso.stdout:
                linea = linea.rstrip()
                if linea:
                    logs.append(linea)
                    # Mostrar últimas 20 líneas
                    log_area.code("\n".join(logs[-20:]), language="bash")

            proceso.wait()

            if proceso.returncode == 0:
                status.update(label="✅ Evaluación completada", state="complete")
                st.success("Reporte generado correctamente.")
                st.rerun()
            else:
                status.update(label="❌ Error en la evaluación", state="error")
                st.error("Hubo un error. Revisa los logs arriba.")

        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(f"No se pudo ejecutar evaluate.py: {e}")

st.divider()

# ─────────────────────────────────────────────
#  CARGAR REPORTE
# ─────────────────────────────────────────────
reporte = cargar_reporte()

if reporte is None:
    st.info("📂 No hay reporte disponible. Corre la evaluación con el botón de arriba.")
    st.stop()

# ─────────────────────────────────────────────
#  METADATA
# ─────────────────────────────────────────────
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("📅 Última evaluación", reporte.get("timestamp", "—"))
with col_m2:
    st.metric("🧪 Casos evaluados", reporte.get("total_casos", 0))
with col_m3:
    promedios = extraer_promedios(reporte)
    aprobadas = sum(1 for v in promedios.values() if v >= UMBRAL)
    st.metric("✅ Métricas aprobadas", f"{aprobadas}/{len(promedios)}")

st.divider()

# ─────────────────────────────────────────────
#  GAUGES — PROMEDIOS
# ─────────────────────────────────────────────
st.subheader("📊 Promedios por Métrica")

if promedios:
    cols = st.columns(len(promedios))
    for i, (nombre, score) in enumerate(promedios.items()):
        color = METRIC_COLORS.get(nombre, "#aaaaaa")
        with cols[i]:
            estado = "✅" if score >= UMBRAL else "❌"
            st.plotly_chart(
                gauge_chart(f"{estado} {nombre}", score, color),
                use_container_width=True,
            )
else:
    st.warning("No se encontraron promedios en el reporte.")

st.divider()

# ─────────────────────────────────────────────
#  TABLA POR CASO
# ─────────────────────────────────────────────
st.subheader("🔍 Resultados por Caso de Prueba")

df = extraer_por_caso(reporte)

if not df.empty:
    # Heatmap de scores
    metric_cols = [c for c in df.columns if c != "Pregunta"]

    if metric_cols:
        fig_heat = px.imshow(
            df[metric_cols].T,
            x=df["Pregunta"],
            y=metric_cols,
            color_continuous_scale=["#ff4444", "#ffaa00", "#00ff88"],
            zmin=0, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_heat.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font={"color": "white"},
            height=300,
            margin=dict(t=20, b=20),
            coloraxis_colorbar=dict(title="Score"),
        )
        fig_heat.update_xaxes(tickangle=-20, tickfont=dict(size=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Tabla detallada
    with st.expander("Ver tabla completa"):
        def color_score(val):
            if val is None:
                return "color: gray"
            if val >= 0.7:
                return "color: #00ff88; font-weight: bold"
            elif val >= 0.5:
                return "color: #ffd166"
            else:
                return "color: #ff6b6b; font-weight: bold"

        styled = df.style.map(color_score, subset=metric_cols)
        st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────
#  DETALLE POR CASO
# ─────────────────────────────────────────────
st.subheader("📋 Detalle por Caso")

casos = reporte.get("resultados", [])
if casos:
    opciones = [f"Caso {i+1}: {c['input'][:50]}..." for i, c in enumerate(casos)]
    seleccion = st.selectbox("Selecciona un caso:", opciones)
    idx = opciones.index(seleccion)
    caso = casos[idx]

    st.markdown(f"**Pregunta:** {caso['input']}")

    for nombre_raw, datos in caso.get("metricas", {}).items():
        nombre = METRIC_LABELS.get(nombre_raw, nombre_raw)
        score = datos.get("score")
        reason = datos.get("reason", "Sin razón disponible")
        passed = datos.get("passed", False)
        error = datos.get("error")

        if error:
            st.error(f"**{nombre}** — Error: {error}")
        else:
            color = "green" if passed else "red"
            icono = "✅" if passed else "❌"
            st.markdown(f"""
            <div style='border:1px solid #333; border-radius:8px; padding:12px; margin:8px 0; background:#111;'>
                <b style='color:{color};'>{icono} {nombre}</b>
                <span style='float:right; font-size:1.2em; color:{color};'><b>{score:.1%}</b></span>
                <br><small style='color:#aaa;'>{reason}</small>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
    <p style='text-align:center; color:#444; font-size:0.8em; font-family:monospace;'>
        Juez: llama3.2:3b (Ollama local) · DeepEval 4.0.5 · Umbral: 50%
    </p>
""", unsafe_allow_html=True)

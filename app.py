import streamlit as st
from main import (
    AgenteOrquestador,
    load_knowledge_base,
    OpikTracker,
)

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MecatrIA — Asistente Inteligente",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: #050810;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #070c18 !important;
    border-right: 1px solid rgba(6,182,212,0.15);
}

/* ── Header principal ── */
.header-wrap {
    text-align: center;
    padding: 32px 0 8px;
}
.header-title {
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(100deg, #22d3ee 0%, #6366f1 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.header-sub {
    color: #64748b;
    font-size: 15px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Tarjetas de estado ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 24px 0 28px;
}
.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg,#22d3ee,#6366f1));
}
.stat-label { font-size: 11px; letter-spacing: 2px; color: #475569; text-transform: uppercase; }
.stat-value { font-size: 28px; font-weight: 800; color: #f1f5f9; margin: 4px 0 0; font-family: 'Space Mono', monospace; }
.stat-desc  { font-size: 12px; color: #64748b; margin-top: 2px; }

/* ── Pipeline visual ── */
.pipeline-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px 20px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.pipe-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid transparent;
    transition: all .3s;
}
.pipe-step.idle    { background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.08); color: #475569; }
.pipe-step.active  { background: rgba(6,182,212,0.12);  border-color: #22d3ee; color: #22d3ee; animation: pulse 1.2s infinite; }
.pipe-step.done    { background: rgba(99,102,241,0.12); border-color: #6366f1; color: #a5b4fc; }
.pipe-arrow { color: #334155; font-size: 18px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* ── Mensajes ── */
.msg-user {
    background: linear-gradient(135deg, rgba(6,182,212,0.12), rgba(99,102,241,0.1));
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 18px 18px 4px 18px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 15px;
}
.msg-ai {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px 18px 18px 18px;
    padding: 20px 24px;
    margin: 12px 0;
    font-size: 15px;
    line-height: 1.7;
}
.msg-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 700;
}
.msg-user .msg-label { color: #22d3ee; }
.msg-ai   .msg-label { color: #a78bfa; }

/* ── Veredicto de seguridad ── */
.verdict {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 12px;
}
.verdict.ok   { background: rgba(34,197,94,0.12);  border: 1px solid #22c55e; color: #4ade80; }
.verdict.warn { background: rgba(234,179,8,0.12);  border: 1px solid #eab308; color: #facc15; }
.verdict.bad  { background: rgba(239,68,68,0.12);  border: 1px solid #ef4444; color: #f87171; }

/* ── Metadata de respuesta ── */
.meta-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.meta-item {
    font-size: 12px;
    color: #475569;
    font-family: 'Space Mono', monospace;
}
.meta-item span { color: #94a3b8; }

/* ── Agente badge sidebar ── */
.agent-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    font-size: 13px;
}
.agent-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Botones rápidos ── */
div[data-testid="stHorizontalBlock"] .stButton button {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    color: #94a3b8;
    border-radius: 10px;
    font-size: 13px;
    width: 100%;
    transition: all .2s;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover {
    background: rgba(6,182,212,0.1);
    border-color: #22d3ee;
    color: #22d3ee;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 20px 0; }

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #22d3ee !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  INICIALIZAR SISTEMA
# ─────────────────────────────────────────────
@st.cache_resource
def init_system():
    kb          = load_knowledge_base()
    opik        = OpikTracker()
    orquestador = AgenteOrquestador(kb)
    return orquestador, opik, kb

orquestador, opik, kb = init_system()


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "total_consultas" not in st.session_state: st.session_state.total_consultas = 0
if "tiempo_total"    not in st.session_state: st.session_state.tiempo_total    = 0.0
if "pipeline_step"   not in st.session_state: st.session_state.pipeline_step   = 0  # 0=idle,1,2,3,4=done
if "quick_prompt"    not in st.session_state: st.session_state.quick_prompt    = ""


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ MecatrIA")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Estado conexión
    st.markdown("**SISTEMA**")
    st.success("🟢 Ollama activo")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Métricas sesión
    st.markdown("**SESIÓN**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Consultas", st.session_state.total_consultas)
    with col_b:
        avg = (
            round(st.session_state.tiempo_total / st.session_state.total_consultas, 1)
            if st.session_state.total_consultas > 0 else 0
        )
        st.metric("Avg (s)", avg)

    st.metric("Proyectos KB", len(kb))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Agentes activos
    st.markdown("**AGENTES ACTIVOS**")
    agentes = [
        ("🔍", "Analista Docs",    "#22d3ee"),
        ("🧠", "Diagnóstico",      "#6366f1"),
        ("🔒", "Verificador Seg.", "#a78bfa"),
        ("🎯", "Orquestador",      "#f59e0b"),
    ]
    for icon, name, color in agentes:
        st.markdown(
            f'<div class="agent-badge">'
            f'<div class="agent-dot" style="background:{color}"></div>'
            f'{icon} {name}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Botón resumen Opik
    if st.button("📊 Ver trazas Opik"):
        resumen = opik.resumen_sesion()
        st.json(resumen)

    # Limpiar chat
    if st.button("🗑️ Limpiar chat"):
        st.session_state.messages = []
        st.session_state.total_consultas = 0
        st.session_state.tiempo_total    = 0.0
        st.rerun()


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-title">⚡ MECATRIA</div>
    <div class="header-sub">Asistente Inteligente · Multi-Agente · RAG Local</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  STATS SUPERIORES
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("PROYECTOS RAG",     str(len(kb)),     "Base de conocimiento Arduino",  "linear-gradient(90deg,#22d3ee,#6366f1)"),
    ("AGENTES",           "4",              "Pipeline multi-agente",         "linear-gradient(90deg,#6366f1,#a78bfa)"),
    ("CONSULTAS",         str(st.session_state.total_consultas), "Esta sesión", "linear-gradient(90deg,#a78bfa,#ec4899)"),
    ("MODELO PRINCIPAL",  "gemma3:4b",      "Diagnóstico técnico",           "linear-gradient(90deg,#f59e0b,#ef4444)"),
]
for col, (label, value, desc, grad) in zip([c1,c2,c3,c4], stats):
    with col:
        st.markdown(
            f'<div class="stat-card" style="--accent:{grad}">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
#  PIPELINE VISUAL
# ─────────────────────────────────────────────
def render_pipeline(step: int):
    steps = [
        (1, "🔍 Analista"),
        (2, "🧠 Diagnóstico"),
        (3, "🔒 Seguridad"),
        (4, "✅ Listo"),
    ]
    html = '<div class="pipeline-wrap"><span style="font-size:12px;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-right:8px">PIPELINE</span>'
    for i, (n, label) in enumerate(steps):
        if step == 0:
            css = "idle"
        elif n < step:
            css = "done"
        elif n == step:
            css = "active"
        else:
            css = "idle"
        html += f'<div class="pipe-step {css}">{label}</div>'
        if i < len(steps) - 1:
            html += '<span class="pipe-arrow">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

pipeline_placeholder = st.empty()
with pipeline_placeholder:
    render_pipeline(st.session_state.pipeline_step)


# ─────────────────────────────────────────────
#  BOTONES RÁPIDOS
# ─────────────────────────────────────────────
st.markdown("**Consultas rápidas de ejemplo:**")
b1, b2, b3, b4 = st.columns(4)
quick_prompts = {
    "🔧 Motor DC no gira":        "Mi motor DC no gira cuando le aplico voltaje",
    "📡 Sensor ultrasónico falla": "El sensor ultrasónico HC-SR04 devuelve valores erráticos",
    "💡 Código LED parpadeante":   "codigo Genera un codigo para hacer parpadear un LED con Arduino",
    "📖 Concepto PWM":             "concepto Explica qué es PWM en Arduino",
}
for col, (label, prompt) in zip([b1, b2, b3, b4], quick_prompts.items()):
    with col:
        if st.button(label):
            st.session_state.quick_prompt = prompt
            st.rerun()


st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HISTORIAL DE MENSAJES
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">'
            f'<div class="msg-label">👨‍💻 Tú</div>'
            f'{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        # Veredicto de seguridad
        veredicto = msg.get("veredicto", "")
        if veredicto == "APROBADO":
            vhtml = '<div class="verdict ok">🟢 SEGURIDAD: APROBADO</div>'
        elif veredicto == "APROBADO CON OBSERVACIONES":
            vhtml = '<div class="verdict warn">🟡 SEGURIDAD: CON OBSERVACIONES</div>'
        elif veredicto == "RECHAZADO":
            vhtml = '<div class="verdict bad">🔴 SEGURIDAD: RECHAZADO</div>'
        else:
            vhtml = ""

        # Metadata
        meta = ""
        if msg.get("tiempos"):
            t = msg["tiempos"]
            meta = (
                f'<div class="meta-row">'
                f'<div class="meta-item">⏱ Total <span>{t.get("total","-")}s</span></div>'
                f'<div class="meta-item">🔍 Analista <span>{t.get("analista_docs","-")}s</span></div>'
                f'<div class="meta-item">🧠 Diagnóstico <span>{t.get("diagnostico","-")}s</span></div>'
                f'<div class="meta-item">🔒 Seguridad <span>{t.get("seguridad","-")}s</span></div>'
                f'</div>'
            )
        if msg.get("documentos"):
            docs_str = " · ".join(msg["documentos"])
            meta += f'<div class="meta-item" style="margin-top:6px">📚 Docs: <span>{docs_str}</span></div>'

        st.markdown(
            f'<div class="msg-ai">'
            f'<div class="msg-label">⚡ MecatrIA</div>'
            f'{msg["content"]}'
            f'{vhtml}'
            f'{meta}'
            f'</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
#  INPUT DEL USUARIO
# ─────────────────────────────────────────────
prompt = st.chat_input("Describe tu problema técnico o pregunta...")

# Usar quick prompt si se presionó un botón
if st.session_state.quick_prompt:
    prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = ""

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()


# ─────────────────────────────────────────────
#  GENERAR RESPUESTA
# ─────────────────────────────────────────────
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    pregunta = st.session_state.messages[-1]["content"]

    # ── Comando: concepto ──
    if pregunta.lower().startswith("concepto "):
        tema = pregunta[9:]
        with st.spinner(f"📖 Explicando '{tema}'..."):
            respuesta = orquestador.diagnostico.explicar_concepto(tema)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": respuesta,
        })
        st.rerun()

    # ── Comando: codigo ──
    elif pregunta.lower().startswith("codigo "):
        desc = pregunta[7:]
        with st.spinner(f"💻 Generando código para '{desc}'..."):
            respuesta = orquestador.diagnostico.generar_codigo(desc)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": respuesta,
        })
        st.rerun()

    # ── Pipeline multi-agente completo ──
    else:
        # Paso 1
        st.session_state.pipeline_step = 1
        with pipeline_placeholder:
            render_pipeline(1)

        status = st.status("⚡ Ejecutando pipeline multi-agente...", expanded=True)

        with status:
            st.write("🔍 **Paso 1/3** — Analista de Documentos analizando...")
            res_docs = orquestador.analista.analizar(pregunta, opik)

            st.session_state.pipeline_step = 2
            st.write("🧠 **Paso 2/3** — Agente de Diagnóstico procesando...")
            res_diag = orquestador.diagnostico.diagnosticar(
                pregunta,
                contexto_documental=res_docs["respuesta"],
                opik_tracker=opik,
            )

            st.session_state.pipeline_step = 3
            st.write("🔒 **Paso 3/3** — Verificador de Seguridad validando...")
            res_seg = orquestador.seguridad.verificar(res_diag["respuesta"], opik)

            st.session_state.pipeline_step = 4
            status.update(label="✅ Pipeline completado", state="complete")

        # Registrar traza del orquestador
        tiempo_total = (
            res_docs["tiempo_s"] + res_diag["tiempo_s"] + res_seg["tiempo_s"]
        )
        opik.log_trace({
            "agente":              "Orquestador",
            "consulta":            pregunta,
            "veredicto_seguridad": res_seg["veredicto"],
            "tiempo_total_s":      round(tiempo_total, 2),
        })

        # Actualizar métricas de sesión
        st.session_state.total_consultas += 1
        st.session_state.tiempo_total    += tiempo_total

        # Guardar mensaje con metadata
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   res_diag["respuesta"],
            "veredicto": res_seg["veredicto"],
            "tiempos": {
                "analista_docs": res_docs["tiempo_s"],
                "diagnostico":   res_diag["tiempo_s"],
                "seguridad":     res_seg["tiempo_s"],
                "total":         round(tiempo_total, 2),
            },
            "documentos": list(set(
                res_docs["documentos_usados"] + res_diag["documentos_usados"]
            )),
        })

        st.session_state.pipeline_step = 0
        st.rerun()
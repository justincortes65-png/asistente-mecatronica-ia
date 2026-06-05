import os
import json
import time
import zipfile
import tempfile
import shutil
import requests
from pathlib import Path
from datetime import datetime
 
# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
 
MODELS = {
    "diagnostico":  "gemma3:4b",
    "codigo":       "qwen2.5-coder:14b",
    "chat":         "llama3.2:3b",
    "embeddings":   "nomic-embed-text:latest",
    "documentos":   "qwen2.5-coder:14b",   # Analista de documentos
    "seguridad":    "llama3.2:3b",          # Verificador de seguridad
}
 
OPIK_PROJECT        = "asistente-mecatronica"
KNOWLEDGE_BASE_PATH = Path("knowledge_base.json")
 
# ─────────────────────────────────────────────
#  CHUNKING OPTIMIZADO
# ─────────────────────────────────────────────
CHUNK_SIZE    = 400   # tokens aproximados por chunk
CHUNK_OVERLAP = 80    # solapamiento entre chunks
 
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide texto en chunks con solapamiento para no perder contexto
    entre fragmentos consecutivos.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks
 
 
# ─────────────────────────────────────────────
#  CARGA DE PROYECTOS ARDUINO
# ─────────────────────────────────────────────
def load_arduino_projects(dataset_path: str = ".") -> list[dict]:
    documents = []
    base = Path(dataset_path)
 
    for ino_file in sorted(base.rglob("*.ino")):
        try:
            code         = ino_file.read_text(encoding="utf-8", errors="ignore")
            project_name = ino_file.parent.name if ino_file.parent != base else ino_file.stem
            name_lower   = project_name.lower()
 
            if   any(k in name_lower for k in ["robot", "follower", "avoider"]):    category = "robotica"
            elif any(k in name_lower for k in ["sensor", "ir", "pir", "ultrasonic", "ldr"]): category = "sensores"
            elif any(k in name_lower for k in ["motor", "servo"]):                   category = "actuadores"
            elif any(k in name_lower for k in ["bluetooth", "remote", "mobile", "voice"]): category = "comunicacion"
            elif any(k in name_lower for k in ["lcd", "led", "alarm", "rtc"]):       category = "interfaz"
            else:                                                                     category = "general"
 
            pins = [
                line.strip()
                for line in code.splitlines()
                if "pinMode" in line or "#define" in line
            ]
 
            # Chunking del código fuente
            code_chunks = chunk_text(code)
 
            doc = {
                "id":       ino_file.stem.replace(" ", "_"),
                "nombre":   project_name,
                "categoria": category,
                "archivo":  str(ino_file),
                "codigo":   code,
                "chunks":   code_chunks,          # ← chunks optimizados
                "n_chunks": len(code_chunks),
                "pines":    pins[:10],
                "resumen":  (
                    f"Proyecto Arduino: {project_name}. "
                    f"Categoria: {category}. "
                    f"Lineas: {len(code.splitlines())}. "
                    f"Chunks: {len(code_chunks)}."
                ),
            }
            documents.append(doc)
            print(f"  Cargado: {project_name} [{category}] — {len(code_chunks)} chunks")
 
        except Exception as e:
            print(f"  Error leyendo {ino_file}: {e}")
 
    return documents
 
 
# ─────────────────────────────────────────────
#  EMBEDDINGS Y KNOWLEDGE BASE
# ─────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": MODELS["embeddings"], "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"  [WARN] Embedding fallo: {e}")
        return []
 
 
def build_knowledge_base(dataset_path: str = "."):
    print("\nConstruyendo base de conocimiento...")
    documents = load_arduino_projects(dataset_path)
    print(f"\nGenerando embeddings para {len(documents)} proyectos...")
 
    for i, doc in enumerate(documents):
        # Embeddings por chunk (para recuperación más precisa)
        chunk_embeddings = []
        for chunk in doc["chunks"]:
            emb = get_embedding(chunk)
            chunk_embeddings.append(emb)
 
        doc["chunk_embeddings"] = chunk_embeddings
        # Embedding del resumen para búsqueda rápida
        doc["embedding"] = get_embedding(
            f"{doc['nombre']}. {doc['resumen']}\n\nCodigo:\n{doc['codigo'][:500]}"
        )
        print(f"  [{i+1}/{len(documents)}] {doc['nombre']} — {len(chunk_embeddings)} chunk-embeddings")
 
    KNOWLEDGE_BASE_PATH.write_text(json.dumps(documents, ensure_ascii=False, indent=2))
    print(f"\nBase de conocimiento guardada en {KNOWLEDGE_BASE_PATH}")
    return documents
 
 
def load_knowledge_base() -> list[dict]:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            "Base de conocimiento no encontrada. Ejecuta: python main.py --build-kb"
        )
    return json.loads(KNOWLEDGE_BASE_PATH.read_text())
 
 
# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────
def ollama_chat(model: str, prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[ERROR] No se pudo conectar a Ollama. Asegurate que este corriendo."
    except Exception as e:
        return f"[ERROR] {e}"
 
 
def cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    return 0.0 if (mag_a == 0 or mag_b == 0) else dot / (mag_a * mag_b)
 
 
def retrieve_documents(query: str, knowledge_base: list, top_k: int = 3) -> list[dict]:
    """Recuperación semántica; fallback a keyword search."""
    query_emb = get_embedding(query)
    if not query_emb:
        return keyword_search(query, knowledge_base, top_k)
 
    scored = []
    for doc in knowledge_base:
        # Buscar también en chunk_embeddings para mayor precisión
        best_chunk_score = 0.0
        for cemb in doc.get("chunk_embeddings", []):
            s = cosine_similarity(query_emb, cemb)
            if s > best_chunk_score:
                best_chunk_score = s
        # Score final: 60% mejor chunk + 40% embedding del resumen
        summary_score = cosine_similarity(query_emb, doc.get("embedding", []))
        final_score   = 0.6 * best_chunk_score + 0.4 * summary_score
        scored.append((final_score, doc))
 
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
 
 
def keyword_search(query: str, knowledge_base: list, top_k: int = 3) -> list[dict]:
    query_words = set(query.lower().split())
    scored = []
    for doc in knowledge_base:
        text  = (doc["nombre"] + " " + doc["resumen"] + " " + doc["codigo"]).lower()
        score = sum(1 for w in query_words if w in text)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
 
 
def build_rag_context(documents: list, max_chunk_chars: int = 1000) -> str:
    """Construye contexto usando chunks relevantes en lugar de corte fijo."""
    parts = []
    for i, doc in enumerate(documents, 1):
        # Usar el primer chunk más representativo
        best_chunk = doc["chunks"][0] if doc.get("chunks") else doc["codigo"][:max_chunk_chars]
        parts.append(
            f"--- Documento {i}: {doc['nombre']} [{doc['categoria']}] ---\n"
            f"{doc['resumen']}\n\nCodigo (chunk principal):\n```cpp\n{best_chunk}\n```"
        )
    return "\n\n".join(parts)
 
 
# ═══════════════════════════════════════════════════════════════
#  AGENTES
# ═══════════════════════════════════════════════════════════════
 
# ── 1. Agente Analista de Documentos ──────────────────────────
class AgenteAnalistaDocumentos:
    """
    Analiza documentación técnica, manuales y datasheets.
    Extrae información relevante para el diagnóstico.
    """
    SYSTEM_PROMPT = """Eres un experto analista de documentacion tecnica en ingenieria mecatronica.
Tu rol es interpretar manuales, datasheets y esquemas tecnicos.
Responde SIEMPRE en español.
Cuando analices documentacion:
1. Extrae los parametros tecnicos clave
2. Identifica advertencias o restricciones importantes
3. Resume los pasos de configuracion relevantes
4. Señala referencias cruzadas utiles"""
 
    def __init__(self, knowledge_base: list):
        self.kb = knowledge_base
 
    def analizar(self, consulta: str, opik_tracker=None) -> dict:
        inicio = time.time()
        docs   = retrieve_documents(consulta, self.kb, top_k=2)
        context = build_rag_context(docs)
 
        prompt = f"""Analiza la siguiente documentacion tecnica en el contexto de esta consulta:
 
CONSULTA: {consulta}
 
DOCUMENTACION DISPONIBLE:
{context}
 
Extrae y resume la informacion tecnica mas relevante para resolver la consulta."""
 
        print(f"  [AnalistaDocumentos] Usando {MODELS['documentos']}...")
        respuesta = ollama_chat(
            model=MODELS["documentos"],
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
        )
 
        resultado = {
            "agente":    "AnalistaDocumentos",
            "consulta":  consulta,
            "respuesta": respuesta,
            "documentos_usados": [d["nombre"] for d in docs],
            "tiempo_s":  round(time.time() - inicio, 2),
            "timestamp": datetime.now().isoformat(),
        }
 
        if opik_tracker:
            opik_tracker.log_trace({**resultado, "modelo": MODELS["documentos"]})
 
        return resultado
 
 
# ── 2. Agente de Diagnóstico ───────────────────────────────────
class AgenteDiagnostico:
    """
    Diagnostica fallas técnicas y genera pasos de solución.
    Puede apoyarse en el análisis previo del AgenteAnalistaDocumentos.
    """
    SYSTEM_PROMPT = """Eres un experto en ingenieria mecatronica y electronica especializado en Arduino.
Tu rol es diagnosticar fallas y sugerir soluciones paso a paso.
Responde SIEMPRE en español.
Cuando recibas un problema:
1. Identifica el tipo de falla (hardware / software / calibracion / comunicacion)
2. Lista posibles causas ordenadas por probabilidad
3. Proporciona pasos de verificacion concretos y numerados
4. Sugiere soluciones con codigo Arduino si aplica"""
 
    def __init__(self, knowledge_base: list):
        self.kb       = knowledge_base
        self.historial = []
 
    def diagnosticar(self, consulta: str, contexto_documental: str = "", opik_tracker=None) -> dict:
        inicio = time.time()
        docs   = retrieve_documents(consulta, self.kb, top_k=3)
        context = build_rag_context(docs)
 
        prompt = f"""El usuario reporta el siguiente problema tecnico:
 
CONSULTA: {consulta}
 
{"ANÁLISIS DOCUMENTAL PREVIO:" + chr(10) + contexto_documental + chr(10) if contexto_documental else ""}
DOCUMENTOS RELEVANTES:
{context}
 
Proporciona un diagnostico detallado con pasos de solucion."""
 
        print(f"  [AgenteDiagnostico] Usando {MODELS['diagnostico']}...")
        respuesta = ollama_chat(
            model=MODELS["diagnostico"],
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
        )
 
        tiempo = round(time.time() - inicio, 2)
        resultado = {
            "agente":    "AgenteDiagnostico",
            "consulta":  consulta,
            "respuesta": respuesta,
            "documentos_usados": [d["nombre"] for d in docs],
            "tiempo_s":  tiempo,
            "timestamp": datetime.now().isoformat(),
        }
 
        if opik_tracker:
            opik_tracker.log_trace({**resultado, "modelo": MODELS["diagnostico"]})
 
        self.historial.append(resultado)
        return resultado
 
    def explicar_concepto(self, concepto: str) -> str:
        prompt = (
            f"Explica este concepto tecnico para un estudiante de ingenieria mecatronica "
            f"con ejemplos Arduino: {concepto}"
        )
        return ollama_chat(model=MODELS["chat"], prompt=prompt)
 
    def generar_codigo(self, descripcion: str) -> str:
        prompt = (
            f"Genera codigo Arduino para: {descripcion}\n"
            f"Incluye: pines con #define, setup(), loop() completos y comentarios."
        )
        return ollama_chat(model=MODELS["codigo"], prompt=prompt)
 
 
# ── 3. Agente Verificador de Seguridad ────────────────────────
class AgenteVerificadorSeguridad:
    """
    Valida que las soluciones propuestas sean seguras antes
    de entregarlas al usuario. Revisa riesgos eléctricos,
    lógicos y de operación.
    """
    SYSTEM_PROMPT = """Eres un experto en seguridad para sistemas electronicos y mecatronicos.
Tu rol es verificar que las soluciones tecnicas propuestas sean seguras.
Responde SIEMPRE en español.
Para cada solucion revisa:
1. Riesgos electricos (cortocircuitos, sobretensiones, corrientes excesivas)
2. Riesgos de software (bucles infinitos, desbordamientos, condiciones de carrera)
3. Riesgos mecanicos si hay actuadores involucrados
4. Indica claramente: APROBADO, APROBADO CON OBSERVACIONES o RECHAZADO
5. Si hay riesgos, explica como mitigarlos"""
 
    def __init__(self):
        pass
 
    def verificar(self, solucion: str, opik_tracker=None) -> dict:
        inicio = time.time()
 
        prompt = f"""Verifica la seguridad de la siguiente solucion tecnica propuesta:
 
SOLUCION A VERIFICAR:
{solucion}
 
Emite tu veredicto de seguridad y lista los riesgos encontrados con sus mitigaciones."""
 
        print(f"  [VerificadorSeguridad] Usando {MODELS['seguridad']}...")
        respuesta = ollama_chat(
            model=MODELS["seguridad"],
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
        )
 
        # Determinar veredicto
        r_upper = respuesta.upper()
        if "RECHAZADO" in r_upper:
            veredicto = "RECHAZADO"
        elif "OBSERVACIONES" in r_upper:
            veredicto = "APROBADO CON OBSERVACIONES"
        else:
            veredicto = "APROBADO"
 
        resultado = {
            "agente":    "VerificadorSeguridad",
            "veredicto": veredicto,
            "respuesta": respuesta,
            "tiempo_s":  round(time.time() - inicio, 2),
            "timestamp": datetime.now().isoformat(),
        }
 
        if opik_tracker:
            opik_tracker.log_trace({**resultado, "modelo": MODELS["seguridad"]})
 
        return resultado
 
 
# ── 4. Agente Orquestador (agente líder) ──────────────────────
class AgenteOrquestador:
    """
    Coordina el pipeline multi-agente:
      1. AgenteAnalistaDocumentos  → extrae contexto documental
      2. AgenteDiagnostico         → genera diagnóstico + solución
      3. AgenteVerificadorSeguridad → valida la solución
      4. Consolida y entrega respuesta final al usuario
    """
 
    def __init__(self, knowledge_base: list):
        self.analista    = AgenteAnalistaDocumentos(knowledge_base)
        self.diagnostico = AgenteDiagnostico(knowledge_base)
        self.seguridad   = AgenteVerificadorSeguridad()
 
    def procesar(self, consulta: str, opik_tracker=None) -> dict:
        inicio_total = time.time()
        print("\n" + "="*60)
        print(f"ORQUESTADOR: procesando consulta")
        print("="*60)
 
        # Paso 1 — Analista de Documentos
        print("\n[Paso 1/3] Analista de Documentos...")
        res_docs = self.analista.analizar(consulta, opik_tracker)
 
        # Paso 2 — Diagnóstico (con contexto del analista)
        print("\n[Paso 2/3] Agente de Diagnostico...")
        res_diag = self.diagnostico.diagnosticar(
            consulta,
            contexto_documental=res_docs["respuesta"],
            opik_tracker=opik_tracker,
        )
 
        # Paso 3 — Verificación de Seguridad
        print("\n[Paso 3/3] Verificador de Seguridad...")
        res_seg = self.seguridad.verificar(res_diag["respuesta"], opik_tracker)
 
        # Consolidar respuesta final
        respuesta_final = self._consolidar(res_docs, res_diag, res_seg)
 
        tiempo_total = round(time.time() - inicio_total, 2)
 
        resultado = {
            "consulta":            consulta,
            "respuesta_final":     respuesta_final,
            "veredicto_seguridad": res_seg["veredicto"],
            "documentos_usados":   list(set(
                res_docs["documentos_usados"] + res_diag["documentos_usados"]
            )),
            "tiempos": {
                "analista_docs": res_docs["tiempo_s"],
                "diagnostico":   res_diag["tiempo_s"],
                "seguridad":     res_seg["tiempo_s"],
                "total":         tiempo_total,
            },
            "detalle": {
                "analisis_documental": res_docs["respuesta"],
                "diagnostico":         res_diag["respuesta"],
                "verificacion":        res_seg["respuesta"],
            },
        }
 
        # Traza del orquestador
        if opik_tracker:
            opik_tracker.log_trace({
                "agente":              "Orquestador",
                "consulta":            consulta,
                "veredicto_seguridad": res_seg["veredicto"],
                "tiempo_total_s":      tiempo_total,
                "timestamp":           datetime.now().isoformat(),
            })
 
        return resultado
 
    def _consolidar(self, res_docs: dict, res_diag: dict, res_seg: dict) -> str:
        veredicto = res_seg["veredicto"]
        partes = [
            "╔══════════════════════════════════════════════════════════╗",
            "║            RESPUESTA DEL ASISTENTE IA                   ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            "📋 DIAGNÓSTICO Y SOLUCIÓN:",
            res_diag["respuesta"],
            "",
            f"🔒 VERIFICACIÓN DE SEGURIDAD: {veredicto}",
        ]
        if veredicto != "APROBADO":
            partes += ["", "⚠️  OBSERVACIONES DE SEGURIDAD:", res_seg["respuesta"]]
        return "\n".join(partes)
 
 
# ═══════════════════════════════════════════════════════════════
#  OPIK TRACKER
# ═══════════════════════════════════════════════════════════════
class OpikTracker:
    LOG_FILE = Path("opik_traces.jsonl")
 
    def __init__(self, project: str = OPIK_PROJECT):
        self.project    = project
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nOpik Tracker iniciado | Proyecto: {project} | Sesion: {self.session_id}")
 
    def log_trace(self, data: dict):
        trace = {"project": self.project, "session_id": self.session_id, **data}
        with open(self.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        try:
            requests.post("http://localhost:5173/api/v1/traces", json=trace, timeout=5)
            print(f"    ✓ Traza [{data.get('agente','?')}] registrada en Opik")
        except Exception:
            print(f"    · Traza [{data.get('agente','?')}] guardada localmente")
 
    def resumen_sesion(self) -> dict:
        traces = []
        if self.LOG_FILE.exists():
            for line in self.LOG_FILE.read_text().splitlines():
                try:
                    t = json.loads(line)
                    if t.get("session_id") == self.session_id:
                        traces.append(t)
                except Exception:
                    pass
 
        tiempos = [t.get("tiempo_s", t.get("tiempo_total_s", 0)) for t in traces]
        agentes = {}
        for t in traces:
            ag = t.get("agente", "desconocido")
            agentes[ag] = agentes.get(ag, 0) + 1
 
        return {
            "total_trazas":     len(traces),
            "tiempo_promedio_s": round(sum(tiempos) / len(tiempos), 2) if tiempos else 0,
            "llamadas_por_agente": agentes,
        }
 
 
# ═══════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
 
    print("=" * 60)
    print("  Asistente IA — Ingenieria Mecatronica (Multi-Agente)")
    print("=" * 60)
 
    # ── Construir knowledge base ──────────────────────────────
    if "--build-kb" in sys.argv:
        idx  = sys.argv.index("--build-kb")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "."
 
        if path.endswith(".zip"):
            print(f"Extrayendo {path}...")
            tmp_dir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(tmp_dir)
                build_knowledge_base(tmp_dir)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            build_knowledge_base(path)
 
        print("\nBase de conocimiento lista. Ejecuta: python main.py")
        sys.exit(0)
 
    # ── Cargar knowledge base ─────────────────────────────────
    try:
        print("Cargando base de conocimiento...")
        kb = load_knowledge_base()
        print(f"  {len(kb)} proyectos cargados")
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
 
    # ── Inicializar sistema multi-agente ──────────────────────
    opik        = OpikTracker()
    orquestador = AgenteOrquestador(kb)
 
    print("\nComandos disponibles:")
    print("  salir              — terminar sesion")
    print("  concepto <tema>    — explicar un concepto tecnico")
    print("  codigo <desc>      — generar codigo Arduino")
    print("  resumen            — ver estadisticas de la sesion")
    print("  <consulta libre>   — diagnostico completo multi-agente")
 
    while True:
        try:
            entrada = input("\n> Consulta: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
 
        if not entrada:
            continue
 
        if entrada.lower() == "salir":
            break
        elif entrada.lower() == "resumen":
            r = opik.resumen_sesion()
            print(f"\nResumen de sesion:")
            print(f"  Total de trazas:      {r['total_trazas']}")
            print(f"  Tiempo promedio (s):  {r['tiempo_promedio_s']}")
            print(f"  Llamadas por agente:  {r['llamadas_por_agente']}")
        elif entrada.lower().startswith("concepto "):
            print(orquestador.diagnostico.explicar_concepto(entrada[9:]))
        elif entrada.lower().startswith("codigo "):
            print(orquestador.diagnostico.generar_codigo(entrada[7:]))
        else:
            resultado = orquestador.procesar(entrada, opik)
 
            print("\n" + resultado["respuesta_final"])
            print(f"\n📂 Documentos usados: {', '.join(resultado['documentos_usados'])}")
            print(f"⏱️  Tiempos — Analista: {resultado['tiempos']['analista_docs']}s | "
                  f"Diagnóstico: {resultado['tiempos']['diagnostico']}s | "
                  f"Seguridad: {resultado['tiempos']['seguridad']}s | "
                  f"Total: {resultado['tiempos']['total']}s")
 
    print(f"\nSesion terminada.")
    r = opik.resumen_sesion()
    print(f"Total trazas: {r['total_trazas']} | "
          f"Tiempo promedio: {r['tiempo_promedio_s']}s")
    print(f"Agentes utilizados: {r['llamadas_por_agente']}")
"""
evaluate.py — Evaluación del sistema multi-agente con DeepEval
Métricas: Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall
Juez: Ollama local (sin OpenAI)
"""

import json
import time
import requests as req

from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM   # ← IMPORTANTE
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

from main import AgenteOrquestador, load_knowledge_base, OpikTracker, retrieve_documents


# ─────────────────────────────────────────────
#  JUEZ LOCAL — OLLAMA  (CORREGIDO)
# ─────────────────────────────────────────────
class OllamaLLM(DeepEvalBaseLLM):
    """Wrapper correcto para usar Ollama como juez en DeepEval."""

    def __init__(self, model: str = "qwen2.5-coder:14b"):
        self.model = model
        super().__init__()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        try:
            resp = req.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            return f"[ERROR] {e}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


# ─────────────────────────────────────────────
#  CASOS DE PRUEBA
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "input": "Mi motor DC no gira cuando le aplico voltaje",
        "expected": "Verificar conexiones, revisar puente H, comprobar alimentación del motor y señal PWM",
    },
    {
        "input": "El sensor ultrasónico HC-SR04 siempre me devuelve la misma distancia",
        "expected": "Revisar conexiones TRIG y ECHO, verificar alimentación 5V, revisar el código de pulso y timeout",
    },
    {
        "input": "Mi servo motor vibra y no se posiciona correctamente",
        "expected": "Verificar señal PWM entre 1000-2000us, revisar alimentación separada, comprobar ángulo máximo del servo",
    },
    {
        "input": "El sensor IR no detecta obstáculos correctamente",
        "expected": "Ajustar potenciómetro de sensibilidad, verificar distancia de detección, revisar interferencia de luz ambiental",
    },
    {
        "input": "Mi Arduino no sube el código y muestra error de puerto COM",
        "expected": "Verificar driver CH340 o FTDI instalado, seleccionar puerto COM correcto, revisar cable USB",
    },
]


# ─────────────────────────────────────────────
#  MÉTRICAS CON JUEZ LOCAL (CORREGIDO)
# ─────────────────────────────────────────────
def get_metrics(juez: OllamaLLM):
    return [
        AnswerRelevancyMetric(model=juez,   threshold=0.5, verbose_mode=True),
        FaithfulnessMetric(model=juez,      threshold=0.5, verbose_mode=True),
        ContextualPrecisionMetric(model=juez, threshold=0.5, verbose_mode=True),
        ContextualRecallMetric(model=juez,  threshold=0.5, verbose_mode=True),
    ]


# ─────────────────────────────────────────────
#  CONSTRUIR TEST CASES
# ─────────────────────────────────────────────
def build_test_cases(kb: list, orquestador: AgenteOrquestador) -> list:
    test_cases = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] Evaluando: {tc['input'][:50]}...")
        inicio = time.time()

        try:
            res_docs = orquestador.analista.analizar(tc["input"])
            res_diag = orquestador.diagnostico.diagnosticar(
                tc["input"],
                contexto_documental=res_docs["respuesta"],
            )
            respuesta_actual = res_diag["respuesta"]

            docs_relevantes = retrieve_documents(tc["input"], kb, top_k=3)
            contexto_rag = [
                f"{d['nombre']}: {d['resumen']}\n{d['codigo'][:400]}"
                for d in docs_relevantes
            ]

            tiempo = round(time.time() - inicio, 2)
            print(f"  ✓ Respuesta obtenida en {tiempo}s")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            respuesta_actual = f"[ERROR] {e}"
            contexto_rag = ["Sin contexto disponible"]

        test_case = LLMTestCase(
            input=tc["input"],
            actual_output=respuesta_actual,
            expected_output=tc["expected"],
            retrieval_context=contexto_rag,
        )
        test_cases.append(test_case)

    return test_cases


# ─────────────────────────────────────────────
#  REPORTE LOCAL JSON
# ─────────────────────────────────────────────
def guardar_reporte(test_cases: list, metrics: list):
    reporte = {
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_casos": len(test_cases),
        "resultados":  [],
    }

    for tc in test_cases:
        caso = {"input": tc.input, "metricas": {}}
        for metric in metrics:
            nombre = metric.__class__.__name__
            try:
                metric.measure(tc)
                caso["metricas"][nombre] = {
                    "score":  round(metric.score, 3),
                    "reason": metric.reason,
                    "passed": metric.score >= metric.threshold,
                }
            except Exception as e:
                caso["metricas"][nombre] = {"error": str(e)}
        reporte["resultados"].append(caso)

    # Promedios
    for metric in metrics:
        nombre = metric.__class__.__name__
        scores = [
            r["metricas"].get(nombre, {}).get("score", 0)
            for r in reporte["resultados"]
            if "score" in r["metricas"].get(nombre, {})
        ]
        if scores:
            reporte[f"promedio_{nombre}"] = round(sum(scores) / len(scores), 3)

    with open("reporte_evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print("\n✅ Reporte guardado en reporte_evaluacion.json")
    return reporte


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DeepEval — Evaluación con Ollama local")
    print("=" * 60)

    # 1. Instanciar juez local
    print("\nInicializando juez Ollama (qwen2.5-coder:14b)...")
    juez = OllamaLLM(model="qwen2.5-coder:14b")

    # 2. Cargar sistema
    print("Cargando knowledge base...")
    kb          = load_knowledge_base()
    opik        = OpikTracker()
    orquestador = AgenteOrquestador(kb)
    print(f"  {len(kb)} proyectos cargados")

    # 3. Construir casos de prueba
    print(f"\nEjecutando {len(TEST_CASES)} casos de prueba...")
    test_cases = build_test_cases(kb, orquestador)

    # 4. Métricas con juez local
    metrics = get_metrics(juez)

    # 5. Guardar reporte
    print("\nCalculando métricas...")
    reporte = guardar_reporte(test_cases, metrics)

    # 6. Resumen en consola
    print("\n" + "=" * 60)
    print("  RESUMEN DE EVALUACIÓN")
    print("=" * 60)
    for key, val in reporte.items():
        if key.startswith("promedio_"):
            nombre = key.replace("promedio_", "").replace("Metric", "")
            emoji  = "✅" if val >= 0.5 else "❌"
            print(f"  {emoji} {nombre:<30} {val:.1%}")

    # 7. Enviar a Confident AI (opcional)
    print("\nEnviando resultados a Confident AI...")
    try:
        dataset = EvaluationDataset(test_cases=test_cases)
        evaluate(dataset, metrics)
        print("✅ Resultados visibles en https://app.confident-ai.com")
    except Exception as e:
        print(f"  [WARN] No se pudo enviar al dashboard: {e}")
        print("  Los resultados están en reporte_evaluacion.json")
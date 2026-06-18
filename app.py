"""
Graphify Core — Backend FastAPI (Producción Real)
==================================================
Trazabilidad en tiempo real vía Webhook + SSE.
Chatbot Graph-RAG con placeholder estructurado para Gemini API.
Servicio del grafo interactivo real generado por Graphify.

Endpoints:
  GET  /                  → Sirve index.html (frontend SPA)
  GET  /graph             → Sirve graphify-out/graph.html (grafo real)
  GET  /api/activity      → Retorna historial de actividades reales
  GET  /api/stream        → SSE: broadcast de nuevas actividades a clientes conectados
  POST /api/webhook/git   → Webhook: recibe actividad real y la propaga vía SSE
  POST /api/chat          → Chatbot Graph-RAG (lee graph.json real)
  GET  /api/health        → Health check del servicio

Sin datos falsos. Sin simuladores. 100% datos reales ingresados por webhook.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
GRAPH_JSON_PATH = BASE_DIR / "graphify-out" / "graph.json"
GRAPH_HTML_PATH = BASE_DIR / "graphify-out" / "graph.html"
FRONTEND_PATH = BASE_DIR / "index.html"

app = FastAPI(
    title="Graphify Core",
    version="1.0.0",
    description="Dashboard interactivo — Trazabilidad real + Graph-RAG Chatbot",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Modelos Pydantic
# ──────────────────────────────────────────────


class WebhookPayload(BaseModel):
    """
    Payload que recibe el webhook de Git.

    Campos obligatorios:
      - developer: nombre del autor del commit/cambio
      - project:   nombre del proyecto afectado
      - activity:  descripción en lenguaje natural del cambio
      - impact:    nivel de impacto ("Bajo" | "Medio" | "Alto")

    Campos opcionales (se autogeneran si no se envían):
      - project_color: color del badge en el dashboard
    """

    developer: str = Field(..., min_length=1, description="Nombre del desarrollador")
    project: str = Field(..., min_length=1, description="Nombre del proyecto")
    activity: str = Field(..., min_length=1, description="Descripción de la actividad")
    impact: str = Field(..., pattern=r"^(Bajo|Medio|Alto)$", description="Nivel de impacto")
    project_color: str | None = Field(
        None,
        description="Color del badge (blue, violet, green, amber, rose). Auto-asignado si no se envía.",
    )


class ChatMessage(BaseModel):
    """Payload del chatbot."""

    message: str = Field(..., min_length=1)


class ActivityItem(BaseModel):
    """Fila de actividad almacenada y transmitida vía SSE."""

    id: str
    developer: str
    developer_avatar: str
    project: str
    project_color: str
    activity: str
    timestamp: str  # ISO-8601
    impact: str


# ──────────────────────────────────────────────
# Almacén en memoria de actividades
# ──────────────────────────────────────────────

activity_log: list[ActivityItem] = []
MAX_ACTIVITY_LOG = 200  # Límite para no saturar RAM

# ─── Cola de broadcast SSE ─────────────────
# Cada cliente SSE conectado registra un asyncio.Queue aquí.
# Cuando llega un webhook, se hace queue.put() a todas las colas.

sse_clients: list[asyncio.Queue] = []


def _broadcast(event: dict) -> None:
    """Envía un evento a todos los clientes SSE conectados."""
    data = json.dumps(event)
    dead_queues: list[asyncio.Queue] = []
    for q in sse_clients:
        # Si la queue está llena (cliente lento), la descartamos
        if q.full():
            dead_queues.append(q)
        else:
            q.put_nowait(data)
    # Limpiar queues muertas
    for q in dead_queues:
        if q in sse_clients:
            sse_clients.remove(q)


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────

# Mapeo de colores para proyectos conocidos
_PROJECT_COLORS: dict[str, str] = {
    "graphify": "blue",
    "api": "blue",
    "frontend": "violet",
    "dashboard": "violet",
    "pipeline": "green",
    "data": "green",
    "auth": "amber",
    "infra": "rose",
    "cloud": "rose",
    "devops": "rose",
}
_FALLBACK_COLORS = ["blue", "violet", "green", "amber", "rose"]


def _infer_project_color(project_name: str) -> str:
    """Infiere el color del badge a partir del nombre del proyecto."""
    name_lower = project_name.lower()
    for keyword, color in _PROJECT_COLORS.items():
        if keyword in name_lower:
            return color
    # Hash determinístico como fallback
    idx = sum(ord(c) for c in name_lower) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def _make_avatar(name: str) -> str:
    """Genera iniciales para el avatar circular a partir del nombre."""
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


def _create_activity_item(payload: WebhookPayload) -> ActivityItem:
    """Construye un ActivityItem a partir del payload del webhook."""
    color = payload.project_color or _infer_project_color(payload.project)
    return ActivityItem(
        id=str(uuid.uuid4())[:8],
        developer=payload.developer,
        developer_avatar=_make_avatar(payload.developer),
        project=payload.project,
        project_color=color,
        activity=payload.activity,
        timestamp=datetime.now(timezone.utc).isoformat(),
        impact=payload.impact,
    )


# ──────────────────────────────────────────────
# Carga y búsqueda en graph.json
# ──────────────────────────────────────────────


def load_graph() -> dict:
    """Carga graph.json desde disco. Retorna dict vacío si no existe."""
    if not GRAPH_JSON_PATH.exists():
        return {"nodes": [], "hyperedges": [], "links": []}
    try:
        with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"nodes": [], "hyperedges": [], "links": []}


def search_graph_context(query: str, graph: dict, top_k: int = 5) -> list[dict]:
    """
    Busca nodos e hiper-aristas relevantes mediante coincidencia
    de palabras clave en labels e IDs.

    Retorna lista de dicts con: type, id, label, score, + campos específicos.
    """
    # Normalizar y tokenizar el query
    query_lower = query.lower()
    words = set(re.findall(r"\w+", query_lower))
    # También agregar bigramas para mejor coincidencia
    tokens = list(words)
    for i in range(len(tokens) - 1):
        words.add(f"{tokens[i]} {tokens[i+1]}")

    results: list[dict] = []

    # Buscar en hiper-aristas
    for edge in graph.get("hyperedges", []):
        label = (edge.get("label") or "").lower()
        edge_id = (edge.get("id") or "").lower()
        source = (edge.get("source_file") or "").lower()
        edge_text = f"{label} {edge_id} {source}"
        score = sum(1 for w in words if w in edge_text)
        if score > 0:
            results.append(
                {
                    "type": "hyperedge",
                    "id": edge.get("id"),
                    "label": edge.get("label"),
                    "nodes": edge.get("nodes", []),
                    "relation": edge.get("relation"),
                    "confidence": edge.get("confidence"),
                    "source_file": edge.get("source_file"),
                    "score": score,
                }
            )

    # Buscar en nodos
    for node in graph.get("nodes", []):
        node_label = (node.get("label") or "").lower()
        node_id = (node.get("id") or "").lower()
        node_cat = (node.get("category") or "").lower()
        node_text = f"{node_label} {node_id} {node_cat}"
        score = sum(1 for w in words if w in node_text)
        if score > 0:
            results.append(
                {
                    "type": "node",
                    "id": node.get("id"),
                    "label": node.get("label"),
                    "category": node.get("category"),
                    "score": score,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ──────────────────────────────────────────────
# PLACEHOLDER: Integración con API de Gemini
# ──────────────────────────────────────────────

# ┌──────────────────────────────────────────────────────────────────────────┐
# │                                                                          │
# │  GEMINI API — INSTRUCCIONES DE INTEGRACIÓN                              │
# │  ═════════════════════════════════════════                               │
# │                                                                          │
# │  1. Instalar SDK:                                                        │
# │     pip install google-generativeai                                      │
# │                                                                          │
# │  2. Inyectar API Key:                                                    │
# │     - Opción A: Variable de entorno GEMINI_API_KEY                      │
# │     - Opción B: Reemplazar directamente el valor en GEMINI_API_KEY      │
# │                                                                          │
# │  3. Descomentar el bloque marcado como [GEMINI-ACTIVE] abajo            │
# │                                                                          │
# │  4. En _generate_response(), cambiar la línea:                          │
# │     response_text = _build_local_response(prompt, context)              │
# │     por:                                                                 │
# │     response_text = await _call_gemini_api(prompt, context)             │
# │                                                                          │
# │  5. En /api/chat, cambiar el campo source:                              │
# │     "source": "graph-local"  →  "source": "gemini-api"                 │
# │                                                                          │
# └──────────────────────────────────────────────────────────────────────────┘

GEMINI_API_KEY = ""  # ← INYECTAR API KEY AQUÍ (o usar env var)
GEMINI_MODEL = "gemini-2.0-flash"

# ── [GEMINI-ACTIVE] Descomentar este bloque para activar Gemini ──
#
# import os
# import httpx
#
# async def _call_gemini_api(prompt: str, context: list[dict]) -> str:
#     """
#     Llamada HTTP real a la API de Gemini.
#
#     Construye un prompt del sistema con el contexto del grafo,
#     envía el mensaje del usuario y retorna la respuesta generativa.
#     """
#     api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
#     if not api_key:
#         return _build_local_response(prompt, context)
#
#     # Construir contexto formateado para el prompt del sistema
#     context_text = _format_context_for_prompt(context)
#
#     system_prompt = (
#         "Eres un asistente experto en el knowledge graph de Graphify. "
#         "Responde en español. Usa el contexto proporcionado del grafo "
#         "para dar respuestas precisas y bien estructuradas. "
#         "Si el contexto no es suficiente, indícalo claramente.\n\n"
#         f"Contexto del grafo:\n{context_text}"
#     )
#
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
#     payload = {
#         "system_instruction": {"parts": [{"text": system_prompt}]},
#         "contents": [{"parts": [{"text": prompt}]}],
#         "generationConfig": {
#             "temperature": 0.4,
#             "maxOutputTokens": 1024,
#         },
#     }
#
#     async with httpx.AsyncClient(timeout=30) as client:
#         response = await client.post(url, json=payload)
#         response.raise_for_status()
#         data = response.json()
#
#     # Extraer texto de la respuesta
#     try:
#         return data["candidates"][0]["content"]["parts"][0]["text"]
#     except (KeyError, IndexError):
#         return "La API de Gemini no retornó una respuesta válida."
#
# ── Fin [GEMINI-ACTIVE] ──


def _format_context_for_prompt(context: list[dict]) -> str:
    """Formatea el contexto del grafo para incluirlo en el prompt de Gemini."""
    if not context:
        return "Sin contexto relevante encontrado."
    lines = []
    for item in context:
        if item["type"] == "hyperedge":
            lines.append(
                f"- Hiper-arista: {item['label']} (relación: {item.get('relation')}, "
                f"confianza: {item.get('confidence')}, nodos: {item.get('nodes')})"
            )
        else:
            lines.append(f"- Nodo: {item['label']} (categoría: {item.get('category')})")
    return "\n".join(lines)


def _build_local_response(prompt: str, context: list[dict]) -> str:
    """
    Respuesta local basada en el contexto del grafo.
    Se usa mientras la API de Gemini no está activa.
    """
    if not context:
        return (
            f'No encontré información relevante en el knowledge graph '
            f'para: **"{prompt}"**.\n\n'
            f'Prueba con términos presentes en el grafo como: '
            f'*press, espalda, hipertrofia, recuperación, rutina, '
            f'sobrecarga, pierna, hombro*.\n\n'
            f'_Activa la API de Gemini para respuestas generativas._'
        )

    lines = [f"Encontré **{len(context)}** coincidencia(s) en el knowledge graph:\n"]
    for item in context:
        if item["type"] == "hyperedge":
            lines.append(
                f"- 🔗 **{item['label']}** ({item.get('relation', 'N/A')}) "
                f"— Confianza: {item.get('confidence', 'N/A')} "
                f"| Fuente: `{item.get('source_file', 'N/A')}`"
            )
        else:
            lines.append(
                f"- 📌 **{item['label']}** — Categoría: {item.get('category', 'N/A')}"
            )
    lines.append(
        "\n_Contexto extraído de graph.json. "
        "Con la API de Gemini activa, la respuesta será generativa._"
    )
    return "\n".join(lines)


async def _generate_response(prompt: str, context: list[dict]) -> str:
    """
    Genera la respuesta del chatbot.

    Por defecto usa la respuesta local basada en el contexto del grafo.
    Para activar Gemini:
      1. Descomentar el bloque [GEMINI-ACTIVE] arriba.
      2. Inyectar la API Key en GEMINI_API_KEY.
      3. Reemplazar la línea de abajo con:
         return await _call_gemini_api(prompt, context)
    """
    return _build_local_response(prompt, context)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@app.get("/")
async def serve_frontend():
    """Sirve el frontend SPA."""
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH, media_type="text/html")
    return JSONResponse(
        {"error": "index.html no encontrado en la raíz del proyecto."},
        status_code=404,
    )


@app.get("/graph")
async def serve_graph():
    """Sirve el grafo interactivo real generado por Graphify."""
    if GRAPH_HTML_PATH.exists():
        return FileResponse(GRAPH_HTML_PATH, media_type="text/html")
    return JSONResponse(
        {"error": "graphify-out/graph.html no encontrado. Ejecuta /graphify primero."},
        status_code=404,
    )


@app.get("/api/activity")
async def get_activity():
    """Retorna el historial de actividades reales (más recientes primero)."""
    return [item.model_dump() for item in reversed(activity_log)]


@app.post("/api/webhook/git")
async def webhook_git(payload: WebhookPayload):
    """
    Webhook de Git — Recibe datos reales y los propaga en tiempo real.

    Ejemplo con curl:
    ─────────────────
    curl -X POST http://localhost:8000/api/webhook/git \
      -H "Content-Type: application/json" \
      -d '{
        "developer": "Juan Pérez",
        "project": "Graphify API",
        "activity": "Implementó endpoint de búsqueda semántica en el grafo",
        "impact": "Alto"
      }'

    El endpoint:
    1. Valida el payload con Pydantic.
    2. Crea el ActivityItem con avatar y color auto-generados.
    3. Lo almacena en el log en memoria.
    4. Lo transmite a todos los clientes SSE conectados.
    """
    item = _create_activity_item(payload)

    # Almacenar
    activity_log.append(item)
    if len(activity_log) > MAX_ACTIVITY_LOG:
        activity_log.pop(0)

    # Broadcast a clientes SSE
    _broadcast(item.model_dump())

    return {
        "status": "ok",
        "message": "Actividad registrada y transmitida",
        "id": item.id,
        "sse_clients": len(sse_clients),
    }


@app.get("/api/stream")
async def stream_activity(request: Request):
    """
    SSE (Server-Sent Events) — Escucha activa de nuevas actividades.

    El frontend abre una conexión persistente a este endpoint.
    Cada vez que POST /api/webhook/git recibe datos reales,
    este endpoint los transmite al instante a todos los clientes.

    No genera datos falsos. Solo retransmite lo que llega por el webhook.
    """
    from sse_starlette.sse import EventSourceResponse  # type: ignore

    # Crear queue para este cliente
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    sse_clients.append(queue)

    async def event_generator() -> AsyncGenerator:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Esperar datos con timeout para detectar desconexión
                    data = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {"event": "activity", "data": data}
                except asyncio.TimeoutError:
                    # Heartbeat para mantener la conexión viva
                    yield {"event": "ping", "data": ""}
        finally:
            # Limpiar queue al desconectarse
            if queue in sse_clients:
                sse_clients.remove(queue)

    return EventSourceResponse(event_generator())


@app.post("/api/chat")
async def chat_endpoint(payload: ChatMessage):
    """
    Chatbot Graph-RAG.

    Flujo:
    1. Recibe el mensaje del usuario.
    2. Carga graph.json real desde disco.
    3. Busca contexto relevante por palabras clave.
    4. Genera respuesta (local o via Gemini si está activado).
    5. Retorna respuesta + contexto + metadatos.
    """
    graph = load_graph()
    context = search_graph_context(payload.message, graph)

    response_text = await _generate_response(payload.message, context)

    return {
        "response": response_text,
        "context_used": context,
        "context_count": len(context),
        "source": "graph-local",  # Cambiar a "gemini-api" al activar Gemini
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("hyperedges", [])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/health")
async def health_check():
    """Health check del servicio."""
    graph = load_graph()
    return {
        "status": "ok",
        "version": "1.0.0",
        "graph_loaded": bool(graph.get("nodes") or graph.get("hyperedges")),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("hyperedges", [])),
        "activity_log_size": len(activity_log),
        "sse_clients_connected": len(sse_clients),
        "gemini_configured": bool(GEMINI_API_KEY),
    }


# ──────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

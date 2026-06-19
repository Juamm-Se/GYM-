"""
Obsidian & Git Intelligence Center v5.0
========================================
Plataforma: Obsidian Vault Grafo + GitHub Webhook Real + Gemini Graph-RAG

Endpoints:
  GET  /                     → Sirve index.html (frontend premium)
  GET  /api/graph-data       → Nodos y conexiones del Vault Obsidian (vis.js)
  GET  /api/activity         → Historial de commits
  GET  /api/stream           → SSE: broadcast en tiempo real desde webhook
  POST /api/webhook/github   → Webhook GitHub: commits reales de git push
  POST /api/chat             → Chatbot Graph-RAG con Gemini 1.5 Flash
  GET  /api/health           → Health check
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════
#  CONFIGURACIÓN — API KEY DE GEMINI
# ═══════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   # ← Set via env var: GEMINI_API_KEY=...
GEMINI_MODEL   = "gemini-2.5-flash"
# ═══════════════════════════════════════════════════

BASE_DIR      = Path(__file__).resolve().parent
VAULT_DIR     = Path(r"C:\Users\sebas\OneDrive - SENA\Escritorio\Practica_Obsidian")
FRONTEND_PATH = BASE_DIR / "index.html"

app = FastAPI(title="Obsidian & Git Intelligence Center", version="5.0.0")

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


class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1)


class CommitItem(BaseModel):
    id: str
    author: str
    author_avatar: str
    repo: str
    message: str
    timestamp: str
    impact: str            # Bajo / Medio / Alto


# ──────────────────────────────────────────────
# Almacén en memoria + Broadcast SSE
# ──────────────────────────────────────────────

commit_log: list[CommitItem] = []
MAX_LOG = 200
sse_clients: list[asyncio.Queue] = []


def _broadcast(event_dict: dict) -> None:
    data = json.dumps(event_dict)
    dead: list[asyncio.Queue] = []
    for q in sse_clients:
        if q.full():
            dead.append(q)
        else:
            q.put_nowait(data)
    for q in dead:
        if q in sse_clients:
            sse_clients.remove(q)


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────


def _avatar(name: str) -> str:
    parts = name.strip().split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else name[:2].upper()


def _infer_impact(message: str) -> str:
    """Infiera el impacto del commit basado en el mensaje."""
    msg = message.lower()
    high_kw = ["breaking", "major", "release", "deploy", "hotfix", "urgent",
               "remove", "delete", "migrate", "refactor", "rewrite", "feat"]
    mid_kw  = ["fix", "update", "change", "add", "improve", "modify", "enhance",
               "merge", "config", "style", "docs"]
    for kw in high_kw:
        if kw in msg:
            return "Alto"
    for kw in mid_kw:
        if kw in msg:
            return "Medio"
    return "Bajo"


# ──────────────────────────────────────────────
# Obsidian Vault — Parseo nativo de [[Enlaces]]
# ──────────────────────────────────────────────


def parse_vault() -> dict:
    """
    Escanea la carpeta 'vault/' buscando archivos .md.
    Extrae los enlaces internos tipo [[Nota]] y construye
    nodos y conexiones para renderizar el grafo con vis.js.

    Formato de salida (compatible con vis.js):
      nodes: [{"id": <string>, "label": <string>}]
      edges: [{"from": <string>, "to": <string>}]
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()  # deduplicación real
    node_map: dict[str, str] = {}  # nombre limpio lower → id
    node_id_counter = 0

    if not VAULT_DIR.exists():
        return {"nodes": nodes, "edges": edges}

    def _clean_name(filename: str) -> str:
        name = filename
        if name.lower().endswith(".md"):
            name = name[:-3]
        name = name.replace("_", " ").replace("-", " ")
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _get_or_create_node(label: str) -> str:
        nonlocal node_id_counter
        key = label.lower()
        if key in node_map:
            return node_map[key]
        node_id = str(node_id_counter)
        node_id_counter += 1
        node_map[key] = node_id
        # Formato exacto que exige vis.js: id + label
        nodes.append({
            "id": node_id,
            "label": label,
        })
        return node_id

    # Primera pasada: crear nodos por cada archivo .md
    md_files = sorted(VAULT_DIR.rglob("*.md"))
    for filepath in md_files:
        clean = _clean_name(filepath.name)
        _get_or_create_node(clean)

    # Segunda pasada: extraer [[enlaces]] y crear edges
    for filepath in md_files:
        source_clean = _clean_name(filepath.name)
        source_id = node_map.get(source_clean.lower())
        if source_id is None:
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Buscar enlaces tipo [[Nota]] y [[Nota|Alias]]
        wikilinks = re.findall(r'\[\[([^\]#|]+)(?:[|#][^\]]*)?\]\]', content)
        for link in wikilinks:
            link_clean = _clean_name(link.strip())  # Normalizar igual que los archivos
            if not link_clean:
                continue
            target_id = _get_or_create_node(link_clean)

            # Solo crear edge si: no es self-loop Y no es duplicado
            if source_id == target_id:
                continue
            edge_key = (source_id, target_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # Formato exacto que exige vis.js: from + to
            edges.append({
                "from": source_id,
                "to": target_id,
            })

    return {"nodes": nodes, "edges": edges}


def _get_vault_text_context(query: str, top_k: int = 5) -> str:
    """
    Busca notas relevantes del Vault para usar como contexto
    en el chat de Gemini (Graph-RAG).
    """
    if not VAULT_DIR.exists():
        return "No se encontró la carpeta 'vault/'."

    tokens = set(re.findall(r"\w+", query.lower()))
    scored_notes: list[tuple[int, str, str]] = []  # (score, title, excerpt)

    for filepath in VAULT_DIR.rglob("*.md"):
        try:
            content = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        title = filepath.stem.replace("_", " ").replace("-", " ").strip()
        text_lower = (title + " " + content).lower()
        score = sum(1 for t in tokens if t in text_lower)

        if score > 0:
            # Extracto: primeras 300 chars relevantes
            excerpt = content[:500].strip()
            scored_notes.append((score, title, excerpt))

    scored_notes.sort(key=lambda x: x[0], reverse=True)
    top_notes = scored_notes[:top_k]

    if not top_notes:
        # Si no hay coincidencias, devolver un resumen general
        all_titles = []
        for filepath in VAULT_DIR.rglob("*.md"):
            title = filepath.stem.replace("_", " ").replace("-", " ").strip()
            all_titles.append(f"- {title}")
        return (
            "No se encontraron notas directamente relevantes.\n"
            "Notas disponibles en el Vault:\n" + "\n".join(all_titles)
        )

    lines = []
    for score, title, excerpt in top_notes:
        lines.append(f"### {title}\n{excerpt}\n")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Gemini API — Graph-RAG
# ──────────────────────────────────────────────


async def _call_gemini(prompt: str, context: str) -> tuple[str, str]:
    if not GEMINI_API_KEY:
        return _local_response(prompt, context), "graph-local"
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        system_prompt = (
            "Eres el asistente experto del Obsidian & Git Intelligence Center, "
            "un sistema de gestión de conocimiento basado en Obsidian Vault. "
            "Las notas del Vault están conectadas como un knowledge graph. "
            "Respondes en español de forma clara, precisa y bien estructurada. "
            "Usas el contexto de las notas para fundamentar tus respuestas. "
            "Si el contexto no es suficiente, sugieres términos alternativos basados en las notas del Vault.\n\n"
            f"Contexto de las notas del Vault:\n{context}"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_prompt}\n\nPregunta del usuario: {prompt}",
        )
        return response.text, "gemini-api"
    except Exception as e:
        local = _local_response(prompt, context)
        return f"{local}\n\n⚠️ _Gemini no disponible ({type(e).__name__}). Mostrando resultado local._", "gemini-fallback"


def _local_response(prompt: str, context: str) -> str:
    if "No se encontraron notas" in context:
        return (
            f'No encontré notas relevantes para: **"{prompt}"**.\n\n'
            f'_Configura GEMINI_API_KEY para respuestas generativas._'
        )
    return (
        f"Encontré contexto en el Vault para: **{prompt}**\n\n"
        f"{context}\n\n"
        f"_Contexto del Vault local. Con Gemini activo la respuesta será generativa._"
    )


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@app.get("/")
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH, media_type="text/html")
    return JSONResponse({"error": "index.html no encontrado"}, status_code=404)


@app.get("/api/health")
async def health_check():
    graph = parse_vault()
    return {
        "status": "ok",
        "version": "5.0.0",
        "mode": "obsidian-git-center",
        "vault_dir": str(VAULT_DIR),
        "vault_exists": VAULT_DIR.exists(),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "commit_log_size": len(commit_log),
        "sse_clients_connected": len(sse_clients),
        "gemini_configured": bool(GEMINI_API_KEY),
    }


@app.get("/api/graph-data")
async def get_graph_data():
    """
    Devuelve nodos y conexiones del Vault Obsidian
    para renderizar con vis.js en el frontend.
    """
    return parse_vault()


@app.get("/api/notes/{note_id}")
async def get_note(note_id: str):
    """
    Devuelve el contenido de una nota del Vault.
    Busca el archivo .md correspondiente usando el ID del nodo
    (que coincide con el label limpio del archivo).
    """
    if not VAULT_DIR.exists():
        return JSONResponse({"error": "Vault no encontrado"}, status_code=404)

    # Buscar el nodo en parse_vault para obtener el label real
    graph = parse_vault()
    label = None
    for node in graph.get("nodes", []):
        if str(node["id"]) == str(note_id):
            label = node["label"]
            break

    if label is None:
        return JSONResponse({"error": f"Nota con ID '{note_id}' no encontrada"}, status_code=404)

    # Reconstruir el nombre de archivo a partir del label
    # Label "01 Ejercicios Pecho" → buscar archivo que matchee
    label_lower = label.lower()

    for filepath in VAULT_DIR.rglob("*.md"):
        # Reconstruir cómo parse_vault limpia el nombre
        clean = filepath.name
        if clean.lower().endswith(".md"):
            clean = clean[:-3]
        clean = clean.replace("_", " ").replace("-", " ")
        clean = re.sub(r"\s+", " ", clean).strip()

        if clean.lower() == label_lower:
            try:
                content = filepath.read_text(encoding="utf-8")
                return {"title": label, "content": content}
            except (OSError, UnicodeDecodeError):
                return JSONResponse({"error": "Error leyendo la nota"}, status_code=500)

    return JSONResponse({"error": f"Archivo para '{label}' no encontrado en vault/"}, status_code=404)


@app.get("/api/activity")
async def get_activity():
    return [item.model_dump() for item in reversed(commit_log)]


# ── Webhook de GitHub — Commits reales ─────────

@app.post("/api/webhook/github")
async def webhook_github(request: Request):
    """
    Receptor REAL de GitHub Webhooks.
    Recibe el payload cuando haces 'git push' y extrae:
      - Repositorio
      - Autor del commit
      - Mensaje del commit
      - Hora
    Envía inmediatamente al frontend vía SSE.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Payload JSON inválido"}, status_code=400)

    event = request.headers.get("x-github-event", "push")
    if event != "push":
        return {"status": "skipped", "reason": f"Evento '{event}' ignorado. Solo 'push'."}

    commits = payload.get("commits", [])
    if not commits:
        return {"status": "skipped", "reason": "Push sin commits."}

    repo_name = payload.get("repository", {}).get("name", "unknown")
    items_created = []

    for commit in commits:
        # Extraer datos reales del commit
        author = (
            commit.get("author", {}).get("name")
            or payload.get("pusher", {}).get("name")
            or "unknown"
        )
        message = commit.get("message", "Sin mensaje")[:120]
        timestamp = commit.get("timestamp", datetime.now(timezone.utc).isoformat())
        impact = _infer_impact(message)

        item = CommitItem(
            id=str(uuid.uuid4())[:8],
            author=author,
            author_avatar=_avatar(author),
            repo=repo_name,
            message=message,
            timestamp=timestamp,
            impact=impact,
        )
        commit_log.append(item)
        if len(commit_log) > MAX_LOG:
            commit_log.pop(0)
        _broadcast(item.model_dump())
        items_created.append(item.model_dump())

    return {
        "status": "ok",
        "repo": repo_name,
        "commits_processed": len(items_created),
        "items": items_created,
        "sse_clients": len(sse_clients),
    }


@app.get("/api/stream")
async def stream_activity(request: Request):
    from sse_starlette.sse import EventSourceResponse

    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    sse_clients.append(queue)

    async def event_generator() -> AsyncGenerator:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {"event": "activity", "data": data}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            if queue in sse_clients:
                sse_clients.remove(queue)

    return EventSourceResponse(event_generator())


@app.post("/api/chat")
async def chat_endpoint(payload: ChatMessage):
    """
    Graph-RAG: usa las notas del Vault Obsidian como contexto
    para que Gemini 1.5 Flash responda preguntas sobre el proyecto.
    """
    context = _get_vault_text_context(payload.message)
    response_text, source = await _call_gemini(payload.message, context)
    graph = parse_vault()
    return {
        "response": response_text,
        "source": source,
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_edges": len(graph.get("edges", [])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

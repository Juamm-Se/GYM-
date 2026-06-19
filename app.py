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

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — RUTA DEL OBSIDIAN VAULT
# ═══════════════════════════════════════════════════════════════════
#  Para cambiar la bóveda que lee el sistema, modifica esta variable:
#
#    OPCION 1 — Ruta absoluta a tu bóveda de Obsidian:
#      OBSIDIAN_VAULT_PATH = Path(r"C:\Users\sebas\Documents\MiVault")
#
#    OPCION 2 — Variable de entorno (recomendado para producción):
#      OBSIDIAN_VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\sebas\OneDrive - SENA\Escritorio\Practica_Obsidian"))
#
#    OPCION 3 — Carpeta local dentro del proyecto:
#      OBSIDIAN_VAULT_PATH = BASE_DIR / "vault"
#
#  ⚠️ TODOS los endpoints (grafo, notas, chat, sugerencias) usan
#     esta única variable. Cámbiala aquí y todo se actualiza.
# ═══════════════════════════════════════════════════════════════════
BASE_DIR             = Path(__file__).resolve().parent
OBSIDIAN_VAULT_PATH  = Path(os.getenv(
    "OBSIDIAN_VAULT_PATH",
    r"C:\Users\sebas\OneDrive - SENA\Escritorio\Practica_Obsidian"
))
FRONTEND_PATH        = BASE_DIR / "index.html"

# ═══════════════════════════════════════════════════
#  CONFIGURACIÓN — API KEY DE GEMINI
# ═══════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   # ← Set via env var
GEMINI_MODEL   = "gemini-2.5-flash"
# ═══════════════════════════════════════════════════

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
# Almacén persistente + Broadcast SSE
# ──────────────────────────────────────────────

COMMIT_LOG_FILE = BASE_DIR / "commits.json"
commit_log: list[CommitItem] = []
MAX_LOG = 200
sse_clients: list[asyncio.Queue] = []


def _load_commits() -> None:
    """Carga commits desde commits.json al iniciar (persistencia).
    Ordena por timestamp descendente para garantizar 'más recientes primero'.
    """
    if COMMIT_LOG_FILE.exists():
        try:
            data = json.loads(COMMIT_LOG_FILE.read_text(encoding="utf-8"))
            items = [CommitItem(**item) for item in data[:MAX_LOG]]
            items.sort(key=lambda x: x.timestamp, reverse=True)
            commit_log.extend(items)
        except (json.JSONDecodeError, OSError, Exception):
            pass  # Archivo corrupto o vacío, iniciar limpio


def _save_commits() -> None:
    """Guarda todos los commits a commits.json (persistencia)."""
    try:
        data = [item.model_dump() for item in commit_log]
        COMMIT_LOG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # No bloquear si falla la escritura


# Cargar commits al importar el módulo
_load_commits()


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

    if not OBSIDIAN_VAULT_PATH.exists():
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
    md_files = sorted(OBSIDIAN_VAULT_PATH.rglob("*.md"))
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
    if not OBSIDIAN_VAULT_PATH.exists():
        return "No se encontró la carpeta 'vault/'."

    tokens = set(re.findall(r"\w+", query.lower()))
    scored_notes: list[tuple[int, str, str]] = []  # (score, title, excerpt)

    for filepath in OBSIDIAN_VAULT_PATH.rglob("*.md"):
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
        for filepath in OBSIDIAN_VAULT_PATH.rglob("*.md"):
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
        "vault_dir": str(OBSIDIAN_VAULT_PATH),
        "vault_exists": OBSIDIAN_VAULT_PATH.exists(),
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


@app.get("/api/vault-suggestions")
async def get_vault_suggestions():
    """
    Devuelve los nombres de las primeras 6 notas del Vault
    para usar como chips de sugerencia dinámicos en el chat.
    Todo depende de OBSIDIAN_VAULT_PATH.
    """
    if not OBSIDIAN_VAULT_PATH.exists():
        return {"suggestions": [], "vault_name": "No encontrado"}

    suggestions = []
    for filepath in sorted(OBSIDIAN_VAULT_PATH.rglob("*.md"))[:6]:
        clean = filepath.name
        if clean.lower().endswith(".md"):
            clean = clean[:-3]
        clean = clean.replace("_", " ").replace("-", " ")
        clean = re.sub(r"\s+", " ", clean).strip()
        suggestions.append(clean)

    vault_name = OBSIDIAN_VAULT_PATH.name
    return {"suggestions": suggestions, "vault_name": vault_name}


@app.get("/api/notes/{note_id}")
async def get_note(note_id: str):
    """
    Devuelve el contenido de una nota del Vault.
    Busca el archivo .md correspondiente usando el ID del nodo
    (que coincide con el label limpio del archivo).
    """
    if not OBSIDIAN_VAULT_PATH.exists():
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

    for filepath in OBSIDIAN_VAULT_PATH.rglob("*.md"):
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
    return [item.model_dump() for item in commit_log]


@app.get("/api/commits")
async def get_commits(page: int = 1, limit: int = 20, search: str = ""):
    """
    Devuelve commits paginados (más recientes primero).
    Query params: ?page=1&limit=20&search=keyword
    Busca en mensaje, autor y repositorio.
    """
    filtered = commit_log
    if search:
        search_lower = search.lower()
        filtered = [
            c for c in commit_log
            if search_lower in c.message.lower()
            or search_lower in c.author.lower()
            or search_lower in c.repo.lower()
        ]
    total = len(filtered)
    total_pages = max(1, (total + limit - 1) // limit)
    page = max(1, min(page, total_pages))
    start = (page - 1) * limit
    end = start + limit
    items = filtered[start:end]
    return {
        "commits": [item.model_dump() for item in items],
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
        "search": search,
    }


@app.get("/api/commit/{commit_id}")
async def get_commit_detail(commit_id: str):
    """
    Devuelve el detalle completo de un commit específico por su ID.
    Incluye info expandida y sugerencias contextuales.
    """
    for item in commit_log:
        if item.id == commit_id:
            # Contar commits del mismo autor y repo
            same_author = sum(1 for c in commit_log if c.author == item.author)
            same_repo = sum(1 for c in commit_log if c.repo == item.repo)
            # Descripción del impacto
            impact_desc = {
                "Alto": "Este commit introduce cambios significativos que pueden afectar la arquitectura, eliminar funcionalidad o requerir migración.",
                "Medio": "Este commit modifica o mejora funcionalidad existente sin cambios arquitectónicos mayores.",
                "Bajo": "Este commit realiza cambios menores: documentación, estilo, o ajustes menores.",
            }
            return {
                **item.model_dump(),
                "same_author_commits": same_author,
                "same_repo_commits": same_repo,
                "impact_description": impact_desc.get(item.impact, ""),
                "time_ago": _time_ago(item.timestamp),
            }
    return JSONResponse({"error": f"Commit '{commit_id}' no encontrado"}, status_code=404)


def _time_ago(ts: str) -> str:
    """Devuelve un string legible del tipo 'hace 5 minutos'."""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = datetime.now(timezone.utc) - dt
        seconds = int(diff.total_seconds())
        if seconds < 60: return "hace un momento"
        if seconds < 3600: return f"hace {seconds // 60} min"
        if seconds < 86400: return f"hace {seconds // 3600} h"
        return f"hace {seconds // 86400} días"
    except Exception:
        return ""


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
        commit_log.insert(0, item)
        if len(commit_log) > MAX_LOG:
            commit_log.pop()
        _broadcast(item.model_dump())
        _save_commits()
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

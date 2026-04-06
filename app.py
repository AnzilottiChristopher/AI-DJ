"""
AI DJ Backend - Production-ready FastAPI server.

Environment variables:
    AIDJ_ENV      "development" or "production" (default: development)
    AIDJ_DOMAIN   Your domain, e.g. "ai-dj.duckdns.org"
    AIDJ_HOST     Bind address (default: 127.0.0.1 in prod, 0.0.0.0 in dev)
    AIDJ_PORT     Port number (default: 8000)

Usage:
    Development:  python app.py
    Production:   set AIDJ_ENV=production && python -m uvicorn app:app --host 127.0.0.1 --port 8000
"""

import os
import sys
import io
import json
import time
import secrets
import uuid
from dataclasses import dataclass, field
from fastapi import FastAPI, WebSocket, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import subprocess
import atexit

from llm.new_llm import LlamaLLM
from music_library import MusicLibrary
from enhanced_audio_manager import AudioManager, PlaybackState
from upload_handler import router as upload_router
from database import init_db, get_connection
from auth import router as auth_router, decode_token
from setlists import router as setlists_router
from feature_utils import sanitize_song_features
from song_similarity import parse_title_artist_from_filename
from typing import Optional, Any
from user_song_maintenance import repair_incomplete_user_song_features

import uvicorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENVIRONMENT = os.getenv("AIDJ_ENV", "development")
DOMAIN = os.getenv("AIDJ_DOMAIN", "ai-dj.duckdns.org")
HOST = os.getenv("AIDJ_HOST", "127.0.0.1" if ENVIRONMENT == "production" else "0.0.0.0")
PORT = int(os.getenv("AIDJ_PORT", "8000"))

app = FastAPI(title="AI DJ Backend")

# ---------------------------------------------------------------------------
# Core services
# ---------------------------------------------------------------------------
music_library = MusicLibrary('music_data/audio', 'music_data/segmented_songs.json')

# Single shared LLM — Ollama serializes anyway; semaphore enforces the queue.
llm = LlamaLLM(music_library=music_library)
llm_semaphore = asyncio.Semaphore(1)

MODEL_PATH = 'models/dj_transition_model'

# ---------------------------------------------------------------------------
# Host rooms + pairing tokens
# ---------------------------------------------------------------------------
PAIRING_TTL_SECONDS = 60 * 60 * 2


@dataclass
class DJRoom:
    room_id: str
    audio_manager: AudioManager
    user_id: Optional[int]
    host_socket: Any = None
    playback_task: Optional[asyncio.Task] = None
    has_started_playback: bool = False
    command_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.time)


@dataclass
class PairingToken:
    token: str
    room_id: str
    user_id: Optional[int]
    expires_at: float
    created_at: float = field(default_factory=time.time)


_rooms: dict[str, DJRoom] = {}
_pairings: dict[str, PairingToken] = {}
_rooms_lock = asyncio.Lock()


def _make_audio_manager():
    return AudioManager(
        music_library,
        model_path=MODEL_PATH,
        enable_auto_play=True,
    )


async def _purge_expired_pairings():
    now = time.time()
    async with _rooms_lock:
        expired = [token for token, pairing in _pairings.items() if pairing.expires_at <= now]
        for token in expired:
            _pairings.pop(token, None)


async def _create_room(user_id: Optional[int]) -> DJRoom:
    room_id = f"room_{uuid.uuid4().hex[:8]}"
    room = DJRoom(
        room_id=room_id,
        audio_manager=_make_audio_manager(),
        user_id=user_id,
    )
    async with _rooms_lock:
        _rooms[room_id] = room
    print(f"[ROOM] Created: {room_id} ({len(_rooms)} active)")
    return room


async def _get_room(room_id: str) -> Optional[DJRoom]:
    async with _rooms_lock:
        return _rooms.get(room_id)


async def _destroy_room(room_id: str):
    async with _rooms_lock:
        room = _rooms.pop(room_id, None)
        stale_tokens = [token for token, pairing in _pairings.items() if pairing.room_id == room_id]
        for token in stale_tokens:
            _pairings.pop(token, None)

    if not room:
        return

    if room.playback_task and not room.playback_task.done():
        room.playback_task.cancel()
        try:
            await room.playback_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[ROOM] Error cancelling playback task for {room_id}: {exc}")

    try:
        room.audio_manager.stop()
    except Exception:
        pass

    print(f"[ROOM] Destroyed: {room_id} ({len(_rooms)} active)")


async def _create_pairing_token(room: DJRoom) -> PairingToken:
    await _purge_expired_pairings()

    while True:
        token = secrets.token_urlsafe(18)
        async with _rooms_lock:
            if token in _pairings:
                continue
            pairing = PairingToken(
                token=token,
                room_id=room.room_id,
                user_id=room.user_id,
                expires_at=time.time() + PAIRING_TTL_SECONDS,
            )
            _pairings[token] = pairing
            return pairing


async def _get_pairing_token(token: str) -> PairingToken:
    await _purge_expired_pairings()
    async with _rooms_lock:
        pairing = _pairings.get(token)
    if not pairing:
        raise HTTPException(status_code=404, detail="Remote pairing link is invalid or expired")
    return pairing


def _serialize_track(item):
    """Best-effort conversion of whatever the queue stores into a UI-friendly object."""
    if item is None:
        return None
    if isinstance(item, dict):
        title = item.get("title") or item.get("name") or item.get("track") or item.get("filename")
        artist = item.get("artist") or item.get("artist_name") or item.get("by")
        payload = {"title": title, "artist": artist}
        if "is_auto_queued" in item:
            payload["is_auto_queued"] = item.get("is_auto_queued")
        return payload
    if hasattr(item, "title") or hasattr(item, "artist"):
        return {
            "title": getattr(item, "title", None),
            "artist": getattr(item, "artist", None),
            "is_auto_queued": getattr(item, "is_auto_queued", None),
        }
    return {"title": str(item), "artist": None}


def _room_state_payload(room: DJRoom, extra: Optional[dict] = None):
    status = room.audio_manager.get_queue_status()
    raw_upcoming = (
        status.get("queue")
        or status.get("upcoming")
        or status.get("upcoming_songs")
        or status.get("next_up")
        or status.get("upcomingQueue")
        or []
    )
    try:
        if hasattr(raw_upcoming, "_queue"):
            raw_upcoming = list(raw_upcoming._queue)
    except Exception:
        pass
    if raw_upcoming is None:
        raw_upcoming = []

    upcoming_tracks = []
    try:
        for item in raw_upcoming:
            serialized = _serialize_track(item)
            if serialized:
                upcoming_tracks.append(serialized)
    except Exception:
        serialized = _serialize_track(raw_upcoming)
        if serialized:
            upcoming_tracks = [serialized]

    payload = {
        "room_id": room.room_id,
        "host_connected": room.host_socket is not None,
        "queue_status": status,
        "current_track": _serialize_track(status.get("current_track")),
        "upcoming": upcoming_tracks,
        "queue": upcoming_tracks,
        "upcoming_count": len(upcoming_tracks),
    }
    if extra:
        payload.update(extra)
    return payload


async def _send_host_json(room: DJRoom, data: dict):
    if not room.host_socket:
        return
    try:
        await room.host_socket.send_json(data)
    except Exception as exc:
        print(f"[ROOM] Failed to send host payload for {room.room_id}: {exc}")


async def _send_host_queue_update(room: DJRoom, extra: Optional[dict] = None):
    await _send_host_json(room, _room_state_payload(room, extra))


async def _sync_transition_to_host(room: DJRoom):
    if room.audio_manager.pending_transition:
        await _send_host_json(room, {
            "type": "transition_planned",
            "transition": room.audio_manager.pending_transition.to_dict(),
        })


async def _ensure_room_playback(room: DJRoom):
    audio_manager = room.audio_manager

    if not room.host_socket or not audio_manager.queue:
        return

    if not audio_manager.is_playing or not room.has_started_playback:
        if audio_manager.is_playing and not room.has_started_playback:
            print("[ROOM] Backend playing but host restarted - forcing clean restart")
            audio_manager.stop()
            audio_manager.current_track = None
            audio_manager.start()
        elif not audio_manager.is_playing:
            if not room.has_started_playback and audio_manager.current_track:
                audio_manager.current_track = None
            audio_manager.start()

        room.playback_task = asyncio.create_task(
            audio_manager.play_queue(room.host_socket)
        )
        room.has_started_playback = True


async def _reset_room(room: DJRoom):
    audio_manager = room.audio_manager
    print(f"[ROOM] Reset requested for {room.room_id}")
    audio_manager.stop()
    audio_manager.queue.clear()
    audio_manager.current_track = None
    audio_manager.pending_transition = None
    audio_manager.transition_audio = None
    audio_manager._pending_transition_for = None
    audio_manager.samples_sent = 0
    audio_manager.current_position = 0.0
    audio_manager.state = PlaybackState.STOPPED

    if room.playback_task and not room.playback_task.done():
        room.playback_task.cancel()
        try:
            await room.playback_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[ROOM] Error cancelling playback during reset: {exc}")

    room.playback_task = None
    room.has_started_playback = False


async def _load_setlist_into_room(room: DJRoom, setlist_id, session_user_id: Optional[int]):
    if not session_user_id:
        return {
            "type": "error",
            "message": "Authentication required to load setlists",
        }

    conn = get_connection()
    setlist_row = conn.execute(
        "SELECT id, name FROM setlists WHERE id = ? AND user_id = ?",
        (setlist_id, session_user_id),
    ).fetchone()

    if not setlist_row:
        conn.close()
        return {
            "type": "error",
            "message": "Setlist not found",
        }

    items = conn.execute(
        "SELECT song_key, song_title, song_artist FROM setlist_items "
        "WHERE setlist_id = ? ORDER BY position",
        (setlist_id,),
    ).fetchall()
    conn.close()

    if not items:
        return {
            "type": "error",
            "message": "Setlist is empty",
        }

    queued_songs = []
    failed_songs = []
    for item in items:
        title = item["song_title"]
        artist = item["song_artist"]
        song_key = item["song_key"]
        lib_entry = music_library.index.get(song_key)
        if lib_entry:
            title, artist = parse_title_artist_from_filename(lib_entry['filename'])

        found = room.audio_manager.add_to_queue(title, artist)
        if found:
            queued_songs.append(title)
        else:
            failed_songs.append(title)

    await _ensure_room_playback(room)
    return _room_state_payload(room, {
        "type": "queue_update",
        "action": "setlist_loaded",
        "message": f"Loaded setlist '{setlist_row['name']}': {len(queued_songs)} songs queued",
        "queued": queued_songs,
        "failed": failed_songs,
    })


def _quick_transition_messages(room: DJRoom):
    try:
        result = room.audio_manager.force_quick_transition()
        if result == 'quick':
            return [{
                "type": "quick_transition_scheduled",
                "message": "Transitioning at next segment boundary...",
            }]
        if result == 'immediate':
            return [{
                "type": "quick_transition_scheduled",
                "message": "Skipping immediately...",
            }]
        return [{
            "type": "error",
            "message": "No song queued to transition to",
        }]
    except Exception as exc:
        return [{
            "type": "error",
            "message": f"Could not schedule quick transition: {str(exc)}",
        }]


async def _process_prompt_command(room: DJRoom, prompt: str, session_user_id: Optional[int]):
    loop = asyncio.get_event_loop()
    async with llm_semaphore:
        result = await loop.run_in_executor(None, llm.classify, prompt)

    intent = result['intent']
    song_info = result['song']

    if intent == 'queue_song':
        found = room.audio_manager.add_to_queue(
            song_info['title'],
            song_info['artist'],
        )
        if not found:
            return [{
                "type": "error",
                "message": f"Song not found: {song_info['title']}",
            }]

        await _ensure_room_playback(room)

        messages = [_room_state_payload(room, {
            "type": "queue_update",
            "action": "added",
            "message": f"Queued: {song_info['title']}",
        })]
        if room.audio_manager.pending_transition:
            messages.append({
                "type": "transition_planned",
                "transition": room.audio_manager.pending_transition.to_dict(),
            })
        return messages

    if intent == 'generate_playlist':
        async with llm_semaphore:
            playlist = await loop.run_in_executor(None, llm.generate_playlist, prompt, session_user_id)

        if not playlist:
            return [{
                "type": "error",
                "message": "Could not generate playlist from your request",
            }]

        queued_songs = []
        failed_songs = []
        for song in playlist:
            found = room.audio_manager.add_to_queue(
                song['title'],
                song.get('artist'),
            )
            if found:
                queued_songs.append(song['title'])
            else:
                failed_songs.append(song['title'])

        await _ensure_room_playback(room)
        return [_room_state_payload(room, {
            "type": "queue_update",
            "action": "playlist_generated",
            "message": f"Queued {len(queued_songs)} songs",
            "queued": queued_songs,
            "failed": failed_songs,
        })]

    if intent == 'play_setlist':
        setlist_name = (song_info.get('title') or '').strip()
        if not session_user_id:
            return [{
                "type": "error",
                "message": "You need to be logged in to load setlists",
            }]
        if not setlist_name:
            return [{
                "type": "error",
                "message": "I couldn't determine which setlist you want. Try saying 'play setlist <name>'",
            }]

        conn = get_connection()
        setlist_row = conn.execute(
            "SELECT id, name FROM setlists WHERE user_id = ? AND LOWER(name) = LOWER(?)",
            (session_user_id, setlist_name),
        ).fetchone()

        if not setlist_row:
            conn.close()
            return [{
                "type": "error",
                "message": f"Setlist '{setlist_name}' not found",
            }]

        items = conn.execute(
            "SELECT song_key, song_title, song_artist FROM setlist_items "
            "WHERE setlist_id = ? ORDER BY position",
            (setlist_row['id'],),
        ).fetchall()
        conn.close()

        if not items:
            return [{
                "type": "error",
                "message": f"Setlist '{setlist_row['name']}' is empty",
            }]

        queued_songs = []
        failed_songs = []
        for item in items:
            title = item["song_title"]
            artist = item["song_artist"]
            song_key = item["song_key"]
            lib_entry = music_library.index.get(song_key)
            if lib_entry:
                title, artist = parse_title_artist_from_filename(lib_entry['filename'])

            found = room.audio_manager.add_to_queue(title, artist)
            if found:
                queued_songs.append(title)
            else:
                failed_songs.append(title)

        await _ensure_room_playback(room)
        return [_room_state_payload(room, {
            "type": "queue_update",
            "action": "setlist_loaded",
            "message": f"Loaded setlist '{setlist_row['name']}': {len(queued_songs)} songs queued",
            "queued": queued_songs,
            "failed": failed_songs,
        })]

    if intent == 'quick_transition':
        return _quick_transition_messages(room)

    if intent == 'set_transition_mode':
        prompt_lower = (prompt or '').lower()
        if any(keyword in prompt_lower for keyword in ["classic", "crossfade", "standard"]):
            requested_mode = "classic"
        elif any(keyword in prompt_lower for keyword in ["dynamic", "warp"]):
            requested_mode = "dynamic"
        else:
            requested_mode = "dynamic"

        success = room.audio_manager.set_transition_mode(requested_mode)
        if success:
            return [_room_state_payload(room, {
                "type": "transition_mode_updated",
                "message": f"Transition mode set to {room.audio_manager.get_transition_mode()}",
                "transition_mode": room.audio_manager.get_transition_mode(),
            })]
        return [{
            "type": "error",
            "message": "Could not update transition mode. Use dynamic or classic.",
        }]

    if intent == 'stop_dj':
        room.audio_manager.stop()
        return [{
            "type": "stopped",
            "message": "Playback stopped",
        }]

    if intent == 'hello':
        return [{
            "type": "greeting",
            "message": "Hey! I'm your AI DJ. Tell me what you want to hear!",
        }]

    if intent == 'help':
        return [{
            "type": "help",
            "message": "You can say things like: 'Play Wake Me Up by Avicii', 'Queue Stargazing', 'Stop the music'. I'll automatically queue similar songs to keep the music going!",
        }]

    return [{
        "type": "unknown",
        "message": "I didn't quite catch that. Try asking me to play a song!",
    }]


async def _process_room_command(room: DJRoom, data: dict, session_user_id: Optional[int], *, is_host: bool):
    async with room.command_lock:
        msg_type = data.get('type')

        if msg_type == 'reorder_queue':
            new_order = data.get('order', [])
            success = room.audio_manager.reorder_queue(new_order)
            if not success:
                return [{
                    "type": "error",
                    "message": "Could not reorder queue",
                }]
            messages = [_room_state_payload(room, {
                "type": "queue_update",
                "action": "reordered",
                "message": "Queue reordered",
            })]
            if room.audio_manager.pending_transition:
                messages.append({
                    "type": "transition_planned",
                    "transition": room.audio_manager.pending_transition.to_dict(),
                })
            return messages

        if msg_type == 'set_transition_mode':
            requested_mode = data.get('mode', '')
            success = room.audio_manager.set_transition_mode(requested_mode)
            if success:
                return [_room_state_payload(room, {
                    "type": "transition_mode_updated",
                    "message": f"Transition mode set to {room.audio_manager.get_transition_mode()}",
                    "transition_mode": room.audio_manager.get_transition_mode(),
                })]
            return [{
                "type": "error",
                "message": "Invalid transition mode. Use 'dynamic' or 'classic'.",
            }]

        if msg_type == 'remove_from_queue':
            index = data.get('index')
            if not isinstance(index, int):
                return [{
                    "type": "error",
                    "message": "Queue index is required",
                }]

            success = room.audio_manager.remove_from_queue(index)
            if not success:
                return [{
                    "type": "error",
                    "message": "Could not remove song from queue",
                }]

            messages = [_room_state_payload(room, {
                "type": "queue_update",
                "action": "removed",
                "message": "Song removed from queue",
            })]
            if room.audio_manager.pending_transition:
                messages.append({
                    "type": "transition_planned",
                    "transition": room.audio_manager.pending_transition.to_dict(),
                })
            return messages

        if msg_type == 'quick_transition':
            return _quick_transition_messages(room)

        if msg_type == 'pause':
            room.audio_manager.pause()
            return [{"type": "paused"}]

        if msg_type == 'resume':
            room.audio_manager.resume()
            return [{"type": "resumed"}]

        if msg_type == 'reset':
            if not is_host:
                return [{
                    "type": "error",
                    "message": "Reset is only available on the desktop host",
                }]
            await _reset_room(room)
            return [{"type": "reset_complete"}]

        if msg_type == 'create_remote_pairing':
            if not is_host:
                return [{
                    "type": "error",
                    "message": "Only the desktop host can create a pairing link",
                }]
            pairing = await _create_pairing_token(room)
            return [{
                "type": "remote_pairing_created",
                "pairing_token": pairing.token,
                "expires_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(pairing.expires_at)),
            }]

        if msg_type == 'load_setlist':
            return [await _load_setlist_into_room(room, data.get('setlist_id'), session_user_id)]

        prompt = data.get('data', '')
        print(f"[ROOM {room.room_id}] Received prompt: {prompt}")
        try:
            return await _process_prompt_command(room, prompt, session_user_id)
        except Exception as exc:
            print(f"[ERROR] LLM processing: {exc}")
            return [{
                "type": "error",
                "message": f"Error processing request: {str(exc)}",
            }]


# Shared mixer + similarity service for REST endpoints (setlist previews etc.)
# These don't need per-user state — they only use the ML model and music_library.
_shared_manager = _make_audio_manager()

# ---------------------------------------------------------------------------
# CORS - automatic based on environment
# ---------------------------------------------------------------------------
if ENVIRONMENT == "production":
    CORS_ORIGINS = [
        f"https://{DOMAIN}",
        f"https://www.{DOMAIN}",
    ]
else:
    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(auth_router)
app.include_router(setlists_router)

# Initialize database on startup
init_db()

# ---------------------------------------------------------------------------
# Ollama lifecycle (dev only - in prod, Ollama runs as its own service)
# ---------------------------------------------------------------------------
ollama_process = None


def _load_user_songs_into_library():
    """Load all user-uploaded songs from DB into the in-memory library at startup."""
    conn = get_connection()
    rows = conn.execute("SELECT user_id, song_data FROM user_songs").fetchall()
    conn.close()
    count = 0
    for row in rows:
        try:
            song_data = json.loads(row["song_data"])
            music_library.add_user_song_hot(song_data, row["user_id"])
            count += 1
        except Exception as e:
            print(f"[STARTUP] Failed to load user song: {e}")
    if count:
        print(f"[STARTUP] Loaded {count} user song(s) into library")


@app.on_event("startup")
async def startup_event():
    global ollama_process

    repaired_rows = repair_incomplete_user_song_features()
    if repaired_rows:
        print(f"[STARTUP] Repaired {repaired_rows} user song(s) with incomplete features")

    # Load user songs from DB into the in-memory library
    _load_user_songs_into_library()

    if ENVIRONMENT == "development":
        print("Starting Ollama server...")
        try:
            ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await asyncio.sleep(2)
            print("Ollama server started successfully")
        except Exception as e:
            print(f"Warning: Could not start Ollama automatically: {e}")
    else:
        print(f"[PROD] Backend starting on {HOST}:{PORT}")
        print(f"[PROD] Domain: {DOMAIN}")
        print(f"[PROD] CORS origins: {CORS_ORIGINS}")
        print(f"[PROD] Expecting Ollama on localhost:11434")


@app.on_event("shutdown")
async def shutdown_event():
    global ollama_process
    if ollama_process:
        print("Shutting down Ollama server...")
        ollama_process.terminate()
        ollama_process.wait()
        print("Ollama server stopped")


def cleanup():
    global ollama_process
    if ollama_process:
        ollama_process.terminate()
        ollama_process.wait()


atexit.register(cleanup)


# ---------------------------------------------------------------------------
# Health check (useful for monitoring)
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    ollama_ok = False
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=3)
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "environment": ENVIRONMENT,
        "ollama": "connected" if ollama_ok else "disconnected",
        "audio_manager": f"{len(_rooms)} active rooms",
        "library_size": len(music_library.index),
    }


# ---------------------------------------------------------------------------
# WebSocket - audio streaming
# ---------------------------------------------------------------------------
@app.websocket("/api/ws/audio")
async def audio_stream(websocket: WebSocket, token: Optional[str] = Query(None)):
    """
    Main WebSocket endpoint for audio streaming.

    Handles:
    - Song queue requests via LLM
    - Audio streaming with transitions
    - Auto-play of similar songs
    - Playback status updates
    """
    await websocket.accept()

    # Resolve user identity from the optional JWT token query param
    session_user_id: Optional[int] = None
    if token:
        try:
            payload = decode_token(token)
            session_user_id = int(payload["sub"])
            print(f"[WS] Authenticated user {session_user_id}")
        except Exception:
            print("[WS] Invalid token provided; treating as guest")

    # Each host WebSocket connection owns one DJ room
    room = await _create_room(session_user_id)
    audio_manager = room.audio_manager
    print(f"[WS] Room: {room.room_id}")

    # FIX FOR AUDIO CUTOUTS: Create a WebSocket wrapper with send locking
    class LockedWebSocket:
        """WebSocket wrapper that prevents send contention between tasks."""
        def __init__(self, ws):
            self._ws = ws
            self._send_lock = asyncio.Lock()
            self._log_buffer = []

        async def send_json(self, data):
            async with self._send_lock:
                await self._ws.send_json(data)

        async def send_bytes(self, data):
            async with self._send_lock:
                await self._ws.send_bytes(data)

        def buffer_log(self, line: str):
            """Buffer a log line for later sending."""
            self._log_buffer.append(line)

        async def flush_logs(self):
            """Send buffered logs to the frontend."""
            if not self._log_buffer:
                return
            lines = self._log_buffer[:]
            self._log_buffer.clear()
            try:
                async with self._send_lock:
                    await self._ws.send_json({
                        "type": "backend_log",
                        "lines": lines,
                    })
            except Exception:
                pass

        async def receive_json(self):
            return await self._ws.receive_json()

        def __getattr__(self, name):
            return getattr(self._ws, name)

    locked_websocket = LockedWebSocket(websocket)

    # Intercept stdout to capture print() output and forward to frontend
    _original_stdout = sys.stdout

    class LogTee(io.TextIOBase):
        """Write to original stdout AND buffer for WebSocket forwarding."""
        def write(self, s):
            _original_stdout.write(s)
            if s.strip():
                locked_websocket.buffer_log(s.rstrip('\n'))
            return len(s)

        def flush(self):
            _original_stdout.flush()

    sys.stdout = LogTee()

    # Periodically flush logs to the frontend
    async def log_flusher():
        while True:
            await asyncio.sleep(0.5)
            await locked_websocket.flush_logs()

    log_flush_task = asyncio.create_task(log_flusher())

    def _serialize_track(item):
        """Best-effort conversion of whatever the queue stores into a UI-friendly object."""
        if item is None:
            return None
        if isinstance(item, dict):
            title = item.get("title") or item.get("name") or item.get("track") or item.get("filename")
            artist = item.get("artist") or item.get("artist_name") or item.get("by")
            return {"title": title, "artist": artist}
        if hasattr(item, "title") or hasattr(item, "artist"):
            return {"title": getattr(item, "title", None), "artist": getattr(item, "artist", None)}
        return {"title": str(item), "artist": None}

    def _queue_payload(extra: dict | None = None):
        status = audio_manager.get_queue_status()
        raw_upcoming = (
            status.get("queue")
            or status.get("upcoming")
            or status.get("upcoming_songs")
            or status.get("next_up")
            or status.get("upcomingQueue")
            or []
        )
        try:
            if hasattr(raw_upcoming, "_queue"):
                raw_upcoming = list(raw_upcoming._queue)
        except Exception:
            pass
        if raw_upcoming is None:
            raw_upcoming = []
        upcoming_tracks = []
        try:
            for x in raw_upcoming:
                s = _serialize_track(x)
                if s:
                    upcoming_tracks.append(s)
        except Exception:
            s = _serialize_track(raw_upcoming)
            if s:
                upcoming_tracks = [s]
        payload = {
            "queue_status": status,
            "upcoming": upcoming_tracks,
            "queue": upcoming_tracks,
            "upcoming_count": len(upcoming_tracks),
        }
        if extra:
            payload.update(extra)
        return payload

    room.host_socket = locked_websocket

    # Send an initial snapshot so the frontend can render the queue immediately
    try:
        await locked_websocket.send_json(_room_state_payload(room, {
            "type": "queue_snapshot",
        }))
    except Exception as e:
        print(f"[WS] initial queue_snapshot send failed: {e}")
        return

    message_handler_task = None

    # Move message handling to a separate async function
    # This allows it to run concurrently with audio playback
    async def handle_messages():
        """
        Handle incoming user messages without blocking audio playback.
        Runs as a separate concurrent task.
        """
        while True:
            try:
                data = await locked_websocket.receive_json()
            except Exception as e:
                print(f"[WS] receive_json failed / disconnected: {e}")
                break

            messages = await _process_room_command(
                room,
                data,
                session_user_id,
                is_host=True,
            )
            for payload in messages:
                await locked_websocket.send_json(payload)

    # Run message handler as a concurrent task instead of blocking
    try:
        message_handler_task = asyncio.create_task(handle_messages())
        print("[WS] Message handler task started")
        await message_handler_task

    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        print("[WS] Cleaning up WebSocket connection...")

        room.host_socket = None
        await _destroy_room(room.room_id)

        if message_handler_task and not message_handler_task.done():
            message_handler_task.cancel()
            try:
                await message_handler_task
            except asyncio.CancelledError:
                print("[CLEANUP] Message handler task cancelled")
            except Exception as e:
                print(f"[CLEANUP] Error cancelling message handler: {e}")

        # Restore original stdout and cancel log flusher
        sys.stdout = _original_stdout
        if log_flush_task and not log_flush_task.done():
            log_flush_task.cancel()
            try:
                await log_flush_task
            except asyncio.CancelledError:
                pass

        print("[WS] Cleanup complete")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """Get active session count. Per-user status is via WebSocket."""
    return {"active_rooms": len(_rooms)}


@app.get("/api/remote/session")
async def get_remote_session(pair: str = Query(...)):
    pairing = await _get_pairing_token(pair)
    room = await _get_room(pairing.room_id)

    if not room or not room.host_socket:
        raise HTTPException(status_code=404, detail="Desktop DJ session is no longer available")

    return _room_state_payload(room, {
        "pair_expires_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(pairing.expires_at)),
    })


@app.post("/api/remote/commands")
async def remote_command(payload: dict, pair: str = Query(...)):
    pairing = await _get_pairing_token(pair)
    room = await _get_room(pairing.room_id)

    if not room or not room.host_socket:
        raise HTTPException(status_code=404, detail="Desktop DJ session is no longer available")

    messages = await _process_room_command(
        room,
        payload,
        pairing.user_id,
        is_host=False,
    )

    for message in messages:
        should_forward = message.get("type") in {
            "queue_update",
            "transition_planned",
            "paused",
            "resumed",
            "quick_transition_scheduled",
            "transition_mode_updated",
            "stopped",
        }
        if should_forward:
            await _send_host_json(room, message)

    return {
        "ok": True,
        "messages": messages,
        "state": _room_state_payload(room, {
            "pair_expires_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(pairing.expires_at)),
        }),
    }


@app.get("/api/library")
async def get_library(token: Optional[str] = Query(None)):
    """Get available songs. Authenticated users also see their own uploads."""
    user_id = None
    if token:
        try:
            payload = decode_token(token)
            user_id = int(payload["sub"])
        except Exception:
            pass

    songs = []
    for key, data in music_library.index.items():
        owner = music_library._user_song_owner.get(key)
        if owner is not None and owner != user_id:
            continue  # skip other users' songs
        title, artist = parse_title_artist_from_filename(data['filename'])
        songs.append({
            "song_key": key,
            "title": title,
            "artist": artist,
            "filename": data['filename'],
            "bpm": data['features'].get('bpm'),
            "key": data['features'].get('key'),
            "is_user_song": owner == user_id,
            "segments": data.get('segments', []),
            "features": data.get('features', {}),
        })
    return {"songs": songs}


# ---------------------------------------------------------------------------
# Setlist helper endpoints (need access to music_library / _shared_manager)
# ---------------------------------------------------------------------------
from pydantic import BaseModel as _BaseModel
from fastapi import Depends as _Depends
from auth import get_current_user as _get_current_user
from fastapi.responses import Response
import numpy as np


class _BestTransitionRequest(_BaseModel):
    song_a_key: str
    song_b_key: str


class _BestTransitionBulkRequest(_BaseModel):
    song_keys: list[str]  # ordered list of song keys in the setlist


class _PreviewTransitionRequest(_BaseModel):
    song_a_key: str
    song_b_key: str
    exit_segment: str | None = None
    entry_segment: str | None = None


def _filter_exit_segments(segments_raw: list) -> list[str]:
    """Filter exit segments to the latter portion of the song.

    Mirrors the live DJ behaviour where transitions are computed after
    playback has passed the early segments.  We keep only segments whose
    start time is in the second half of the song (by time), which naturally
    excludes intro / early verse while keeping cool-down, outro, later
    choruses, etc.  Falls back to the last two segments if the filter would
    leave nothing.
    """
    if not segments_raw:
        return []
    last_end = max(s.get('end', s.get('start', 0)) for s in segments_raw)
    if last_end <= 0:
        return [s['name'] for s in segments_raw]
    midpoint = last_end / 2.0
    later = [s['name'] for s in segments_raw if s.get('start', 0) >= midpoint]
    if not later:
        # Fallback: use last two segments
        later = [s['name'] for s in segments_raw[-2:]]
    return later


@app.post("/api/setlists/best-transition")
async def best_transition(req: _BestTransitionRequest, user: dict = _Depends(_get_current_user)):
    """Use the ML model to find the best exit/entry segments for a song pair."""
    song_a = music_library.index.get(req.song_a_key)
    song_b = music_library.index.get(req.song_b_key)

    if not song_a or not song_b:
        raise HTTPException(status_code=404, detail="One or both songs not found in library")

    segments_a_raw = song_a.get('segments', [])
    segments_b = [s['name'] for s in song_b.get('segments', [])]

    # Filter exit segments to the latter half of song A, matching the live
    # DJ's position-based filtering that naturally skips intro/early verse.
    exit_segments = _filter_exit_segments(segments_a_raw)

    if not exit_segments or not segments_b:
        raise HTTPException(status_code=400, detail="One or both songs have no segment data")

    if not _shared_manager.mixer:
        raise HTTPException(status_code=503, detail="Transition model not loaded")

    rankings = _shared_manager.mixer.ranking_service.get_transition_rankings(
        song_a_features=song_a['features'],
        song_b_features=song_b['features'],
        song_a_segments=exit_segments,
        song_b_segments=segments_b,
        top_n=5,
    )

    if not rankings:
        raise HTTPException(status_code=400, detail="No valid transitions found")

    return {
        "best": rankings[0].to_dict(),
        "alternatives": [r.to_dict() for r in rankings[1:]],
    }


@app.post("/api/setlists/best-transitions-bulk")
async def best_transitions_bulk(req: _BestTransitionBulkRequest, user: dict = _Depends(_get_current_user)):
    """Get best transitions for every adjacent pair in a setlist."""
    if len(req.song_keys) < 2:
        return {"transitions": []}

    if not _shared_manager.mixer:
        raise HTTPException(status_code=503, detail="Transition model not loaded")

    results = []
    for i in range(len(req.song_keys) - 1):
        key_a = req.song_keys[i]
        key_b = req.song_keys[i + 1]
        song_a = music_library.index.get(key_a)
        song_b = music_library.index.get(key_b)

        if not song_a or not song_b:
            results.append({"song_a_key": key_a, "song_b_key": key_b, "error": "Song not found"})
            continue

        segments_a_raw = song_a.get('segments', [])
        segments_b = [s['name'] for s in song_b.get('segments', [])]
        exit_segments = _filter_exit_segments(segments_a_raw)

        if not exit_segments or not segments_b:
            results.append({"song_a_key": key_a, "song_b_key": key_b, "error": "No segment data"})
            continue

        rankings = _shared_manager.mixer.ranking_service.get_transition_rankings(
            song_a_features=song_a['features'],
            song_b_features=song_b['features'],
            song_a_segments=exit_segments,
            song_b_segments=segments_b,
            top_n=1,
        )

        if rankings:
            results.append({
                "song_a_key": key_a,
                "song_b_key": key_b,
                "best": rankings[0].to_dict(),
            })
        else:
            results.append({"song_a_key": key_a, "song_b_key": key_b, "error": "No valid transition"})

    return {"transitions": results}


@app.post("/api/setlists/preview-transition")
async def preview_transition(req: _PreviewTransitionRequest, user: dict = _Depends(_get_current_user)):
    """Generate an ~8 second audio preview of a transition between two songs."""
    import soundfile as sf
    from transition_mixer import TransitionPlan, Segment

    song_a = music_library.index.get(req.song_a_key)
    song_b = music_library.index.get(req.song_b_key)

    if not song_a or not song_b:
        raise HTTPException(status_code=404, detail="One or both songs not found in library")

    if not _shared_manager.mixer:
        raise HTTPException(status_code=503, detail="Transition model not loaded")

    segments_a = song_a.get('segments', [])
    segments_b = song_b.get('segments', [])

    if not segments_a or not segments_b:
        raise HTTPException(status_code=400, detail="One or both songs have no segment data")

    # Resolve the exit and entry segments
    # If user specified segments, use those; otherwise use the model to pick
    exit_seg_dict = None
    entry_seg_dict = None

    if req.exit_segment:
        exit_seg_dict = next((s for s in segments_a if s['name'] == req.exit_segment), None)
    if req.entry_segment:
        entry_seg_dict = next((s for s in segments_b if s['name'] == req.entry_segment), None)

    # If either is missing, use the model to find the best pair
    if not exit_seg_dict or not entry_seg_dict:
        seg_names_a = [s['name'] for s in segments_a]
        seg_names_b = [s['name'] for s in segments_b]

        # If one was specified, constrain to only that side
        if req.exit_segment and exit_seg_dict:
            seg_names_a = [req.exit_segment]
        if req.entry_segment and entry_seg_dict:
            seg_names_b = [req.entry_segment]

        rankings = _shared_manager.mixer.ranking_service.get_transition_rankings(
            song_a_features=song_a['features'],
            song_b_features=song_b['features'],
            song_a_segments=seg_names_a,
            song_b_segments=seg_names_b,
            top_n=1,
        )
        if not rankings:
            raise HTTPException(status_code=400, detail="No valid transition found for this song pair")

        best = rankings[0]
        if not exit_seg_dict:
            exit_seg_dict = next((s for s in segments_a if s['name'] == best.exit_segment), segments_a[-1])
        if not entry_seg_dict:
            entry_seg_dict = next((s for s in segments_b if s['name'] == best.entry_segment), segments_b[0])

    exit_segment = Segment(name=exit_seg_dict['name'], start=exit_seg_dict['start'], end=exit_seg_dict['end'])
    entry_segment = Segment(name=entry_seg_dict['name'], start=entry_seg_dict['start'], end=entry_seg_dict['end'])

    # Calculate crossfade duration
    crossfade_duration = _shared_manager.mixer._calculate_crossfade_duration(
        song_a['features'], song_b['features'], exit_segment, entry_segment
    )

    # Build the transition plan directly (bypassing compute_transition which
    # filters by current_position — not relevant for a preview)
    transition_start = max(0.0, exit_segment.end - crossfade_duration)
    song_a_bpm = song_a['features'].get('bpm', 120.0)
    song_b_bpm = song_b['features'].get('bpm', 120.0)

    plan = TransitionPlan(
        song_a_title=req.song_a_key,
        song_b_title=req.song_b_key,
        exit_segment=exit_segment,
        entry_segment=entry_segment,
        predicted_score=0.0,
        crossfade_duration=crossfade_duration,
        transition_start_time=transition_start,
        song_b_start_offset=entry_segment.start,
        song_a_bpm=song_a_bpm,
        song_b_bpm=song_b_bpm,
    )

    # Load audio and generate preview
    loop = asyncio.get_event_loop()

    def _generate_preview():
        sr = _shared_manager.mixer.sample_rate
        audio_a, _ = sf.read(str(song_a['path']), dtype='float32', always_2d=False)
        audio_b, _ = sf.read(str(song_b['path']), dtype='float32', always_2d=False)

        # Convert stereo to mono if needed
        if audio_a.ndim > 1:
            audio_a = audio_a.mean(axis=1)
        if audio_b.ndim > 1:
            audio_b = audio_b.mean(axis=1)

        # Create the crossfade
        mixed = _shared_manager.mixer.prepare_mixed_audio(audio_a, audio_b, plan, use_dynamic=False)

        # Build ~8 second preview: 2s pre + crossfade + 2s post
        pre = mixed['pre_transition']
        crossfade = mixed['crossfade']
        post = mixed['post_transition']

        pre_samples = min(2 * sr, len(pre))
        post_samples = min(2 * sr, len(post))

        preview = np.concatenate([
            pre[-pre_samples:] if pre_samples > 0 else np.array([], dtype=np.float32),
            crossfade,
            post[:post_samples] if post_samples > 0 else np.array([], dtype=np.float32),
        ])

        # Convert to int16 PCM bytes
        preview_int16 = np.clip(preview * 32767, -32768, 32767).astype(np.int16)
        return preview_int16.tobytes(), sr

    audio_bytes, sr = await loop.run_in_executor(None, _generate_preview)

    return Response(
        content=audio_bytes,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(sr),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


@app.post("/api/auto-play/toggle")
async def toggle_auto_play():
    """Toggle auto-play for all active sessions."""
    async with _sessions_lock:
        sessions_copy = dict(_sessions)
    for mgr in sessions_copy.values():
        mgr.enable_auto_play = not mgr.enable_auto_play
    state = next(iter(sessions_copy.values())).enable_auto_play if sessions_copy else True
    return {
        "auto_play_enabled": state,
        "message": f"Auto-play {'enabled' if state else 'disabled'} for {len(sessions_copy)} sessions"
    }


@app.get("/api/auto-play/status")
async def get_auto_play_status():
    """Get auto-play status."""
    return {
        "enabled": True,
        "active_sessions": len(_sessions),
        "similarity_service_ready": True,
    }


@app.post("/api/library/add-user-song")
async def add_user_song(payload: dict):
    """Hot-add a user-owned song into the in-memory library."""
    try:
        song_data = dict(payload["song_data"])
        user_id = int(payload["user_id"])
        song_data["features"] = sanitize_song_features(song_data.get("features"))
        print(f"[ADD USER SONG] Adding: {song_data['song_name']} for user {user_id}")

        normalized_key = music_library.add_user_song_hot(song_data, user_id)

        # Update similarity index in any active session that has one loaded
        async with _sessions_lock:
            sessions_copy = dict(_sessions)
        for mgr in sessions_copy.values():
            if mgr.similarity_service:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    mgr.similarity_service.add_song_embedding,
                    normalized_key,
                    {
                        'filename': song_data['song_name'],
                        'features': song_data['features'],
                        'segments': song_data['segments']
                    }
                )
                break  # only need to update once

        return {
            "success": True,
            "message": f"'{normalized_key}' added for user {user_id}",
            "song_count": len(music_library.index),
        }
    except Exception as e:
        print(f"[ADD USER SONG ERROR] {e}")
        return {"success": False, "message": str(e)}


@app.post("/api/library/add-song")
async def add_song(song_data: dict):
    try:
        song_data = dict(song_data)
        song_data["features"] = sanitize_song_features(song_data.get("features"))
        print(f"[ADD SONG] Adding: {song_data['song_name']}")

        normalized_key = music_library.add_song_hot(song_data)
        print(f"[ADD SONG] Added to library index")

        async with _sessions_lock:
            sessions_copy = dict(_sessions)
        for mgr in sessions_copy.values():
            if mgr.similarity_service:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    mgr.similarity_service.add_song_embedding,
                    normalized_key,
                    {
                        'filename': song_data['song_name'],
                        'features': song_data['features'],
                        'segments': song_data['segments']
                    }
                )
                print(f"[ADD SONG] Added to similarity index")
                break
        total_songs = len(music_library.index)
        print(f"[ADD SONG] COMPLETE: {total_songs} songs in library")

        return {
            "success": True,
            "message": f"'{normalized_key}' is now fully available",
            "song_count": total_songs,
            "similarity_updated": True,
        }
    except KeyError as e: 
        print(f"[ADD SONG ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Failed to add song: {str(e)}"
        }
    except Exception as e:
        print(f"[ADD SONG ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Failed to add song: {str(e)}"
        }

@app.post("/api/library/reload")
async def reload_library():
    """
    Reload music library to pick up newly uploaded songs.
    Call this after uploading a new song to make it available immediately.
    """
    try:
        print("[RELOAD] Reloading music library...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, music_library.reload)
        # Rebuild similarity embeddings in all active sessions
        async with _sessions_lock:
            sessions_copy = dict(_sessions)
        for mgr in sessions_copy.values():
            if mgr.similarity_service:
                print("[RELOAD] Rebuilding similarity embeddings...")
                await loop.run_in_executor(None, mgr.similarity_service._build_embeddings)
                break  # one rebuild is enough — all share same library
        song_count = len(music_library.index)
        print(f"[RELOAD] Complete: {song_count} songs available")
        return {
            "success": True,
            "message": f"Library reloaded: {song_count} songs",
            "song_count": song_count
        }
    except Exception as e:
        print(f"[RELOAD ERROR] {e}")
        return {
            "success": False,
            "message": f"Failed to reload library: {str(e)}"
        }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

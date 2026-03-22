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
from fastapi import FastAPI, WebSocket, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import subprocess
import atexit

from llm.new_llm import LlamaLLM
from music_library import MusicLibrary
from enhanced_audio_manager import AudioManager, PlaybackState
import music_library
from upload_handler import router as upload_router
from database import init_db, get_connection
from auth import router as auth_router, decode_token
from setlists import router as setlists_router
from song_similarity import parse_title_artist_from_filename
from typing import Optional

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
# music_library = MusicLibrary('music_data/audio', 'music_data/segmented_songs.json')
# llm = LlamaLLM(music_library=music_library)
#
# MODEL_PATH = 'models/dj_transition_model'
# audio_manager = AudioManager(
#     music_library,
#     model_path=MODEL_PATH,
#     enable_auto_play=True
# )

llm = LlamaLLM(music_library=music_library)
llm_semaphore = asyncio.Semaphore(1)

MODEL_PATH = 'models/dj_transition_model'

_sessions: dict = {}
_sessions_lock = asyncio.Semaphore(1)

def _make_audio_manager():
    return AudioManager(
            music_library,
            model_path=MODEL_PATH,
            enable_auto_play=True,
            )
    
async def _get_session(session_id: str):
    async with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = _make_audio_manager()
            print(f"[SESSION] Created: {session_id} ({len(_sessions)} active)")
        return _sessions[session_id]

async def _destroy_session(session_id: str): 
    async with _sessions_lock:
        mgr = _sessions.pop(session_id, None)
    if mgr: 
        try:
            mgr.stop()
        except Exception:
            pass
        print(f"[SESSION] Destroyed: {session_id} ({len(_sessions)} active)")

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
        # "audio_manager": "playing" if audio_manager.is_playing else "idle",
        "audio_manager": f"{len(_sessions)} active sessions",
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

    import uuid 
    session_id = f"{session_user_id or 'guest'}_{uuid.uuid4().hex[:8]}"
    audio_manager = await _get_session(session_id)
    print(f"[WS] Session: {session_id}")

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

    # Send an initial snapshot so the frontend can render the queue immediately
    try:
        await locked_websocket.send_json(_queue_payload({
            "type": "queue_snapshot",
        }))
    except Exception as e:
        print(f"[WS] initial queue_snapshot send failed: {e}")
        return

    # Define variables that will be shared between tasks
    playback_task = None
    message_handler_task = None

    # Track if THIS WebSocket has started playback
    # This prevents issues when frontend refreshes and reconnects
    has_started_playback = False

    # Move message handling to a separate async function
    # This allows it to run concurrently with audio playback
    async def handle_messages():
        """
        Handle incoming user messages without blocking audio playback.
        Runs as a separate concurrent task.
        """
        nonlocal playback_task, has_started_playback

        while True:
            try:
                data = await locked_websocket.receive_json()
            except Exception as e:
                print(f"[WS] receive_json failed / disconnected: {e}")
                break

            # Direct actions (no LLM needed)
            msg_type = data.get('type')
            if msg_type == 'reorder_queue':
                new_order = data.get('order', [])
                success = audio_manager.reorder_queue(new_order)
                if success:
                    await locked_websocket.send_json(_queue_payload({
                        "type": "queue_update",
                        "action": "reordered",
                        "message": "Queue reordered",
                    }))
                    if audio_manager.pending_transition:
                        await locked_websocket.send_json({
                            "type": "transition_planned",
                            "transition": audio_manager.pending_transition.to_dict()
                        })
                continue

            if msg_type == 'set_transition_mode':
                requested_mode = data.get('mode', '')
                success = audio_manager.set_transition_mode(requested_mode)
                if success:
                    await locked_websocket.send_json(_queue_payload({
                        "type": "transition_mode_updated",
                        "message": f"Transition mode set to {audio_manager.get_transition_mode()}",
                        "transition_mode": audio_manager.get_transition_mode(),
                    }))
                else:
                    await locked_websocket.send_json({
                        "type": "error",
                        "message": "Invalid transition mode. Use 'dynamic' or 'classic'."
                    })
                continue

            if msg_type == 'remove_from_queue':
                index = data.get('index')
                if isinstance(index, int):
                    success = audio_manager.remove_from_queue(index)
                    if success:
                        await locked_websocket.send_json(_queue_payload({
                            "type": "queue_update",
                            "action": "removed",
                            "message": "Song removed from queue",
                        }))
                        if audio_manager.pending_transition:
                            await locked_websocket.send_json({
                                "type": "transition_planned",
                                "transition": audio_manager.pending_transition.to_dict()
                            })
                continue

            if msg_type == 'pause':
                audio_manager.pause()
                await locked_websocket.send_json({"type": "paused"})
                continue

            if msg_type == 'resume':
                audio_manager.resume()
                await locked_websocket.send_json({"type": "resumed"})
                continue

            if msg_type == 'reset':
                print("[WS] Client reset - clearing all state")
                audio_manager.stop()
                audio_manager.queue.clear()
                audio_manager.current_track = None
                audio_manager.pending_transition = None
                audio_manager.transition_audio = None
                audio_manager._pending_transition_for = None
                audio_manager.samples_sent = 0
                audio_manager.current_position = 0.0
                audio_manager.state = PlaybackState.STOPPED
                if playback_task and not playback_task.done():
                    playback_task.cancel()
                    playback_task = None
                has_started_playback = False
                await locked_websocket.send_json({"type": "reset_complete"})
                continue

            if msg_type == 'load_setlist':
                setlist_id = data.get('setlist_id')
                if not session_user_id:
                    await locked_websocket.send_json({
                        "type": "error",
                        "message": "Authentication required to load setlists"
                    })
                    continue

                conn = get_connection()
                setlist_row = conn.execute(
                    "SELECT id, name FROM setlists WHERE id = ? AND user_id = ?",
                    (setlist_id, session_user_id),
                ).fetchone()

                if not setlist_row:
                    conn.close()
                    await locked_websocket.send_json({
                        "type": "error",
                        "message": "Setlist not found"
                    })
                    continue

                items = conn.execute(
                    "SELECT song_key, song_title, song_artist FROM setlist_items "
                    "WHERE setlist_id = ? ORDER BY position",
                    (setlist_id,),
                ).fetchall()
                conn.close()

                if not items:
                    await locked_websocket.send_json({
                        "type": "error",
                        "message": "Setlist is empty"
                    })
                    continue


                queued_songs = []
                failed_songs = []
                for item in items:
                    # Resolve proper title/artist from the library via song_key
                    title = item["song_title"]
                    artist = item["song_artist"]
                    song_key = item["song_key"]
                    lib_entry = music_library.index.get(song_key)
                    if lib_entry:
                        title, artist = parse_title_artist_from_filename(lib_entry['filename'])

                    found = audio_manager.add_to_queue(title, artist)
                    if found:
                        queued_songs.append(title)
                    else:
                        failed_songs.append(title)

                await locked_websocket.send_json(_queue_payload({
                    "type": "queue_update",
                    "action": "setlist_loaded",
                    "message": f"Loaded setlist '{setlist_row['name']}': {len(queued_songs)} songs queued",
                    "queued": queued_songs,
                    "failed": failed_songs,
                }))

                if (not audio_manager.is_playing or not has_started_playback) and queued_songs:
                    if audio_manager.is_playing and not has_started_playback:
                        audio_manager.stop()
                        audio_manager.current_track = None
                        audio_manager.start()
                    elif not audio_manager.is_playing:
                        if not has_started_playback and audio_manager.current_track:
                            audio_manager.current_track = None
                        audio_manager.start()

                    playback_task = asyncio.create_task(
                        audio_manager.play_queue(locked_websocket)
                    )
                    has_started_playback = True
                continue

            prompt = data.get('data', '')
            print(f"[WS] Received prompt: {prompt}")

            try:
                # Run LLM in thread pool to prevent blocking event loop
                loop = asyncio.get_event_loop()
                # result = await loop.run_in_executor(None, llm.classify, prompt)
                intent = result['intent']
                song_info = result['song']

                if intent == 'queue_song':
                    found = audio_manager.add_to_queue(
                        song_info['title'],
                        song_info['artist']
                    )

                    if found:
                        await locked_websocket.send_json(_queue_payload({
                            "type": "queue_update",
                            "action": "added",
                            "message": f"Queued: {song_info['title']}",
                        }))

                        if audio_manager.pending_transition:
                            await locked_websocket.send_json({
                                "type": "transition_planned",
                                "transition": audio_manager.pending_transition.to_dict()
                            })

                        if not audio_manager.is_playing or not has_started_playback:
                            if audio_manager.is_playing and not has_started_playback:
                                print("[WS] Backend playing but new WebSocket - forcing clean restart")
                                audio_manager.stop()
                                audio_manager.current_track = None
                                audio_manager.start()
                            elif not audio_manager.is_playing:
                                if not has_started_playback and audio_manager.current_track:
                                    print(f"[WS] New WebSocket, clearing old current_track: {audio_manager.current_track.title if audio_manager.current_track else 'None'}")
                                    audio_manager.current_track = None
                                audio_manager.start()

                            playback_task = asyncio.create_task(
                                audio_manager.play_queue(locked_websocket)
                            )
                            has_started_playback = True
                            print(f"[WS] Started playback task for this WebSocket")
                    else:
                        await locked_websocket.send_json({
                            "type": "error",
                            "message": f"Song not found: {song_info['title']}"
                        })

                elif intent == 'generate_playlist':
                    playlist = await loop.run_in_executor(None, llm.generate_playlist, prompt, session_user_id)

                    if not playlist:
                        await locked_websocket.send_json({
                            "type": "error",
                            "message": "Could not generate playlist from your request"
                        })
                    else:
                        queued_songs = []
                        failed_songs = []

                        for song in playlist:
                            found = audio_manager.add_to_queue(
                                song['title'],
                                song.get('artist')
                            )
                            if found:
                                queued_songs.append(song['title'])
                            else:
                                failed_songs.append(song['title'])

                        await locked_websocket.send_json(_queue_payload({
                            "type": "queue_update",
                            "action": "playlist_generated",
                            "message": f"Queued {len(queued_songs)} songs",
                            "queued": queued_songs,
                            "failed": failed_songs,
                        }))

                        if (not audio_manager.is_playing or not has_started_playback) and queued_songs:
                            if audio_manager.is_playing and not has_started_playback:
                                print("[WS] Backend playing but new WebSocket - restarting for playlist")
                                audio_manager.stop()
                                audio_manager.current_track = None
                                audio_manager.start()
                            elif not audio_manager.is_playing:
                                if not has_started_playback and audio_manager.current_track:
                                    print(f"[WS] New WebSocket (playlist), clearing old track")
                                    audio_manager.current_track = None
                                audio_manager.start()

                            playback_task = asyncio.create_task(
                                audio_manager.play_queue(locked_websocket)
                            )
                            has_started_playback = True

                elif intent == 'play_setlist':
                    setlist_name = (song_info.get('title') or '').strip()
                    if not session_user_id:
                        await locked_websocket.send_json({
                            "type": "error",
                            "message": "You need to be logged in to load setlists"
                        })
                    elif not setlist_name:
                        await locked_websocket.send_json({
                            "type": "error",
                            "message": "I couldn't determine which setlist you want. Try saying 'play setlist <name>'"
                        })
                    else:
                        conn = get_connection()
                        setlist_row = conn.execute(
                            "SELECT id, name FROM setlists WHERE user_id = ? AND LOWER(name) = LOWER(?)",
                            (session_user_id, setlist_name),
                        ).fetchone()

                        if not setlist_row:
                            conn.close()
                            await locked_websocket.send_json({
                                "type": "error",
                                "message": f"Setlist '{setlist_name}' not found"
                            })
                        else:
                            items = conn.execute(
                                "SELECT song_key, song_title, song_artist FROM setlist_items "
                                "WHERE setlist_id = ? ORDER BY position",
                                (setlist_row['id'],),
                            ).fetchall()
                            conn.close()

                            if not items:
                                await locked_websocket.send_json({
                                    "type": "error",
                                    "message": f"Setlist '{setlist_row['name']}' is empty"
                                })
                            else:
                                queued_songs = []
                                failed_songs = []
                                for item in items:
                                    title = item["song_title"]
                                    artist = item["song_artist"]
                                    song_key = item["song_key"]
                                    lib_entry = music_library.index.get(song_key)
                                    if lib_entry:
                                        title, artist = parse_title_artist_from_filename(lib_entry['filename'])

                                    found = audio_manager.add_to_queue(title, artist)
                                    if found:
                                        queued_songs.append(title)
                                    else:
                                        failed_songs.append(title)

                                await locked_websocket.send_json(_queue_payload({
                                    "type": "queue_update",
                                    "action": "setlist_loaded",
                                    "message": f"Loaded setlist '{setlist_row['name']}': {len(queued_songs)} songs queued",
                                    "queued": queued_songs,
                                    "failed": failed_songs,
                                }))

                                if (not audio_manager.is_playing or not has_started_playback) and queued_songs:
                                    if audio_manager.is_playing and not has_started_playback:
                                        audio_manager.stop()
                                        audio_manager.current_track = None
                                        audio_manager.start()
                                    elif not audio_manager.is_playing:
                                        if not has_started_playback and audio_manager.current_track:
                                            audio_manager.current_track = None
                                        audio_manager.start()

                                    playback_task = asyncio.create_task(
                                        audio_manager.play_queue(locked_websocket)
                                    )
                                    has_started_playback = True

                elif intent == 'quick_transition':
                    try:
                        result = audio_manager.force_quick_transition()
                        if result == "quick":
                            await locked_websocket.send_json({
                                "type": "quick_transition_scheduled",
                                "message": "Transitioning at next segment boundary..."
                            })
                        elif result == "immediate":
                            await locked_websocket.send_json({
                                "type": "quick_transition_scheduled",
                                "message": "Skipping immediately..."
                            })
                        else:
                            await locked_websocket.send_json({
                                "type": "error",
                                "message": "No song queued to transition to"
                            })
                    except Exception as e:
                        await locked_websocket.send_json({
                            "type": "error",
                            "message": f"Could not schedule quick transition: {str(e)}"
                        })

                elif intent == 'set_transition_mode':
                    prompt_lower = (prompt or "").lower()
                    if any(keyword in prompt_lower for keyword in ["classic", "crossfade", "standard"]):
                        requested_mode = "classic"
                    elif any(keyword in prompt_lower for keyword in ["dynamic", "warp"]):
                        requested_mode = "dynamic"
                    else:
                        requested_mode = "dynamic"

                    success = audio_manager.set_transition_mode(requested_mode)
                    if success:
                        await locked_websocket.send_json(_queue_payload({
                            "type": "transition_mode_updated",
                            "message": f"Transition mode set to {audio_manager.get_transition_mode()}",
                            "transition_mode": audio_manager.get_transition_mode(),
                        }))
                    else:
                        await locked_websocket.send_json({
                            "type": "error",
                            "message": "Could not update transition mode. Use dynamic or classic."
                        })

                elif intent == 'stop_dj':
                    audio_manager.stop()
                    await locked_websocket.send_json({
                        "type": "stopped",
                        "message": "Playback stopped"
                    })

                elif intent == 'hello':
                    await locked_websocket.send_json({
                        "type": "greeting",
                        "message": "Hey! I'm your AI DJ. Tell me what you want to hear!"
                    })

                elif intent == 'help':
                    await locked_websocket.send_json({
                        "type": "help",
                        "message": "You can say things like: 'Play Wake Me Up by Avicii', 'Queue Stargazing', 'Stop the music'. I'll automatically queue similar songs to keep the music going!"
                    })

                else:
                    await locked_websocket.send_json({
                        "type": "unknown",
                        "message": "I didn't quite catch that. Try asking me to play a song!"
                    })

            except Exception as e:
                print(f"[ERROR] LLM processing: {e}")
                await locked_websocket.send_json({
                    "type": "error",
                    "message": f"Error processing request: {str(e)}"
                })

    # Run message handler as a concurrent task instead of blocking
    try:
        message_handler_task = asyncio.create_task(handle_messages())
        print("[WS] Message handler task started")
        await message_handler_task

    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        print("[WS] Cleaning up WebSocket connection...")

        try:
            audio_manager.stop()
        except Exception as e:
            print(f"[CLEANUP] Error stopping audio manager: {e}")

        if playback_task and not playback_task.done():
            playback_task.cancel()
            try:
                await playback_task
            except asyncio.CancelledError:
                print("[CLEANUP] Playback task cancelled")
            except Exception as e:
                print(f"[CLEANUP] Error cancelling playback task: {e}")

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
    """Get current playback status including auto-play info."""
    return audio_manager.get_queue_status()


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
# Setlist helper endpoints (need access to music_library / audio_manager)
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

    if not audio_manager.mixer:
        raise HTTPException(status_code=503, detail="Transition model not loaded")

    rankings = audio_manager.mixer.ranking_service.get_transition_rankings(
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

    if not audio_manager.mixer:
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

        rankings = audio_manager.mixer.ranking_service.get_transition_rankings(
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

    if not audio_manager.mixer:
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

        rankings = audio_manager.mixer.ranking_service.get_transition_rankings(
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
    crossfade_duration = audio_manager.mixer._calculate_crossfade_duration(
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
        sr = audio_manager.mixer.sample_rate
        audio_a, _ = sf.read(str(song_a['path']), dtype='float32', always_2d=False)
        audio_b, _ = sf.read(str(song_b['path']), dtype='float32', always_2d=False)

        # Convert stereo to mono if needed
        if audio_a.ndim > 1:
            audio_a = audio_a.mean(axis=1)
        if audio_b.ndim > 1:
            audio_b = audio_b.mean(axis=1)

        # Create the crossfade
        mixed = audio_manager.mixer.prepare_mixed_audio(audio_a, audio_b, plan, use_dynamic=False)

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
    """Toggle auto-play feature on/off."""
    audio_manager.enable_auto_play = not audio_manager.enable_auto_play
    return {
        "auto_play_enabled": audio_manager.enable_auto_play,
        "message": f"Auto-play {'enabled' if audio_manager.enable_auto_play else 'disabled'}"
    }


@app.get("/api/auto-play/status")
async def get_auto_play_status():
    """Get auto-play status and recently played songs."""
    return {
        "enabled": audio_manager.enable_auto_play,
        "recently_played": audio_manager.recently_played,
        "similarity_service_ready": audio_manager.similarity_service is not None
    }


@app.post("/api/library/add-user-song")
async def add_user_song(payload: dict):
    """Hot-add a user-owned song into the in-memory library."""
    try:
        song_data = payload["song_data"]
        user_id = int(payload["user_id"])
        print(f"[ADD USER SONG] Adding: {song_data['song_name']} for user {user_id}")

        normalized_key = music_library.add_user_song_hot(song_data, user_id)

        if audio_manager.similarity_service:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                audio_manager.similarity_service.add_song_embedding,
                normalized_key,
                {
                    'filename': song_data['song_name'],
                    'features': song_data['features'],
                    'segments': song_data['segments']
                }
            )

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
        print(f"[ADD SONG] Adding: {song_data['song_name']}")

        normalized_key = music_library.add_song_hot(song_data)
        print(f"[ADD SONG] Added to library index")

        if audio_manager.similarity_service:
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                None,
                audio_manager.similarity_service.add_song_embedding,
                normalized_key,
                {
                    'filename': song_data['song_name'],
                    'features': song_data['features'],
                    'segments': song_data['segments']
                }
            )
            print(f"[ADD SONG] Added to similarity index")
        total_songs = len(music_library.index)
        print(f"[ADD SONG] COMPLETE: {total_songs} songs in library")

        return {
            "success": True,
            "message": f"'{normalized_key}' is now fully available",
            "song_count": total_songs,
            "similarity_updated": audio_manager.similarity_service is not None
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
    global music_library, audio_manager, llm
    try:
        print("[RELOAD] Reloading music library...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, music_library.reload)
        if audio_manager.similarity_service:
            print("[RELOAD] Rebuilding similarity embeddings...")
            await loop.run_in_executor(None, audio_manager.similarity_service._build_embeddings)
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

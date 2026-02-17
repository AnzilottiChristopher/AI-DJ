"""
AI DJ Backend — Production-ready FastAPI server.

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
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import subprocess
import atexit

from llm.new_llm import LlamaLLM
from music_library import MusicLibrary
from enhanced_audio_manager import AudioManager
from upload_handler import router as upload_router

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
llm = LlamaLLM(music_library=music_library)

MODEL_PATH = 'models/dj_transition_model'
audio_manager = AudioManager(
    music_library,
    model_path=MODEL_PATH,
    enable_auto_play=True
)

# ---------------------------------------------------------------------------
# CORS — automatic based on environment
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

# ---------------------------------------------------------------------------
# Ollama lifecycle (dev only — in prod, Ollama runs as its own service)
# ---------------------------------------------------------------------------
ollama_process = None


@app.on_event("startup")
async def startup_event():
    global ollama_process
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
        "audio_manager": "playing" if audio_manager.is_playing else "idle",
        "library_size": len(music_library.index),
    }


# ---------------------------------------------------------------------------
# WebSocket — audio streaming
# ---------------------------------------------------------------------------
@app.websocket("/api/ws/audio")
async def audio_stream(websocket: WebSocket):
    """
    Main WebSocket endpoint for audio streaming.

    Handles:
    - Song queue requests via LLM
    - Audio streaming with transitions
    - Auto-play of similar songs
    - Playback status updates
    """
    await websocket.accept()

    # FIX FOR AUDIO CUTOUTS: Create a WebSocket wrapper with send locking
    class LockedWebSocket:
        """WebSocket wrapper that prevents send contention between tasks."""
        def __init__(self, ws):
            self._ws = ws
            self._send_lock = asyncio.Lock()

        async def send_json(self, data):
            async with self._send_lock:
                await self._ws.send_json(data)

        async def send_bytes(self, data):
            async with self._send_lock:
                await self._ws.send_bytes(data)

        async def receive_json(self):
            return await self._ws.receive_json()

        def __getattr__(self, name):
            return getattr(self._ws, name)

    locked_websocket = LockedWebSocket(websocket)

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

            prompt = data.get('data', '')
            print(f"[WS] Received prompt: {prompt}")

            try:
                # Run LLM in thread pool to prevent blocking event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, llm.classify, prompt)
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
                    playlist = await loop.run_in_executor(None, llm.generate_playlist, prompt)

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

                elif intent == 'quick_transition':
                    try:
                        success = audio_manager.force_quick_transition()
                        if success:
                            await locked_websocket.send_json({
                                "type": "quick_transition_scheduled",
                                "message": "Transitioning at next segment boundary..."
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

        print("[WS] Cleanup complete")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """Get current playback status including auto-play info."""
    return audio_manager.get_queue_status()


@app.get("/api/library")
async def get_library():
    """Get list of available songs."""
    songs = []
    for key, data in music_library.index.items():
        songs.append({
            "title": key,
            "filename": data['filename'],
            "bpm": data['features'].get('bpm'),
            "key": data['features'].get('key'),
        })
    return {"songs": songs}


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


@app.post("/api/library/add-song")
async def add_song(song_data: dict):
    try:
        song_name = song_data['song_name']
        print(f"[ADD SONG] Adding: {song_name}")

        music_library.index[song_name] = {
            'filename': song_data['song_name'],
            'features': song_data['features'],
            'segments': song_data['segments']
        }
        print(f"[ADD SONG] Added to library index")

        if audio_manager.similarity_service:
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                None,
                audio_manager.similarity_service.add_song_embedding,
                song_name,
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
            "message": f"'{song_name}' is now fully available",
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

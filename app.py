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

app = FastAPI(title="AI DJ Backend")

# music_library = MusicLibrary('music_data/audio', 'music_data/segmented_alex_pre_analysis_results_converted.json')
music_library = MusicLibrary('music_data/audio', 'music_data/segmented_songs.json')
llm = LlamaLLM(music_library=music_library)

MODEL_PATH = 'models/dj_transition_model'  # or wherever your model is
audio_manager = AudioManager(
    music_library, 
    model_path=MODEL_PATH,
    enable_auto_play=True  # Enable auto-play feature
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)

# Ollama process management
ollama_process = None

@app.on_event("startup")
async def startup_event():
    global ollama_process
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
            # Receiving doesn't need a lock
            return await self._ws.receive_json()
        
        # Delegate other methods to the original websocket
        def __getattr__(self, name):
            return getattr(self._ws, name)
    
    # Wrap the websocket with locking
    locked_websocket = LockedWebSocket(websocket)

    def _serialize_track(item):
        """Best-effort conversion of whatever the queue stores into a UI-friendly object."""
        if item is None:
            return None
        # Common case: already a dict
        if isinstance(item, dict):
            title = item.get("title") or item.get("name") or item.get("track") or item.get("filename")
            artist = item.get("artist") or item.get("artist_name") or item.get("by")
            return {
                "title": title,
                "artist": artist,
            }
        # If the queue stores a custom object
        if hasattr(item, "title") or hasattr(item, "artist"):
            return {
                "title": getattr(item, "title", None),
                "artist": getattr(item, "artist", None),
            }
        # Fallback: string
        return {
            "title": str(item),
            "artist": None,
        }

    def _queue_payload(extra: dict | None = None):
        status = audio_manager.get_queue_status()

        # Different versions of the backend used different keys; normalize here.
        raw_upcoming = (
            status.get("queue")
            or status.get("upcoming")
            or status.get("upcoming_songs")
            or status.get("next_up")
            or status.get("upcomingQueue")
            or []
        )

        # If the queue is an asyncio.Queue (or another queue-like object), convert it to a list.
        # asyncio.Queue keeps items in a private deque at `._queue`.
        try:
            if hasattr(raw_upcoming, "_queue"):
                raw_upcoming = list(raw_upcoming._queue)
        except Exception:
            pass

        # Some implementations might store upcoming as a single item; normalize to a list.
        if raw_upcoming is None:
            raw_upcoming = []

        # UI-friendly list of upcoming tracks, in order.
        upcoming_tracks = []
        try:
            for x in raw_upcoming:
                s = _serialize_track(x)
                if s:
                    upcoming_tracks.append(s)
        except Exception:
            # If raw_upcoming isn't iterable for some reason
            s = _serialize_track(raw_upcoming)
            if s:
                upcoming_tracks = [s]

        # Keep the original status, but also provide the normalized list.
        payload = {
            "queue_status": status,
            "upcoming": upcoming_tracks,        # <-- frontend should read this for the ordered list
            "queue": upcoming_tracks,           # <-- alias for convenience / backwards compat
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
        # Client disconnected before we could send
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
        nonlocal playback_task, has_started_playback  # Allow modification from outer scope
        
        while True:
            try:
                data = await locked_websocket.receive_json()
            except Exception as e:
                # Client disconnected or sent invalid JSON
                print(f"[WS] receive_json failed / disconnected: {e}")
                break

            prompt = data.get('data', '')
            print(f"[WS] Received prompt: {prompt}")

            try:
                # Run LLM in thread pool to prevent blocking event loop
                # The LLM classification is CPU-intensive and synchronous, which would
                # freeze the audio streaming if run directly in the async context
                loop = asyncio.get_event_loop()
                
                # Run classify in a thread pool executor (non-blocking)
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

                        # Check if a transition is being prepared
                        if audio_manager.pending_transition:
                            await locked_websocket.send_json({
                                "type": "transition_planned",
                                "transition": audio_manager.pending_transition.to_dict()
                            })

                        # Start playback if:
                        # 1. Backend says not playing, OR
                        # 2. This WebSocket hasn't started its own playback task yet
                        #    (handles frontend refresh where old task is orphaned)
                        if not audio_manager.is_playing or not has_started_playback:
                            # If backend thinks it's playing but we haven't started playback,
                            # it means the old WebSocket was disconnected. Start fresh.
                            if audio_manager.is_playing and not has_started_playback:
                                print("[WS] Backend playing but new WebSocket - forcing clean restart")
                                audio_manager.stop()  # Stop the orphaned task
                                audio_manager.current_track = None  # Clear old track
                                audio_manager.start()
                            elif not audio_manager.is_playing:
                                # Even if stopped, clear current_track if this is a new WebSocket
                                # (handles refresh where cleanup stopped playback but left track)
                                if not has_started_playback and audio_manager.current_track:
                                    print(f"[WS] New WebSocket, clearing old current_track: {audio_manager.current_track.title if audio_manager.current_track else 'None'}")
                                    audio_manager.current_track = None
                                audio_manager.start()
                            
                            # Use locked_websocket to prevent send contention
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
                    # Handle playlist generation (also CPU-intensive, run in thread pool)
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

                        # Start playback if not already playing (or new WebSocket)
                        if (not audio_manager.is_playing or not has_started_playback) and queued_songs:
                            if audio_manager.is_playing and not has_started_playback:
                                print("[WS] Backend playing but new WebSocket - restarting for playlist")
                                audio_manager.stop()
                                audio_manager.current_track = None
                                audio_manager.start()
                            elif not audio_manager.is_playing:
                                # Clear old track if new WebSocket
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
    # message handling no longer blocks audio streaming
    try:
        # Start the message handler as a background task
        message_handler_task = asyncio.create_task(handle_messages())
        print("[WS] Message handler task started")
        
        # Wait for the message handler to finish (usually means disconnect)
        # Audio playback happens in its own task (playback_task) which runs independently
        await message_handler_task
        
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        # Enhanced cleanup to handle both tasks
        print("[WS] Cleaning up WebSocket connection...")
        
        # Stop audio manager
        try:
            audio_manager.stop()
        except Exception as e:
            print(f"[CLEANUP] Error stopping audio manager: {e}")

        # Cancel playback task if running
        if playback_task and not playback_task.done():
            playback_task.cancel()
            try:
                await playback_task
            except asyncio.CancelledError:
                print("[CLEANUP] Playback task cancelled")
            except Exception as e:
                print(f"[CLEANUP] Error cancelling playback task: {e}")
        
        # Cancel message handler task if running
        if message_handler_task and not message_handler_task.done():
            message_handler_task.cancel()
            try:
                await message_handler_task
            except asyncio.CancelledError:
                print("[CLEANUP] Message handler task cancelled")
            except Exception as e:
                print(f"[CLEANUP] Error cancelling message handler: {e}")
        
        print("[WS] Cleanup complete")


@app.get("/api/status")
async def get_status():
    """Get current playback status including auto-play info."""
    return audio_manager.get_queue_status()


@app.get("/api/library")
async def get_library():
    """Get list of available songs.

    Returns UI-friendly fields:
    - id: normalized lookup key used by the backend (e.g., "hey brother avicii")
    - title: human-readable title parsed from filename
    - artist: human-readable artist parsed from filename
    - bpm/key: extracted features when available
    - filename: underlying wav filename
    """

    def _title_case(s: str) -> str:
        # Keep it simple: replace separators and title-case.
        return s.replace('-', ' ').replace('_', ' ').strip().title()

    songs = []

    for normalized_key, data in music_library.index.items():
        filename = data.get('filename')
        features = data.get('features') or {}

        # Parse title/artist from the filename format: "title-words_artist-words.wav"
        display_title = normalized_key
        display_artist = None

        try:
            if filename:
                stem = filename
                if stem.lower().endswith('.wav'):
                    stem = stem[:-4]

                if '_' in stem:
                    title_part, artist_part = stem.split('_', 1)
                    display_title = _title_case(title_part)
                    display_artist = _title_case(artist_part) if artist_part else None
                else:
                    display_title = _title_case(stem)
        except Exception:
            # Fallback: show the normalized key if parsing fails
            display_title = normalized_key
            display_artist = None

        songs.append({
            "id": normalized_key,                 # use this when sending queue requests
            "title": display_title,
            "artist": display_artist,
            "filename": filename,
            "bpm": features.get('bpm'),
            "key": features.get('key'),
            "scale": features.get('scale'),
        })

    # Sort for a nicer UI
    songs.sort(key=lambda s: ((s.get('artist') or ''), (s.get('title') or '')))

    return {"count": len(songs), "songs": songs}


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
@app.post("/api/library/reload")
async def reload_library():
    """
    Reload music library to pick up newly uploaded songs.
    
    Call this after uploading a new song to make it available immediately.
    """
    global music_library, audio_manager, llm
    try:
        print("[RELOAD] Reloading music library...")
        
        # Reload the music library (re-reads JSON and rebuilds index)
        music_library.reload()
        
        # Rebuild similarity embeddings if similarity service exists
        if audio_manager.similarity_service:
            print("[RELOAD] Rebuilding similarity embeddings...")
            audio_manager.similarity_service._build_embeddings(music_library)
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

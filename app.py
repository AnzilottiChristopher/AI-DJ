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
    playback_task = None
    
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get('data', '')
            
            print(f"[WS] Received prompt: {prompt}")
            
            try:
                # Classify the intent
                result = llm.classify(prompt)
                intent = result['intent']
                song_info = result['song']
                
                if intent == 'queue_song':
                    found = audio_manager.add_to_queue(
                        song_info['title'], 
                        song_info['artist']
                    )
                    
                    if found:
                        # Send queue update
                        await websocket.send_json({
                            "type": "queued",
                            "message": f"Queued: {song_info['title']}",
                            "queue": audio_manager.get_queue_status()
                        })
                        
                        # Check if a transition is being prepared
                        if audio_manager.pending_transition:
                            await websocket.send_json({
                                "type": "transition_planned",
                                "transition": audio_manager.pending_transition.to_dict()
                            })
                        
                        # Start playback if not already playing
                        if not audio_manager.is_playing:
                            audio_manager.start()
                            playback_task = asyncio.create_task(
                                audio_manager.play_queue(websocket)
                            )
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Song not found: {song_info['title']}"
                        })
                        
                elif intent == 'generate_playlist':
                    # NEW: Handle playlist generation
                    playlist = llm.generate_playlist(prompt)
                    
                    if not playlist:
                        await websocket.send_json({
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
                        
                        await websocket.send_json({
                            "type": "playlist_generated",
                            "message": f"Queued {len(queued_songs)} songs",
                            "queued": queued_songs,
                            "failed": failed_songs,
                            "queue": audio_manager.get_queue_status()
                        })
                        
                        # Start playback if not already playing
                        if not audio_manager.is_playing and queued_songs:
                            audio_manager.start()
                            playback_task = asyncio.create_task(
                                audio_manager.play_queue(websocket)
                            )
                
                
                elif intent == 'stop_dj':
                    audio_manager.stop()
                    await websocket.send_json({
                        "type": "stopped",
                        "message": "Playback stopped"
                    })
                
                elif intent == 'hello':
                    await websocket.send_json({
                        "type": "greeting",
                        "message": "Hey! I'm your AI DJ. Tell me what you want to hear!"
                    })
                
                elif intent == 'help':
                    await websocket.send_json({
                        "type": "help",
                        "message": "You can say things like: 'Play Wake Me Up by Avicii', 'Queue Stargazing', 'Stop the music'. I'll automatically queue similar songs to keep the music going!"
                    })
                
                else:
                    await websocket.send_json({
                        "type": "unknown",
                        "message": "I didn't quite catch that. Try asking me to play a song!"
                    })
                
            except Exception as e:
                print(f"[ERROR] LLM processing: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing request: {str(e)}"
                })
            
            # Acknowledge receipt
            await websocket.send_json({
                "status": "received",
                "prompt": prompt
            })
    
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        if playback_task:
            playback_task.cancel()
        try:
            await websocket.close()
        except:
            pass


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
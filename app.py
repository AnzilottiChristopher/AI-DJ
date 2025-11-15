from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np
import asyncio
import subprocess
import atexit
from llm.new_llm import LlamaLLM
from music_library import MusicLibrary
from audio_manager import AudioManager
import uvicorn 

app = FastAPI()
llm = LlamaLLM()
music_library = MusicLibrary('music_data/audio', 'music_data/metadata.json')
audio_manager = AudioManager(music_library)

# depending on which port the front end is running on, just adjust this part
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# the following two methods are for the llm running
ollama_process = None

@app.on_event("startup")
async def startup_event():
    global ollama_process
    print("starting Ollama server...")
    try:
        # start ollama in the background
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
        print("shutting down Ollama server...")
        ollama_process.terminate()
        ollama_process.wait()
        print("ollama server stopped")

def cleanup():
    global ollama_process
    if ollama_process:
        ollama_process.terminate()
        ollama_process.wait()

atexit.register(cleanup)

# this will likely be changed in the future
@app.websocket("/api/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data['data']
            
            print(f"received prompt: {prompt}")
            
            try:
                result = llm.classify(prompt)
                
                intent = result['intent']
                song_info = result['song'] # this includes both title and artist
                
                # later this should be handled by some utility method or something
                if intent == 'queue_song':
                    found = audio_manager.add_to_queue(song_info['title'], song_info['artist'])
                    
                    if found:
                        await websocket.send_json({
                            "type": "queued",
                            "message": f"Queued: {song_info['title']}",
                            "queue": audio_manager.get_queue_status()
                        })
                        
                        # start the play back if not playing already
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
                
            except Exception as e:
                print(f"error in llm: {e}")
            
            
            await websocket.send_json({"status": "received", "prompt": prompt})
        
    except Exception as e:
        print(f"error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


# pay attention to what ports you are running on
# as you may need to change this on your own
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
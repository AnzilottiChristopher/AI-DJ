from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np
import asyncio
import uvicorn 

app = FastAPI()

# depending on which port the front end is running on, just adjust this part
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# this will likely be changed in the future
@app.websocket("/api/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data['data']
            
            print(f"received prompt: {prompt}")
            
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
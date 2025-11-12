from flask import Flask, request, jsonify
from flask_cors import CORS
from llm.new_llm import LlamaLLM
from concurrent.futures import ThreadPoolExecutor
from DJ_Functions import DJFunctions
from pathlib import Path

app = Flask(__name__)
CORS(app)  # ONLY needed if React runs on a different port (e.g., 3000)



@app.post("/api/chat")
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    # Super simple "bot" response
    # bot_reply = f"You said: {user_message}"
    future = executor.submit(generate_llm_response, user_message)
    # future.add_done_callback(lambda f: return jsonify({"reply":f.result()}))
    return jsonify({"reply": future.result()})

def generate_llm_response(user_text: str) -> str:
    print(f"Received user message: {user_text}")
    try:
        return llm.process_response(user_text)
    except Exception as e:
        # keep UI alive even if model errors
        return f"[ERROR] {type(e).__name__}: {e}"
    
@app.post("/api/play")
def play():
    track_path = Path("wav_files/wakemeup-avicii.wav")
    title = track_path.stem
    dj = DJFunctions([track_path])
    dj.play(title)

if __name__ == "__main__":
    llm = LlamaLLM()
    # track_path = Path("wav_files/wakemeup-avicii.wav")
    # title = track_path.stem
    # dj = DJFunctions([track_path])
    # dj.play(title)
    executor = ThreadPoolExecutor(max_workers=2)
    app.run(debug=True)
    
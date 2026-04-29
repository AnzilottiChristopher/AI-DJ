# AI DJ Backend

A production-ready FastAPI backend that powers an AI-assisted DJ system. It handles real-time audio playback, intelligent song transitions, LLM-driven track selection via Ollama, user authentication, setlist management, and song uploads — all served over WebSockets and a REST API.

---

## Prerequisites

- **Python 3.11**
- **[Ollama](https://ollama.com/)** — must be installed and running locally for LLM-powered track selection
- **Conda** (recommended) or pip

---

## Setup

### Option 1: Conda (Recommended)

The repo includes an `environment.yml` with all dependencies pinned.

```bash
git clone https://github.com/AnzilottiChristopher/AI-DJ.git
cd AI-DJ

conda env create -f environment.yml
conda activate AI-DJ
```

### Option 2: pip

```bash
git clone https://github.com/AnzilottiChristopher/AI-DJ.git
cd AI-DJ

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Directory Structure

```
AI-DJ/
├── app.py                      # Main entry point — run this
├── enhanced_audio_manager.py   # Core audio playback and queue management
├── music_library.py            # Music index and library loading
├── song_similarity.py          # Embedding-based song similarity
├── transition_mixer.py         # Crossfade and transition logic
├── dj_transition_model.py      # ML model for transition scoring
├── dj_transition_service.py    # Transition planning service
├── auth.py                     # JWT authentication routes
├── database.py                 # SQLite database init and connection
├── setlists.py                 # Setlist CRUD routes
├── upload_handler.py           # Song upload routes
├── feature_utils.py            # Audio feature sanitization helpers
├── user_song_maintenance.py    # User song feature repair utilities
├── llm/
│   └── new_llm.py              # Ollama LLM integration
├── models/
│   └── dj_transition_model/    # Trained XGBoost transition model
└── music_data/
    ├── audio/                  # Your MP3/WAV music files go here
    └── segmented_songs.json    # Pre-computed song segments and features
```

> **Note:** Several files in the repo (`old_app.py`, `audio_manager.py`, `audio_processor.py`, `pyo_devices.py`, `DJ_Functions.py`) are legacy and not actively used by the current `app.py`.

---

## Music Data

Before running, you need audio files and their pre-analyzed features:

1. Place your audio files in `music_data/audio/`
2. Ensure `music_data/segmented_songs.json` exists with segment and feature data for each track

The app expects songs to have BPM, key, energy, and segment metadata for transition scoring to work. If you have raw audio files, you'll need to run your feature extraction pipeline first to populate `segmented_songs.json`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AIDJ_ENV` | `development` | Set to `production` for prod mode |
| `AIDJ_DOMAIN` | `ai-dj.duckdns.org` | Your domain (used for CORS/SSL) |
| `AIDJ_HOST` | `0.0.0.0` (dev) / `127.0.0.1` (prod) | Bind address |
| `AIDJ_PORT` | `8000` | Port to listen on |

---

## Running the App

### Development

```bash
python app.py
```

The server will start at `http://0.0.0.0:8000`.

### Production

```bash
set AIDJ_ENV=production  # Windows
# or
export AIDJ_ENV=production  # Linux/Mac

python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

---

## Ollama Setup

The LLM component requires Ollama to be running. Install it from [ollama.com](https://ollama.com), pull a model, and make sure the service is active before starting the backend:

```bash
ollama serve
```

The app connects to Ollama via `langchain-ollama` to handle natural language DJ requests and intelligent queue suggestions.

---

## API Overview

The backend exposes:

- **WebSocket** — real-time DJ room control and playback state
- `POST /api/auth/...` — registration and login (JWT-based)
- `GET/POST /api/setlists/...` — setlist management
- `POST /api/upload` — song upload
- `POST /api/library/reload` — hot-reload the music library
- `POST /api/library/add-song` / `add-user-song` — add songs to the in-memory library at runtime
- `GET /api/queue/status` — current playback queue
- `POST /api/auto-play/toggle` — toggle auto DJ mode
- `POST /api/transition/preview` — generate a crossfade audio preview between two tracks

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web framework and ASGI server |
| `langchain-ollama` + `ollama` | LLM integration for track selection |
| `librosa` + `soundfile` | Audio analysis and feature extraction |
| `scikit-learn` + `xgboost` | Transition scoring model |
| `websockets` | Real-time client communication |
| `bcrypt` + `PyJWT` | Auth and token management |
| `pydub` | Audio manipulation |
| `numpy` / `scipy` | Signal processing |

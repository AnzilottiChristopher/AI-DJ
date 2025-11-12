# new_llm.py
from typing import Callable, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, re

class LlamaLLM:
    def __init__(self):
        self.llm = ChatOllama(model="phi3:mini", temperature=0)
        self._queue = []
        self.prompt = ChatPromptTemplate.from_template(
            """You are an intent classifier for a DJ app.
        Return STRICT JSON with fields:
        - intent: one of ["hello","start_dj","stop_dj","help","queue_song","none"]
        - reason: short string
        - song: object with fields
        - title: string | null
        - artist: string | null

        User input: "{user_input}"

        Rules:
        - If the user asks to start/queue/launch dj/music/mixing, intent="start_dj".
        - If they ask to stop/end/quit dj/music, intent="stop_dj".
        - If they greet (hello/hi/hey), intent="hello".
        - If they ask for help or commands, intent="help".
        - If they ask to queue/play/add a specific song, intent="queue_song" and fill song.title and (if supplied) song.artist.
        - Otherwise, intent="none".
        Return ONLY JSON, no code fences, no prose.

        Examples:
        Input: "queue up levels by avicii"
        Output: {{ "intent":"queue_song", "reason":"User asked to queue a specific track", "song": {{ "title": "Levels", "artist": "Avicii" }} }}

        Input: "add 'blinding lights' to the queue"
        Output: {{ "intent":"queue_song", "reason":"User asked to queue a specific track", "song": {{ "title": "Blinding Lights", "artist": null }} }}

        Input: "start the dj"
        Output: {{ "intent":"start_dj", "reason":"Start request", "song": {{ "title": null, "artist": null }} }}
        """
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

        # map intents -> bound methods; queue_song expects kwargs
        self.COMMANDS: Dict[str, Callable[..., None]] = {
            "hello": self.cmd_hello,
            "start_dj": self.cmd_start_dj,
            "stop_dj": self.cmd_stop_dj,
            "help": self.cmd_help,
            "queue_song": self.cmd_queue_song,
            "none": self.cmd_none,
        }
        

    # ---------- commands ----------
    def cmd_hello(self): print("[CMD] Hello! This is the hello command.")
    def cmd_start_dj(self): print("[CMD] Starting DJ pipeline... (placeholder)")
    def cmd_stop_dj(self): print("[CMD] Stopping DJ pipeline... (placeholder)")
    def cmd_help(self): print("[CMD] Available commands: hello, start_dj, stop_dj, help, queue_song")
    def cmd_none(self): print("[NO ACTION] No matching command.")

    def queue_track(self, title: Optional[str], artist: Optional[str]):
        if not title: 
            print("[WARN] No title extracted; ignoring.")
            return False
        self._queue.append({"title": title, "artist": artist})
        print("[QUEUE] ->", self._queue[-1])
        return True

    def cmd_queue_song(self, *, title, artist):
        ok = self.queue_track(title, artist)
        if ok:
            print(f"[CMD] Queued: {title}" + (f" — {artist}" if artist else ""))
        self.print_queue()
    
    def print_queue(self):
        if not self._queue:
            print("[QUEUE] The queue is currently empty.")
            return
        print("[QUEUE] Current queue:")
        for i, song in enumerate(self._queue, start=1):
            title = song.get("title") or "Unknown"
            artist = song.get("artist")
            print(f"  {i}. {title}" + (f" — {artist}" if artist else ""))

    # ---------- helpers ----------
    def extract_json(self, s: str) -> dict:
        if not s or not s.strip():
            return {"intent": "none", "reason": "Empty model output.", "song": {"title": None, "artist": None}}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"intent": "none", "reason": "Could not parse JSON.", "song": {"title": None, "artist": None}}

    # ---------- API ----------
    def classify(self, text: str) -> dict:
        raw = self.chain.invoke({"user_input": text})  # str from StrOutputParser
        data = self.extract_json(raw)

        # normalize / defaults
        intent = data.get("intent", "none")
        if intent not in self.COMMANDS:
            intent = "none"
        song = data.get("song") or {}
        title = song.get("title")
        artist = song.get("artist")

        return {
            "intent": intent,
            "reason": data.get("reason", ""),
            "song": {"title": title, "artist": artist},
        }

    def dispatch(self, intent: str, *, song: Optional[dict] = None):
        if intent == "queue_song":
            title = (song or {}).get("title")
            artist = (song or {}).get("artist")
            return self.COMMANDS[intent](title=title, artist=artist)
        return self.COMMANDS[intent]()

    def process_response(self, text: str) -> str:
        res = self.classify(text)
        self.dispatch(res["intent"], song=res.get("song"))
        # craft a friendly UI message
        if res["intent"] == "queue_song":
            t = res["song"].get("title") or "Unknown title"
            a = res["song"].get("artist")
            detail = f" — {a}" if a else ""
            return f"[INTENT] queue_song → {t}{detail}"
        return f"[INTENT] {res['intent']} — {res['reason']}"

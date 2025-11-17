# new_llm.py
from typing import Callable, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, re

class LlamaLLM:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.3, repeat_penalty=1.1)
        self._queue = []
        self.prompt = ChatPromptTemplate.from_template(
            """You are a JSON-only response bot for a DJ app. Classify the user's intent and extract song info if applicable.

        User says: "{user_input}"

        Return ONLY this JSON format:
        {{"intent":"INTENT_TYPE", "reason":"brief explanation", "song":{{"title":"SONG_NAME_OR_NULL", "artist":"ARTIST_NAME_OR_NULL"}}}}

        Intent types:
        - "queue_song": User wants to play/queue/add a SPECIFIC song (e.g., "play waiting for love", "add levels by avicii")
        - "start_dj": User wants to start DJ/music with NO specific song (e.g., "start the dj", "play some music")
        - "stop_dj": User wants to stop the music (e.g., "stop", "end the music")
        - "hello": User is greeting (e.g., "hi", "hello")
        - "help": User asks for help (e.g., "what can you do", "help")
        - "none": Anything else

        Rules:
        - If user mentions a SPECIFIC song name, intent is "queue_song"
        - Use EXACT song name from user's input (capitalize properly)
        - Correct spelling errors intelligently
        - Extract artist if provided, otherwise null
        - For queue_song, always fill song.title; for other intents, set both to null

        Examples:
        "play stargazing by kygo" → {{"intent":"queue_song", "reason":"specific song request", "song":{{"title":"Stargazing", "artist":"Kygo"}}}}
        "start the music" → {{"intent":"start_dj", "reason":"generic start request", "song":{{"title":null, "artist":null}}}}
        "stop" → {{"intent":"stop_dj", "reason":"stop request", "song":{{"title":null, "artist":null}}}}

        Return ONLY the JSON, nothing else."""
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
    # def extract_json(self, s: str) -> dict:
    #     if not s or not s.strip():
    #         return {"intent": "none", "reason": "Empty model output.", "song": {"title": None, "artist": None}}
    #     try:
    #         return json.loads(s)
    #     except json.JSONDecodeError:
    #         pass
    #     m = re.search(r"\{.*\}", s, re.S)
    #     if m:
    #         try:
    #             return json.loads(m.group(0))
    #         except json.JSONDecodeError:
    #             pass
    #     return {"intent": "none", "reason": "Could not parse JSON.", "song": {"title": None, "artist": None}}

    # ---------- helpers ----------
    def extract_json(self, s: str) -> dict:
        if not s or not s.strip():
            return {"intent": "none", "reason": "Empty model output.", "song": {"title": None, "artist": None}}
        
        # Print raw output for debugging
        # print(f"[DEBUG] Raw LLM output: {s[:200]}...")  # First 200 chars
        
        # Try direct parse first
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        
        # Remove markdown code fences
        s_clean = s.strip()
        if s_clean.startswith("```"):
            # Remove ```json or ``` at start
            s_clean = re.sub(r'^```(?:json)?\s*\n?', '', s_clean)
            # Remove ``` at end
            s_clean = re.sub(r'\n?```\s*$', '', s_clean)
        
        # Try parsing cleaned version
        try:
            return json.loads(s_clean.strip())
        except json.JSONDecodeError:
            pass
        
        # Search for JSON object anywhere in string
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    
        print(f"[ERROR] Could not parse JSON from: {s}")
        return {"intent": "none", "reason": "Could not parse JSON.", "song": {"title": None, "artist": None}}
    # ---------- API ----------
    # def classify(self, text: str) -> dict:
    #     print("Classifying prompt!")
    #     raw = self.chain.invoke({"user_input": text})  # str from StrOutputParser
    #     data = self.extract_json(raw)

    #     # normalize / defaults
    #     intent = data.get("intent", "none")
    #     if intent not in self.COMMANDS:
    #         intent = "none"
    #     song = data.get("song") or {}
    #     title = song.get("title")
    #     artist = song.get("artist")

    #     return {
    #         "intent": intent,
    #         "reason": data.get("reason", ""),
    #         "song": {"title": title, "artist": artist},
    #     }
    def classify(self, text: str) -> dict:
        # print("="*50)
        # print(f"[DEBUG] CLASSIFY CALLED")
        # print(f"[DEBUG] Input text: '{text}'")
        # print("="*50)
        
        # Invoke the chain
        raw = self.chain.invoke({"user_input": text})
        
        # print(f"[DEBUG] Raw LLM output (full): '{raw}'")
        # print("="*50)
        
        data = self.extract_json(raw)
        print(f"[DEBUG] Parsed JSON: {data}")
        
        # normalize / defaults
        intent = data.get("intent", "none")
        if intent not in self.COMMANDS:
            intent = "none"
        song = data.get("song") or {}
        title = song.get("title")
        artist = song.get("artist")

        result = {
            "intent": intent,
            "reason": data.get("reason", ""),
            "song": {"title": title, "artist": artist},
        }
        
        # print(f"[DEBUG] Final result: {result}")
        # print("="*50)
        
        return result

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

# new_llm.py - Updated version
from typing import Callable, Dict, Any, Optional, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, re

class LlamaLLM:
    def __init__(self, music_library=None):
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.3, repeat_penalty=1.1)
        self._queue = []
        self.music_library = music_library
        
        # Original classification prompt
        self.prompt = ChatPromptTemplate.from_template(
            """You are a JSON-only response bot for a DJ app. Classify the user's intent and extract song info if applicable.

User says: "{user_input}"

Return ONLY this JSON format:
{{"intent":"INTENT_TYPE", "reason":"brief explanation", "song":{{"title":"SONG_NAME_OR_NULL", "artist":"ARTIST_NAME_OR_NULL"}}}}

Intent types:
- "queue_song": User wants to play/queue/add a SPECIFIC song (e.g., "play waiting for love", "add levels by avicii")
- "generate_playlist": User wants multiple songs based on a mood/vibe/criteria (e.g., "give me upbeat songs", "play something danceable", "make me a playlist for a party")
- "start_dj": User wants to start DJ/music with NO specific song (e.g., "start the dj", "play some music")
- "stop_dj": User wants to stop the music (e.g., "stop", "end the music")
- "hello": User is greeting (e.g., "hi", "hello")
- "help": User asks for help (e.g., "what can you do", "help")
- "none": Anything else

Rules:
- If user mentions a SPECIFIC song name, intent is "queue_song"
- If user asks for songs based on mood, vibe, energy, or style WITHOUT naming a specific song, intent is "generate_playlist"
- Use EXACT song name from user's input (capitalize properly)
- For queue_song, always fill song.title; for other intents, set both to null

Examples:
"play stargazing by kygo" → {{"intent":"queue_song", "reason":"specific song request", "song":{{"title":"Stargazing", "artist":"Kygo"}}}}
"give me some high energy songs" → {{"intent":"generate_playlist", "reason":"mood-based playlist request", "song":{{"title":null, "artist":null}}}}
"play something danceable" → {{"intent":"generate_playlist", "reason":"vibe-based request", "song":{{"title":null, "artist":null}}}}
"start the music" → {{"intent":"start_dj", "reason":"generic start request", "song":{{"title":null, "artist":null}}}}

Return ONLY the JSON, nothing else."""
        )
        
        # Playlist generation prompt - this is where the magic happens
        self.playlist_prompt = ChatPromptTemplate.from_template(
            """You are a DJ assistant. Given a user's request and a song library, select the best matching songs.

SONG LIBRARY:
{song_library}

METADATA GUIDE:
- danceability: higher = more danceable (typical range 0.9-1.3)
- loudness: 0-1 scale, higher = louder/more intense
- bpm: beats per minute (higher = faster)
- energy: onset_rate, higher = more energetic/busy
- key/scale: musical key and whether major (happy) or minor (darker)

USER REQUEST: "{user_input}"

Select 3-5 songs that best match this request. Consider:
1. The metadata values that relate to the request
2. Your knowledge of these songs/artists if you recognize them
3. How well the songs would flow together

Return ONLY a JSON array of filenames from the library, no explanation:
["song1.wav", "song2.wav", "song3.wav"]

If no songs match well, return the 3 most generally appropriate songs.
Return ONLY the JSON array, nothing else."""
        )
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.playlist_chain = self.playlist_prompt | self.llm | StrOutputParser()

        self.COMMANDS: Dict[str, Callable[..., None]] = {
            "hello": self.cmd_hello,
            "start_dj": self.cmd_start_dj,
            "stop_dj": self.cmd_stop_dj,
            "help": self.cmd_help,
            "queue_song": self.cmd_queue_song,
            "generate_playlist": self.cmd_generate_playlist,
            "none": self.cmd_none,
        }

    # ---------- commands ----------
    def cmd_hello(self): print("[CMD] Hello! This is the hello command.")
    def cmd_start_dj(self): print("[CMD] Starting DJ pipeline... (placeholder)")
    def cmd_stop_dj(self): print("[CMD] Stopping DJ pipeline... (placeholder)")
    def cmd_help(self): print("[CMD] Available commands: hello, start_dj, stop_dj, help, queue_song, generate_playlist")
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
    
    def cmd_generate_playlist(self, *, playlist: List[Dict]):
        """Handle playlist generation result."""
        print(f"[CMD] Generated playlist with {len(playlist)} songs")
        for song in playlist:
            self.queue_track(song.get('title'), song.get('artist'))
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

    def extract_json(self, s: str) -> dict:
        if not s or not s.strip():
            return {"intent": "none", "reason": "Empty model output.", "song": {"title": None, "artist": None}}
        
        s_clean = s.strip()
        if s_clean.startswith("```"):
            s_clean = re.sub(r'^```(?:json)?\s*\n?', '', s_clean)
            s_clean = re.sub(r'\n?```\s*$', '', s_clean)
        
        try:
            return json.loads(s_clean.strip())
        except json.JSONDecodeError:
            pass
        
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    
        print(f"[ERROR] Could not parse JSON from: {s}")
        return {"intent": "none", "reason": "Could not parse JSON.", "song": {"title": None, "artist": None}}
    
    def extract_playlist_json(self, s: str) -> List[str]:
        """Extract a JSON array of filenames from LLM response."""
        if not s or not s.strip():
            return []
        
        s_clean = s.strip()
        # Remove markdown code fences
        if s_clean.startswith("```"):
            s_clean = re.sub(r'^```(?:json)?\s*\n?', '', s_clean)
            s_clean = re.sub(r'\n?```\s*$', '', s_clean)
        
        try:
            result = json.loads(s_clean.strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        
        # Try to find array in the string
        m = re.search(r'\[.*?\]', s, re.S)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        
        print(f"[ERROR] Could not parse playlist JSON from: {s}")
        return []

    def classify(self, text: str) -> dict:
        raw = self.chain.invoke({"user_input": text})
        data = self.extract_json(raw)
        
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
        
        print(f"[DEBUG] Classified intent: {intent}")
        return result
    
    def generate_playlist(self, user_request: str) -> List[Dict]:
        """
        Generate a playlist based on the user's request.
        
        Args:
            user_request: The user's natural language request
            
        Returns:
            List of dicts with 'title' and 'artist' keys
        """
        if not self.music_library:
            print("[ERROR] No music library available for playlist generation")
            return []
        
        # Get the library in LLM-friendly format
        song_library = self.music_library.get_library_for_llm()
        
        print(f"[PLAYLIST] Generating playlist for: '{user_request}'")
        print(f"[PLAYLIST] Library has {len(self.music_library.index)} songs")
        
        # Call the playlist chain
        raw = self.playlist_chain.invoke({
            "song_library": song_library,
            "user_input": user_request
        })
        
        print(f"[PLAYLIST] Raw LLM response: {raw}")
        
        # Parse the response
        filenames = self.extract_playlist_json(raw)
        
        if not filenames:
            print("[PLAYLIST] No valid filenames returned, using fallback")
            # Fallback: return first 3 songs
            filenames = [data['filename'] for _, data in list(self.music_library.index.items())[:3]]
        
        # Validate and convert filenames to title/artist format
        playlist = []
        for filename in filenames:
            # Find the song in the library
            found = False
            for key, data in self.music_library.index.items():
                if data['filename'] == filename:
                    # Parse title and artist from the key
                    # Key format is like "hey brother avicii"
                    parts = key.rsplit(' ', 1)  # Split from the right to get artist
                    if len(parts) == 2:
                        # Try to figure out where title ends and artist begins
                        # This is tricky - let's use the filename which has underscore separator
                        name = filename.replace('.wav', '')
                        if '_' in name:
                            title_part, artist_part = name.rsplit('_', 1)
                            title = title_part.replace('-', ' ').title()
                            artist = artist_part.replace('-', ' ').title()
                        else:
                            title = key.title()
                            artist = None
                    else:
                        title = key.title()
                        artist = None
                    
                    playlist.append({"title": title, "artist": artist, "filename": filename})
                    found = True
                    break
            
            if not found:
                print(f"[PLAYLIST] Warning: filename not found in library: {filename}")
        
        print(f"[PLAYLIST] Generated playlist: {[p['title'] for p in playlist]}")
        return playlist

    def dispatch(self, intent: str, *, song: Optional[dict] = None, playlist: Optional[List[Dict]] = None):
        if intent == "queue_song":
            title = (song or {}).get("title")
            artist = (song or {}).get("artist")
            return self.COMMANDS[intent](title=title, artist=artist)
        elif intent == "generate_playlist":
            return self.COMMANDS[intent](playlist=playlist or [])
        return self.COMMANDS[intent]()

    def process_response(self, text: str) -> str:
        res = self.classify(text)
        
        # If it's a playlist request, generate the playlist
        if res["intent"] == "generate_playlist":
            playlist = self.generate_playlist(text)
            self.dispatch(res["intent"], playlist=playlist)
            if playlist:
                titles = [p['title'] for p in playlist]
                return f"[PLAYLIST] Generated {len(playlist)} songs: {', '.join(titles)}"
            return "[PLAYLIST] Could not generate playlist"
        
        self.dispatch(res["intent"], song=res.get("song"))
        
        if res["intent"] == "queue_song":
            t = res["song"].get("title") or "Unknown title"
            a = res["song"].get("artist")
            detail = f" — {a}" if a else ""
            return f"[INTENT] queue_song → {t}{detail}"
        return f"[INTENT] {res['intent']} — {res['reason']}"
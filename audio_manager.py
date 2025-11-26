# audio_manager.py
import asyncio
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, List
import struct

class AudioManager:
    def __init__(self, music_library):
        self.music_library = music_library
        self.queue: List[dict] = []
        self.current_track: Optional[dict] = None
        self.is_playing = False
        self.sample_rate = 44100
        self.chunk_size = 4096  # samples per chunk
        
    def add_to_queue(self, title, artist) -> bool:

        track_data = self.music_library.search(title, artist)
        if track_data:
            self.queue.append({
                'title': title,
                'artist': artist,
                'track_data': track_data
            })
            print(f"[QUEUE] Added: {title}" + (f" by {artist}" if artist else ""))
            return True
        else:
            print(f"[QUEUE] Not found: {title}")
            return False
    
    def get_next_track(self) -> Optional[dict]:
        """Get next track from queue"""
        if self.queue:
            return self.queue.pop(0)
        return None
    
    async def stream_audio(self, websocket, track_data: dict):
        """Stream audio data to websocket in chunks"""
        try:
            # Load audio file
            audio, sr = sf.read(str(track_data['path']))
            
            # Convert to stereo if mono
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio], axis=1)
            
            # Resample if needed (simplified - you may want proper resampling)
            if sr != self.sample_rate:
                print(f"[WARN] Sample rate mismatch: {sr} vs {self.sample_rate}")
            
            # Convert to int16 for efficient transmission
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Send metadata first
            duration = len(audio) / sr
            await websocket.send_json({
                "type": "track_start",
                "track": {
                    "title": track_data.get('title', 'Unknown'),
                    "bpm": track_data['features']['bpm'],
                    "key": track_data['features']['key'],
                    "duration": duration,
                    "sample_rate": self.sample_rate
                }
            })
            
            # Stream audio in chunks
            total_samples = len(audio_int16)
            for i in range(0, total_samples, self.chunk_size):
                if not self.is_playing:
                    break
                    
                chunk = audio_int16[i:i + self.chunk_size]
                
                # Convert to bytes
                audio_bytes = chunk.tobytes()
                
                # Send as binary data
                await websocket.send_bytes(audio_bytes)
                print('sent some bytes')
                
                # Small delay to simulate real-time playback
                # (chunk_size / sample_rate / channels)
                await asyncio.sleep(self.chunk_size / self.sample_rate / 2 * 0.9)
            
            # Signal track end
            await websocket.send_json({
                "type": "track_end"
            })
            
            print(f"[STREAM] Finished streaming")
            
        except Exception as e:
            print(f"[STREAM ERROR] {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    async def play_queue(self, websocket):
        """Play all songs in queue sequentially"""
        self.is_playing = True
        
        while self.is_playing and (self.queue or self.current_track):
            if not self.current_track:
                self.current_track = self.get_next_track()
                
            if self.current_track:
                await self.stream_audio(websocket, self.current_track['track_data'])
                self.current_track = None
            else:
                break
        
        self.is_playing = False
        await websocket.send_json({
            "type": "queue_empty",
            "message": "Queue finished"
        })
    
    def start(self):
        """Start playback"""
        self.is_playing = True
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
    
    def get_queue_status(self) -> dict:
        """Get current queue status"""
        return {
            "is_playing": self.is_playing,
            "current_track": self.current_track.get('title') if self.current_track else None,
            "queue_length": len(self.queue),
            "queue": [
                {"title": q['title'], "artist": q['artist']} 
                for q in self.queue
            ]
        }

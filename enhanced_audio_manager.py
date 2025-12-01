"""
Enhanced Audio Manager with DJ Transition Support

This module manages audio playback with intelligent mixing between tracks.
When a new song is queued while one is playing, it computes the optimal
transition and streams the mixed audio seamlessly.
"""

import asyncio
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from transition_mixer import TransitionMixer, TransitionPlan, generate_placeholder_segments


class PlaybackState(Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    TRANSITIONING = "transitioning"


@dataclass 
class TrackInfo:
    """Information about a track in the queue or playing."""
    title: str
    artist: str
    track_data: Dict
    audio: Optional[np.ndarray] = None
    duration: float = 0.0


class EnhancedAudioManager:
    """
    Audio manager with intelligent DJ transitions.
    
    Key features:
    - Streams audio via WebSocket
    - Computes optimal transitions using ML model
    - Creates smooth crossfades between tracks
    - Tracks playback position for timing
    """
    
    def __init__(self, music_library, model_path: str = None):
        """
        Initialize the audio manager.
        
        Args:
            music_library: MusicLibrary instance for looking up songs
            model_path: Path to trained transition model (optional, enables smart mixing)
        """
        self.music_library = music_library
        self.queue: List[TrackInfo] = []
        self.current_track: Optional[TrackInfo] = None
        self.state = PlaybackState.STOPPED
        
        # Audio settings
        self.sample_rate = 44100
        self.chunk_size = 4096  # samples per chunk
        
        # Playback tracking
        self.current_position = 0.0  # seconds into current track
        self.samples_sent = 0
        
        # Transition mixing
        self.mixer: Optional[TransitionMixer] = None
        self.pending_transition: Optional[TransitionPlan] = None
        self.transition_audio: Optional[Dict] = None
        
        if model_path:
            try:
                self.mixer = TransitionMixer(model_path, self.sample_rate)
                print(f"[AUDIO] Transition mixer loaded from {model_path}")
            except Exception as e:
                print(f"[AUDIO] Could not load transition mixer: {e}")
                self.mixer = None
    
    def _load_audio(self, track_data: Dict) -> tuple[np.ndarray, float]:
        """Load audio file and return (audio_array, duration)."""
        audio, sr = sf.read(str(track_data['path']))
        
        # Convert mono to stereo
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        
        # Resample if needed (basic - for production use librosa)
        if sr != self.sample_rate:
            print(f"[AUDIO] Warning: Sample rate mismatch {sr} vs {self.sample_rate}")
        
        duration = len(audio) / sr
        return audio, duration
    
    def _ensure_segments(self, track_data: Dict, duration: float) -> Dict:
        """Ensure track has segment data, generating placeholders if needed."""
        if 'segments' not in track_data or not track_data['segments']:
            bpm = track_data.get('features', {}).get('bpm', 120.0)
            track_data['segments'] = generate_placeholder_segments(duration, bpm)
            print(f"[AUDIO] Generated placeholder segments for track")
        return track_data
    
    def add_to_queue(self, title: str, artist: str) -> bool:
        """
        Add a song to the queue.
        
        If a song is currently playing and mixer is available,
        this will trigger transition computation.
        
        Returns True if song was found and added.
        """
        track_data = self.music_library.search(title, artist)
        
        if not track_data:
            print(f"[QUEUE] Not found: {title}")
            return False
        
        track_info = TrackInfo(
            title=title,
            artist=artist or "Unknown Artist",
            track_data=track_data
        )
        
        self.queue.append(track_info)
        print(f"[QUEUE] Added: {title}" + (f" by {artist}" if artist else ""))
        
        # If we're playing and have a mixer, prepare the transition
        if self.state == PlaybackState.PLAYING and self.mixer and self.current_track:
            asyncio.create_task(self._prepare_transition(track_info))
        
        return True
    
    async def _prepare_transition(self, next_track: TrackInfo):
        """
        Prepare transition to the next track asynchronously.
        """
        if not self.current_track or not self.mixer:
            return
        
        print(f"[MIXER] Preparing transition: {self.current_track.title} → {next_track.title}")
        
        try:
            # Load next track audio
            next_audio, next_duration = self._load_audio(next_track.track_data)
            next_track.audio = next_audio
            next_track.duration = next_duration
            
            # Ensure both tracks have segment data
            current_data = self._ensure_segments(
                self.current_track.track_data.copy(),
                self.current_track.duration
            )
            current_data['title'] = self.current_track.title
            
            next_data = self._ensure_segments(
                next_track.track_data.copy(),
                next_duration
            )
            next_data['title'] = next_track.title
            
            # Compute optimal transition
            plan = self.mixer.compute_transition(
                song_a_data=current_data,
                song_b_data=next_data,
                current_position=self.current_position
            )
            
            if plan:
                self.pending_transition = plan
                
                # Pre-compute the mixed audio
                self.transition_audio = self.mixer.prepare_mixed_audio(
                    self.current_track.audio,
                    next_audio,
                    plan
                )
                
                print(f"[MIXER] Transition ready: will start at {plan.transition_start_time:.1f}s")
            else:
                print("[MIXER] Could not compute transition, will use simple cut")
                self.pending_transition = None
                
        except Exception as e:
            print(f"[MIXER] Error preparing transition: {e}")
            self.pending_transition = None
    
    def get_next_track(self) -> Optional[TrackInfo]:
        """Get next track from queue."""
        if self.queue:
            return self.queue.pop(0)
        return None
    
    async def stream_audio(self, websocket, track_info: TrackInfo):
        """
        Stream audio data to WebSocket.
        
        Handles both normal playback and transitions.
        """
        try:
            # Load audio if not already loaded
            if track_info.audio is None:
                track_info.audio, track_info.duration = self._load_audio(track_info.track_data)
            
            # Ensure segments exist
            self._ensure_segments(track_info.track_data, track_info.duration)
            
            # Convert to int16 for transmission
            audio_int16 = (track_info.audio * 32767).astype(np.int16)
            
            # Send track metadata
            await websocket.send_json({
                "type": "track_start",
                "track": {
                    "title": track_info.title,
                    "artist": track_info.artist,
                    "bpm": track_info.track_data['features'].get('bpm', 120),
                    "key": track_info.track_data['features'].get('key', 'C'),
                    "duration": track_info.duration,
                    "sample_rate": self.sample_rate
                }
            })
            
            # Reset position tracking
            self.samples_sent = 0
            self.current_position = 0.0
            
            total_samples = len(audio_int16)
            i = 0
            
            while i < total_samples and self.state != PlaybackState.STOPPED:
                # Update position
                self.current_position = self.samples_sent / self.sample_rate
                
                # Check if we should start a transition
                if self._should_start_transition():
                    # Transition to the mixed audio
                    await self._stream_transition(websocket)
                    return  # _stream_transition handles the rest
                
                # Normal chunk streaming
                chunk = audio_int16[i:i + self.chunk_size]
                await websocket.send_bytes(chunk.tobytes())
                
                i += self.chunk_size
                self.samples_sent += len(chunk)
                
                # Pace the streaming
                await asyncio.sleep(self.chunk_size / self.sample_rate / 2 * 0.9)
            
            # Track ended naturally
            await websocket.send_json({"type": "track_end"})
            print(f"[STREAM] Finished: {track_info.title}")
            
        except Exception as e:
            print(f"[STREAM ERROR] {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    def _should_start_transition(self) -> bool:
        """Check if we should start the transition now."""
        if not self.pending_transition or not self.transition_audio:
            return False
        
        # Start transition when we reach the planned time
        # Add a small buffer for processing
        buffer = 0.5  # seconds
        return self.current_position >= (self.pending_transition.transition_start_time - buffer)
    
    async def _stream_transition(self, websocket):
        """
        Stream the transition (crossfade) and continue with the next song.
        """
        if not self.transition_audio or not self.pending_transition:
            return
        
        self.state = PlaybackState.TRANSITIONING
        plan = self.pending_transition
        
        print(f"[MIXER] Starting transition at {self.current_position:.1f}s")
        
        # Notify frontend about the transition
        await websocket.send_json({
            "type": "transition_start",
            "transition": plan.to_dict()
        })
        
        try:
            # Stream the crossfade audio
            crossfade = self.transition_audio['crossfade']
            crossfade_int16 = (crossfade * 32767).astype(np.int16)
            
            for i in range(0, len(crossfade_int16), self.chunk_size):
                if self.state == PlaybackState.STOPPED:
                    break
                    
                chunk = crossfade_int16[i:i + self.chunk_size]
                await websocket.send_bytes(chunk.tobytes())
                await asyncio.sleep(self.chunk_size / self.sample_rate / 2 * 0.9)
            
            # Transition complete - now stream the rest of song B
            await websocket.send_json({
                "type": "transition_complete",
                "now_playing": {
                    "title": plan.song_b_title,
                }
            })
            
            # Get the next track from queue and stream remaining audio
            next_track = self.get_next_track()
            if next_track and self.transition_audio['post_transition'] is not None:
                # Update current track
                self.current_track = next_track
                
                # Send new track info
                await websocket.send_json({
                    "type": "track_start",
                    "track": {
                        "title": next_track.title,
                        "artist": next_track.artist,
                        "bpm": next_track.track_data['features'].get('bpm', 120),
                        "key": next_track.track_data['features'].get('key', 'C'),
                        "duration": next_track.duration,
                        "sample_rate": self.sample_rate,
                        "is_continuation": True  # Frontend knows we're mid-song
                    }
                })
                
                # Stream remaining audio
                post_audio = self.transition_audio['post_transition']
                post_int16 = (post_audio * 32767).astype(np.int16)
                
                self.samples_sent = 0
                self.current_position = self.transition_audio['timing']['song_b_continue_sample'] / self.sample_rate
                
                for i in range(0, len(post_int16), self.chunk_size):
                    if self.state == PlaybackState.STOPPED:
                        break
                    
                    self.current_position = (self.transition_audio['timing']['song_b_continue_sample'] + self.samples_sent) / self.sample_rate
                    
                    chunk = post_int16[i:i + self.chunk_size]
                    await websocket.send_bytes(chunk.tobytes())
                    self.samples_sent += len(chunk)
                    await asyncio.sleep(self.chunk_size / self.sample_rate / 2 * 0.9)
                
                await websocket.send_json({"type": "track_end"})
            
            # Clear transition state
            self.pending_transition = None
            self.transition_audio = None
            self.state = PlaybackState.PLAYING
            
        except Exception as e:
            print(f"[TRANSITION ERROR] {e}")
            self.state = PlaybackState.PLAYING
    
    async def play_queue(self, websocket):
        """Play all songs in queue with transitions."""
        self.state = PlaybackState.PLAYING
        
        while self.state != PlaybackState.STOPPED:
            if not self.current_track:
                self.current_track = self.get_next_track()
            
            if self.current_track:
                await self.stream_audio(websocket, self.current_track)
                self.current_track = None
            else:
                # Queue is empty
                break
        
        self.state = PlaybackState.STOPPED
        await websocket.send_json({
            "type": "queue_empty",
            "message": "Queue finished"
        })
    
    def start(self):
        """Start playback."""
        self.state = PlaybackState.PLAYING
    
    def stop(self):
        """Stop playback."""
        self.state = PlaybackState.STOPPED
        self.pending_transition = None
        self.transition_audio = None
    
    def get_queue_status(self) -> Dict:
        """Get current queue and playback status."""
        return {
            "state": self.state.value,
            "current_track": self.current_track.title if self.current_track else None,
            "current_position": round(self.current_position, 1),
            "pending_transition": self.pending_transition.to_dict() if self.pending_transition else None,
            "queue_length": len(self.queue),
            "queue": [
                {"title": t.title, "artist": t.artist}
                for t in self.queue
            ]
        }


# Backwards compatibility - original AudioManager interface
class AudioManager(EnhancedAudioManager):
    """
    Drop-in replacement for the original AudioManager.
    Adds transition support while maintaining the same API.
    """
    
    def __init__(self, music_library, model_path: str = None):
        # Try to find model in common locations
        if model_path is None:
            possible_paths = [
                'trained_dj_model',
                'models/trained_dj_model',
                '../models/trained_dj_model',
            ]
            for path in possible_paths:
                try:
                    super().__init__(music_library, path)
                    return
                except:
                    continue
            
            # No model found, initialize without mixing
            print("[AUDIO] No transition model found, mixing disabled")
            super().__init__(music_library, None)
        else:
            super().__init__(music_library, model_path)
    
    @property
    def is_playing(self) -> bool:
        """Backwards compatibility property."""
        return self.state != PlaybackState.STOPPED
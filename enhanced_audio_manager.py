"""
Enhanced Audio Manager with DJ Transition Support

This module manages audio playback with intelligent mixing between tracks.
When a new song is queued while one is playing, it computes the optimal
transition and streams the mixed audio seamlessly.

Features:
- Auto-play: Automatically queues similar songs when queue is empty
- Smart transitions: Uses ML model to compute optimal crossfade points
- User override: User requests always take priority over auto-queued songs
"""

import asyncio
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from transition_mixer import TransitionMixer, TransitionPlan, generate_placeholder_segments
from song_similarity import SongSimilarityService


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
    song_key: str = ""  # Key in music library index
    audio: Optional[np.ndarray] = None
    duration: float = 0.0
    is_auto_queued: bool = False  # True if auto-queued by similarity


class EnhancedAudioManager:
    """
    Audio manager with intelligent DJ transitions and auto-play.
    
    Key features:
    - Streams audio via WebSocket
    - Computes optimal transitions using ML model
    - Creates smooth crossfades between tracks
    - Auto-queues similar songs when queue is empty
    - Tracks playback position for timing
    """
    
    def __init__(self, music_library, model_path: str = None, enable_auto_play: bool = True):
        """
        Initialize the audio manager.
        
        Args:
            music_library: MusicLibrary instance for looking up songs
            model_path: Path to trained transition model (optional, enables smart mixing)
            enable_auto_play: Whether to automatically queue similar songs
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
        
        # Auto-play / similarity
        self.enable_auto_play = enable_auto_play
        self.similarity_service: Optional[SongSimilarityService] = None
        self.recently_played: List[str] = []  # Track song keys to avoid repeats
        self.max_history = 10  # How many songs to remember for avoiding repeats
        
        # WebSocket reference for sending transition notifications
        self._websocket = None
        
        # Initialize transition mixer
        if model_path:
            try:
                self.mixer = TransitionMixer(model_path, self.sample_rate)
                print(f"[AUDIO] Transition mixer loaded from {model_path}")
            except Exception as e:
                print(f"[AUDIO] Could not load transition mixer: {e}")
                self.mixer = None
        
        # Initialize similarity service for auto-play
        if enable_auto_play:
            try:
                self.similarity_service = SongSimilarityService(music_library)
                print(f"[AUDIO] Auto-play enabled with similarity service")
            except Exception as e:
                print(f"[AUDIO] Could not initialize similarity service: {e}")
                self.similarity_service = None
    
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
    
    def _get_song_key(self, track_data: Dict) -> str:
        """Get the song key from track data for similarity lookups."""
        if self.similarity_service:
            key = self.similarity_service.get_song_key_from_track_data(track_data)
            if key:
                return key
        return ""
    
    def _add_to_history(self, song_key: str):
        """Add a song to the recently played history."""
        if song_key and song_key not in self.recently_played:
            self.recently_played.append(song_key)
            # Trim history if too long
            if len(self.recently_played) > self.max_history:
                self.recently_played.pop(0)
    
    def _auto_queue_next_song(self) -> bool:
        """
        Automatically queue the next similar song.
        
        Returns True if a song was successfully queued.
        """
        if not self.enable_auto_play or not self.similarity_service:
            return False
        
        if not self.current_track or not self.current_track.song_key:
            return False
        
        # Don't auto-queue if there's already something in the queue
        if self.queue:
            return False
        
        # Find the next similar song
        next_key = self.similarity_service.get_next_song(
            self.current_track.song_key,
            exclude_keys=self.recently_played
        )
        
        if not next_key:
            print("[AUTO-PLAY] No similar songs available")
            return False
        
        # Get the track data from the library
        track_data = self.music_library.index.get(next_key)
        if not track_data:
            print(f"[AUTO-PLAY] Could not find track data for: {next_key}")
            return False
        
        # Get properly formatted title and artist from similarity service
        song_info = self.similarity_service.get_song_info(next_key)
        if song_info:
            title = song_info['title']
            artist = song_info['artist']
        else:
            # Fallback: try to parse from filename directly
            from song_similarity import parse_title_artist_from_filename
            filename = track_data.get('filename', next_key)
            title, artist = parse_title_artist_from_filename(filename)
        
        track_info = TrackInfo(
            title=title,
            artist=artist,
            track_data=track_data,
            song_key=next_key,
            is_auto_queued=True
        )
        
        self.queue.append(track_info)
        print(f"[AUTO-PLAY] Queued: {title} by {artist}")
        
        return True
    
    def add_to_queue(self, title: str, artist: str) -> bool:
        """
        Add a song to the queue.
        
        If a song is currently playing and mixer is available,
        this will trigger transition computation.
        
        User-requested songs always override auto-queued songs.
        
        Returns True if song was found and added.
        """
        track_data = self.music_library.search(title, artist)
        
        if not track_data:
            print(f"[QUEUE] Not found: {title}")
            return False
        
        # Get song key for similarity tracking
        song_key = self._get_song_key(track_data)
        
        track_info = TrackInfo(
            title=title,
            artist=artist or "Unknown Artist",
            track_data=track_data,
            song_key=song_key,
            is_auto_queued=False  # User requested
        )
        
        # If there's an auto-queued song in the queue, replace it
        if self.queue and self.queue[0].is_auto_queued:
            old_track = self.queue[0]
            print(f"[QUEUE] Replacing auto-queued '{old_track.title}' with user request '{title}'")
            self.queue[0] = track_info
            # Clear any pending transition since we're changing the next song
            self.pending_transition = None
            self.transition_audio = None
        else:
            # Queue is empty or has user-requested songs, just append
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
                
                # Send notification to frontend
                if self._websocket:
                    try:
                        await self._websocket.send_json({
                            "type": "transition_planned",
                            "transition": plan.to_dict(),
                            "next_track": {
                                "title": next_track.title,
                                "artist": next_track.artist,
                                "is_auto_queued": next_track.is_auto_queued
                            }
                        })
                        print(f"[MIXER] Sent transition_planned to frontend")
                    except Exception as e:
                        print(f"[MIXER] Could not send transition notification: {e}")
            else:
                print("[MIXER] Could not compute transition, will use simple cut")
                self.pending_transition = None
                
        except Exception as e:
            print(f"[MIXER] Error preparing transition: {e}")
            import traceback
            traceback.print_exc()
            self.pending_transition = None
    
    def get_next_track(self) -> Optional[TrackInfo]:
        """Get next track from queue."""
        if self.queue:
            return self.queue.pop(0)
        return None
    
    async def _notify_auto_queue(self, track_info: TrackInfo):
        """Notify frontend about auto-queued track."""
        if self._websocket:
            try:
                await self._websocket.send_json({
                    "type": "auto_queued",
                    "message": f"Up next: {track_info.title}",
                    "track": {
                        "title": track_info.title,
                        "artist": track_info.artist,
                        "is_auto_queued": True
                    },
                    "queue": self.get_queue_status()
                })
            except Exception as e:
                print(f"[AUTO-PLAY] Could not send notification: {e}")
    
    async def stream_audio(self, websocket, track_info: TrackInfo):
        """
        Stream audio data to WebSocket.
        
        Handles both normal playback and transitions.
        """
        # Store websocket reference for transition notifications
        self._websocket = websocket
        
        try:
            # Load audio if not already loaded
            if track_info.audio is None:
                track_info.audio, track_info.duration = self._load_audio(track_info.track_data)
            
            # Ensure segments exist
            self._ensure_segments(track_info.track_data, track_info.duration)
            
            # Add to history for similarity exclusion
            if track_info.song_key:
                self._add_to_history(track_info.song_key)
            
            # Convert to int16 for transmission
            audio_clipped = np.clip(track_info.audio, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16) 
            
            # Send track metadata FIRST (before auto-queue messages)
            # This ensures frontend processes track_start before receiving queued track info
            await websocket.send_json({
                "type": "track_start",
                "track": {
                    "title": track_info.title,
                    "artist": track_info.artist,
                    "bpm": track_info.track_data['features'].get('bpm', 120),
                    "key": track_info.track_data['features'].get('key', 'C'),
                    "duration": track_info.duration,
                    "sample_rate": self.sample_rate,
                    "is_auto_queued": track_info.is_auto_queued
                }
            })
            
            # Auto-queue the next song AFTER track_start is sent
            # This way the queued track won't be cleared by onTrackStart
            if self._auto_queue_next_song():
                # Notify frontend about the auto-queued track
                if self.queue:
                    await self._notify_auto_queue(self.queue[0])

            # Prepare transition to the next song in queue (whether user-queued or auto-queued)
            if self.queue and self.mixer:
                await self._prepare_transition(self.queue[0])

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
                await asyncio.sleep(self.chunk_size / self.sample_rate * 0.98)
            
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
            
            # CRITICAL FIX: Clip audio before int16 conversion to prevent overflow
            crossfade_clipped = np.clip(crossfade, -1.0, 1.0)
            crossfade_int16 = (crossfade_clipped * 32767).astype(np.int16)
            
            for i in range(0, len(crossfade_int16), self.chunk_size):
                if self.state == PlaybackState.STOPPED:
                    break
                    
                chunk = crossfade_int16[i:i + self.chunk_size]
                await websocket.send_bytes(chunk.tobytes())
                await asyncio.sleep(self.chunk_size / self.sample_rate * 0.98)
            
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
                # Capture current transition data BEFORE preparing next transition
                post_audio = self.transition_audio['post_transition']
                song_b_continue_sample = self.transition_audio['timing']['song_b_continue_sample']

                # Update current track
                self.current_track = next_track

                # Add to history
                if next_track.song_key:
                    self._add_to_history(next_track.song_key)

                # Auto-queue the next song now that we've moved to a new track
                if self._auto_queue_next_song():
                    if self.queue:
                        await self._notify_auto_queue(self.queue[0])

                # Prepare transition to the next song in queue (whether user-queued or auto-queued)
                if self.queue and self.mixer:
                    await self._prepare_transition(self.queue[0])

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
                        "is_continuation": True,  # Frontend knows we're mid-song
                        "is_auto_queued": next_track.is_auto_queued
                    }
                })

                # CRITICAL FIX: Clip audio before int16 conversion
                post_clipped = np.clip(post_audio, -1.0, 1.0)
                post_int16 = (post_clipped * 32767).astype(np.int16)

                self.samples_sent = 0
                self.current_position = song_b_continue_sample / self.sample_rate

                # Debug: Log post-transition streaming info
                post_duration = len(post_int16) / self.sample_rate
                next_transition_time = self.pending_transition.transition_start_time if self.pending_transition else None
                print(f"[DEBUG] Starting post-transition for {self.current_track.title}")
                print(f"[DEBUG]   Post-transition duration: {post_duration:.1f}s")
                print(f"[DEBUG]   Current position starts at: {self.current_position:.1f}s")
                print(f"[DEBUG]   Next transition at: {next_transition_time}s")

                for i in range(0, len(post_int16), self.chunk_size):
                    if self.state == PlaybackState.STOPPED:
                        break

                    self.current_position = (song_b_continue_sample + self.samples_sent) / self.sample_rate

                    # Check if we should start the next transition (for chained playlist songs)
                    if self._should_start_transition():
                        print(f"[DEBUG] Transition triggered at position {self.current_position:.1f}s")
                        await self._stream_transition(websocket)
                        return  # Recursive transition handles the rest

                    chunk = post_int16[i:i + self.chunk_size]
                    await websocket.send_bytes(chunk.tobytes())
                    self.samples_sent += len(chunk)
                    await asyncio.sleep(self.chunk_size / self.sample_rate * 0.98)
                
                print(f"[DEBUG] Post-transition complete for {self.current_track.title}, position: {self.current_position:.1f}s")
                await websocket.send_json({"type": "track_end"})

                # Prepare transition for the next auto-queued song
                if self.queue and self.mixer:
                    await self._prepare_transition(self.queue[0])
            
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
        
        # Store websocket reference
        self._websocket = websocket
        
        while self.state != PlaybackState.STOPPED:
            if not self.current_track:
                self.current_track = self.get_next_track()
            
            if self.current_track:
                await self.stream_audio(websocket, self.current_track)
                self.current_track = None
            else:
                # Queue is empty - with auto-play this shouldn't happen often
                # but if similarity service isn't working, we end here
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
            "current_track": {
                "title": self.current_track.title,
                "artist": self.current_track.artist,
                "is_auto_queued": self.current_track.is_auto_queued
            } if self.current_track else None,
            "current_position": round(self.current_position, 1),
            "pending_transition": self.pending_transition.to_dict() if self.pending_transition else None,
            "queue_length": len(self.queue),
            "queue": [
                {
                    "title": t.title, 
                    "artist": t.artist,
                    "is_auto_queued": t.is_auto_queued
                }
                for t in self.queue
            ],
            "auto_play_enabled": self.enable_auto_play,
            "recently_played": self.recently_played[-5:]  # Last 5 songs
        }


# Backwards compatibility - original AudioManager interface
class AudioManager(EnhancedAudioManager):
    """
    Drop-in replacement for the original AudioManager.
    Adds transition support and auto-play while maintaining the same API.
    """
    
    def __init__(self, music_library, model_path: str = None, enable_auto_play: bool = True):
        # Try to find model in common locations
        if model_path is None:
            possible_paths = [
                'trained_dj_model',
                'models/trained_dj_model',
                '../models/trained_dj_model',
                'models/dj_transition_model',
            ]
            for path in possible_paths:
                try:
                    super().__init__(music_library, path, enable_auto_play)
                    return
                except:
                    continue
            
            # No model found, initialize without mixing
            print("[AUDIO] No transition model found, mixing disabled")
            super().__init__(music_library, None, enable_auto_play)
        else:
            super().__init__(music_library, model_path, enable_auto_play)
    
    @property
    def is_playing(self) -> bool:
        """Backwards compatibility property."""
        return self.state != PlaybackState.STOPPED
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
from datetime import time
import soundfile as sf
import numpy as np
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from transition_mixer import TransitionMixer, TransitionPlan, generate_placeholder_segments
from song_similarity import SongSimilarityService
from audio_processor import AudioProcessor


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
    effects_config: Dict[str, Any] = field(default_factory=dict)  # Audio effects to apply


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
    
    def __init__(self, music_library, model_path: Optional[str] = None, enable_auto_play: bool = True):
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

        self._pending_transition_for: Optional[TrackInfo] = None
        
        # Audio settings
        self.sample_rate = 44100
        #self.chunk_size = 4096  # samples per chunk
        self.chunk_size = 16384
        
        # Playback tracking
        self.current_position = 0.0  # seconds into current track
        self.samples_sent = 0
        
        # Transition mixing
        self.mixer: Optional[TransitionMixer] = None
        self.pending_transition: Optional[TransitionPlan] = None
        self.transition_audio: Optional[Dict] = None
        self.use_dynamic_transitions: bool = True
        
        # Audio processing (tempo adjustment, filtering)
        self.processor = AudioProcessor(self.sample_rate)
        
        # Global effect settings (applied to all incoming tracks)
        self.global_effects: Dict[str, Any] = {}  # e.g., {'tempo_factor': 0.9, 'filter': 'lowpass', 'filter_freq': 5000}
        
        # Auto-play / similarity
        self.enable_auto_play = enable_auto_play
        self.similarity_service: Optional[SongSimilarityService] = None
        self.recently_played: List[str] = []  # Track song keys to avoid repeats
        self.max_history = 10  # How many songs to remember for avoiding repeats
        
        # WebSocket reference for sending transition notifications
        self._websocket = None

        # Skip tracking: count how many times skip has been pressed for the current song
        self._skip_count = 0
        
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
    
    def _load_audio(self, track_data: Dict, effects_config: Optional[Dict] = None) -> tuple[np.ndarray, float]:
        """Load audio file, apply effects, and return (audio_array, duration)."""
        audio, sr = sf.read(str(track_data['path']))
        
        # Convert mono to stereo
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
            # Normalize shape (N,2) or (N,) for all cases
            from audio_processor import AudioProcessor
            audio = AudioProcessor(self.sample_rate)._normalize_audio_shape(audio)
        
        # Resample if needed (basic - for production use librosa)
        if sr != self.sample_rate:
            print(f"[AUDIO] Warning: Sample rate mismatch {sr} vs {self.sample_rate}")
        
        duration = len(audio) / sr
        
        # Apply effects (from track config or global settings)
        effects = effects_config or self.global_effects
        if effects:
            audio = self._apply_effects_to_audio(audio, effects)
        
        return audio, duration
    
    def _ensure_segments(self, track_data: Dict, duration: float) -> Dict:
        """Ensure track has segment data, generating placeholders if needed."""
        if 'segments' not in track_data or not track_data['segments']:
            bpm = track_data.get('features', {}).get('bpm', 120.0)
            track_data['segments'] = generate_placeholder_segments(duration, bpm)
            print(f"[AUDIO] Generated placeholder segments for track")
        return track_data
    
    def _apply_effects_to_audio(self, audio: np.ndarray, effects: Dict[str, Any]) -> np.ndarray:
        """Apply audio effects based on configuration dict."""
        processed = audio.copy()
        
        # Tempo adjustment
        if 'tempo_factor' in effects:
            factor = effects['tempo_factor']
            print(f"[EFFECTS] Adjusting tempo to {factor*100:.0f}%")
            processed = self.processor.slow_down_tempo(processed, factor)
        
        # BPM matching
        if 'match_bpm' in effects:
            current_bpm, target_bpm = effects['match_bpm']
            print(f"[EFFECTS] Matching BPM: {current_bpm} → {target_bpm}")
            processed = self.processor.adjust_bpm(processed, current_bpm, target_bpm)
        
        # Low-pass filter
        if 'lowpass_freq' in effects:
            freq = effects['lowpass_freq']
            print(f"[EFFECTS] Applying low-pass filter at {freq}Hz")
            processed = self.processor.apply_lowpass_filter(processed, freq)
        
        # High-pass filter
        if 'highpass_freq' in effects:
            freq = effects['highpass_freq']
            print(f"[EFFECTS] Applying high-pass filter at {freq}Hz")
            processed = self.processor.apply_highpass_filter(processed, freq)
        
        # Band-pass filter
        if 'bandpass' in effects:
            low, high = effects['bandpass']
            print(f"[EFFECTS] Applying band-pass filter {low}Hz - {high}Hz")
            processed = self.processor.apply_bandpass_filter(processed, low, high)
        
        # Notch filter (hum removal)
        if 'notch_freq' in effects:
            freq = effects['notch_freq']
            q = effects.get('notch_q', 30)
            print(f"[EFFECTS] Applying notch filter at {freq}Hz")
            processed = self.processor.apply_notch_filter(processed, freq, q)
        
        # Bass boost
        if 'bass_boost_db' in effects:
            db = effects['bass_boost_db']
            print(f"[EFFECTS] Boosting bass by {db}dB")
            processed = self.processor.bass_boost(processed, db)
        
        # Vocal enhancement
        if 'vocal_enhancement' in effects and effects['vocal_enhancement']:
            print(f"[EFFECTS] Enhancing vocals")
            processed = self.processor.vocal_enhancement(processed)
        
        # Normalize
        if 'normalize_db' in effects:
            db = effects['normalize_db']
            print(f"[EFFECTS] Normalizing to {db}dB")
            processed = self.processor.normalize(processed, db)
        
        return processed
    
    def set_global_effects(self, effects: Dict[str, Any]):
        """Set global effects to apply to all incoming tracks."""
        self.global_effects = effects
        print(f"[EFFECTS] Global effects updated: {effects}")
    
    def clear_global_effects(self):
        """Clear all global effects."""
        self.global_effects = {}
        print(f"[EFFECTS] Global effects cleared")

    def set_transition_mode(self, mode: str) -> bool:
        """
        Set transition mode at runtime.

        Args:
            mode: 'dynamic' (warp-style) or 'classic' (equal-power crossfade)

        Returns:
            True if mode was applied, False if invalid
        """
        normalized = str(mode).strip().lower()
        if normalized in ("dynamic", "warp"):
            self.use_dynamic_transitions = True
            print("[MIXER] Transition mode set to dynamic")
            self._push_queue_update("transition_mode_changed")
            return True
        if normalized in ("classic", "crossfade", "standard"):
            self.use_dynamic_transitions = False
            print("[MIXER] Transition mode set to classic")
            self._push_queue_update("transition_mode_changed")
            return True
        return False

    def get_transition_mode(self) -> str:
        """Get current transition mode ('dynamic' or 'classic')."""
        return "dynamic" if self.use_dynamic_transitions else "classic"
    
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
        self._push_queue_update("auto_queued")
        
        return True
    
    
    def add_to_queue(
        self, 
        title: str, 
        artist: str,
        effects: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a song to the queue with optional effects.
        
        If a song is currently playing and mixer is available,
        this will trigger transition computation.
        
        User-requested songs always override auto-queued songs.
        
        Args:
            title: Song title
            artist: Artist name
            effects: Optional dict of audio effects to apply:
                    - 'tempo_factor': float (0.5 = half speed, 2.0 = double)
                    - 'match_bpm': (current_bpm, target_bpm)
                    - 'lowpass_freq': float (Hz)
                    - 'highpass_freq': float (Hz)
                    - 'bandpass': (low_freq, high_freq)
                    - 'vocal_enhancement': bool
                    - 'bass_boost_db': float
                    - 'normalize_db': float
        
        Returns True if song was found and added.
        
        Example:
            manager.add_to_queue("Song", "Artist", effects={'tempo_factor': 0.9})
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
            is_auto_queued=False,  # User requested
            effects_config=effects or {}
        )
        
        # Append to the back so user songs play in order after any existing
        # queued songs (including auto-queued ones).
        is_next = False
        if not self.queue:
            self.queue.append(track_info)
            is_next = True
        else:
            self.queue.append(track_info)

        print(f"[QUEUE] Added: {title}" + (f" by {artist}" if artist else ""))

        # Only prepare a transition if this song is actually next to play
        if is_next and self.state == PlaybackState.PLAYING and self.mixer and self.current_track:
            asyncio.create_task(self._prepare_transition(self.queue[0]))

        return True

    def reorder_queue(self, new_order: list) -> bool:
        """
        Reorder the queue based on a list of indices representing the new order.
        If position 0 changes, invalidate the pending transition and re-prepare.
        """
        if not self.queue or len(new_order) != len(self.queue):
            return False
        if sorted(new_order) != list(range(len(self.queue))):
            return False

        old_first = self.queue[0]
        self.queue = [self.queue[i] for i in new_order]
        new_first = self.queue[0]

        # If the next-up song changed, invalidate transition and re-plan
        if old_first is not new_first:
            self._pending_transition_for = None
            self.pending_transition = None
            self.transition_audio = None
            print(f"[QUEUE] Next song changed: {old_first.title} -> {new_first.title}, re-planning transition")
            if self.state == PlaybackState.PLAYING and self.mixer and self.current_track:
                asyncio.create_task(self._prepare_transition(new_first))

        self._push_queue_update("reordered")
        return True

    async def _prepare_transition(self, next_track: TrackInfo, force_quick: bool = False):
        """
        Prepare transition to the next track asynchronously.
        """
        if not self.current_track or not self.mixer:
            return
        
        transition_type = "QUICK" if force_quick else "NORMAL"
        print(f"[MIXER] Preparing transition: {self.current_track.title} -> {next_track.title}")
        
        try:
            def compute():
                # Load next track audio with its effects config
                next_audio, next_duration = self._load_audio(next_track.track_data, next_track.effects_config)
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
                # Pass force_quick parameter to compute_transition
                plan = self.mixer.compute_transition(
                    song_a_data=current_data,
                    song_b_data=next_data,
                    current_position=self.current_position,
                    force_next_segment=force_quick
                )
                
                if plan: 
                    transition_audio = self.mixer.prepare_mixed_audio(
                            self.current_track.audio,
                            next_audio,
                            plan,
                            use_dynamic=self.use_dynamic_transitions
                            )
                    return plan, transition_audio
                return None, None
            
            plan, transition_audio = await asyncio.to_thread(compute)

            self._pending_transition_for = next_track

            if plan:
                self.pending_transition = plan
                self.transition_audio = transition_audio
                
                # Pre-compute the mixed audio
                # self.transition_audio = self.mixer.prepare_mixed_audio(
                #     self.current_track.audio,
                #     next_audio,
                #     plan,
                #     use_dynamic=self.use_dynamic_transitions
                # )
                
                transition_timing = "SOON (next segment)" if force_quick else "at end of song"
                print(f"[MIXER] Transition ready ({transition_timing}): will start at {plan.transition_start_time:.1f}s")
                
                # Send notification to frontend
                if self._websocket:
                    try:
                        await self._websocket.send_json({
                            "type": "transition_planned",
                            "transition": plan.to_dict(),
                            "is_quick": force_quick,
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

            if next_track in self.queue:
                self.queue.remove(next_track)
                print(f"[MIXER] Removed unplayable track from queue: {next_track.title}")
                if self.queue and self.mixer:
                    asyncio.create_task(self._prepare_transition(self.queue[0]))

    def force_quick_transition(self) -> str:
        """
        Force a quick transition at the next available segments.
        On the first skip, finds a nearby segment to transition at.
        On a second skip for the same song, triggers an immediate crossfade.

        Returns:
            str: "quick" if first skip scheduled, "immediate" if force-skip initiated,
                 "" (empty) if cannot skip
        """

        # Must be playing with a mixer to transition
        if self.state != PlaybackState.PLAYING or not self.mixer or not self.current_track:
            print("[SKIP] Cannot force transition - not playing or no mixer")
            return ""

        # If queue is empty, try to auto-queue a similar song first
        if not self.queue:
            if self._auto_queue_next_song():
                print("[SKIP] Auto-queued a song for skip")
                if self._websocket:
                    asyncio.create_task(self._notify_auto_queue(self.queue[0]))
            else:
                print("[SKIP] Cannot skip - no songs in queue and auto-queue failed")
                return ""

        self._skip_count += 1

        if self._skip_count >= 2:
            # Double-skip: immediate crossfade from current position
            print(f"[FORCE SKIP] Immediate crossfade from {self.current_track.title}")
            asyncio.create_task(self._prepare_immediate_skip())
            return "immediate"
        else:
            # First skip: find nearby segment
            print(f"[QUICK TRANSITION] Forcing quick transition from {self.current_track.title}")
            asyncio.create_task(self._prepare_transition(
                self.queue[0],
                force_quick=True
            ))
            return "quick"

    async def _prepare_immediate_skip(self):
        """
        Prepare an immediate crossfade from the current playback position.
        Uses a simple 5-second equal-power crossfade, bypassing segment analysis.
        """
        if not self.current_track or not self.queue or not self.mixer:
            return

        next_track = self.queue[0]

        try:
            def compute():
                # Load next track audio if needed
                if next_track.audio is None:
                    next_audio, next_duration = self._load_audio(
                        next_track.track_data, next_track.effects_config
                    )
                    next_track.audio = next_audio
                    next_track.duration = next_duration

                # Determine song B start point:
                # Use planned entry point if a transition was already computed
                song_b_start_sample = 0
                if (self.pending_transition and
                    self.pending_transition.song_b_title == next_track.title):
                    song_b_start_sample = int(
                        self.pending_transition.song_b_start_offset * self.sample_rate
                    )
                    print(f"[FORCE SKIP] Using planned entry point at "
                          f"{self.pending_transition.song_b_start_offset:.1f}s in {next_track.title}")

                current_sample = int(self.current_position * self.sample_rate)

                # Use the mixer's immediate crossfade (always classic, 5s)
                return self.mixer.create_immediate_crossfade(
                    audio_a=self.current_track.audio,
                    audio_b=next_track.audio,
                    current_sample=current_sample,
                    song_b_start_sample=song_b_start_sample,
                    crossfade_duration=5.0,
                )

            transition_audio = await asyncio.to_thread(compute)

            # Build a minimal TransitionPlan for the streaming machinery
            current_sample = int(self.current_position * self.sample_rate)
            song_b_offset = transition_audio['timing']['song_b_continue_sample'] / self.sample_rate
            crossfade_dur = transition_audio['timing']['crossfade_duration']

            plan = TransitionPlan(
                song_a_title=self.current_track.title,
                song_b_title=next_track.title,
                exit_segment=None,
                entry_segment=None,
                predicted_score=0.0,
                crossfade_duration=crossfade_dur,
                transition_start_time=self.current_position,  # NOW
                song_b_start_offset=song_b_offset - crossfade_dur,
                song_a_bpm=self.current_track.track_data['features'].get('bpm', 120),
                song_b_bpm=next_track.track_data['features'].get('bpm', 120),
            )

            self.pending_transition = plan
            self.transition_audio = transition_audio

            print(f"[FORCE SKIP] Immediate crossfade ready - "
                  f"{crossfade_dur:.1f}s crossfade starting NOW")

            # Notify frontend
            if self._websocket:
                await self._websocket.send_json({
                    "type": "force_skip_initiated",
                    "message": f"Skipping to {next_track.title} immediately...",
                    "transition": plan.to_dict(),
                })

        except Exception as e:
            print(f"[FORCE SKIP ERROR] {e}")
            import traceback
            traceback.print_exc()

    
    def get_next_track(self) -> Optional[TrackInfo]:
        """Get next track from queue."""
        if self.queue:
            next_track = self.queue.pop(0)
            self._push_queue_update("dequeued")
            return next_track
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
            # Load audio if not already loaded (with effects applied automatically)
            if track_info.audio is None:
                track_info.audio, track_info.duration = self._load_audio(
                    track_info.track_data, 
                    track_info.effects_config
                )
            
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

            # Reset position tracking and skip count for new song
            self.samples_sent = 0
            self.current_position = 0.0
            self._skip_count = 0
            
            total_samples = len(audio_int16)
            i = 0
            stream_start_time = time.monotonic()
            
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
                # await asyncio.sleep(self.chunk_size / self.sample_rate * 0.98)
                expected_time = stream_start_time + (self.samples_sent / self.sample_rate) * 0.990
                sleep_time = expected_time - time.monotonic()
                if sleep_time > 0:
                    await asyncio.sleep(self.chunk_size / self.sample_rate * 0.985)
            
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

            crossfade_start_time = time.monotonic()
            crossfade_samples_sent = 0
            
            for i in range(0, len(crossfade_int16), self.chunk_size):
                if self.state == PlaybackState.STOPPED:
                    break
                    
                chunk = crossfade_int16[i:i + self.chunk_size]
                await websocket.send_bytes(chunk.tobytes())
                # await asyncio.sleep(self.chunk_size / self.sample_rate * 0.98)
                crossfade_samples_sent += len(chunk)
                expected_time = crossfade_start_time + (crossfade_samples_sent / self.sample_rate) * 0.990
                sleep_time = expected_time - time.monotonic()
                if sleep_time > 0:
                    await asyncio.sleep(self.chunk_size / self.sample_rate * 0.985)
            
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

                # CRITICAL FIX: Clip audio before int16 conversion
                post_clipped = np.clip(post_audio, -1.0, 1.0)
                post_int16 = (post_clipped * 32767).astype(np.int16)

                self.samples_sent = 0
                self.current_position = song_b_continue_sample / self.sample_rate
                self._skip_count = 0  # Reset skip count for new song

                # Send track_start BEFORE preparing next transition, so the frontend
                # clears the old transition info before receiving the new one.
                await websocket.send_json({
                    "type": "track_start",
                    "track": {
                        "title": next_track.title,
                        "artist": next_track.artist,
                        "bpm": next_track.track_data['features'].get('bpm', 120),
                        "key": next_track.track_data['features'].get('key', 'C'),
                        "duration": next_track.duration,
                        "sample_rate": self.sample_rate,
                        "is_continuation": True,
                        "is_auto_queued": next_track.is_auto_queued,
                        "start_offset": round(self.current_position, 2)
                    }
                })

                # Now prepare the next transition - its transition_planned message
                # will arrive after track_start, so the frontend won't clear it.
                if self.queue and self.mixer:
                    await self._prepare_transition(self.queue[0])

                # Debug: Log post-transition streaming info
                post_duration = len(post_int16) / self.sample_rate
                next_transition_time = self.pending_transition.transition_start_time if self.pending_transition else None
                print(f"[DEBUG] Starting post-transition for {self.current_track.title}")
                print(f"[DEBUG]   Post-transition duration: {post_duration:.1f}s")
                print(f"[DEBUG]   Current position starts at: {self.current_position:.1f}s")
                print(f"[DEBUG]   Next transition at: {next_transition_time}s")

                self.state = PlaybackState.PLAYING
                
                post_start_time = time.monotonic()

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
                    
                    expected_time = post_start_time + (self.samples_sent / self.sample_rate) * 0.990 
                    sleep_time = expected_time - time.monotonic()
                    if sleep_time > 0:
                        await asyncio.sleep(self.chunk_size / self.sample_rate * 0.985)
                
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
        self._push_queue_update("stopped")
    
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
            "transition_mode": self.get_transition_mode(),
            "recently_played": self.recently_played[-5:]  # Last 5 songs
        }
    
    def _push_queue_update(self, reason: str = "queue_changed"):
        """
        Best-effort push of current queue status to the connected frontend.
        Safe to call from sync code (schedules async send).
        """
        if not self._websocket:
            return

        payload = {
            "type": "queue_update",
            "reason": reason,
            "queue_status": self.get_queue_status()
        }

        async def _send():
            try:
                await self._websocket.send_json(payload)
            except Exception as e:
                print(f"[QUEUE] Could not send queue_update: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send())
        except RuntimeError:
            # No running event loop (rare in your app); ignore
            pass
    
    # ==================== AUDIO PROCESSING METHODS ====================
    
    def slow_down_track(self, track_data: Dict, speed_factor: float) -> np.ndarray:
        """
        Slow down a track's tempo using TSMD time-stretching.
        
        Args:
            track_data: Track data dict with 'path' key
            speed_factor: Speed multiplier (0.5 = half speed, 0.8 = 80% speed)
        
        Returns:
            Slowed audio array
        
        Example:
            audio = manager.slow_down_track(track_data, 0.8)
        """
        audio, sr = sf.read(str(track_data['path']))
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        return self.processor.slow_down_tempo(audio, speed_factor)
    
    def adjust_track_bpm(self, track_data: Dict, current_bpm: float, target_bpm: float) -> np.ndarray:
        """
        Adjust a track to match target BPM.
        
        Args:
            track_data: Track data dict
            current_bpm: Current tempo
            target_bpm: Target tempo
        
        Returns:
            Tempo-adjusted audio
        
        Example:
            # Slow from 120 BPM to 100 BPM
            audio = manager.adjust_track_bpm(track_data, 120, 100)
        """
        audio, sr = sf.read(str(track_data['path']))
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        return self.processor.adjust_bpm(audio, current_bpm, target_bpm)
    
    def apply_filter_to_track(
        self, 
        track_data: Dict, 
        filter_type: str, 
        **kwargs
    ) -> np.ndarray:
        """
        Apply audio filter to a track.
        
        Args:
            track_data: Track data dict
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'notch')
            **kwargs: Filter parameters
                - lowpass: cutoff_freq, order
                - highpass: cutoff_freq, order
                - bandpass: low_freq, high_freq, order
                - notch: freq, Q
        
        Returns:
            Filtered audio
        
        Examples:
            # Low-pass filter (remove highs)
            audio = manager.apply_filter_to_track(track, 'lowpass', cutoff_freq=5000)
            
            # Band-pass filter (keep vocals)
            audio = manager.apply_filter_to_track(track, 'bandpass', low_freq=100, high_freq=8000)
            
            # Remove 60Hz hum
            audio = manager.apply_filter_to_track(track, 'notch', freq=60)
        """
        audio, sr = sf.read(str(track_data['path']))
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        
        filter_methods = {
            'lowpass': self.processor.apply_lowpass_filter,
            'highpass': self.processor.apply_highpass_filter,
            'bandpass': self.processor.apply_bandpass_filter,
            'notch': self.processor.apply_notch_filter,
        }
        
        if filter_type not in filter_methods:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return filter_methods[filter_type](audio, **kwargs)
    
    def apply_vocal_enhancement(self, track_data: Dict) -> np.ndarray:
        """
        Enhance vocals in a track using band-pass filter (100Hz - 8kHz).
        
        Returns:
            Enhanced audio
        """
        audio, sr = sf.read(str(track_data['path']))
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        return self.processor.vocal_enhancement(audio)
    
    def apply_bass_boost(self, track_data: Dict, boost_db: float = 6) -> np.ndarray:
        """
        Boost bass frequencies in a track.
        
        Args:
            track_data: Track data dict
            boost_db: Boost amount in dB (default 6dB)
        
        Returns:
            Bass-boosted audio
        """
        audio, sr = sf.read(str(track_data['path']))
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        return self.processor.bass_boost(audio, boost_db)
    
    def normalize_audio(self, track_data: Dict, target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target loudness level.
        
        Args:
            track_data: Track data dict
            target_db: Target level in dB (default -3dB for headroom)
        
        Returns:
            Normalized audio
        """
        audio, sr = sf.read(str(track_data['path']))
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        return self.processor.normalize(audio, target_db)

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

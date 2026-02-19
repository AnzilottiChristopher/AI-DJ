"""
Real-time Warp Transition Generator

Adapts the offline warp_transition.py logic for real-time streaming.
Pre-computes transition segments with gradual tempo and filter changes.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from audiotsm import phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
from typing import Dict, Optional, Tuple
import math


class RealtimeWarpTransition:
    """
    Generates smooth DJ-style transitions with:
    - Gradual tempo changes (both tracks meet in middle)
    - Time-varying filters (HPF on outgoing, LPF on incoming)
    - Smooth crossfade
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.block_size = 2048  # Filter update granularity
        
    def prepare_transition(
        self,
        track_a_audio: np.ndarray,
        track_b_audio: np.ndarray,
        bpm_a: float,
        bpm_b: float,
        transition_start_sample: int,
        transition_duration_sec: float = 12.0,
        n_steps: int = 48,
        hp_start: float = 20.0,
        hp_end: float = 950.0,
        lp_start: float = 18000.0,
        lp_end: float = 950.0,
        curve_power: float = 1.8,
        filter_order: int = 4
    ) -> Dict:
        """
        Pre-compute the entire transition with tempo/filter effects.
        
        Args:
            track_a_audio: Outgoing track audio (mono or stereo)
            track_b_audio: Incoming track audio (mono or stereo)  
            bpm_a: BPM of track A
            bpm_b: BPM of track B
            transition_start_sample: Start position in track A
            transition_duration_sec: Duration of transition
            n_steps: Number of steps for smooth transition
            hp_start/end: Highpass filter range for outgoing
            lp_start/end: Lowpass filter range for incoming
            curve_power: Crossfade curve shape (higher = more dramatic)
            filter_order: Filter order
            
        Returns:
            Dict with:
                - 'transition_audio': Mixed transition segment
                - 'duration_samples': Length in samples
                - 'transition_start_time': Start time in seconds
                - 'post_transition_sample_b': Where to continue track B after
        """
        # Ensure mono for processing (can extend to stereo)
        track_a = self._ensure_mono(track_a_audio)
        track_b = self._ensure_mono(track_b_audio)
        
        transition_samples = int(transition_duration_sec * self.sample_rate)
        step_len = transition_samples // n_steps
        
        # Calculate middle BPM both tracks will meet at
        mid_bpm = (bpm_a + bpm_b) / 2
        mid_speed_a = mid_bpm / bpm_a
        mid_speed_b = mid_bpm / bpm_b
        
        print(f"[WARP] Preparing transition: {bpm_a:.1f}→{mid_bpm:.1f}←{bpm_b:.1f} BPM")
        print(f"[WARP] {n_steps} steps × {step_len} samples = {transition_samples} total")
        
        out_segments = []
        
        for i in range(n_steps):
            # Extract segments for this step
            a_start = transition_start_sample + i * step_len
            a_end = a_start + step_len
            b_start = i * step_len
            b_end = b_start + step_len
            
            seg_a = track_a[a_start:a_end] if a_end <= len(track_a) else np.zeros(step_len)
            seg_b = track_b[b_start:b_end] if b_end <= len(track_b) else np.zeros(step_len)
            
            # Interpolate tempo for this step
            progress = i / max(1, n_steps - 1)
            speed_a = np.interp(progress, [0, 1], [1.0, mid_speed_a])
            speed_b = np.interp(progress, [0, 1], [1.0, mid_speed_b])
            
            # Time-stretch using audiotsm
            seg_a_stretch = self._time_stretch(seg_a, speed_a)
            seg_b_stretch = self._time_stretch(seg_b, speed_b)
            
            # Resample to exact step length
            seg_a_stretch = self._resample_to_length(seg_a_stretch, step_len)
            seg_b_stretch = self._resample_to_length(seg_b_stretch, step_len)
            
            # Apply time-varying filters
            hp_cutoff = np.interp(progress, [0, 1], [hp_start, hp_end])
            lp_cutoff = np.interp(progress, [0, 1], [lp_start, lp_end])
            
            seg_a_filt = self._apply_highpass(seg_a_stretch, hp_cutoff, filter_order)
            seg_b_filt = self._apply_lowpass(seg_b_stretch, lp_cutoff, filter_order)
            
            # Crossfade with EDM-style curve
            fade = self._edm_curve(progress, curve_power)
            mixed = seg_a_filt * (1.0 - fade) + seg_b_filt * fade
            
            out_segments.append(mixed)
        
        # Concatenate all segments
        transition_audio = np.concatenate(out_segments)
        
        # Apply peak normalization to prevent clipping
        peak = np.max(np.abs(transition_audio))
        if peak > 0.98:
            transition_audio = transition_audio * (0.98 / peak)
        
        return {
            'transition_audio': transition_audio,
            'duration_samples': len(transition_audio),
            'transition_start_time': transition_start_sample / self.sample_rate,
            'post_transition_sample_b': n_steps * step_len,
            'bpm_a': bpm_a,
            'bpm_b': bpm_b,
            'mid_bpm': mid_bpm
        }
    
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert to mono if stereo."""
        if audio.ndim == 2:
            return np.mean(audio, axis=1)
        return audio
    
    def _time_stretch(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Time-stretch audio using audiotsm phase vocoder."""
        if len(audio) == 0:
            return audio
            
        # audiotsm expects (channels, samples)
        audio_2d = audio[np.newaxis, :].astype(np.float32)
        
        reader = ArrayReader(audio_2d)
        writer = ArrayWriter(channels=1)
        tsm = phasevocoder(channels=1, speed=speed)
        
        try:
            tsm.run(reader, writer)
            return writer.data.flatten().astype(np.float32)
        except Exception as e:
            print(f"[WARP] Time-stretch failed: {e}, returning original")
            return audio
    
    def _resample_to_length(self, audio: np.ndarray, target_len: int) -> np.ndarray:
        """Resample audio to exact target length using linear interpolation."""
        if len(audio) == target_len:
            return audio
        if len(audio) == 0:
            return np.zeros(target_len, dtype=np.float32)
            
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, audio).astype(np.float32)
    
    def _apply_highpass(self, audio: np.ndarray, cutoff: float, order: int) -> np.ndarray:
        """Apply highpass filter."""
        if len(audio) == 0:
            return audio
        cutoff = np.clip(cutoff, 5.0, self.sample_rate * 0.49)
        sos = butter(order, cutoff, btype='high', fs=self.sample_rate, output='sos')
        return sosfilt(sos, audio).astype(np.float32)
    
    def _apply_lowpass(self, audio: np.ndarray, cutoff: float, order: int) -> np.ndarray:
        """Apply lowpass filter."""
        if len(audio) == 0:
            return audio
        cutoff = np.clip(cutoff, 5.0, self.sample_rate * 0.49)
        sos = butter(order, cutoff, btype='low', fs=self.sample_rate, output='sos')
        return sosfilt(sos, audio).astype(np.float32)
    
    def _edm_curve(self, t: float, power: float) -> float:
        """EDM-style crossfade curve with smooth build."""
        # Smoothstep + power curve for dramatic EDM feel
        s = t * t * (3.0 - 2.0 * t)  # Smoothstep
        return float(np.power(s, power))
    
    def prepare_beat_aligned_transition(
        self,
        track_a_audio: np.ndarray,
        track_b_audio: np.ndarray,
        track_a_segments: list,
        track_b_segments: list,
        bpm_a: float,
        bpm_b: float,
        transition_bars: int = 16,
        beats_per_bar: int = 4,
        **kwargs
    ) -> Optional[Dict]:
        """
        Prepare transition aligned to beat grid (uses segment data).
        
        Args:
            track_a/b_audio: Audio arrays
            track_a/b_segments: Segment data with beat/downbeat info
            bpm_a/b: BPMs
            transition_bars: How many bars to transition over
            beats_per_bar: Usually 4
            **kwargs: Additional args passed to prepare_transition
            
        Returns:
            Transition dict or None if beat alignment fails
        """
        # Find a good downbeat in track A to start transition
        transition_beats = transition_bars * beats_per_bar
        transition_duration_sec = (transition_beats / bpm_a) * 60
        
        # Try to find downbeat position in track A segments
        # (Simplified - you can enhance this with actual beat detection)
        downbeat_sample = self._find_transition_downbeat(
            track_a_segments, 
            beats_per_bar,
            transition_beats,
            self.sample_rate
        )
        
        if downbeat_sample is None:
            # Fallback to time-based
            print("[WARP] Beat alignment failed, using time-based transition")
            downbeat_sample = int(len(track_a_audio) * 0.75)  # Start at 75% through song
        
        return self.prepare_transition(
            track_a_audio,
            track_b_audio,
            bpm_a,
            bpm_b,
            downbeat_sample,
            transition_duration_sec,
            **kwargs
        )
    
    def _find_transition_downbeat(
        self, 
        segments: list, 
        beats_per_bar: int,
        needed_beats: int,
        sample_rate: int
    ) -> Optional[int]:
        """Find a good downbeat position for transition start."""
        if not segments:
            return None
        
        # Look for segments marked as downbeats in the last third of the song
        total_duration = segments[-1]['end'] if segments else 0
        search_start = total_duration * 0.6
        
        for seg in segments:
            if seg.get('start', 0) > search_start:
                # Check if this is a downbeat (first beat of bar)
                if seg.get('is_downbeat', False) or seg.get('beat_position', 0) % beats_per_bar == 0:
                    return int(seg['start'] * sample_rate)
        
        # Fallback: use any segment in the search region
        for seg in segments:
            if seg.get('start', 0) > search_start:
                return int(seg['start'] * sample_rate)
        
        return None


# ==================== Integration Helper ====================

def integrate_warp_transition_into_audio_manager(audio_manager_instance):
    """
    Helper function to add warp transition capability to existing EnhancedAudioManager.
    
    Usage:
        manager = EnhancedAudioManager(music_library)
        integrate_warp_transition_into_audio_manager(manager)
        # Now manager has warp transition methods
    """
    warp = RealtimeWarpTransition(audio_manager_instance.sample_rate)
    
    def prepare_warp_transition(next_track: 'TrackInfo'):
        """Prepare warp-style transition when next track is queued."""
        if not audio_manager_instance.current_track:
            return None
        
        current = audio_manager_instance.current_track
        
        # Get BPMs
        bpm_a = current.track_data['features'].get('bpm', 120.0)
        bpm_b = next_track.track_data['features'].get('bpm', 120.0)
        
        # Determine transition start (e.g., 30s before end of current track)
        transition_lead_time = 30.0  # seconds before end
        track_duration = len(current.audio) / audio_manager_instance.sample_rate
        transition_start_time = max(0, track_duration - transition_lead_time)
        transition_start_sample = int(transition_start_time * audio_manager_instance.sample_rate)
        
        print(f"[WARP] Preparing transition from {current.title} to {next_track.title}")
        
        # Pre-compute the transition
        transition_data = warp.prepare_transition(
            track_a_audio=current.audio,
            track_b_audio=next_track.audio,
            bpm_a=bpm_a,
            bpm_b=bpm_b,
            transition_start_sample=transition_start_sample,
            transition_duration_sec=12.0,
            n_steps=48
        )
        
        # Store for streaming
        audio_manager_instance.warp_transition_data = transition_data
        audio_manager_instance.warp_pending = True
        
        return transition_data
    
    # Add methods to audio manager instance
    audio_manager_instance.warp_transition = warp
    audio_manager_instance.prepare_warp_transition = prepare_warp_transition
    audio_manager_instance.warp_transition_data = None
    audio_manager_instance.warp_pending = False
    
    print("[WARP] Integrated warp transition into audio manager")


# ==================== Example Usage ====================

if __name__ == "__main__":
    """
    Example: Generate offline warp transition (same as warp_transition.py)
    """
    import soundfile as sf
    
    FILE_A = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Ellie Goulding - Lights.wav"
    FILE_B = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Major Lazer - Lean On.wav"
    OUTPUT = "realtime_warp_demo.wav"
    
    # Load tracks
    track_a, sr = sf.read(FILE_A)
    track_b, _ = sf.read(FILE_B)
    
    # Create transition
    warp = RealtimeWarpTransition(sample_rate=sr)
    
    transition_start = int(60 * sr)  # Start at 1:00
    
    result = warp.prepare_transition(
        track_a_audio=track_a,
        track_b_audio=track_b,
        bpm_a=120.0,
        bpm_b=98.0,
        transition_start_sample=transition_start,
        transition_duration_sec=16.0,  # Longer for dramatic effect
        n_steps=64  # More steps = smoother
    )
    
    # Assemble final mix
    pre_transition = track_a[:transition_start]
    transition = result['transition_audio']
    post_transition = track_b[result['post_transition_sample_b']:]
    
    final_mix = np.concatenate([pre_transition, transition, post_transition])
    
    # Save
    sf.write(OUTPUT, final_mix, sr)
    print(f"✓ Saved warp transition to {OUTPUT}")
    print(f"  Duration: {len(final_mix)/sr:.1f}s")
    print(f"  Transition: {len(transition)/sr:.1f}s")

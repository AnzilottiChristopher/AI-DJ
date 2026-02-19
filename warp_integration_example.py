"""
Example: Integrating Warp Transitions into EnhancedAudioManager

This shows how to modify your enhanced_audio_manager.py to use
the real-time warp transition system for DJ-style tempo + filter effects.
"""

# ==================== STEP 1: Add import ====================
# Add this to the top of enhanced_audio_manager.py:
from realtime_warp_transition import RealtimeWarpTransition


# ==================== STEP 2: Initialize in __init__ ====================
# In EnhancedAudioManager.__init__, add:

class EnhancedAudioManager:
    def __init__(self, music_library, model_path=None, enable_auto_play=True):
        # ... existing initialization ...
        
        # Add warp transition support
        self.warp_transition = RealtimeWarpTransition(self.sample_rate)
        self.warp_transition_data = None
        self.use_warp_transitions = True  # Toggle for warp vs standard transitions
        
        print("[AUDIO] Warp transition system initialized")


# ==================== STEP 3: Modify _prepare_transition ====================
# Replace or extend the existing _prepare_transition method:

    async def _prepare_transition(self, next_track: TrackInfo):
        """
        Prepare transition to next track.
        Uses warp transitions if enabled, otherwise standard crossfade.
        """
        if not self.current_track or not self.current_track.audio.any():
            return
        
        print(f"[MIXER] Preparing transition to: {next_track.title}")
        
        # Load next track audio if not loaded
        if next_track.audio is None:
            next_track.audio, next_track.duration = self._load_audio(
                next_track.track_data,
                next_track.effects_config
            )
        
        if self.use_warp_transitions:
            # Use warp-style transition with tempo + filter effects
            await self._prepare_warp_transition(next_track)
        else:
            # Use standard transition mixer (existing code)
            await self._prepare_standard_transition(next_track)
    
    async def _prepare_warp_transition(self, next_track: TrackInfo):
        """Prepare warp-style transition with tempo and filter effects."""
        current = self.current_track
        
        # Get BPMs from track metadata
        bpm_a = current.track_data['features'].get('bpm', 120.0)
        bpm_b = next_track.track_data['features'].get('bpm', 120.0)
        
        # Calculate transition timing
        # Start transition 30 seconds before end of current track
        transition_lead_time = 30.0
        track_duration = len(current.audio) / self.sample_rate
        transition_start_time = max(10.0, track_duration - transition_lead_time)
        transition_start_sample = int(transition_start_time * self.sample_rate)
        
        print(f"[WARP] Computing transition: {bpm_a:.1f} BPM → {bpm_b:.1f} BPM")
        
        # Pre-compute the entire transition with tempo/filter effects
        transition_data = self.warp_transition.prepare_transition(
            track_a_audio=current.audio,
            track_b_audio=next_track.audio,
            bpm_a=bpm_a,
            bpm_b=bpm_b,
            transition_start_sample=transition_start_sample,
            transition_duration_sec=16.0,  # 16 second transition
            n_steps=64,  # 64 steps for smooth effect
            hp_start=20.0,
            hp_end=950.0,
            lp_start=18000.0,
            lp_end=950.0,
            curve_power=1.8  # EDM-style build
        )
        
        # Store the pre-computed transition
        self.warp_transition_data = transition_data
        self.transition_audio = {
            'crossfade': transition_data['transition_audio'],
            'post_transition': next_track.audio[transition_data['post_transition_sample_b']:],
            'timing': {
                'transition_start_time': transition_data['transition_start_time'],
                'song_b_continue_sample': transition_data['post_transition_sample_b']
            }
        }
        
        # Create a minimal transition plan for compatibility
        from transition_mixer import TransitionPlan
        self.pending_transition = TransitionPlan(
            transition_start_time=transition_data['transition_start_time'],
            crossfade_duration=len(transition_data['transition_audio']) / self.sample_rate,
            song_a_title=current.title,
            song_b_title=next_track.title,
            song_a_segment_idx=0,
            song_b_segment_idx=0,
            alignment_score=1.0
        )
        
        print(f"[WARP] Transition prepared: {transition_data['duration_samples']/self.sample_rate:.1f}s")
        print(f"[WARP] Will start at {transition_start_time:.1f}s")
    
    async def _prepare_standard_transition(self, next_track: TrackInfo):
        """Standard transition using existing mixer (original code)."""
        # This is your existing _prepare_transition logic
        if not self.mixer:
            return
        
        # ... existing mixer code ...
        pass


# ==================== STEP 4: Streaming works automatically ====================
# The existing _stream_transition and stream_audio methods work without changes!
# They just stream the pre-computed self.transition_audio['crossfade']


# ==================== USAGE EXAMPLES ====================

# Example 1: Basic usage with default settings
def example_basic_usage():
    """
    from enhanced_audio_manager import EnhancedAudioManager
    from music_library import MusicLibrary
    
    library = MusicLibrary()
    manager = EnhancedAudioManager(library)
    
    # Warp transitions are enabled by default
    # Just queue tracks normally and they'll use tempo + filter effects
    """
    pass


# Example 2: Toggle between warp and standard transitions
def example_toggle_transitions():
    """
    manager = EnhancedAudioManager(library)
    
    # Use warp transitions (tempo + filters)
    manager.use_warp_transitions = True
    
    # Or use standard crossfades
    manager.use_warp_transitions = False
    """
    pass


# Example 3: Customize warp transition parameters
def example_custom_parameters():
    """
    In _prepare_warp_transition, you can customize:
    
    transition_data = self.warp_transition.prepare_transition(
        # ... audio args ...
        transition_duration_sec=20.0,  # Longer transition
        n_steps=128,                    # Smoother (more steps)
        hp_start=20.0,                  # Start with full bass
        hp_end=1500.0,                  # More aggressive highpass
        lp_start=20000.0,               # Full spectrum incoming
        lp_end=800.0,                   # Meet at lower frequency
        curve_power=2.5,                # More dramatic build
        filter_order=6                  # Steeper filters
    )
    """
    pass


# Example 4: Use beat-aligned transitions
def example_beat_aligned():
    """
    # If you have segment data with beat info, use beat-aligned version:
    
    async def _prepare_warp_transition(self, next_track: TrackInfo):
        current = self.current_track
        
        transition_data = self.warp_transition.prepare_beat_aligned_transition(
            track_a_audio=current.audio,
            track_b_audio=next_track.audio,
            track_a_segments=current.track_data.get('segments', []),
            track_b_segments=next_track.track_data.get('segments', []),
            bpm_a=current.track_data['features'].get('bpm', 120.0),
            bpm_b=next_track.track_data['features'].get('bpm', 120.0),
            transition_bars=16,  # 16-bar transition (EDM standard)
            beats_per_bar=4
        )
        
        # ... rest same as before ...
    """
    pass


# ==================== CONFIGURATION OPTIONS ====================

CONFIG = {
    # Transition timing
    'transition_lead_time': 30.0,      # Start transition N seconds before track ends
    'transition_duration': 16.0,       # Duration of transition in seconds
    'n_steps': 64,                     # Number of processing steps (higher = smoother)
    
    # Filter settings
    'hp_start': 20.0,                  # Outgoing track highpass start (Hz)
    'hp_end': 950.0,                   # Outgoing track highpass end (Hz)
    'lp_start': 18000.0,               # Incoming track lowpass start (Hz)
    'lp_end': 950.0,                   # Incoming track lowpass end (Hz)
    'filter_order': 4,                 # Butterworth filter order
    
    # Crossfade curve
    'curve_power': 1.8,                # Power curve for EDM build feel (1.0 = linear, 2.0 = aggressive)
    
    # Beat alignment
    'transition_bars': 16,             # Bars to transition over (8, 16, 32)
    'beats_per_bar': 4,                # Usually 4 for EDM
    'use_beat_alignment': True,        # Try to align to downbeats
}


# ==================== TESTING ====================

if __name__ == "__main__":
    print("""
    To integrate warp transitions into your audio manager:
    
    1. Copy realtime_warp_transition.py to your project
    
    2. Add to enhanced_audio_manager.py:
       - Import: from realtime_warp_transition import RealtimeWarpTransition
       - In __init__: self.warp_transition = RealtimeWarpTransition(self.sample_rate)
       - In __init__: self.use_warp_transitions = True
       - Add _prepare_warp_transition method (see above)
       - Modify _prepare_transition to call warp version
    
    3. The streaming code works without changes!
       - _stream_transition already handles pre-computed audio
       - Just streams self.transition_audio['crossfade'] in chunks
    
    4. Test with your app.py:
       - Queue two tracks with different BPMs
       - Listen for smooth tempo changes + filter sweeps
       - Should sound like a professional DJ transition
    
    For beat-aligned transitions (advanced):
    - Ensure segments have 'is_downbeat' or 'beat_position' metadata
    - Use prepare_beat_aligned_transition instead
    
    Customization:
    - Adjust CONFIG values above to taste
    - Longer transition_duration for chill vibes
    - Higher curve_power for dramatic EDM builds
    - More n_steps for ultra-smooth (but slower to compute)
    """)

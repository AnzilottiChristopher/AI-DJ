"""
Integration Test: Audio Processing with DJ Transitions

Shows how effects are automatically applied during streaming without
modifying stream_audio() - everything works cohesively.
"""

# Example 1: Global Effects (applied to all incoming tracks)
# ============================================================
# manager = EnhancedAudioManager(music_library, model_path)
# 
# # Set global tempo slowdown for all incoming tracks
# manager.set_global_effects({
#     'tempo_factor': 0.9,  # 90% speed (10% slower)
# })
# 
# # Now every track you queue will be slowed down automatically
# manager.add_to_queue("Song 1", "Artist")
# await manager.stream_audio(websocket, manager.queue[0])
# # ^ The audio is automatically slowed during loading!


# Example 2: Per-Track Effects (override global)
# ============================================================
# manager = EnhancedAudioManager(music_library, model_path)
# 
# # Add track with specific effects (overrides global)
# manager.add_to_queue("Song 1", "Artist", effects={
#     'tempo_factor': 0.8,  # Slow this track to 80%
#     'lowpass_freq': 5000,  # Smooth highs
# })
# 
# manager.add_to_queue("Song 2", "Artist", effects={
#     'bandpass': (100, 8000),  # Vocals only
# })
# 
# await manager.stream_audio(websocket, manager.queue[0])
# # ^ Effects applied automatically during load!


# Example 3: Matching BPM for Smooth Transitions
# ============================================================
# Say current track is 120 BPM, next should be 100 BPM
# manager = EnhancedAudioManager(music_library, model_path)
# 
# # Set global BPM normalization
# manager.set_global_effects({
#     'match_bpm': (120, 100),  # Always match to 100 BPM
# })
# 
# manager.add_to_queue("Incoming Track", "Artist")
# # When it streams, it will automatically match the BPM!


# Example 4: DJ Workflow - Smooth Mixing
# ============================================================
# manager = EnhancedAudioManager(music_library, model_path)
# 
# # Current playing track
# manager.add_to_queue("Track 1", "Artist", effects={
#     'tempo_factor': 1.0,
# })
# 
# # Next track - prepare for smooth transition
# manager.add_to_queue("Track 2", "Artist", effects={
#     'tempo_factor': 1.05,  # Slightly sped up to match beat
#     'highpass_freq': 100,   # Remove muddiness
#     'normalize_db': -3.0,   # Consistent loudness
# })
# 
# # The transition_mixer + processors work together seamlessly
# # No need to modify stream_audio!


# Example 5: Complex Effect Chain for Radio DJ
# ============================================================
# manager = EnhancedAudioManager(music_library, model_path)
# 
# manager.add_to_queue("Next Song", "Artist", effects={
#     'vocal_enhancement': True,      # Boost vocals
#     'notch_freq': 60,               # Remove hum
#     'highpass_freq': 50,            # Clean rumble
#     'bass_boost_db': 3,             # Punch
#     'normalize_db': -2,             # Loud!
# })


# Example 6: Clear effects when needed
# ============================================================
# manager.set_global_effects({'tempo_factor': 0.9})
# # ... later ...
# manager.clear_global_effects()  # Back to normal


# Key Features:
# =============
# 1. Effects config stored in TrackInfo.effects_config
# 2. _load_audio() applies effects automatically
# 3. stream_audio() calls _load_audio(), gets processed audio
# 4. Works with transitions seamlessly
# 5. Effects printed to console for debugging
# 6. Global + per-track effects system for flexibility
# 7. No modifications to stream_audio() needed!


# Available Effects Dictionary Keys:
# ===================================
EFFECTS_REFERENCE = {
    'tempo_factor': 'float (0.5=half speed, 1.0=normal, 2.0=double)',
    'match_bpm': 'tuple (current_bpm, target_bpm)',
    'lowpass_freq': 'float Hz - remove frequencies above',
    'highpass_freq': 'float Hz - remove frequencies below',
    'bandpass': 'tuple (low_hz, high_hz)',
    'notch_freq': 'float Hz - remove specific frequency (hum)',
    'notch_q': 'float - notch filter Q factor (default 30)',
    'bass_boost_db': 'float dB - how much to boost bass',
    'vocal_enhancement': 'bool - boost 100-8000Hz',
    'normalize_db': 'float dB - target loudness level',
}


# Example 7: Typical DJ Transition Workflow
# ==========================================
"""
Step-by-step what happens:

1. User queues next song:
   manager.add_to_queue("Song B", "Artist", effects={
       'highpass_freq': 100,
       'tempo_factor': 0.95,
   })

2. add_to_queue() creates TrackInfo with effects_config set

3. stream_audio() is called with that TrackInfo

4. Inside stream_audio():
   - if track_info.audio is None:
   - Calls _load_audio(track_data, track_info.effects_config)
   
5. Inside _load_audio():
   - Loads raw audio from file
   - Gets effects = track_info.effects_config (or {} if empty)
   - If effects is not empty, calls _apply_effects_to_audio()
   
6. _apply_effects_to_audio() checks for each effect type:
   - tempo_factor → processor.slow_down_tempo()
   - highpass_freq → processor.apply_highpass_filter()
   - etc.
   
7. Returns processed audio with all effects applied

8. stream_audio() continues normally with processed audio

9. Audio streamed to WebSocket is already processed!

10. Transition mixer works with the processed audio

No modification to stream_audio needed! ✓
"""


if __name__ == "__main__":
    print(__doc__)
    print("\nEffect Keys Reference:")
    for key, desc in EFFECTS_REFERENCE.items():
        print(f"  {key}: {desc}")

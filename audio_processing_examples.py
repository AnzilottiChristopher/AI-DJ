"""
Audio Processing Usage Examples

Shows how to use tempo adjustment and filtering with your DJ system.
"""

import numpy as np
from audio_processor import AudioProcessor
from enhanced_audio_manager import EnhancedAudioManager

# Example 1: Direct processor usage
# ================================================

processor = AudioProcessor(sample_rate=44100)

# Slow down audio to 80% tempo (20% slower)
# audio_slowed = processor.slow_down_tempo(audio, speed_factor=0.8)

# Adjust audio from 120 BPM to 100 BPM
# audio_adjusted = processor.adjust_bpm(audio, current_bpm=120, target_bpm=100)

# Apply low-pass filter (remove frequencies above 5kHz)
# audio_filtered = processor.apply_lowpass_filter(audio, cutoff_freq=5000)

# Apply high-pass filter (remove frequencies below 100Hz)
# audio_filtered = processor.apply_highpass_filter(audio, cutoff_freq=100)

# Apply band-pass filter (keep 100Hz - 8kHz for vocals)
# audio_filtered = processor.apply_bandpass_filter(audio, low_freq=100, high_freq=8000)

# Remove 60Hz hum with notch filter
# audio_filtered = processor.apply_notch_filter(audio, freq=60, Q=30)


# Example 2: Using with Enhanced Audio Manager
# ================================================

def example_with_manager(manager: EnhancedAudioManager, track_data: dict):
    """
    Example of using audio processing with the DJ system.
    
    Args:
        manager: EnhancedAudioManager instance
        track_data: Track data dict with 'path' key
    """
    
    # Slow down a track
    # slowed_audio = manager.slow_down_track(track_data, speed_factor=0.9)
    
    # Adjust BPM to match another track
    # adjusted_audio = manager.adjust_track_bpm(track_data, current_bpm=120, target_bpm=110)
    
    # Apply filters
    # vocal_filtered = manager.apply_filter_to_track(track_data, 'bandpass', low_freq=100, high_freq=8000)
    # lo_filtered = manager.apply_filter_to_track(track_data, 'lowpass', cutoff_freq=5000)
    # hi_filtered = manager.apply_filter_to_track(track_data, 'highpass', cutoff_freq=100)
    
    # Enhance vocals
    # enhanced = manager.apply_vocal_enhancement(track_data)
    
    # Boost bass
    # bass_boosted = manager.apply_bass_boost(track_data, boost_db=6)
    
    # Normalize loudness
    # normalized = manager.normalize_audio(track_data, target_db=-3.0)
    pass


# Example 3: Chain multiple effects
# ================================================

def chain_effects(audio: np.ndarray, processor: AudioProcessor) -> np.ndarray:
    """
    Example of chaining multiple audio effects together.
    """
    # 1. Normalize input
    audio = processor.normalize(audio, target_db=-6.0)
    
    # 2. Remove low frequency rumble
    audio = processor.apply_highpass_filter(audio, cutoff_freq=50)
    
    # 3. Enhance vocals
    audio = processor.apply_bandpass_filter(audio, low_freq=200, high_freq=4000)
    
    # 4. Slightly reduce treble
    audio = processor.treble_cut(audio, reduction_db=2)
    
    # 5. Boost overall level
    audio = processor.normalize(audio, target_db=-3.0)
    
    return audio


# Example 4: Dynamic mixing with filtering
# ================================================

def dynamic_mix_example(
    audio1: np.ndarray, 
    audio2: np.ndarray, 
    processor: AudioProcessor,
    transition_duration: float = 2.0
):
    """
    Example of mixing two tracks with smart filtering.
    
    For smooth transitions, you might want to:
    - Low-pass the outgoing track (remove harsh highs)
    - High-pass the incoming track (remove muddiness)
    - Gradually transition between them
    """
    
    # Filter track 1 (smooth it out for crossfade out)
    audio1_smooth = processor.apply_lowpass_filter(audio1, cutoff_freq=6000)
    
    # Filter track 2 (clean it up for crossfade in)
    audio2_clean = processor.apply_highpass_filter(audio2, cutoff_freq=80)
    
    # Normalize both
    audio1_smooth = processor.normalize(audio1_smooth)
    audio2_clean = processor.normalize(audio2_clean)
    
    # Crossfade them
    mixed = processor.crossfade(audio1_smooth, audio2_clean, transition_duration)
    
    # Final normalize
    return processor.normalize(mixed, target_db=-3.0)


# Example 5: Real-world DJ scenario
# ================================================

def dj_transition_scenario(
    current_track: dict,
    next_track: dict,
    manager: EnhancedAudioManager,
    current_bpm: float = 120,
    next_bpm: float = 110,
):
    """
    Real DJ workflow: matching tempos and filtering for smooth mix.
    """
    
    # 1. Load tracks
    audio1, sr1 = manager.processor.sample_rate, manager.processor.sample_rate
    audio2, sr2 = manager.processor.sample_rate, manager.processor.sample_rate
    
    # 2. Match tempos first
    audio1_adjusted = manager.adjust_track_bpm(current_track, current_bpm, 100)
    audio2_adjusted = manager.adjust_track_bpm(next_track, next_bpm, 100)
    
    # 3. Apply EQ for smooth transition
    # Highpass outgoing track (remove mud)
    audio1_eq = manager.apply_filter_to_track(current_track, 'highpass', cutoff_freq=100)
    
    # Lowpass incoming track (reduce harshness)  
    audio2_eq = manager.apply_filter_to_track(next_track, 'lowpass', cutoff_freq=5000)
    
    # 4. Mix with crossfade
    # This would be handled by your transition_mixer
    
    return audio1_eq, audio2_eq


# Example 6: Specific filter frequencies reference
# ================================================

FILTER_PRESETS = {
    'bass_boost': {
        'filter_type': 'lowpass',
        'cutoff_freq': 250,
        'description': 'Isolate bass frequencies for boost'
    },
    'vocal_isolation': {
        'filter_type': 'bandpass',
        'low_freq': 200,
        'high_freq': 4000,
        'description': 'Isolate vocal range'
    },
    'treble_cut': {
        'filter_type': 'lowpass',
        'cutoff_freq': 8000,
        'description': 'Remove harsh highs'
    },
    'rumble_reduction': {
        'filter_type': 'highpass',
        'cutoff_freq': 50,
        'description': 'Remove subsonic rumble'
    },
    'hum_removal': {
        'filter_type': 'notch',
        'freq': 60,  # 60Hz for US/Japan, 50Hz for Europe
        'Q': 30,
        'description': 'Remove AC hum'
    },
    'midrange_enhancement': {
        'filter_type': 'bandpass',
        'low_freq': 500,
        'high_freq': 2500,
        'description': 'Enhance presency/presence'
    },
}


if __name__ == "__main__":
    # Quick reference for common use cases
    print("Audio Processing Examples")
    print("=" * 50)
    print()
    print("1. Tempo Adjustment:")
    print("   - slow_down_tempo(audio, 0.8)  # 20% slower")
    print("   - adjust_bpm(audio, 120, 100)  # 120 BPM → 100 BPM")
    print()
    print("2. Filtering:")
    print("   - apply_lowpass_filter(audio, 5000)   # Remove highs")
    print("   - apply_highpass_filter(audio, 100)   # Remove lows")
    print("   - apply_bandpass_filter(audio, 100, 8000)  # Keep range")
    print("   - apply_notch_filter(audio, 60)  # Remove hum")
    print()
    print("3. Common Presets:")
    for name, settings in FILTER_PRESETS.items():
        print(f"   - {name}: {settings['description']}")
    print()
    print("See code examples above for usage patterns.")

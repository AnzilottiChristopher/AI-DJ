import numpy as np
import soundfile as sf
from audio_processor import AudioProcessor

# Paths to your downloaded songs (replace with your actual file paths)
SONG_A_PATH = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Ellie Goulding - Lights.wav"
SONG_B_PATH = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Major Lazer - Lean On.wav"  # Song 2: Lean On
OUTPUT_PATH = 'Lights_to_LeanOn_transition.wav'

# BPM values (replace with actual BPMs if known)
BPM_A = 120  # Example BPM for Lights
BPM_B = 98   # Example BPM for Lean On

# Transition parameters
CROSSFADE_DURATION_SEC = 10  # Duration of crossfade in seconds

# Load songs
song_a, sr_a = sf.read(SONG_A_PATH)
song_b, sr_b = sf.read(SONG_B_PATH)

# Ensure sample rates match
if sr_a != sr_b:
    raise ValueError(f"Sample rates do not match: {sr_a} vs {sr_b}")
sample_rate = sr_a

# Initialize processor
processor = AudioProcessor(sample_rate=sample_rate)

# Adjust BPM of song_a to match song_b
song_a_matched = processor.adjust_bpm(song_a, BPM_A, BPM_B)

# Normalize shapes
song_a_matched = processor._normalize_audio_shape(song_a_matched)
song_b = processor._normalize_audio_shape(song_b)

# Calculate crossfade samples
crossfade_samples = int(CROSSFADE_DURATION_SEC * sample_rate)

# Prepare segments for transition
segment_a = song_a_matched[-crossfade_samples:]
segment_b = song_b[:crossfade_samples]

# Create crossfade
fade_out = np.linspace(1, 0, crossfade_samples)
fade_in = np.linspace(0, 1, crossfade_samples)

crossfade = segment_a * fade_out[:, None] + segment_b * fade_in[:, None]

# Concatenate: song_a (up to crossfade), crossfade, song_b (after crossfade)
transition_audio = np.concatenate([
    song_a_matched[:-crossfade_samples],
    crossfade,
    song_b[crossfade_samples:]
], axis=0)

# Save to .wav file
sf.write(OUTPUT_PATH, transition_audio, sample_rate)

print(f"Transition created and saved to {OUTPUT_PATH}")

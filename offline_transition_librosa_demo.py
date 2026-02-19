import numpy as np
import soundfile as sf
import librosa
import audiotsm
from scipy.signal import sosfilt, butter
from audiotsm.io.array import ArrayReader, ArrayWriter

# Paths to your downloaded songs (replace with your actual file paths)
SONG_A_PATH = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Ellie Goulding - Lights.wav"
SONG_B_PATH = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Major Lazer - Lean On.wav"
OUTPUT_PATH = 'Lights_to_LeanOn_librosa_transition_filtered.wav'

# BPM values (replace with actual BPMs if known)
BPM_A = 120  # Example BPM for Lights
BPM_B = 98   # Example BPM for Lean On

# Transition parameters
CROSSFADE_DURATION_SEC = 16  # Duration of crossfade in seconds
TRANSITION_START_SEC = 60    # Transition starts at 1:00 into song_a

# Filtering parameters
LOWPASS_CUTOFF = 3000  # Hz
HIGHPASS_CUTOFF = 200  # Hz
FILTER_ORDER = 4

# Filtering functions (now using sos)
def butter_lowpass_sos(cutoff, fs, order=5):
    return butter(order, cutoff / (0.5 * fs), btype='low', analog=False, output='sos')

def butter_highpass_sos(cutoff, fs, order=5):
    return butter(order, cutoff / (0.5 * fs), btype='high', analog=False, output='sos')

def apply_filter_sos(data, fs, cutoff, order, filter_type='low'):
    if filter_type == 'low':
        sos = butter_lowpass_sos(cutoff, fs, order)
    else:
        sos = butter_highpass_sos(cutoff, fs, order)
    return sosfilt(sos, data)

# Vectorized gradual filtering using sos

def vectorized_gradual_lowpass_sos(data, fs, cutoff_start, cutoff_end, order):
    n = len(data)
    cutoffs = np.linspace(cutoff_start, cutoff_end, n)
    filtered = np.zeros_like(data)
    unique_cutoffs = np.linspace(cutoff_start, cutoff_end, 32)
    filter_bank = [butter_lowpass_sos(c, fs, order) for c in unique_cutoffs]
    idxs = np.searchsorted(unique_cutoffs, cutoffs)
    for i, sos in enumerate(filter_bank):
        mask = idxs == i
        if np.any(mask):
            filtered[mask] = sosfilt(sos, data[mask])
    return filtered

def vectorized_gradual_highpass_sos(data, fs, cutoff_start, cutoff_end, order):
    n = len(data)
    # Clamp cutoffs to at least 1 Hz
    cutoffs = np.clip(np.linspace(cutoff_start, cutoff_end, n), 1, None)
    filtered = np.zeros_like(data)
    unique_cutoffs = np.clip(np.linspace(cutoff_start, cutoff_end, 32), 1, None)
    filter_bank = [butter_highpass_sos(c, fs, order) for c in unique_cutoffs]
    idxs = np.searchsorted(unique_cutoffs, cutoffs)
    for i, sos in enumerate(filter_bank):
        mask = idxs == i
        if np.any(mask):
            filtered[mask] = sosfilt(sos, data[mask])
    return filtered

# Gradual highpass on outgoing (20Hz to 500Hz), highpass on incoming (500Hz to 20Hz)
def vectorized_gradual_highpass_sos(data, fs, cutoff_start, cutoff_end, order):
    n = len(data)
    # Clamp cutoffs to at least 1 Hz
    cutoffs = np.clip(np.linspace(cutoff_start, cutoff_end, n), 1, None)
    filtered = np.zeros_like(data)
    unique_cutoffs = np.clip(np.linspace(cutoff_start, cutoff_end, 32), 1, None)
    filter_bank = [butter_highpass_sos(c, fs, order) for c in unique_cutoffs]
    idxs = np.searchsorted(unique_cutoffs, cutoffs)
    for i, sos in enumerate(filter_bank):
        mask = idxs == i
        if np.any(mask):
            filtered[mask] = sosfilt(sos, data[mask])
    return filtered

# Load songs
song_a, sr_a = librosa.load(SONG_A_PATH, sr=None, mono=True)
song_b, sr_b = librosa.load(SONG_B_PATH, sr=None, mono=True)
print("Loaded Songs")
if sr_a != sr_b:
    raise ValueError(f"Sample rates do not match: {sr_a} vs {sr_b}")
sample_rate = sr_a

crossfade_samples = int(CROSSFADE_DURATION_SEC * sample_rate)
transition_start_sample = int(TRANSITION_START_SEC * sample_rate)

# Define speed factor
speed_factor = BPM_B / BPM_A


# --- Smooth tempo and filter transition implementation ---
N_STEPS = 48  # Number of steps for smoothness
step_len = crossfade_samples // N_STEPS
out_segments = []

# Outgoing: from BPM_A to midpoint, highpass 20->500Hz
# Incoming: from BPM_B to midpoint, lowpass 20kHz->500Hz
mid_bpm = (BPM_A + BPM_B) / 2
mid_speed_a = mid_bpm / BPM_A
mid_speed_b = mid_bpm / BPM_B

for i in range(N_STEPS):
    # Stepwise indices
    a_start = transition_start_sample + i * step_len
    a_end = a_start + step_len
    b_start = i * step_len
    b_end = b_start + step_len
    seg_a = song_a[a_start:a_end]
    seg_b = song_b[b_start:b_end]
    # Tempo for this step
    speed_a = np.interp(i, [0, N_STEPS-1], [1.0, mid_speed_a])
    speed_b = np.interp(i, [0, N_STEPS-1], [1.0, mid_speed_b])
    # Time-stretch
    seg_a_2d = seg_a[np.newaxis, :]
    reader_a = ArrayReader(seg_a_2d)
    writer_a = ArrayWriter(channels=1)
    tsmer_a = audiotsm.wsola(channels=1, speed=speed_a)
    tsmer_a.run(reader_a, writer_a)
    seg_a_stretch = writer_a.data.flatten()
    # Pad/cut to step_len
    if len(seg_a_stretch) != step_len:
        seg_a_stretch = np.interp(np.linspace(0, len(seg_a_stretch)-1, step_len), np.arange(len(seg_a_stretch)), seg_a_stretch)

    seg_b_2d = seg_b[np.newaxis, :]
    reader_b = ArrayReader(seg_b_2d)
    writer_b = ArrayWriter(channels=1)
    tsmer_b = audiotsm.wsola(channels=1, speed=speed_b)
    tsmer_b.run(reader_b, writer_b)
    seg_b_stretch = writer_b.data.flatten()
    if len(seg_b_stretch) != step_len:
        seg_b_stretch = np.interp(np.linspace(0, len(seg_b_stretch)-1, step_len), np.arange(len(seg_b_stretch)), seg_b_stretch)

    # Filter for this step
    hp_cutoff = np.interp(i, [0, N_STEPS-1], [20, 500])
    lp_cutoff = np.interp(i, [0, N_STEPS-1], [20000, 500])
    seg_a_filt = apply_filter_sos(seg_a_stretch, sample_rate, hp_cutoff, FILTER_ORDER, filter_type='high')
    seg_b_filt = apply_filter_sos(seg_b_stretch, sample_rate, lp_cutoff, FILTER_ORDER, filter_type='low')

    # Crossfade for this step
    fade = i / (N_STEPS-1)
    out = seg_a_filt * (1-fade) + seg_b_filt * fade
    out_segments.append(out)

# Concatenate all transition steps
transition_section = np.concatenate(out_segments)
# After transition, use original song_b
song_b_post = song_b[N_STEPS*step_len:]

# Final output: song_a up to transition, transition, song_b remainder
transition_audio = np.concatenate([
    song_a[:transition_start_sample],
    transition_section,
    song_b_post
], axis=0)

# Save to .wav file
sf.write(OUTPUT_PATH, transition_audio, sample_rate)

print(f"Transition with filtering and tempo matching (only during transition) saved to {OUTPUT_PATH}")

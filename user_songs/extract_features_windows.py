import librosa
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

def load_song(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        return y, sr
    except Exception as e:
        print(f"Could not load {file_path}: {e}")
        return None, None

def extract_features_for_window(y, sr, start_time, end_time):
    # Convert times to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the audio segment
    y_segment = y[start_sample:end_sample]

    if len(y_segment) == 0:
        return None

    features = {}

    # 1. ENERGY FEATURES (Critical for build-up vs drop vs cooloff)
    rms = librosa.feature.rms(y=y_segment)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['rms_max'] = float(np.max(rms))

    # Energy slope (is energy increasing or decreasing?)
    if len(rms) > 1:
        features['rms_slope'] = float(np.polyfit(np.arange(len(rms)), rms, 1)[0])
    else:
        features['rms_slope'] = 0.0

    # 2. SPECTRAL FEATURES (Timbral characteristics)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

    zcr = librosa.feature.zero_crossing_rate(y_segment)[0]
    features['zcr_mean'] = float(np.mean(zcr))

    # 3. RHYTHMIC FEATURES
    onset_env = librosa.onset.onset_strength(y=y_segment, sr=sr)
    features['onset_strength_mean'] = float(np.mean(onset_env))
    features['onset_strength_max'] = float(np.max(onset_env))

    # 4. FREQUENCY BAND ENERGY
    # Compute spectrogram
    D = np.abs(librosa.stft(y_segment))
    freqs = librosa.fft_frequencies(sr=sr)

    # Sub-bass energy (drops have lots of this)
    sub_bass_mask = (freqs >= 20) & (freqs < 60)
    sub_bass_energy = np.mean(np.sum(D[sub_bass_mask, :], axis=0))

    # Total energy
    total_energy = np.mean(np.sum(D, axis=0)) + 1e-6

    # Sub-bass ratio
    features['sub_bass_ratio'] = float(sub_bass_energy / total_energy)

    # High frequency energy (build-ups often have risers)
    high_mask = (freqs >= 2000) & (freqs < 8000)
    high_energy = np.mean(np.sum(D[high_mask, :], axis=0))
    features['high_freq_ratio'] = float(high_energy / total_energy)

    return features

def extract_positional_features(window_center, duration, peak_time):
    features = {}

    # Time-based features
    features['time_from_start'] = float(window_center)
    features['time_to_end'] = float(duration - window_center)
    features['position_ratio'] = float(window_center / duration)

    # Distance from peak energy moment
    features['distance_from_peak'] = float(abs(window_center - peak_time))

    # Boolean indicators
    features['is_near_start'] = 1.0 if window_center < 0.15 * duration else 0.0
    features['is_near_end'] = 1.0 if window_center > 0.85 * duration else 0.0

    return features

def extract_song_features(audio_path, window_size=5.0, hop_size=2.5):
    # Load audio
    y, sr = load_song(audio_path)
    if y is None:
        return None

    duration = len(y) / sr

    # Find peak energy time (for positional features)
    rms_full = librosa.feature.rms(y=y)[0]
    peak_frame = np.argmax(rms_full)
    peak_time = librosa.frames_to_time(peak_frame, sr=sr)

    # Create windows
    windows = []
    window_start = 0.0

    while window_start < duration:
        window_end = min(window_start + window_size, duration)
        window_center = (window_start + window_end) / 2

        # Extract audio features for this window
        audio_features = extract_features_for_window(y, sr, window_start, window_end)

        if audio_features is None:
            break

        # Extract positional features
        position_features = extract_positional_features(window_center, duration, peak_time)

        # Combine all features
        all_features = {**audio_features, **position_features}
        all_features['window_start'] = float(window_start)
        all_features['window_end'] = float(window_end)

        windows.append(all_features)

        # Move to next window
        window_start += hop_size

    return windows

def assign_labels_to_windows(windows, segments):
    labeled_windows = []

    for window in windows:
        window_center = (window['window_start'] + window['window_end']) / 2

        # Find which segment this window belongs to
        label = None
        for segment in segments:
            if segment['start'] <= window_center < segment['end']:
                label = segment['name']
                break

        # If window is past last segment, assign to last segment (outro)
        if label is None:
            label = segments[-1]['name']

        labeled_windows.append({
            'label': label,
            'features': window
        })

    return labeled_windows

def process_all_songs(labels_file, audio_dir, output_file):
    # Load labels
    labels_path = Path(labels_file)
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_file}")
        print("   Run normalize_labels_windows.py first!")
        return None

    with open(labels_file, 'r') as f:
        data = json.load(f)

    songs = data['songs']

    # Check audio directory
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        return None

    # Count .wav files
    wav_files = list(audio_path.glob("*.wav"))
    print(f"Found {len(wav_files)} .wav files in audio directory")

    all_training_data = []

    print(f"\nExtracting features from {len(songs)} songs...")
    print(f"Window size: 5 seconds, Overlap: 50%\n")

    for song_data in tqdm(songs):
        song_name = song_data['song_name']
        audio_file = audio_path / song_name

        if not audio_file.exists():
            print(f"⚠️  Audio file not found: {audio_file}")
            continue

        # Extract features
        windows = extract_song_features(audio_file, window_size=5.0, hop_size=2.5)

        if windows is None:
            print(f"❌ Failed to process: {song_name}")
            continue

        # Assign labels
        labeled_windows = assign_labels_to_windows(windows, song_data['segments'])

        # Add song name to each window
        for item in labeled_windows:
            item['song_name'] = song_name

        all_training_data.extend(labeled_windows)

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_training_data, f, indent=2)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"✅ Feature extraction complete!")
    print(f"{'='*60}")
    print(f"Total windows: {len(all_training_data)}")
    print(f"Output file: {output_file}")

    # Label distribution
    label_counts = {}
    for item in all_training_data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nLabel distribution:")
    for label in ['intro', 'buildup', 'drop', 'cooloff', 'outro']:
        count = label_counts.get(label, 0)
        pct = (count / len(all_training_data)) * 100 if len(all_training_data) > 0 else 0
        print(f"  {label:10s}: {count:4d} windows ({pct:5.1f}%)")

    return all_training_data

def main():
    print("="*60)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*60)

    # Paths (relative to user_songs folder)
    labels_file = "../music_data/normalized_labels.json"
    audio_directory = "./user_uploaded_songs/"
    output_file = "../music_data/training_features.json"

    print(f"\nLabels file: {labels_file}")
    print(f"Audio directory: {audio_directory}")
    print(f"Output file: {output_file}\n")

    # Process all songs
    try:
        training_data = process_all_songs(labels_file, audio_directory, output_file)

        if training_data:
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS! Ready for Phase 3 (Model Training)")
            print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

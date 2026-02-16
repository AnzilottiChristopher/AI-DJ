import sys
import json
import numpy as np
import pickle
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import feature extraction
try:
    from extract_features import extract_features
    FEATURES_AVAILABLE = True
except ImportError:
    print("[WARNING] extract_features not available - features will be empty")
    FEATURES_AVAILABLE = False
    def extract_features(path):
        return {}


def load_model(model_dir):
    """Load trained model and metadata."""
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        print("\nRun train_rf.py first!")
        return None, None, None
    
    print("Loading model...")
    
    try:
        # Load Random Forest
        with open(model_dir / "rf_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(model_dir / "feature_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        with open(model_dir / "model_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        print("[OK] Model loaded")
        return model, scaler, metadata
    
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None, None, None


def extract_features_for_window(y, sr, start_time, end_time):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_segment = y[start_sample:end_sample]
    
    if len(y_segment) == 0:
        return None
    
    features = {}
    
    # 1. ENERGY FEATURES
    rms = librosa.feature.rms(y=y_segment)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['rms_max'] = float(np.max(rms))
    
    if len(rms) > 1:
        features['rms_slope'] = float(np.polyfit(np.arange(len(rms)), rms, 1)[0])
    else:
        features['rms_slope'] = 0.0
    
    # 2. SPECTRAL FEATURES
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
    D = np.abs(librosa.stft(y_segment))
    freqs = librosa.fft_frequencies(sr=sr)
    
    sub_bass_mask = (freqs >= 20) & (freqs < 60)
    sub_bass_energy = np.mean(np.sum(D[sub_bass_mask, :], axis=0))
    
    total_energy = np.mean(np.sum(D, axis=0)) + 1e-6
    features['sub_bass_ratio'] = float(sub_bass_energy / total_energy)
    
    high_mask = (freqs >= 2000) & (freqs < 8000)
    high_energy = np.mean(np.sum(D[high_mask, :], axis=0))
    features['high_freq_ratio'] = float(high_energy / total_energy)
    
    return features


def extract_positional_features(window_center, duration, peak_time):
    features = {}
    
    features['time_from_start'] = float(window_center)
    features['time_to_end'] = float(duration - window_center)
    features['position_ratio'] = float(window_center / duration)
    features['distance_from_peak'] = float(abs(window_center - peak_time))
    features['is_near_start'] = 1.0 if window_center < 0.15 * duration else 0.0
    features['is_near_end'] = 1.0 if window_center > 0.85 * duration else 0.0
    
    return features


def extract_song_features(audio_path, window_size=5.0, hop_size=2.5):
    print(f"\nExtracting features from: {Path(audio_path).name}")
    
    # Load audio
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        print(f"[ERROR] Error loading audio: {e}")
        return None
    
    duration = len(y) / sr
    print(f"Duration: {duration:.1f} seconds")
    
    # Find peak energy time
    rms_full = librosa.feature.rms(y=y)[0]
    peak_frame = np.argmax(rms_full)
    peak_time = librosa.frames_to_time(peak_frame, sr=sr)
    
    # Create windows
    windows = []
    window_start = 0.0
    
    print("Extracting features...", end="", flush=True)
    
    while window_start < duration:
        window_end = min(window_start + window_size, duration)
        window_center = (window_start + window_end) / 2
        
        # Extract features
        audio_features = extract_features_for_window(y, sr, window_start, window_end)
        if audio_features is None:
            break
        
        position_features = extract_positional_features(window_center, duration, peak_time)
        
        # Combine
        all_features = {**audio_features, **position_features}
        all_features['window_start'] = float(window_start)
        all_features['window_end'] = float(window_end)
        
        windows.append(all_features)
        window_start += hop_size
    
    print(f" {len(windows)} windows [OK]")
    
    return windows


def predict_segments(windows, model, scaler, metadata):
    feature_names = metadata['feature_names']
    
    print("\nPredicting segments...")
    
    # Extract features in correct order
    X = []
    for window in windows:
        feature_vector = [window[name] for name in feature_names]
        X.append(feature_vector)
    
    X = np.array(X)
    
    # Normalize
    X_scaled = scaler.transform(X)
    
    # Predict
    predicted_labels = model.predict(X_scaled)
    
    # Convert to segments
    segments = []
    current_label = None
    segment_start = None
    
    for i, (label, window) in enumerate(zip(predicted_labels, windows)):
        if label != current_label:
            # End previous segment
            if current_label is not None:
                segments.append({
                    'name': current_label,
                    'start': segment_start,
                    'end': window['window_start']
                })
            
            # Start new segment
            current_label = label
            segment_start = window['window_start']
    
    # Add final segment
    if current_label is not None:
        segments.append({
            'name': current_label,
            'start': segment_start,
            'end': windows[-1]['window_end']
        })
    
    print(f"[OK] Found {len(segments)} segments")
    
    return segments


def save_segments(segments, song_path, output_file):
    song_name = Path(song_path).name
    
    # Extract features from audio file
    if FEATURES_AVAILABLE:
        print("\n[FEATURES] Extracting musical features...")
        features = extract_features(song_path)
    else:
        features = {}
    
    output_data = {
        'song_name': song_name,
        'features': features,  
        'segments': segments
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[OK] Saved segments to: {output_file}")
    
    # Show extracted features
    if features:
        print("\n[OK] Extracted features:")
        if features.get('bpm'):
            print(f"  BPM: {features['bpm']:.1f}")
        if features.get('key') and features.get('scale'):
            print(f"  Key: {features['key']} {features['scale']}")
        if features.get('danceability'):
            print(f"  Danceability: {features['danceability']:.2f}")


def display_segments(segments):
    print("\n" + "="*60)
    print("DETECTED SEGMENTS")
    print("="*60)
    
    for i, seg in enumerate(segments):
        duration = seg['end'] - seg['start']
        print(f"{i+1:2d}. {seg['start']:6.1f}s - {seg['end']:6.1f}s  ({duration:5.1f}s)  {seg['name']:10s}")
    
    # Summary
    print("\n" + "-"*60)
    print("Summary:")
    label_counts = {}
    for seg in segments:
        label = seg['name']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label in ['intro', 'buildup', 'drop', 'cooloff', 'outro']:
        count = label_counts.get(label, 0)
        if count > 0:
            print(f"  {label:10s}: {count} segment(s)")


def main():
    print("="*60)
    print("AI SONG SEGMENTATION")
    print("="*60)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python segment_new_song.py <path/to/song.wav>")
        print("\nExample:")
        print("  python segment_new_song.py ../music_data/audio/Test-Song_Artist.wav")
        return
    
    audio_path = sys.argv[1]
    
    # Check if file exists
    if not Path(audio_path).exists():
        print(f"\n[ERROR] Audio file not found: {audio_path}")
        return
    
    # Load model (looks for ./models from user_songs/ folder)
    model_dir = "./models"
    model, scaler, metadata = load_model(model_dir)
    
    if model is None:
        return
    
    # Extract features
    windows = extract_song_features(audio_path)
    
    if windows is None:
        return
    
    # Predict segments
    segments = predict_segments(windows, model, scaler, metadata)
    
    # Display results
    display_segments(segments)
    
    # Save to file (creates {song_name}_segments.json in current directory)
    output_file = Path(audio_path).stem + "_segments.json"
    save_segments(segments, audio_path, output_file)
    
    print("\n" + "="*60)
    print("[SUCCESS] SEGMENTATION COMPLETE!")
    print("="*60)
    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    main()

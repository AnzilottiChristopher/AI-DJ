import librosa
import numpy as np
from scipy import signal
import sys
from pathlib import Path


def extract_features(audio_path):
    try:
        print("[FEATURES] Extracting musical features...")
        
        # Load audio
        y, sr = librosa.load(str(audio_path))
        
        features = {}
        
        # 1. BPM (Tempo)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['bpm'] = float(tempo)
        
        # 2. Key and Scale
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = np.argmax(np.sum(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        features['key'] = keys[key_idx]
        
        # Key strength (how confident we are about the key)
        chroma_sum = np.sum(chroma, axis=1)
        features['key_strength'] = float(chroma_sum[key_idx] / np.sum(chroma_sum))
        
        # Scale (major/minor) - simplified estimation
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        major_corr = np.corrcoef(chroma_sum, major_profile)[0, 1]
        minor_corr = np.corrcoef(chroma_sum, minor_profile)[0, 1]
        features['scale'] = 'major' if major_corr > minor_corr else 'minor'
        
        # 3. Loudness (RMS energy in dB)
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20 * np.log10(np.mean(rms) + 1e-6)

        normalized_loudness = np.clip((rms_db + 60) / 60, 0, 1)
        features['loudness'] = float(normalized_loudness)
        
        # 4. Danceability (based on beat strength and regularity)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        beat_strength = np.mean(tempogram)
        features['danceability'] = float(np.clip(beat_strength / 10, 0, 1))
        
        # 5. Spectral Centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        # 6. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        # 7. Dissonance (based on spectral flatness and roughness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['dissonance'] = float(np.mean(spectral_flatness))
        
        # 8. Onset Rate (how many musical events per second)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        features['onset_rate'] = float(len(onset_frames) / duration)
        
        print("[FEATURES] Feature extraction complete")
        return features
        
    except Exception as e:
        print(f"[FEATURES] Error extracting features: {e}")
        # Return empty features on error
        return {
            "bpm": None,
            "key": None,
            "scale": None,
            "key_strength": None,
            "loudness": None,
            "danceability": None,
            "spectral_centroid": None,
            "spectral_rolloff": None,
            "dissonance": None,
            "onset_rate": None,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <path/to/song.wav>")
        return
    
    audio_path = Path(sys.argv[1])
    
    if not audio_path.exists():
        print(f"[ERROR] File not found: {audio_path}")
        return
    
    features = extract_features(audio_path)
    
    print("\n" + "="*60)
    print("EXTRACTED FEATURES")
    print("="*60)
    for key, value in features.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.2f}")
            else:
                print(f"  {key:20s}: {value}")
        else:
            print(f"  {key:20s}: Not detected")
    print("="*60)


if __name__ == "__main__":
    main()

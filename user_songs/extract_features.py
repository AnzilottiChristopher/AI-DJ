import librosa
import numpy as np
from scipy import signal

def extract_audio_features(audio_path):
    """Extract comprehensive audio features for DJ analysis"""
    
    # Load audio
    y, sr = librosa.load(audio_path)
    
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
    features['loudness'] = float(20 * np.log10(np.mean(rms) + 1e-6))
    
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
    
    return features

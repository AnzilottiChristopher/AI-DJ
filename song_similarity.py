"""
Song Similarity Service

Embeds songs as feature vectors and finds similar tracks using cosine similarity.
This enables auto-play functionality by suggesting the next best song to play.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Key encoding for circle of fifths (same as transition model)
KEY_ORDER = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
KEY_TO_INDEX = {key: i for i, key in enumerate(KEY_ORDER)}
KEY_ALIASES = {
    'C#': 'Db', 'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb',
    'Gb': 'F#', 'Cb': 'B', 'Fb': 'E'
}


@dataclass
class SongEmbedding:
    """A song with its computed embedding vector."""
    song_key: str  # The key in the music library index
    title: str
    artist: str
    features: Dict
    embedding: np.ndarray


def parse_title_artist_from_filename(filename: str) -> tuple[str, str]:
    """
    Parse proper title and artist from filename.
    
    Filename format: "song-title_artist-name.wav"
    Example: "hey-brother_avicii.wav" -> ("Hey Brother", "Avicii")
    Example: "glad-you-came_the-wanted.wav" -> ("Glad You Came", "The Wanted")
    
    Returns:
        Tuple of (title, artist) with proper capitalization
    """
    # Remove .wav extension
    name = filename.replace('.wav', '').replace('.mp3', '')
    
    # Split by underscore to separate title and artist
    if '_' in name:
        parts = name.split('_', 1)  # Split only on first underscore
        title_part = parts[0]
        artist_part = parts[1] if len(parts) > 1 else "Unknown Artist"
    else:
        # No underscore, treat whole thing as title
        title_part = name
        artist_part = "Unknown Artist"
    
    # Convert hyphens to spaces and title case
    title = title_part.replace('-', ' ').title()
    artist = artist_part.replace('-', ' ').title()
    
    return title, artist


class SongSimilarityService:
    """
    Service for finding similar songs based on audio features.
    
    Uses normalized feature vectors and cosine similarity to find
    tracks that would flow well together in a DJ set.
    """
    
    def __init__(self, music_library):
        """
        Initialize the similarity service.
        
        Args:
            music_library: MusicLibrary instance with song metadata
        """
        self.music_library = music_library
        self.embeddings: Dict[str, SongEmbedding] = {}
        
        # Feature weights for similarity calculation
        # Higher weight = more important for similarity
        self.feature_weights = {
            'bpm': 2.0,           # BPM matching is critical for DJing
            'danceability': 1.5,  # Energy level matters
            'loudness': 1.0,
            'key_x': 1.2,         # Key compatibility
            'key_y': 1.2,
            'scale': 0.8,
            'spectral_centroid': 0.7,
            'spectral_rolloff': 0.5,
            'dissonance': 0.6,
            'onset_rate': 0.8,
            'key_strength': 0.4,
        }
        
        # Build embeddings for all songs
        self._build_embeddings()
    
    def _normalize_key(self, key: str) -> str:
        """Normalize key name to standard format."""
        key = key.strip()
        return KEY_ALIASES.get(key, key)
    
    def _encode_key_circular(self, key: str) -> Tuple[float, float]:
        """
        Encode musical key as x,y coordinates on a circle.
        
        This preserves the circular nature of the circle of fifths,
        so keys that are harmonically close have similar encodings.
        """
        key = self._normalize_key(key)
        
        if key not in KEY_TO_INDEX:
            # Default to C if unknown
            idx = 0
        else:
            idx = KEY_TO_INDEX[key]
        
        # Convert to angle (12 keys = full circle)
        angle = (idx / 12.0) * 2 * np.pi
        
        # Return x, y coordinates
        return np.cos(angle), np.sin(angle)
    
    def _extract_embedding(self, features: Dict) -> np.ndarray:
        """
        Extract a normalized feature vector from song features.
        
        The embedding captures:
        - BPM (normalized to ~0-1 range for typical DJ music)
        - Key (circular encoding)
        - Scale (binary)
        - Energy characteristics (loudness, danceability)
        - Spectral characteristics
        """
        # BPM: normalize to 0-1 range (assuming 80-160 BPM range)
        bpm = features.get('bpm', 120.0)
        bpm_norm = (bpm - 80) / 80  # 80 BPM -> 0, 160 BPM -> 1
        
        # Key: circular encoding
        key = features.get('key', 'C')
        key_x, key_y = self._encode_key_circular(key)
        
        # Scale: binary (major=1, minor=0)
        scale = 1.0 if features.get('scale', 'major') == 'major' else 0.0
        
        # Other features (already roughly normalized or we normalize)
        key_strength = features.get('key_strength', 0.8)
        loudness = features.get('loudness', 0.8)
        danceability = features.get('danceability', 1.0) / 1.5  # Normalize ~0-1
        
        # Spectral features (normalize to rough 0-1 range)
        spectral_centroid = features.get('spectral_centroid', 1500) / 3000
        spectral_rolloff = features.get('spectral_rolloff', 2000) / 4000
        
        # Other
        dissonance = features.get('dissonance', 0.45)
        onset_rate = features.get('onset_rate', 3.5) / 10  # Normalize
        
        # Build embedding vector
        embedding = np.array([
            bpm_norm * self.feature_weights['bpm'],
            key_x * self.feature_weights['key_x'],
            key_y * self.feature_weights['key_y'],
            scale * self.feature_weights['scale'],
            key_strength * self.feature_weights['key_strength'],
            loudness * self.feature_weights['loudness'],
            danceability * self.feature_weights['danceability'],
            spectral_centroid * self.feature_weights['spectral_centroid'],
            spectral_rolloff * self.feature_weights['spectral_rolloff'],
            dissonance * self.feature_weights['dissonance'],
            onset_rate * self.feature_weights['onset_rate'],
        ], dtype=np.float32)
        
        return embedding
    
    def _build_embeddings(self):
        """Build embeddings for all songs in the library."""
        print("[SIMILARITY] Building song embeddings...")
        
        for song_key, song_data in self.music_library.index.items():
            features = song_data.get('features', {})
            embedding = self._extract_embedding(features)
            
            # Parse title and artist from filename
            filename = song_data.get('filename', '')
            title, artist = parse_title_artist_from_filename(filename)
            
            self.embeddings[song_key] = SongEmbedding(
                song_key=song_key,
                title=title,
                artist=artist,
                features=features,
                embedding=embedding
            )
        
        print(f"[SIMILARITY] Built embeddings for {len(self.embeddings)} songs")
    
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    
    def find_similar_songs(
        self,
        current_song_key: str,
        exclude_keys: Optional[List[str]] = None,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar songs to the current song.
        
        Args:
            current_song_key: Key of the current song in the library index
            exclude_keys: List of song keys to exclude (e.g., recently played)
            top_n: Number of similar songs to return
            
        Returns:
            List of (song_key, similarity_score) tuples, sorted by similarity
        """
        if current_song_key not in self.embeddings:
            print(f"[SIMILARITY] Warning: Song not found: {current_song_key}")
            return []
        
        exclude_keys = exclude_keys or []
        exclude_set = set(exclude_keys)
        exclude_set.add(current_song_key)  # Don't recommend the same song
        
        current_embedding = self.embeddings[current_song_key].embedding
        
        similarities = []
        for song_key, song_emb in self.embeddings.items():
            if song_key in exclude_set:
                continue
            
            similarity = self._cosine_similarity(current_embedding, song_emb.embedding)
            similarities.append((song_key, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def get_next_song(
        self,
        current_song_key: str,
        exclude_keys: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Get the single best next song to play.
        
        Args:
            current_song_key: Key of the current song
            exclude_keys: Songs to exclude (e.g., recently played)
            
        Returns:
            Song key of the recommended next song, or None if no options
        """
        similar = self.find_similar_songs(current_song_key, exclude_keys, top_n=1)
        
        if similar:
            next_key, score = similar[0]
            print(f"[SIMILARITY] Recommended next: '{next_key}' (similarity: {score:.3f})")
            return next_key
        
        return None
    
    def get_song_key_from_track_data(self, track_data: Dict) -> Optional[str]:
        """
        Find the song key in our index that matches the given track data.
        
        This is needed because track_data might come from a search result
        and we need to find its corresponding key in our embeddings.
        """
        target_path = str(track_data.get('path', ''))
        
        for song_key, song_data in self.music_library.index.items():
            if str(song_data.get('path', '')) == target_path:
                return song_key
        
        return None
    
    def get_song_info(self, song_key: str) -> Optional[Dict]:
        """
        Get properly formatted song information for a song key.
        
        Returns:
            Dict with 'title', 'artist', 'song_key' or None if not found
        """
        if song_key not in self.embeddings:
            return None
        
        emb = self.embeddings[song_key]
        return {
            'title': emb.title,
            'artist': emb.artist,
            'song_key': song_key
        }


# Convenience function for quick testing
def demo_similarity(music_library):
    """Demo the similarity service."""
    service = SongSimilarityService(music_library)
    
    print("\n=== Song Similarity Demo ===\n")
    
    # Pick a random song and find similar ones
    for song_key in list(service.embeddings.keys())[:3]:
        print(f"Songs similar to '{song_key}':")
        similar = service.find_similar_songs(song_key, top_n=3)
        for other_key, score in similar:
            print(f"  - '{other_key}' (similarity: {score:.3f})")
        print()


if __name__ == '__main__':
    # Quick test
    from music_library import MusicLibrary
    
    lib = MusicLibrary(
        'music_data/audio',
        'music_data/segmented_alex_pre_analysis_results_converted.json'
    )
    
    demo_similarity(lib)
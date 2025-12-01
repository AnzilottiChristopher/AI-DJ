"""
DJ Transition Ranking Model using XGBoost

This module provides a machine learning model for ranking DJ transitions between songs.
It uses song audio features and segment types to predict transition quality ratings.

Usage:
    # Training
    model = DJTransitionModel()
    model.load_training_data('alex_pre_analysis_results.json', 'transition-results.json')
    model.train()
    model.save('dj_transition_model.json')
    
    # Inference
    model = DJTransitionModel.load('dj_transition_model.json')
    rankings = model.rank_transitions(song_a_features, song_b_features, available_segments)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings

warnings.filterwarnings('ignore')


# Musical key compatibility matrix based on music theory
# Keys are arranged in circle of fifths order
KEY_ORDER = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
KEY_TO_INDEX = {key: i for i, key in enumerate(KEY_ORDER)}

# Alternative key names mapping
KEY_ALIASES = {
    'C#': 'Db', 'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb',
    'Gb': 'F#', 'Cb': 'B', 'Fb': 'E'
}

# Segment energy levels (0-1 scale, representing intensity)
SEGMENT_ENERGY = {
    'intro': 0.2,
    'verse': 0.4,
    'verse1': 0.4,
    'verse2': 0.45,
    'pre-chorus': 0.55,
    'build-up': 0.7,
    'build-up1': 0.7,
    'build-up2': 0.75,
    'chorus': 0.85,
    'chorus1': 0.85,
    'chorus2': 0.88,
    'beat-drop': 0.95,
    'drop': 0.95,
    'breakdown': 0.5,
    'bridge': 0.5,
    'cool-down': 0.35,
    'outro': 0.15,
}

# Segment type categories for one-hot encoding
SEGMENT_TYPES = [
    'intro', 'verse', 'pre-chorus', 'build-up', 'chorus', 
    'beat-drop', 'breakdown', 'bridge', 'cool-down', 'outro'
]


@dataclass
class SongFeatures:
    """Container for song audio features."""
    bpm: float
    key: str
    scale: str  # 'major' or 'minor'
    key_strength: float
    loudness: float
    danceability: float
    spectral_centroid: float
    spectral_rolloff: float
    dissonance: float
    onset_rate: float
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SongFeatures':
        """Create SongFeatures from a dictionary."""
        return cls(
            bpm=data.get('bpm', 120.0),
            key=data.get('key', 'C'),
            scale=data.get('scale', 'major'),
            key_strength=data.get('key_strength', 0.5),
            loudness=data.get('loudness', 0.5),
            danceability=data.get('danceability', 1.0),
            spectral_centroid=data.get('spectral_centroid', 1500.0),
            spectral_rolloff=data.get('spectral_rolloff', 2000.0),
            dissonance=data.get('dissonance', 0.45),
            onset_rate=data.get('onset_rate', 3.5)
        )


@dataclass
class TransitionPrediction:
    """Container for a single transition prediction."""
    exit_segment: str
    entry_segment: str
    predicted_rating: float
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DJTransitionModel:
    """
    XGBoost-based model for ranking DJ transitions between songs.
    
    The model learns from human-rated transitions to predict the quality
    of transitioning from one song segment to another, considering:
    - BPM compatibility
    - Musical key harmony
    - Energy flow between segments
    - Audio characteristic similarities
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[xgb.XGBRegressor] = None
        self.song_features: Dict[str, SongFeatures] = {}
        self.feature_names: List[str] = []
        self.training_stats: Dict[str, Any] = {}
        self._exit_segment_encoder = LabelEncoder()
        self._entry_segment_encoder = LabelEncoder()
        self._is_fitted = False
        
    def _normalize_key(self, key: str) -> str:
        """Normalize key name to standard format."""
        key = key.strip()
        return KEY_ALIASES.get(key, key)
    
    def _get_key_distance(self, key_a: str, key_b: str) -> int:
        """
        Calculate the distance between two keys on the circle of fifths.
        Returns value 0-6 (0 = same key, 6 = tritone apart).
        """
        key_a = self._normalize_key(key_a)
        key_b = self._normalize_key(key_b)
        
        if key_a not in KEY_TO_INDEX or key_b not in KEY_TO_INDEX:
            return 6  # Max distance for unknown keys
            
        idx_a = KEY_TO_INDEX[key_a]
        idx_b = KEY_TO_INDEX[key_b]
        
        distance = abs(idx_a - idx_b)
        return min(distance, 12 - distance)  # Wrap around circle
    
    def _get_key_compatibility(self, key_a: str, scale_a: str, 
                                key_b: str, scale_b: str) -> float:
        """
        Calculate harmonic compatibility score (0-1).
        Considers both key distance and scale relationship.
        """
        distance = self._get_key_distance(key_a, key_b)
        
        # Base compatibility from key distance
        # 0 distance = 1.0, 6 distance = 0.0
        key_compat = 1.0 - (distance / 6.0)
        
        # Scale relationship bonus
        if scale_a == scale_b:
            scale_bonus = 0.1
        else:
            # Relative major/minor (3 semitones apart) are compatible
            scale_bonus = 0.05
            
        return min(1.0, key_compat + scale_bonus)
    
    def _get_segment_energy(self, segment: str) -> float:
        """Get the energy level of a segment type."""
        # Normalize segment name
        segment_lower = segment.lower().strip()
        
        # Direct match
        if segment_lower in SEGMENT_ENERGY:
            return SEGMENT_ENERGY[segment_lower]
        
        # Partial match (e.g., 'verse2' -> 'verse')
        for seg_type, energy in SEGMENT_ENERGY.items():
            if segment_lower.startswith(seg_type.rstrip('12')):
                return energy
                
        # Default to mid-energy
        return 0.5
    
    def _get_segment_category(self, segment: str) -> str:
        """Categorize a segment into one of the main types."""
        segment_lower = segment.lower().strip()
        
        # Remove numbers for categorization
        for seg_type in SEGMENT_TYPES:
            if segment_lower.startswith(seg_type) or seg_type.startswith(segment_lower.rstrip('0123456789')):
                return seg_type
        
        # Map specific segments
        if 'drop' in segment_lower:
            return 'beat-drop'
        if 'break' in segment_lower:
            return 'breakdown'
            
        return 'verse'  # Default
    
    def _calculate_bpm_compatibility(self, bpm_a: float, bpm_b: float) -> Tuple[float, float, float]:
        """
        Calculate BPM compatibility metrics.
        Returns: (ratio, difference, compatibility_score)
        """
        # Handle half-time/double-time relationships
        ratios = [bpm_a / bpm_b, (bpm_a * 2) / bpm_b, bpm_a / (bpm_b * 2)]
        
        # Find the ratio closest to 1.0
        best_ratio = min(ratios, key=lambda x: abs(x - 1.0))
        
        # Calculate how far from perfect match
        deviation = abs(best_ratio - 1.0)
        
        # Compatibility score (1.0 = perfect, 0.0 = incompatible)
        # DJs typically can mix within ~5% BPM difference comfortably
        compatibility = max(0.0, 1.0 - (deviation / 0.15))
        
        return best_ratio, bpm_a - bpm_b, compatibility
    
    def _extract_features(self, song_a: SongFeatures, song_b: SongFeatures,
                          exit_segment: str, entry_segment: str) -> np.ndarray:
        """
        Extract feature vector for a transition.
        """
        features = []
        
        # BPM features
        bpm_ratio, bpm_diff, bpm_compat = self._calculate_bpm_compatibility(
            song_a.bpm, song_b.bpm
        )
        features.extend([
            bpm_ratio,                    # BPM ratio
            bpm_diff,                     # Raw BPM difference
            bpm_compat,                   # BPM compatibility score
            abs(bpm_diff),                # Absolute BPM difference
        ])
        
        # Key/harmonic features
        key_compat = self._get_key_compatibility(
            song_a.key, song_a.scale, song_b.key, song_b.scale
        )
        key_distance = self._get_key_distance(song_a.key, song_b.key)
        same_scale = 1.0 if song_a.scale == song_b.scale else 0.0
        
        features.extend([
            key_compat,                   # Harmonic compatibility
            key_distance / 6.0,           # Normalized key distance
            same_scale,                   # Same scale indicator
            song_a.key_strength,          # Song A key strength
            song_b.key_strength,          # Song B key strength
            (song_a.key_strength + song_b.key_strength) / 2,  # Avg key strength
        ])
        
        # Energy/intensity features
        exit_energy = self._get_segment_energy(exit_segment)
        entry_energy = self._get_segment_energy(entry_segment)
        energy_diff = entry_energy - exit_energy
        
        features.extend([
            exit_energy,                  # Exit segment energy
            entry_energy,                 # Entry segment energy
            energy_diff,                  # Energy difference (direction)
            abs(energy_diff),             # Energy jump magnitude
            song_a.loudness,              # Song A loudness
            song_b.loudness,              # Song B loudness
            song_b.loudness - song_a.loudness,  # Loudness change
        ])
        
        # Danceability features
        features.extend([
            song_a.danceability,
            song_b.danceability,
            song_b.danceability - song_a.danceability,
            abs(song_b.danceability - song_a.danceability),
        ])
        
        # Spectral features (texture/brightness)
        features.extend([
            song_a.spectral_centroid / 3000.0,  # Normalized
            song_b.spectral_centroid / 3000.0,
            (song_b.spectral_centroid - song_a.spectral_centroid) / 3000.0,
            song_a.spectral_rolloff / 4000.0,
            song_b.spectral_rolloff / 4000.0,
        ])
        
        # Dissonance and rhythm
        features.extend([
            song_a.dissonance,
            song_b.dissonance,
            abs(song_b.dissonance - song_a.dissonance),
            song_a.onset_rate / 10.0,     # Normalized onset rate
            song_b.onset_rate / 10.0,
        ])
        
        # Segment type encoding (simplified)
        exit_cat = self._get_segment_category(exit_segment)
        entry_cat = self._get_segment_category(entry_segment)
        
        # Transition pattern features
        # Good patterns: outro->intro, build-up->drop, verse->chorus
        is_natural_flow = 0.0
        if (exit_cat, entry_cat) in [
            ('outro', 'intro'),
            ('build-up', 'beat-drop'),
            ('build-up', 'chorus'),
            ('verse', 'chorus'),
            ('verse', 'build-up'),
            ('cool-down', 'verse'),
            ('cool-down', 'intro'),
            ('breakdown', 'build-up'),
            ('chorus', 'verse'),
        ]:
            is_natural_flow = 1.0
        
        # Jarring patterns
        is_jarring = 0.0
        if (exit_cat, entry_cat) in [
            ('intro', 'beat-drop'),
            ('verse', 'beat-drop'),
            ('beat-drop', 'intro'),
            ('chorus', 'intro'),
        ]:
            is_jarring = 1.0
            
        features.extend([
            is_natural_flow,
            is_jarring,
        ])
        
        # One-hot encode segment categories
        for seg_type in SEGMENT_TYPES:
            features.append(1.0 if exit_cat == seg_type else 0.0)
        for seg_type in SEGMENT_TYPES:
            features.append(1.0 if entry_cat == seg_type else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _get_feature_names(self) -> List[str]:
        """Get names for all features."""
        names = [
            # BPM features
            'bpm_ratio', 'bpm_diff', 'bpm_compat', 'bpm_diff_abs',
            # Key features
            'key_compat', 'key_distance_norm', 'same_scale', 
            'key_strength_a', 'key_strength_b', 'key_strength_avg',
            # Energy features
            'exit_energy', 'entry_energy', 'energy_diff', 'energy_jump',
            'loudness_a', 'loudness_b', 'loudness_change',
            # Danceability
            'dance_a', 'dance_b', 'dance_diff', 'dance_diff_abs',
            # Spectral
            'spectral_centroid_a', 'spectral_centroid_b', 'spectral_centroid_diff',
            'spectral_rolloff_a', 'spectral_rolloff_b',
            # Dissonance and rhythm
            'dissonance_a', 'dissonance_b', 'dissonance_diff',
            'onset_rate_a', 'onset_rate_b',
            # Transition patterns
            'is_natural_flow', 'is_jarring',
        ]
        
        # Add segment one-hot names
        for seg_type in SEGMENT_TYPES:
            names.append(f'exit_{seg_type}')
        for seg_type in SEGMENT_TYPES:
            names.append(f'entry_{seg_type}')
            
        return names
    
    def load_training_data(self, songs_file: str, transitions_file: str) -> pd.DataFrame:
        """
        Load and prepare training data from JSON files.
        
        Args:
            songs_file: Path to song features JSON
            transitions_file: Path to transition ratings JSON
            
        Returns:
            DataFrame with extracted features and ratings
        """
        # Load song features
        with open(songs_file, 'r') as f:
            songs_data = json.load(f)
        
        # Index songs by name
        self.song_features = {}
        for song in songs_data['songs']:
            name = song['song_name']
            self.song_features[name] = SongFeatures.from_dict(song['features'])
        
        print(f"Loaded {len(self.song_features)} songs")
        
        # Load transitions
        with open(transitions_file, 'r') as f:
            transitions_data = json.load(f)
        
        # Build training dataset
        X_list = []
        y_list = []
        metadata = []
        
        for trans in transitions_data['transitions']:
            song_a_name = trans['song_a']
            song_b_name = trans['song_b']
            
            if song_a_name not in self.song_features:
                print(f"Warning: Song not found: {song_a_name}")
                continue
            if song_b_name not in self.song_features:
                print(f"Warning: Song not found: {song_b_name}")
                continue
                
            song_a = self.song_features[song_a_name]
            song_b = self.song_features[song_b_name]
            
            features = self._extract_features(
                song_a, song_b,
                trans['exit_segment'],
                trans['entry_segment']
            )
            
            X_list.append(features)
            y_list.append(trans['rating'])
            metadata.append({
                'song_a': song_a_name,
                'song_b': song_b_name,
                'exit_segment': trans['exit_segment'],
                'entry_segment': trans['entry_segment'],
            })
        
        self.feature_names = self._get_feature_names()
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Prepared {len(X)} training samples")
        print(f"Rating distribution: min={y.min()}, max={y.max()}, mean={y.mean():.2f}")
        
        # Create DataFrame for inspection
        df = pd.DataFrame(X, columns=self.feature_names)
        df['rating'] = y
        for key in ['song_a', 'song_b', 'exit_segment', 'entry_segment']:
            df[key] = [m[key] for m in metadata]
        
        self._X = X
        self._y = y
        self._metadata = metadata
        
        return df
    
    def train(self, test_size: float = 0.2, 
              n_estimators: int = 200,
              max_depth: int = 6,
              learning_rate: float = 0.1,
              verbose: bool = True) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            test_size: Fraction of data to use for validation
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            verbose: Print training progress
            
        Returns:
            Dictionary with training metrics
        """
        if not hasattr(self, '_X'):
            raise ValueError("No training data loaded. Call load_training_data first.")
        
        X, y = self._X, self._y
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=self.random_state,
            n_jobs=-1,
            early_stopping_rounds=20,
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        train_mae = np.mean(np.abs(train_pred - y_train))
        val_mae = np.mean(np.abs(val_pred - y_val))
        
        # Correlation (how well we rank)
        train_corr = np.corrcoef(train_pred, y_train)[0, 1]
        val_corr = np.corrcoef(val_pred, y_val)[0, 1]
        
        self.training_stats = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_corr': train_corr,
            'val_corr': val_corr,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_features': X.shape[1],
        }
        
        self._is_fitted = True
        
        if verbose:
            print("\n=== Training Results ===")
            print(f"Training RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}")
            print(f"Validation RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")
            print(f"Training Correlation: {train_corr:.3f}")
            print(f"Validation Correlation: {val_corr:.3f}")
        
        return self.training_stats
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance rankings."""
        if not self._is_fitted:
            raise ValueError("Model not trained yet.")
            
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)
    
    def predict_rating(self, song_a: SongFeatures, song_b: SongFeatures,
                       exit_segment: str, entry_segment: str) -> float:
        """
        Predict the rating for a single transition.
        
        Args:
            song_a: Features of the source song
            song_b: Features of the destination song
            exit_segment: Segment type to exit from
            entry_segment: Segment type to enter on
            
        Returns:
            Predicted rating (1-10 scale)
        """
        if not self._is_fitted:
            raise ValueError("Model not trained yet.")
            
        features = self._extract_features(song_a, song_b, exit_segment, entry_segment)
        prediction = self.model.predict(features.reshape(1, -1))[0]
        
        # Clip to valid rating range
        return float(np.clip(prediction, 1.0, 10.0))
    
    def rank_transitions(self, song_a: SongFeatures, song_b: SongFeatures,
                         exit_segments: List[str], 
                         entry_segments: List[str]) -> List[TransitionPrediction]:
        """
        Rank all possible transitions between two songs.
        
        Args:
            song_a: Features of the source song
            song_b: Features of the destination song
            exit_segments: Available segments to exit from in song A
            entry_segments: Available segments to enter on in song B
            
        Returns:
            List of TransitionPrediction objects, sorted by predicted rating (descending)
        """
        if not self._is_fitted:
            raise ValueError("Model not trained yet.")
        
        predictions = []
        
        # Generate all possible combinations
        for exit_seg in exit_segments:
            for entry_seg in entry_segments:
                rating = self.predict_rating(song_a, song_b, exit_seg, entry_seg)
                predictions.append(TransitionPrediction(
                    exit_segment=exit_seg,
                    entry_segment=entry_seg,
                    predicted_rating=rating
                ))
        
        # Sort by rating descending
        predictions.sort(key=lambda x: x.predicted_rating, reverse=True)
        
        return predictions
    
    def rank_transitions_from_dicts(self, song_a_features: Dict, song_b_features: Dict,
                                     exit_segments: List[str],
                                     entry_segments: List[str]) -> List[Dict]:
        """
        Rank transitions using dictionaries instead of SongFeatures objects.
        Convenience method for API integration.
        
        Args:
            song_a_features: Dict with song A features (bpm, key, scale, etc.)
            song_b_features: Dict with song B features
            exit_segments: Available exit segments
            entry_segments: Available entry segments
            
        Returns:
            List of dicts with transition rankings
        """
        song_a = SongFeatures.from_dict(song_a_features)
        song_b = SongFeatures.from_dict(song_b_features)
        
        predictions = self.rank_transitions(song_a, song_b, exit_segments, entry_segments)
        
        return [p.to_dict() for p in predictions]
    
    def save(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save model (will create .json for XGBoost and .pkl for metadata)
        """
        if not self._is_fitted:
            raise ValueError("Model not trained yet.")
        
        base_path = Path(filepath)
        
        # Save XGBoost model
        model_path = base_path.with_suffix('.json')
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'song_features': {k: asdict(v) for k, v in self.song_features.items()},
        }
        
        metadata_path = base_path.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DJTransitionModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded DJTransitionModel instance
        """
        base_path = Path(filepath)
        model_path = base_path.with_suffix('.json')
        metadata_path = base_path.with_suffix('.metadata.pkl')
        
        instance = cls()
        
        # Load XGBoost model
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(str(model_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        instance.feature_names = metadata['feature_names']
        instance.training_stats = metadata['training_stats']
        instance.song_features = {
            k: SongFeatures.from_dict(v) 
            for k, v in metadata.get('song_features', {}).items()
        }
        instance._is_fitted = True
        
        print(f"Model loaded from {model_path}")
        
        return instance


def main():
    """Main function to train and save the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DJ Transition Model')
    parser.add_argument('--songs', type=str, required=True,
                        help='Path to song features JSON')
    parser.add_argument('--transitions', type=str, required=True,
                        help='Path to transition ratings JSON')
    parser.add_argument('--output', type=str, default='dj_transition_model',
                        help='Output path for saved model')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of boosting rounds')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Create and train model
    model = DJTransitionModel()
    
    print("Loading training data...")
    df = model.load_training_data(args.songs, args.transitions)
    
    print("\nTraining model...")
    stats = model.train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    
    print("\nFeature Importance:")
    print(model.get_feature_importance())
    
    print(f"\nSaving model to {args.output}...")
    model.save(args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

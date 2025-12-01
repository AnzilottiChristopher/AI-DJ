"""
DJ Transition Ranking Service

A lightweight service wrapper for the DJ Transition Model.
Designed for easy integration with FastAPI backend.

Usage:
    from dj_transition_service import TransitionRankingService
    
    # Initialize (loads model)
    service = TransitionRankingService('path/to/model')
    
    # Get ranked transitions
    rankings = service.get_transition_rankings(
        song_a_features={'bpm': 128, 'key': 'G', 'scale': 'major', ...},
        song_b_features={'bpm': 126, 'key': 'A', 'scale': 'minor', ...},
        song_a_segments=['chorus', 'outro', 'cool-down'],
        song_b_segments=['intro', 'verse', 'build-up']
    )
    
    # Returns: [{'exit': 'outro', 'entry': 'intro', 'score': 9.2}, ...]
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

from dj_transition_model import DJTransitionModel, SongFeatures


@dataclass
class TransitionRecommendation:
    """Single transition recommendation."""
    exit_segment: str
    entry_segment: str
    score: float
    rank: int
    
    def to_dict(self) -> Dict:
        return {
            'exit_segment': self.exit_segment,
            'entry_segment': self.entry_segment,
            'score': round(self.score, 2),
            'rank': self.rank
        }


class TransitionRankingService:
    """
    Service for ranking DJ transitions between songs.
    
    Provides a clean API for backend integration, handling model loading,
    feature preparation, and prediction formatting.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the service.
        
        Args:
            model_path: Path to trained model. If None, model must be loaded later.
        """
        self._model: Optional[DJTransitionModel] = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk."""
        self._model = DJTransitionModel.load(model_path)
        print(f"TransitionRankingService: Model loaded from {model_path}")
    
    def is_ready(self) -> bool:
        """Check if the service is ready to make predictions."""
        return self._model is not None and self._model._is_fitted
    
    def _validate_features(self, features: Dict) -> Dict:
        """Validate and fill in default values for song features."""
        defaults = {
            'bpm': 120.0,
            'key': 'C',
            'scale': 'major',
            'key_strength': 0.8,
            'loudness': 0.8,
            'danceability': 1.0,
            'spectral_centroid': 1500.0,
            'spectral_rolloff': 2000.0,
            'dissonance': 0.45,
            'onset_rate': 3.5
        }
        
        validated = defaults.copy()
        for key in defaults:
            if key in features and features[key] is not None:
                validated[key] = features[key]
        
        return validated
    
    def get_transition_rankings(
        self,
        song_a_features: Dict,
        song_b_features: Dict,
        song_a_segments: List[str],
        song_b_segments: List[str],
        top_n: Optional[int] = None
    ) -> List[TransitionRecommendation]:
        """
        Get ranked transition recommendations between two songs.
        
        Args:
            song_a_features: Audio features of the current song
                            (bpm, key, scale, loudness, danceability, etc.)
            song_b_features: Audio features of the next song
            song_a_segments: Available exit segments in song A
            song_b_segments: Available entry segments in song B
            top_n: If set, return only top N recommendations
            
        Returns:
            List of TransitionRecommendation objects, sorted by score (best first)
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate features
        song_a_validated = self._validate_features(song_a_features)
        song_b_validated = self._validate_features(song_b_features)
        
        # Get predictions from model
        raw_rankings = self._model.rank_transitions_from_dicts(
            song_a_validated,
            song_b_validated,
            song_a_segments,
            song_b_segments
        )
        
        # Convert to TransitionRecommendation objects
        recommendations = [
            TransitionRecommendation(
                exit_segment=r['exit_segment'],
                entry_segment=r['entry_segment'],
                score=r['predicted_rating'],
                rank=i + 1
            )
            for i, r in enumerate(raw_rankings)
        ]
        
        if top_n is not None:
            recommendations = recommendations[:top_n]
        
        return recommendations
    
    def get_best_transition(
        self,
        song_a_features: Dict,
        song_b_features: Dict,
        song_a_segments: List[str],
        song_b_segments: List[str]
    ) -> TransitionRecommendation:
        """
        Get the single best transition between two songs.
        
        Convenience method for getting just the top recommendation.
        """
        rankings = self.get_transition_rankings(
            song_a_features,
            song_b_features,
            song_a_segments,
            song_b_segments,
            top_n=1
        )
        return rankings[0]
    
    def get_transition_rankings_json(
        self,
        song_a_features: Dict,
        song_b_features: Dict,
        song_a_segments: List[str],
        song_b_segments: List[str],
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Get ranked transitions as JSON-serializable dicts.
        
        Same as get_transition_rankings but returns dicts instead of dataclasses.
        """
        recommendations = self.get_transition_rankings(
            song_a_features,
            song_b_features,
            song_a_segments,
            song_b_segments,
            top_n
        )
        return [r.to_dict() for r in recommendations]
    
    def get_compatibility_score(
        self,
        song_a_features: Dict,
        song_b_features: Dict
    ) -> Dict[str, float]:
        """
        Get overall compatibility metrics between two songs.
        
        Returns a dict with:
        - overall_score: General mixability score (1-10)
        - bpm_compatibility: How well BPMs match (0-1)
        - key_compatibility: Harmonic compatibility (0-1)
        - energy_compatibility: Energy level match (0-1)
        """
        song_a = SongFeatures.from_dict(self._validate_features(song_a_features))
        song_b = SongFeatures.from_dict(self._validate_features(song_b_features))
        
        # Calculate BPM compatibility
        bpm_ratio = song_a.bpm / song_b.bpm
        bpm_ratios = [bpm_ratio, bpm_ratio * 2, bpm_ratio / 2]
        best_ratio = min(bpm_ratios, key=lambda x: abs(x - 1.0))
        bpm_compat = max(0.0, 1.0 - abs(best_ratio - 1.0) / 0.15)
        
        # Key compatibility (using model's internal method)
        key_compat = self._model._get_key_compatibility(
            song_a.key, song_a.scale,
            song_b.key, song_b.scale
        )
        
        # Energy compatibility (based on loudness and danceability)
        energy_diff = abs(song_a.loudness - song_b.loudness)
        dance_diff = abs(song_a.danceability - song_b.danceability)
        energy_compat = max(0.0, 1.0 - (energy_diff + dance_diff * 0.3) / 0.5)
        
        # Overall score
        overall = (bpm_compat * 0.4 + key_compat * 0.35 + energy_compat * 0.25) * 10
        
        return {
            'overall_score': round(overall, 2),
            'bpm_compatibility': round(bpm_compat, 3),
            'key_compatibility': round(key_compat, 3),
            'energy_compatibility': round(energy_compat, 3)
        }


# FastAPI integration example
def create_fastapi_routes(app, service: TransitionRankingService):
    """
    Add transition ranking routes to a FastAPI app.
    
    Usage:
        from fastapi import FastAPI
        from dj_transition_service import TransitionRankingService, create_fastapi_routes
        
        app = FastAPI()
        service = TransitionRankingService('model_path')
        create_fastapi_routes(app, service)
    """
    from fastapi import HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    
    class SongFeaturesInput(BaseModel):
        bpm: float
        key: str
        scale: str
        key_strength: Optional[float] = 0.8
        loudness: Optional[float] = 0.8
        danceability: Optional[float] = 1.0
        spectral_centroid: Optional[float] = 1500.0
        spectral_rolloff: Optional[float] = 2000.0
        dissonance: Optional[float] = 0.45
        onset_rate: Optional[float] = 3.5
    
    class TransitionRequest(BaseModel):
        song_a: SongFeaturesInput
        song_b: SongFeaturesInput
        song_a_segments: List[str]
        song_b_segments: List[str]
        top_n: Optional[int] = None
    
    class TransitionResponse(BaseModel):
        exit_segment: str
        entry_segment: str
        score: float
        rank: int
    
    @app.post("/api/transitions/rank", response_model=List[TransitionResponse])
    async def rank_transitions(request: TransitionRequest):
        """Rank possible transitions between two songs."""
        try:
            rankings = service.get_transition_rankings_json(
                song_a_features=request.song_a.model_dump(),
                song_b_features=request.song_b.model_dump(),
                song_a_segments=request.song_a_segments,
                song_b_segments=request.song_b_segments,
                top_n=request.top_n
            )
            return rankings
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/transitions/best")
    async def get_best_transition(request: TransitionRequest):
        """Get the single best transition between two songs."""
        try:
            best = service.get_best_transition(
                song_a_features=request.song_a.model_dump(),
                song_b_features=request.song_b.model_dump(),
                song_a_segments=request.song_a_segments,
                song_b_segments=request.song_b_segments
            )
            return best.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    class CompatibilityRequest(BaseModel):
        song_a: SongFeaturesInput
        song_b: SongFeaturesInput
    
    @app.post("/api/transitions/compatibility")
    async def get_compatibility(request: CompatibilityRequest):
        """Get compatibility metrics between two songs."""
        try:
            return service.get_compatibility_score(
                song_a_features=request.song_a.model_dump(),
                song_b_features=request.song_b.model_dump()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # Demo usage
    print("DJ Transition Ranking Service - Demo")
    print("=" * 50)
    
    # Load the service
    service = TransitionRankingService('/home/claude/trained_dj_model')
    
    # Example songs
    song_a = {
        'bpm': 128.0,
        'key': 'G',
        'scale': 'major',
        'key_strength': 0.93,
        'loudness': 0.92,
        'danceability': 1.19,
        'spectral_centroid': 1525.0,
        'spectral_rolloff': 2115.0,
        'dissonance': 0.47,
        'onset_rate': 3.8
    }
    
    song_b = {
        'bpm': 122.0,
        'key': 'A',
        'scale': 'major',
        'key_strength': 0.97,
        'loudness': 0.80,
        'danceability': 1.04,
        'spectral_centroid': 1198.0,
        'spectral_rolloff': 1260.0,
        'dissonance': 0.45,
        'onset_rate': 4.1
    }
    
    # Get compatibility
    print("\n📊 Song Compatibility:")
    compat = service.get_compatibility_score(song_a, song_b)
    print(f"  Overall Score: {compat['overall_score']}/10")
    print(f"  BPM Match: {compat['bpm_compatibility']:.1%}")
    print(f"  Key Match: {compat['key_compatibility']:.1%}")
    print(f"  Energy Match: {compat['energy_compatibility']:.1%}")
    
    # Get transition rankings
    print("\n🎵 Top 5 Transitions:")
    rankings = service.get_transition_rankings_json(
        song_a, song_b,
        song_a_segments=['chorus', 'cool-down', 'outro'],
        song_b_segments=['intro', 'verse', 'build-up'],
        top_n=5
    )
    
    for r in rankings:
        print(f"  {r['rank']}. {r['exit_segment']} → {r['entry_segment']} (score: {r['score']})")
    
    # Get best transition
    print("\n✨ Best Transition:")
    best = service.get_best_transition(
        song_a, song_b,
        song_a_segments=['chorus', 'cool-down', 'outro'],
        song_b_segments=['intro', 'verse', 'build-up']
    )
    print(f"  {best.exit_segment} → {best.entry_segment} (score: {best.score:.2f})")

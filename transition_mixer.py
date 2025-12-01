"""
Transition Mixer - Handles intelligent DJ transitions between songs.

This module computes optimal transition points using the ML model and
creates smooth audio crossfades between tracks.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

# Import our trained model service
from dj_transition_service import TransitionRankingService, TransitionRecommendation


@dataclass
class Segment:
    """Represents a segment within a song."""
    name: str
    start: float  # seconds
    end: float    # seconds
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def to_samples(self, sample_rate: int) -> Tuple[int, int]:
        """Convert to sample indices."""
        return int(self.start * sample_rate), int(self.end * sample_rate)


@dataclass
class TransitionPlan:
    """Complete plan for transitioning between two songs."""
    song_a_title: str
    song_b_title: str
    exit_segment: Segment
    entry_segment: Segment
    predicted_score: float
    crossfade_duration: float  # seconds
    
    # Timing info
    transition_start_time: float  # when to start the crossfade (seconds into song A)
    song_b_start_offset: float    # where to start song B (seconds)
    
    def to_dict(self) -> Dict:
        return {
            'song_a': self.song_a_title,
            'song_b': self.song_b_title,
            'exit_segment': self.exit_segment.name,
            'entry_segment': self.entry_segment.name,
            'score': round(self.predicted_score, 2),
            'crossfade_duration': self.crossfade_duration,
            'transition_start_time': round(self.transition_start_time, 2),
            'song_b_start_offset': round(self.song_b_start_offset, 2),
        }


class TransitionMixer:
    """
    Handles computing and executing DJ transitions between songs.
    
    Uses the trained XGBoost model to find optimal transition points,
    then creates smooth audio crossfades.
    """
    
    def __init__(self, model_path: str, sample_rate: int = 44100):
        """
        Initialize the mixer.
        
        Args:
            model_path: Path to trained transition model
            sample_rate: Audio sample rate (default 44100)
        """
        self.sample_rate = sample_rate
        self.ranking_service = TransitionRankingService(model_path)
        
        # Default crossfade settings
        self.default_crossfade_duration = 8.0  # seconds
        self.min_crossfade_duration = 4.0
        self.max_crossfade_duration = 16.0
        
    def _parse_segments(self, segments_data: List[Dict]) -> List[Segment]:
        """Parse segment data from metadata into Segment objects."""
        segments = []
        for seg in segments_data:
            segments.append(Segment(
                name=seg['name'],
                start=seg['start'],
                end=seg['end']
            ))
        return segments
    
    def _get_available_exit_segments(self, segments: List[Segment], 
                                      current_position: float) -> List[Segment]:
        """
        Get segments that can still be used for exit (haven't passed yet).
        
        Args:
            segments: All segments in the song
            current_position: Current playback position in seconds
            
        Returns:
            List of segments that start after current position
        """
        # We want segments whose START is still ahead of us
        # Give a small buffer (2 seconds) to allow preparation time
        buffer = 2.0
        available = [s for s in segments if s.start > current_position + buffer]
        return available
    
    def _calculate_crossfade_duration(self, song_a_features: Dict, 
                                       song_b_features: Dict,
                                       exit_segment: Segment,
                                       entry_segment: Segment) -> float:
        """
        Calculate appropriate crossfade duration based on song characteristics.
        """
        # Base duration on BPM - faster songs = shorter crossfade
        avg_bpm = (song_a_features.get('bpm', 120) + song_b_features.get('bpm', 120)) / 2
        
        # At 120 BPM, use 8 seconds. Scale inversely with BPM
        base_duration = 8.0 * (120 / avg_bpm)
        
        # Adjust based on segment types
        # Build-up to drop = shorter, more impactful
        # Outro to intro = can be longer, smoother
        exit_name = exit_segment.name.lower()
        entry_name = entry_segment.name.lower()
        
        if 'build' in exit_name and 'drop' in entry_name:
            base_duration *= 0.5  # Quick, punchy transition
        elif 'outro' in exit_name and 'intro' in entry_name:
            base_duration *= 1.2  # Smooth, gradual transition
        elif 'chorus' in exit_name or 'chorus' in entry_name:
            base_duration *= 0.8  # Don't linger on high-energy sections
            
        # Clamp to reasonable range
        return max(self.min_crossfade_duration, 
                   min(self.max_crossfade_duration, base_duration))
    
    def compute_transition(self, 
                           song_a_data: Dict,
                           song_b_data: Dict,
                           current_position: float = 0.0) -> Optional[TransitionPlan]:
        """
        Compute the optimal transition between two songs.
        
        Args:
            song_a_data: Current song metadata (features, segments)
            song_b_data: Next song metadata
            current_position: Current playback position in song A (seconds)
            
        Returns:
            TransitionPlan with optimal transition details, or None if no valid transition
        """
        # Parse segments
        segments_a = self._parse_segments(song_a_data.get('segments', []))
        segments_b = self._parse_segments(song_b_data.get('segments', []))
        
        if not segments_a or not segments_b:
            print("[MIXER] Warning: Missing segment data, using fallback")
            return self._fallback_transition(song_a_data, song_b_data, current_position)
        
        # Get available exit segments (ones we haven't passed yet)
        available_exits = self._get_available_exit_segments(segments_a, current_position)
        
        if not available_exits:
            print("[MIXER] No exit segments available ahead of current position")
            return None
        
        # Get segment names for the model
        exit_segment_names = [s.name for s in available_exits]
        entry_segment_names = [s.name for s in segments_b]
        
        # Use ML model to rank transitions
        rankings = self.ranking_service.get_transition_rankings(
            song_a_features=song_a_data['features'],
            song_b_features=song_b_data['features'],
            song_a_segments=exit_segment_names,
            song_b_segments=entry_segment_names,
            top_n=5  # Get top 5 options
        )
        
        if not rankings:
            print("[MIXER] No valid transitions found")
            return None
        
        # Pick the best one
        best = rankings[0]
        
        # Find the actual segment objects
        exit_segment = next(s for s in available_exits if s.name == best.exit_segment)
        entry_segment = next(s for s in segments_b if s.name == best.entry_segment)
        
        # Calculate crossfade duration
        crossfade_duration = self._calculate_crossfade_duration(
            song_a_data['features'],
            song_b_data['features'],
            exit_segment,
            entry_segment
        )
        
        # Determine exact timing
        # Start transition at the END of exit segment minus crossfade duration
        # This way the crossfade completes right as we enter the new segment
        transition_start = exit_segment.end - crossfade_duration
        
        # Ensure we don't start before current position
        transition_start = max(transition_start, current_position + 2.0)
        
        # Song B starts at the beginning of entry segment
        song_b_start_offset = entry_segment.start
        
        plan = TransitionPlan(
            song_a_title=song_a_data.get('title', 'Unknown'),
            song_b_title=song_b_data.get('title', 'Unknown'),
            exit_segment=exit_segment,
            entry_segment=entry_segment,
            predicted_score=best.score,
            crossfade_duration=crossfade_duration,
            transition_start_time=transition_start,
            song_b_start_offset=song_b_start_offset
        )
        
        print(f"[MIXER] Computed transition: {exit_segment.name} → {entry_segment.name} "
              f"(score: {best.score:.1f}, starts at {transition_start:.1f}s)")
        
        return plan
    
    def _fallback_transition(self, song_a_data: Dict, song_b_data: Dict, 
                             current_position: float) -> TransitionPlan:
        """
        Create a simple fallback transition when segment data is missing.
        Uses the last portion of song A and beginning of song B.
        """
        # Estimate song duration from audio file if available
        song_a_duration = song_a_data.get('duration', 180.0)  # Default 3 min
        
        # Transition in the last 30 seconds of the song
        transition_start = max(current_position + 10, song_a_duration - 30)
        
        return TransitionPlan(
            song_a_title=song_a_data.get('title', 'Unknown'),
            song_b_title=song_b_data.get('title', 'Unknown'),
            exit_segment=Segment('outro', transition_start, song_a_duration),
            entry_segment=Segment('intro', 0, 15),
            predicted_score=5.0,  # Neutral score for fallback
            crossfade_duration=8.0,
            transition_start_time=transition_start,
            song_b_start_offset=0.0
        )
    
    def create_crossfade(self, 
                         audio_a: np.ndarray,
                         audio_b: np.ndarray,
                         plan: TransitionPlan) -> Tuple[np.ndarray, int, int]:
        """
        Create the actual crossfade audio between two tracks.
        
        Args:
            audio_a: Full audio array for song A
            audio_b: Full audio array for song B
            plan: TransitionPlan with timing details
            
        Returns:
            Tuple of:
            - crossfade_audio: The mixed crossfade segment
            - crossfade_start_sample: Where in song A the crossfade begins
            - song_b_continue_sample: Where in song B to continue after crossfade
        """
        sr = self.sample_rate
        
        # Convert times to samples
        crossfade_samples = int(plan.crossfade_duration * sr)
        transition_start_sample = int(plan.transition_start_time * sr)
        song_b_start_sample = int(plan.song_b_start_offset * sr)
        
        # Ensure we don't exceed array bounds
        transition_start_sample = min(transition_start_sample, len(audio_a) - crossfade_samples)
        transition_start_sample = max(0, transition_start_sample)
        
        # Extract the segments to crossfade
        segment_a = audio_a[transition_start_sample:transition_start_sample + crossfade_samples]
        
        # For song B, we start at the entry segment
        song_b_end_sample = min(song_b_start_sample + crossfade_samples, len(audio_b))
        segment_b = audio_b[song_b_start_sample:song_b_end_sample]
        
        # Ensure both segments are the same length
        min_len = min(len(segment_a), len(segment_b))
        segment_a = segment_a[:min_len]
        segment_b = segment_b[:min_len]
        
        # Create crossfade curves (equal-power crossfade)
        t = np.linspace(0, 1, min_len)
        
        # Equal-power curves for smooth energy transition
        # fade_out: cos curve from 1 to 0
        # fade_in: sin curve from 0 to 1
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)
        
        # Handle stereo/mono
        if len(segment_a.shape) == 2:
            fade_out = fade_out[:, np.newaxis]
            fade_in = fade_in[:, np.newaxis]
        
        # Apply crossfade
        crossfade_audio = (segment_a * fade_out) + (segment_b * fade_in)
        
        # Where song B continues after crossfade
        song_b_continue_sample = song_b_start_sample + min_len
        
        return crossfade_audio, transition_start_sample, song_b_continue_sample
    
    def prepare_mixed_audio(self,
                            audio_a: np.ndarray,
                            audio_b: np.ndarray,
                            plan: TransitionPlan) -> Dict[str, Any]:
        """
        Prepare all audio segments needed for streaming with transition.
        
        Returns a dict with:
        - 'pre_transition': Audio from song A before crossfade
        - 'crossfade': The mixed crossfade audio
        - 'post_transition': Remaining audio from song B
        - 'timing': Timing information
        """
        crossfade, start_sample, continue_sample = self.create_crossfade(
            audio_a, audio_b, plan
        )
        
        # Pre-transition: everything in song A before the crossfade
        pre_transition = audio_a[:start_sample]
        
        # Post-transition: everything in song B after the crossfade
        post_transition = audio_b[continue_sample:]
        
        return {
            'pre_transition': pre_transition,
            'crossfade': crossfade,
            'post_transition': post_transition,
            'timing': {
                'pre_duration': len(pre_transition) / self.sample_rate,
                'crossfade_duration': len(crossfade) / self.sample_rate,
                'post_duration': len(post_transition) / self.sample_rate,
                'transition_start_sample': start_sample,
                'song_b_continue_sample': continue_sample,
            },
            'plan': plan.to_dict()
        }


def generate_placeholder_segments(duration: float, bpm: float = 120.0) -> List[Dict]:
    """
    Generate placeholder segment data for a song.
    
    This is a temporary solution until real segment analysis is implemented.
    Creates a typical EDM song structure based on duration and BPM.
    
    Args:
        duration: Song duration in seconds
        bpm: Song BPM for timing calculations
        
    Returns:
        List of segment dicts with name, start, end
    """
    # Typical EDM structure (rough percentages)
    # intro: 0-8%, verse1: 8-20%, build-up1: 20-28%, chorus1: 28-40%
    # verse2: 40-50%, build-up2: 50-58%, chorus2: 58-75%
    # breakdown: 75-85%, outro: 85-100%
    
    segments = [
        {'name': 'intro', 'pct_start': 0.0, 'pct_end': 0.08},
        {'name': 'verse', 'pct_start': 0.08, 'pct_end': 0.20},
        {'name': 'build-up', 'pct_start': 0.20, 'pct_end': 0.28},
        {'name': 'chorus', 'pct_start': 0.28, 'pct_end': 0.40},
        {'name': 'verse2', 'pct_start': 0.40, 'pct_end': 0.50},
        {'name': 'build-up2', 'pct_start': 0.50, 'pct_end': 0.58},
        {'name': 'beat-drop', 'pct_start': 0.58, 'pct_end': 0.75},
        {'name': 'cool-down', 'pct_start': 0.75, 'pct_end': 0.88},
        {'name': 'outro', 'pct_start': 0.88, 'pct_end': 1.0},
    ]
    
    result = []
    for seg in segments:
        result.append({
            'name': seg['name'],
            'start': round(duration * seg['pct_start'], 2),
            'end': round(duration * seg['pct_end'], 2)
        })
    
    return result
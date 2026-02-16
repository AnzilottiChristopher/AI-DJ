import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path


class TempoTransitionDJ:
    """
    AI DJ class that performs smooth tempo transitions between two songs.
    Gradually adjusts tempo of first song to meet in the middle with second song,
    then adjusts back to natural tempo of second song.
    """
    
    def __init__(self, track_a_path, track_b_path, output_path="transition_output.wav"):
        """
        Initialize with two track paths.
        
        Args:
            track_a_path: Path to first song (will transition out)
            track_b_path: Path to second song (will transition in)
            output_path: Where to save the final mixed output
        """
        self.track_a_path = track_a_path
        self.track_b_path = track_b_path
        self.output_path = output_path
        
        # Load audio files
        print("Loading tracks...")

        self.track_a, self.sr = librosa.load(track_a_path, sr=44100, mono=True)
        self.track_b, _ = librosa.load(track_b_path, sr=44100, mono=True)
        
        # Detect tempos
        print("Detecting tempos...")
        self.tempo_a = librosa.feature.tempo(y=self.track_a, sr=self.sr)[0]
        self.tempo_b = librosa.feature.tempo(y=self.track_b, sr=self.sr)[0]
        
        print(f"Track A tempo: {self.tempo_a:.1f} BPM")
        print(f"Track B tempo: {self.tempo_b:.1f} BPM")
        
    def calculate_meeting_tempo(self, bias=0.5):
        """
        Calculate the meeting point tempo.
        
        Args:
            bias: 0.0 = favor track A tempo, 1.0 = favor track B tempo, 0.5 = exact middle
        
        Returns:
            Meeting tempo in BPM
        """
        meeting_tempo = self.tempo_a * (1 - bias) + self.tempo_b * bias
        print(f"Meeting tempo: {meeting_tempo:.1f} BPM (bias: {bias})")
        return meeting_tempo
    
    def ease_in_out_curve(self, n_points):
        """
        Create a smooth easing curve for tempo transitions.
        
        Args:
            n_points: Number of points in the curve
            
        Returns:
            Array of smoothly interpolated values from 0 to 1
        """
        t = np.linspace(0, 1, n_points)
        # Smooth S-curve (ease-in-out)
        return t * t * (3.0 - 2.0 * t)
    
    def apply_smooth_time_stretch(self, audio, start_rate, end_rate, duration_seconds):
        """
        Apply gradual time stretching using phase vocoder with overlapping windows.
        This version reduces glitches by using larger, overlapping segments.
        
        Args:
            audio: Input audio array
            start_rate: Starting stretch rate (1.0 = normal)
            end_rate: Ending stretch rate (1.0 = normal)
            duration_seconds: Duration of the transition
            
        Returns:
            Time-stretched audio
        """
        print(f"  Stretching from rate {start_rate:.3f} to {end_rate:.3f} over {duration_seconds}s")
        
        # Use fewer, larger segments with overlap to reduce glitches
        n_segments = 50  # Reduced from 50
        overlap_ratio = 0.5  # 50% overlap between segments
        
        # Calculate segment parameters
        max_samples = int(duration_seconds * self.sr)
        audio_segment = audio[:max_samples]
        
        # Create smooth tempo curve
        curve = self.ease_in_out_curve(n_segments)
        rates = start_rate + (end_rate - start_rate) * curve
        
        # Calculate segment and hop sizes
        segment_length = len(audio_segment) // (n_segments * (1 - overlap_ratio))
        hop_length = int(segment_length * (1 - overlap_ratio))
        
        # Process with overlap-add
        output_length = int(len(audio_segment) * np.mean(1.0 / rates))
        output = np.zeros(output_length)
        window = np.hanning(segment_length)
        
        current_output_pos = 0
        
        for i, rate in enumerate(rates):
            start_idx = int(i * hop_length)
            end_idx = min(start_idx + segment_length, len(audio_segment))
            
            if start_idx >= len(audio_segment):
                break
            
            # Extract segment
            segment = audio_segment[start_idx:end_idx]
            
            # Apply window to reduce edge artifacts
            if len(segment) == segment_length:
                windowed_segment = segment * window
            else:
                # Handle last segment
                partial_window = window[:len(segment)]
                windowed_segment = segment * partial_window
            
            # Time stretch this segment

            stretched = librosa.effects.time_stretch(windowed_segment, rate=rate)
            
            # Overlap-add into output
            output_end = min(current_output_pos + len(stretched), len(output))
            output[current_output_pos:output_end] += stretched[:output_end - current_output_pos]
            
            current_output_pos += int(len(stretched) * (1 - overlap_ratio))
        
        return output[:current_output_pos]
    
    def apply_simple_gradual_stretch(self, audio, start_rate, end_rate, duration_seconds):
        """
        Simpler approach: divide into fewer segments with crossfading.
        This is more stable and less glitchy.
        
        Args:
            audio: Input audio array
            start_rate: Starting stretch rate (1.0 = normal)
            end_rate: Ending stretch rate (1.0 = normal)
            duration_seconds: Duration of the transition
            
        Returns:
            Time-stretched audio
        """
        print(f"  Stretching from rate {start_rate:.3f} to {end_rate:.3f} over {duration_seconds}s")
        
        # Use very few segments for stability
        n_segments = 8  # Small number of segments
        
        max_samples = int(duration_seconds * self.sr)
        audio_segment = audio[:max_samples]
        
        # Create smooth tempo curve
        curve = self.ease_in_out_curve(n_segments + 1)
        rates = start_rate + (end_rate - start_rate) * curve
        
        # Calculate segment length
        segment_length = len(audio_segment) // n_segments
        crossfade_length = segment_length // 4  # 25% crossfade
        
        stretched_parts = []
        
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = min(start_idx + segment_length, len(audio_segment))
            
            if start_idx >= len(audio_segment):
                break
            
            segment = audio_segment[start_idx:end_idx]
            
            # Use average of current and next rate for this segment
            avg_rate = (rates[i] + rates[i + 1]) / 2
            
            # Time stretch this segment
            stretched = librosa.effects.time_stretch(segment, rate=avg_rate)
            
            # Apply crossfade with previous segment (except first)
            if i > 0 and len(stretched_parts) > 0:
                prev_segment = stretched_parts[-1]
                
                # Create crossfade
                fade_len = min(crossfade_length, len(prev_segment), len(stretched))
                
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = np.linspace(0, 1, fade_len)
                
                # Apply crossfade
                prev_segment[-fade_len:] = (prev_segment[-fade_len:] * fade_out + 
                                            stretched[:fade_len] * fade_in)
                stretched = stretched[fade_len:]  # Remove crossfaded portion
            
            stretched_parts.append(stretched)
        
        return np.concatenate(stretched_parts)
    
    def create_crossfade(self, audio_a, audio_b, crossfade_duration):
        """
        Create a crossfade between two audio segments.
        
        Args:
            audio_a: First audio (fading out)
            audio_b: Second audio (fading in)
            crossfade_duration: Duration of crossfade in seconds
            
        Returns:
            Crossfaded audio
        """
        crossfade_samples = int(crossfade_duration * self.sr)
        
        # Ensure both segments are long enough
        min_length = min(len(audio_a), len(audio_b), crossfade_samples)
        
        # Take the end of audio_a and beginning of audio_b
        segment_a = audio_a[-min_length:]
        segment_b = audio_b[:min_length]
        
        # Create fade curves (equal power crossfade)
        fade_out = np.linspace(1, 0, min_length) ** 0.5
        fade_in = np.linspace(0, 1, min_length) ** 0.5
        
        # Apply fades and mix
        crossfaded = segment_a * fade_out + segment_b * fade_in
        
        return crossfaded
    
    def create_transition(self, 
                         transition_point_a=30.0,
                         transition_duration=16.0,
                         crossfade_duration=8.0,
                         tempo_bias=0.5,
                         use_simple_stretch=True):
        """
        Create the complete transition between tracks.
        
        Args:
            transition_point_a: When to start transition in track A (seconds)
            transition_duration: Total duration of tempo transition (seconds)
            crossfade_duration: Duration of crossfade (seconds)
            tempo_bias: 0.0-1.0, where to set meeting tempo
            use_simple_stretch: If True, use simpler (less glitchy) stretching method
            
        Returns:
            Final mixed audio
        """
        print("\n=== Creating Transition ===")
        
        # Choose stretching method
        stretch_method = (self.apply_simple_gradual_stretch if use_simple_stretch 
                         else self.apply_smooth_time_stretch)
        
        # Calculate meeting tempo and stretch rates
        meeting_tempo = self.calculate_meeting_tempo(tempo_bias)
        rate_a_to_meeting = meeting_tempo / self.tempo_a
        rate_meeting_to_b = self.tempo_b / meeting_tempo
        
        # PHASE 1: Pre-transition (track A at normal tempo)
        print("\nPhase 1: Pre-transition")
        transition_start_sample = int(transition_point_a * self.sr)
        pre_transition = self.track_a[:transition_start_sample]
        
        # PHASE 2: Transition track A to meeting tempo
        print("\nPhase 2: Track A transitioning to meeting tempo")
        half_transition = transition_duration / 2
        
        track_a_remaining = self.track_a[transition_start_sample:]
        track_a_transition = stretch_method(
            track_a_remaining,
            start_rate=1.0,
            end_rate=rate_a_to_meeting,
            duration_seconds=half_transition
        )
        
        # PHASE 3: Hold track A at meeting tempo for crossfade
        print("\nPhase 3: Preparing crossfade at meeting tempo")
        
        # Get more audio at meeting tempo for smoother crossfade
        after_transition_idx = int(half_transition * self.sr)
        remaining_a = track_a_remaining[after_transition_idx:]
        track_a_at_meeting = librosa.effects.time_stretch(
            remaining_a[:int(crossfade_duration * self.sr * 2)],
            rate=rate_a_to_meeting
        )
        
        # Combine transition and sustained meeting tempo
        track_a_full = np.concatenate([track_a_transition, track_a_at_meeting])
        
        # PHASE 4: Track B starts at meeting tempo, transitions to natural
        print("\nPhase 4: Track B transitioning from meeting to natural tempo")
        
        # Start with track B at meeting tempo for crossfade
        track_b_for_crossfade = librosa.effects.time_stretch(
            self.track_b[:int(crossfade_duration * self.sr * 2)],
            rate=1.0/rate_meeting_to_b
        )
        
        # Then transition to natural tempo
        track_b_after_crossfade = self.track_b[int(crossfade_duration * self.sr):]
        track_b_transition = stretch_method(
            track_b_after_crossfade,
            start_rate=1.0/rate_meeting_to_b,
            end_rate=1.0,
            duration_seconds=half_transition
        )
        
        # Combine crossfade portion and transition
        track_b_full = np.concatenate([track_b_for_crossfade, track_b_transition])
        
        # PHASE 5: Create crossfade
        print("\nPhase 5: Creating crossfade")
        crossfade = self.create_crossfade(
            track_a_full,
            track_b_full,
            crossfade_duration
        )
        
        # PHASE 6: Post-transition (track B at natural tempo)
        print("\nPhase 6: Post-transition")
        post_transition_start = (int(crossfade_duration * self.sr) + 
                                int(half_transition * self.sr))
        post_transition = self.track_b[post_transition_start:
                                      post_transition_start + int(30 * self.sr)]
        
        # Combine all phases
        print("\nCombining all phases...")
        
        # Before crossfade: pre_transition + track_a up to crossfade point
        crossfade_start_in_a = len(track_a_full) - int(crossfade_duration * self.sr)
        if crossfade_start_in_a < 0:
            crossfade_start_in_a = 0
        
        before_crossfade = np.concatenate([
            pre_transition,
            track_a_full[:crossfade_start_in_a]
        ])
        
        # After crossfade: track_b after crossfade + post_transition
        after_crossfade_start = int(crossfade_duration * self.sr)
        after_crossfade = np.concatenate([
            track_b_full[after_crossfade_start:],
            post_transition
        ])
        
        final_mix = np.concatenate([
            before_crossfade,
            crossfade,
            after_crossfade
        ])
        
        # Normalize to prevent clipping
        final_mix = final_mix / np.max(np.abs(final_mix)) * 0.95
        
        return final_mix
    
    def save_output(self, audio):
        """
        Save the final mixed audio to file.
        
        Args:
            audio: Audio array to save
        """
        print(f"\nSaving output to {self.output_path}")
        sf.write(self.output_path, audio, self.sr)
        print(f"Output saved! Duration: {len(audio)/self.sr:.1f} seconds")
    
    def run(self, transition_point_a=30.0, transition_duration=16.0, 
            crossfade_duration=8.0, tempo_bias=0.5, use_simple_stretch=True):
        """
        Run the complete transition process.
        
        Args:
            transition_point_a: When to start transition in track A (seconds)
            transition_duration: Total duration of tempo transition (seconds)
            crossfade_duration: Duration of crossfade (seconds)
            tempo_bias: 0.0-1.0, where to set meeting tempo
            use_simple_stretch: If True, use simpler (less glitchy) method
        """
        print("="*50)
        print("AI DJ Tempo Transition")
        print("="*50)
        
        # Create transition
        mixed_audio = self.create_transition(
            transition_point_a=transition_point_a,
            transition_duration=transition_duration,
            crossfade_duration=crossfade_duration,
            tempo_bias=tempo_bias,
            use_simple_stretch=use_simple_stretch
        )
        
        # Save output
        self.save_output(mixed_audio)
        
        print("\n" + "="*50)
        print("Transition complete!")
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Initialize with your tracks (replace with actual paths)
    dj = TempoTransitionDJ(
        track_a_path="C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/lights_ellie-goulding.wav",
        track_b_path="C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/lean-on_major-lazer-dj-snake-mØ.wav"
    )
    # Run the transition
    # transition_point_a: where in song A to start (seconds)
    # transition_duration: how long the tempo shift takes (seconds)
    # crossfade_duration: how long the crossfade is (seconds)
    # tempo_bias: 0.0 = meet at song A's tempo, 1.0 = meet at song B's tempo, 0.5 = middle
    dj.run(
        transition_point_a=210.0,      # Start transition 30s into first song
        transition_duration=12.0,      # Spend 16s changing tempo
        crossfade_duration=6.0,        # 8s crossfade
        tempo_bias=0.7,
        use_simple_stretch=True         # Use simpler stretching to reduce glitches                  
    )
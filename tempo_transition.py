import librosa
import numpy as np
import soundfile as sf
from audiotsm import wsola, phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
from scipy import signal


class TempoTransitionDJ:
    """
    AI DJ class for smooth tempo transitions - designed for streaming/real-time use.
    Creates Apple AutoMix-style warpy transitions.
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize DJ engine.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sr = sample_rate
        
    def detect_tempo(self, audio):
        """Detect tempo of audio."""
        tempo = librosa.beat.tempo(y=audio, sr=self.sr)[0]
        return tempo
    
    def time_stretch_smooth(self, audio, speed, use_phasevocoder=False):
        """
        Time stretch using audiotsm with no chunking - smoother results.
        
        Args:
            audio: Input audio array (mono)
            speed: Speed factor (1.0 = normal, <1.0 = slower, >1.0 = faster)
            use_phasevocoder: Use phase vocoder for more "warpy" effect
            
        Returns:
            Stretched audio
        """
        # Ensure audio is 2D
        if audio.ndim == 1:
            audio_2d = audio.reshape(1, -1)
        else:
            audio_2d = audio
        
        # Create reader and writer
        reader = ArrayReader(audio_2d)
        writer = ArrayWriter(1)
        
        # Choose algorithm
        if use_phasevocoder:
            # Phase vocoder gives that "warpy" effect
            tsm = phasevocoder(channels=1, speed=speed)
        else:
            # WSOLA is cleaner for most transitions
            tsm = wsola(channels=1, speed=speed)
        
        # Process entire segment at once (no chunking)
        tsm.run(reader, writer, flush=True)
        
        return writer.data[0]
    
    def create_tempo_curve(self, start_tempo, end_tempo, duration_seconds, curve_type='ease_in_out'):
        """
        Create a tempo automation curve.
        
        Args:
            start_tempo: Starting tempo in BPM
            end_tempo: Ending tempo in BPM
            duration_seconds: Duration of transition
            curve_type: 'linear', 'ease_in_out', 'exponential', 'automix_warp'
            
        Returns:
            Array of tempo values over time
        """
        n_points = int(duration_seconds * 10)  # 10 points per second
        t = np.linspace(0, 1, n_points)
        
        if curve_type == 'linear':
            curve = t
        elif curve_type == 'ease_in_out':
            # Smooth S-curve
            curve = t * t * (3.0 - 2.0 * t)
        elif curve_type == 'exponential':
            # More dramatic acceleration
            curve = np.exp(3 * t) / np.exp(3)
        elif curve_type == 'automix_warp':
            # Apple-style: quick warp, settle, quick warp back
            # Create a curve that moves fast at start/end, slow in middle
            curve = 0.5 * (1 + np.sin(np.pi * (t - 0.5)))
        else:
            curve = t
        
        tempos = start_tempo + (end_tempo - start_tempo) * curve
        return tempos
    
    def apply_variable_tempo_stretch(self, audio, tempo_curve, original_tempo):
        """
        Apply time stretching with variable tempo over time.
        This is the key for smooth, non-glitchy transitions.
        
        Args:
            audio: Input audio
            tempo_curve: Array of target tempos over time
            original_tempo: Original tempo of the audio
            
        Returns:
            Stretched audio with variable tempo
        """
        # Instead of chunking, we'll use a single pass with interpolated speeds
        # Calculate average speed for this segment
        speed_curve = tempo_curve / original_tempo
        avg_speed = np.mean(speed_curve)
        
        # For smoothest results, apply average stretch
        # (Real-time variable speed would require custom WSOLA implementation)
        stretched = self.time_stretch_smooth(audio, avg_speed)
        
        return stretched
    
    def create_transition_segment(self, 
                                  track_a, 
                                  track_b, 
                                  tempo_a, 
                                  tempo_b,
                                  transition_duration=8.0,
                                  crossfade_duration=4.0,
                                  meeting_bias=0.5,
                                  warp_style=True):
        """
        Create a transition segment between two tracks.
        This returns just the transition portion for streaming.
        
        Args:
            track_a: Audio from first track (should be ~transition_duration seconds)
            track_b: Audio from second track (should be ~transition_duration seconds)
            tempo_a: Tempo of track A
            tempo_b: Tempo of track B
            transition_duration: How long the tempo shift takes
            crossfade_duration: How long the crossfade is
            meeting_bias: Where to meet (0.0 = track A tempo, 1.0 = track B tempo)
            warp_style: Use Apple AutoMix-style warpy effect
            
        Returns:
            Transition audio segment
        """
        # Calculate meeting tempo
        meeting_tempo = tempo_a * (1 - meeting_bias) + tempo_b * meeting_bias
        
        half_transition = transition_duration / 2
        
        # === TRACK A PROCESSING ===
        
        # Phase 1: Transition A from its tempo to meeting tempo
        phase1_samples = int(half_transition * self.sr)
        track_a_phase1 = track_a[:phase1_samples]
        
        speed_a_to_meeting = meeting_tempo / tempo_a
        
        if warp_style:
            # Use phase vocoder for warpy effect
            track_a_stretched = self.time_stretch_smooth(
                track_a_phase1, 
                speed_a_to_meeting, 
                use_phasevocoder=True
            )
        else:
            track_a_stretched = self.time_stretch_smooth(
                track_a_phase1, 
                speed_a_to_meeting
            )
        
        # Phase 2: Hold at meeting tempo during crossfade
        crossfade_samples = int(crossfade_duration * self.sr)
        track_a_phase2 = track_a[phase1_samples:phase1_samples + crossfade_samples]
        track_a_at_meeting = self.time_stretch_smooth(track_a_phase2, speed_a_to_meeting)
        
        # Combine A's phases
        track_a_full = np.concatenate([track_a_stretched, track_a_at_meeting])
        
        # === TRACK B PROCESSING ===
        
        # Phase 1: Start B at meeting tempo during crossfade
        track_b_phase1 = track_b[:crossfade_samples]
        speed_meeting_to_b = tempo_b / meeting_tempo
        
        track_b_at_meeting = self.time_stretch_smooth(
            track_b_phase1, 
            1.0 / speed_meeting_to_b  # Slow down to meeting tempo
        )
        
        # Phase 2: Transition B from meeting tempo back to its natural tempo
        phase2_samples = int(half_transition * self.sr)
        track_b_phase2 = track_b[crossfade_samples:crossfade_samples + phase2_samples]
        
        if warp_style:
            track_b_stretched = self.time_stretch_smooth(
                track_b_phase2, 
                speed_meeting_to_b,
                use_phasevocoder=True
            )
        else:
            track_b_stretched = self.time_stretch_smooth(
                track_b_phase2, 
                speed_meeting_to_b
            )
        
        # Combine B's phases
        track_b_full = np.concatenate([track_b_at_meeting, track_b_stretched])
        
        # === CROSSFADE ===
        
        # Find crossfade region
        fade_start_a = len(track_a_stretched)
        fade_end_a = len(track_a_full)
        
        fade_a = track_a_full[fade_start_a:fade_end_a]
        fade_b = track_b_full[:len(fade_a)]
        
        # Match lengths
        min_len = min(len(fade_a), len(fade_b))
        fade_a = fade_a[:min_len]
        fade_b = fade_b[:min_len]
        
        # Equal power crossfade
        fade_out = np.cos(np.linspace(0, np.pi/2, min_len)) ** 2
        fade_in = np.sin(np.linspace(0, np.pi/2, min_len)) ** 2
        
        crossfaded = fade_a * fade_out + fade_b * fade_in
        
        # === ASSEMBLE ===
        
        transition = np.concatenate([
            track_a_full[:fade_start_a],  # A transitioning
            crossfaded,                     # Crossfade
            track_b_full[min_len:]         # B transitioning back
        ])
        
        return transition
    
    def get_transition_for_stream(self,
                                   track_a_audio,
                                   track_b_audio,
                                   transition_point_a,
                                   transition_duration=8.0,
                                   crossfade_duration=4.0,
                                   warp_style=True):
        """
        Get a transition segment for insertion into an audio stream.
        This is what you'd call from your AI DJ transition model.
        
        Args:
            track_a_audio: Full audio of track A
            track_b_audio: Full audio of track B
            transition_point_a: Time in track A to start transition (seconds)
            transition_duration: Duration of tempo transition
            crossfade_duration: Duration of crossfade
            warp_style: Use warpy Apple AutoMix effect
            
        Returns:
            Dictionary with:
                - 'pre_transition': Track A before transition starts
                - 'transition': The transition segment
                - 'post_transition': Track B after transition ends
                - 'transition_start_time': When transition starts
                - 'transition_end_time': When transition ends
        """
        # Detect tempos
        tempo_a = self.detect_tempo(track_a_audio)
        tempo_b = self.detect_tempo(track_b_audio)
        
        print(f"Track A tempo: {tempo_a:.1f} BPM")
        print(f"Track B tempo: {tempo_b:.1f} BPM")
        
        # Extract segments
        transition_start_sample = int(transition_point_a * self.sr)
        
        # Get enough audio for the transition
        buffer_samples = int((transition_duration + crossfade_duration + 2) * self.sr)
        
        track_a_segment = track_a_audio[transition_start_sample:
                                       transition_start_sample + buffer_samples]
        track_b_segment = track_b_audio[:buffer_samples]
        
        # Create transition
        transition = self.create_transition_segment(
            track_a_segment,
            track_b_segment,
            tempo_a,
            tempo_b,
            transition_duration=transition_duration,
            crossfade_duration=crossfade_duration,
            warp_style=warp_style
        )
        
        # Calculate timing
        transition_end_sample = int((crossfade_duration + transition_duration / 2) * self.sr)
        
        return {
            'pre_transition': track_a_audio[:transition_start_sample],
            'transition': transition,
            'post_transition': track_b_audio[transition_end_sample:],
            'transition_start_time': transition_point_a,
            'transition_end_time': transition_point_a + len(transition) / self.sr,
            'tempo_a': tempo_a,
            'tempo_b': tempo_b
        }


# Example usage for your AI DJ streaming system
if __name__ == "__main__":
    # Initialize DJ engine
    dj = TempoTransitionDJ(sample_rate=44100)
    
    # Load tracks
    print("Loading tracks...")
    track_a, sr = librosa.load("C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/let-me-love-you_dj-snake-justin-bieber.wav", sr=44100, mono=True)
    track_b, _ = librosa.load("C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/lean-on_major-lazer-dj-snake-mØ.wav", sr=44100, mono=True)
    # Get transition segment
    result = dj.get_transition_for_stream(
        track_a_audio=track_a,
        track_b_audio=track_b,
        transition_point_a=164.0,
        transition_duration=8.0,
        crossfade_duration=4.0,
        warp_style=True  # Apple AutoMix-style warp
    )
    
    # For streaming, you would feed these segments to your audio output:
    # 1. Stream result['pre_transition']
    # 2. Stream result['transition']
    # 3. Stream result['post_transition']
    
    # Or save to demonstrate:
    final_mix = np.concatenate([
        result['pre_transition'],
        result['transition'],
        result['post_transition'][:int(30 * sr)]  # Just first 30s of track B
    ])
    
    # Normalize
    final_mix = final_mix / np.max(np.abs(final_mix)) * 0.95
    
    sf.write("automix_style_transition2.wav", final_mix, sr)
    print("\n✓ Transition complete!")
    print(f"Transition starts at: {result['transition_start_time']:.1f}s")
    print(f"Transition ends at: {result['transition_end_time']:.1f}s")
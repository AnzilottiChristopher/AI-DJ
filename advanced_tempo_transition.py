import librosa
import numpy as np
import soundfile as sf
from audiotsm import wsola, phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
from scipy import signal
from scipy.interpolate import interp1d


class AdvancedTempoTransitionDJ:
    """
    Advanced AI DJ with complex multi-layer transitions.
    Features: harmonic mixing, EQ crossfading, multi-stage tempo curves, 
    beat-synced effects, and dynamic energy matching.
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize advanced DJ engine.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sr = sample_rate
        
    def detect_tempo(self, audio):
        """Detect tempo of audio."""
        tempo = librosa.beat.tempo(y=audio, sr=self.sr)[0]
        return tempo
    
    def detect_beats(self, audio):
        """Detect beat positions."""
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        beat_frames = librosa.frames_to_samples(beats)
        return beat_frames
    
    def get_harmonic_content(self, audio):
        """Extract harmonic/percussive components."""
        harmonic, percussive = librosa.effects.hpss(audio)
        return harmonic, percussive
    
    def apply_eq_curve(self, audio, curve_type='low_pass', strength=0.5):
        """
        Apply EQ curves for smooth frequency transitions.
        
        Args:
            audio: Input audio
            curve_type: 'low_pass', 'high_pass', 'band_pass'
            strength: 0.0 to 1.0, how much to apply
        """
        if len(audio) == 0:
            return audio
        
        # Clamp strength to valid range
        strength = np.clip(strength, 0.0, 1.0)
            
        nyquist = self.sr / 2
        
        try:
            if curve_type == 'low_pass':
                # Ensure cutoff is in valid range (20 Hz to nyquist - 100 Hz)
                cutoff = nyquist * (1.0 - strength * 0.6)
                cutoff = np.clip(cutoff, 100, nyquist - 100)
                sos = signal.butter(4, cutoff, btype='low', fs=self.sr, output='sos')
                
            elif curve_type == 'high_pass':
                # Ensure cutoff is in valid range
                cutoff = nyquist * strength * 0.2 + 20  # Minimum 20 Hz
                cutoff = np.clip(cutoff, 20, nyquist - 100)
                sos = signal.butter(4, cutoff, btype='high', fs=self.sr, output='sos')
                
            elif curve_type == 'band_pass':
                low = np.clip(nyquist * 0.1, 20, nyquist - 200)
                high = np.clip(nyquist * 0.5, low + 100, nyquist - 100)
                sos = signal.butter(2, [low, high], btype='band', fs=self.sr, output='sos')
            
            filtered = signal.sosfilt(sos, audio)
            
            # Blend with original based on strength
            return audio * (1 - strength) + filtered * strength
            
        except Exception as e:
            # If filter fails, just return original audio
            print(f"Warning: EQ filter failed ({e}), returning original audio")
            return audio
    
    def time_stretch_smooth(self, audio, speed, use_phasevocoder=False):
        """
        Time stretch using audiotsm.
        
        Args:
            audio: Input audio array (mono)
            speed: Speed factor (1.0 = normal)
            use_phasevocoder: Use phase vocoder for warpy effect
            
        Returns:
            Stretched audio
        """
        if len(audio) == 0:
            return audio
            
        if audio.ndim == 1:
            audio_2d = audio.reshape(1, -1)
        else:
            audio_2d = audio
        
        reader = ArrayReader(audio_2d)
        writer = ArrayWriter(1)
        
        if use_phasevocoder:
            tsm = phasevocoder(channels=1, speed=speed)
        else:
            tsm = wsola(channels=1, speed=speed)
        
        tsm.run(reader, writer, flush=True)
        
        return writer.data[0]
    
    def create_multi_stage_curve(self, duration_seconds, stages):
        """
        Create a multi-stage transition curve.
        
        Args:
            duration_seconds: Total duration
            stages: List of dicts with 'duration_ratio', 'curve_type', 'intensity'
        
        Returns:
            Curve values from 0 to 1
        """
        total_points = max(10, int(duration_seconds * 100))
        curve = []
        
        current_value = 0.0
        
        for stage in stages:
            stage_points = max(1, int(total_points * stage['duration_ratio']))
            
            t = np.linspace(0, 1, stage_points)
            
            # Generate stage curve
            if stage['curve_type'] == 'ease_in':
                stage_curve = t ** 2
            elif stage['curve_type'] == 'ease_out':
                stage_curve = 1 - (1 - t) ** 2
            elif stage['curve_type'] == 'ease_in_out':
                stage_curve = t * t * (3.0 - 2.0 * t)
            elif stage['curve_type'] == 'linear':
                stage_curve = t
            elif stage['curve_type'] == 'bounce':
                stage_curve = t + 0.1 * np.sin(t * np.pi * 2)
                stage_curve = np.clip(stage_curve, 0, 1)
            elif stage['curve_type'] == 'warp':
                stage_curve = 0.5 * (1 + np.sin(np.pi * (t - 0.5)))
            else:
                stage_curve = t
            
            # Scale by intensity
            stage_range = stage['intensity']
            scaled_curve = current_value + stage_curve * stage_range
            curve.extend(scaled_curve)
            
            current_value = scaled_curve[-1]
        
        return np.array(curve)
    
    def match_lengths(self, *arrays):
        """Match the lengths of multiple arrays by trimming to shortest."""
        min_len = min(len(arr) for arr in arrays)
        return tuple(arr[:min_len] for arr in arrays)
    
    def create_layered_crossfade(self, audio_a, audio_b, duration, layers=3):
        """
        Create a multi-layered crossfade with different frequency bands and timing.
        
        Args:
            audio_a: Outgoing audio
            audio_b: Incoming audio
            duration: Crossfade duration in seconds
            layers: Number of frequency layers (2-5 recommended)
            
        Returns:
            Crossfaded audio
        """
        n_samples = int(duration * self.sr)
        
        # Trim to same length
        min_len = min(len(audio_a), len(audio_b), n_samples)
        seg_a = audio_a[:min_len]
        seg_b = audio_b[:min_len]
        
        # Separate into frequency bands
        output = np.zeros(min_len)
        
        if layers == 2:
            bands = [(0, 250), (250, 22050)]
        elif layers == 3:
            bands = [(0, 250), (250, 2000), (2000, 22050)]
        elif layers == 4:
            bands = [(0, 60), (60, 250), (250, 4000), (4000, 22050)]
        elif layers == 5:
            bands = [(0, 60), (60, 250), (250, 2000), (2000, 6000), (6000, 22050)]
        else:
            bands = [(0, 250), (250, 2000), (2000, 22050)]
        
        nyquist = self.sr / 2
        
        # Process each band with different timing
        for i, (low, high) in enumerate(bands):
            # Ensure valid frequency range
            low_clipped = np.clip(low, 20, nyquist - 200)
            high_clipped = np.clip(high, low_clipped + 100, nyquist - 100)
            
            try:
                # Filter both tracks to this band
                sos = signal.butter(4, [low_clipped, high_clipped], btype='band', fs=self.sr, output='sos')
                band_a = signal.sosfilt(sos, seg_a)
                band_b = signal.sosfilt(sos, seg_b)
            except:
                # If filter fails, use original
                band_a = seg_a
                band_b = seg_b
            
            # Create offset fade curves for each band
            offset_ratio = i / len(bands)
            
            # Bass fades slower and later
            if i == 0:
                fade_start = int(min_len * 0.4)
                fade_len = min_len - fade_start
                fade_curve = np.ones(min_len)
                if fade_len > 0:
                    fade_curve[fade_start:] = np.linspace(1, 0, fade_len) ** 1.5
                fade_curve_in = 1 - fade_curve
                
            # Highs fade faster and earlier
            elif i == len(bands) - 1:
                fade_start = int(min_len * 0.1)
                fade_len = min_len - fade_start
                fade_curve = np.ones(min_len)
                if fade_len > 0:
                    fade_curve[fade_start:] = np.linspace(1, 0, fade_len) ** 0.5
                fade_curve_in = 1 - fade_curve
                
            # Mids in between
            else:
                fade_start = int(min_len * (0.2 + offset_ratio * 0.2))
                fade_len = min_len - fade_start
                fade_curve = np.ones(min_len)
                if fade_len > 0:
                    fade_curve[fade_start:] = np.linspace(1, 0, fade_len)
                fade_curve_in = 1 - fade_curve
            
            # Apply fades and sum
            output += band_a * fade_curve + band_b * fade_curve_in
        
        return output
    
    def apply_beat_synced_effect(self, audio, beats, effect_type='filter_sweep'):
        """
        Apply beat-synced effects during transition.
        
        Args:
            audio: Input audio
            beats: Beat positions (sample indices)
            effect_type: 'filter_sweep', 'echo', 'reverse', 'stutter'
            
        Returns:
            Audio with effects
        """
        output = audio.copy()
        
        if effect_type == 'filter_sweep':
            # Sweep filter frequency on each beat
            for i, beat in enumerate(beats):
                if beat >= len(output):
                    break
                
                sweep_len = min(int(0.5 * self.sr), len(output) - beat)
                if sweep_len <= 0:
                    continue
                    
                sweep_freq = np.linspace(200, 8000, sweep_len)
                
                try:
                    for j, freq in enumerate(sweep_freq):
                        if beat + j >= len(output):
                            break
                        
                        freq_clipped = np.clip(freq, 100, self.sr/2 - 100)
                        sos = signal.butter(2, freq_clipped, btype='low', fs=self.sr, output='sos')
                        segment = output[beat:beat+j+1]
                        filtered = signal.sosfilt(sos, segment)
                        output[beat:beat+j+1] = filtered * 0.3 + segment * 0.7
                except:
                    pass  # Skip if filter fails
        
        elif effect_type == 'echo':
            # Add rhythmic echo on beats
            for beat in beats:
                if beat >= len(output):
                    break
                
                echo_delay = int(0.125 * self.sr)
                echo_strength = 0.4
                
                for repeat in range(1, 4):
                    echo_pos = beat + repeat * echo_delay
                    if echo_pos < len(output):
                        output[echo_pos] += output[beat] * (echo_strength ** repeat)
        
        elif effect_type == 'stutter':
            # Stutter effect on every 4th beat
            for i, beat in enumerate(beats):
                if i % 4 == 3 and beat < len(output):
                    stutter_len = min(int(0.0625 * self.sr), len(output) - beat)
                    if stutter_len <= 0:
                        continue
                    stutter_segment = output[beat:beat+stutter_len]
                    
                    for rep in range(1, 4):
                        insert_pos = beat + rep * stutter_len
                        if insert_pos + stutter_len < len(output):
                            output[insert_pos:insert_pos+stutter_len] = stutter_segment * 0.7
        
        return output
    
    def create_complex_transition(self,
                                  track_a,
                                  track_b,
                                  tempo_a,
                                  tempo_b,
                                  transition_duration=12.0,
                                  crossfade_duration=6.0,
                                  transition_style='complex'):
        """
        Create a complex multi-layered transition.
        
        Args:
            track_a: Audio from first track
            track_b: Audio from second track
            tempo_a: Tempo of track A
            tempo_b: Tempo of track B
            transition_duration: Duration of tempo transition
            crossfade_duration: Duration of crossfade
            transition_style: 'simple', 'complex', 'ultra_complex'
            
        Returns:
            Transition audio
        """
        meeting_tempo = (tempo_a + tempo_b) / 2
        
        # Define multi-stage tempo curves
        if transition_style == 'simple':
            stages_a = [
                {'duration_ratio': 1.0, 'curve_type': 'ease_in_out', 'intensity': 1.0}
            ]
            stages_b = stages_a
            
        elif transition_style == 'complex':
            stages_a = [
                {'duration_ratio': 0.3, 'curve_type': 'ease_in', 'intensity': 0.4},
                {'duration_ratio': 0.4, 'curve_type': 'warp', 'intensity': 0.5},
                {'duration_ratio': 0.3, 'curve_type': 'ease_out', 'intensity': 0.1}
            ]
            stages_b = [
                {'duration_ratio': 0.3, 'curve_type': 'ease_in', 'intensity': 0.1},
                {'duration_ratio': 0.4, 'curve_type': 'warp', 'intensity': 0.5},
                {'duration_ratio': 0.3, 'curve_type': 'ease_out', 'intensity': 0.4}
            ]
            
        else:  # ultra_complex
            stages_a = [
                {'duration_ratio': 0.2, 'curve_type': 'ease_in', 'intensity': 0.3},
                {'duration_ratio': 0.2, 'curve_type': 'bounce', 'intensity': 0.2},
                {'duration_ratio': 0.2, 'curve_type': 'warp', 'intensity': 0.3},
                {'duration_ratio': 0.2, 'curve_type': 'linear', 'intensity': 0.15},
                {'duration_ratio': 0.2, 'curve_type': 'ease_out', 'intensity': 0.05}
            ]
            stages_b = [
                {'duration_ratio': 0.2, 'curve_type': 'ease_in', 'intensity': 0.05},
                {'duration_ratio': 0.2, 'curve_type': 'linear', 'intensity': 0.15},
                {'duration_ratio': 0.2, 'curve_type': 'warp', 'intensity': 0.3},
                {'duration_ratio': 0.2, 'curve_type': 'bounce', 'intensity': 0.2},
                {'duration_ratio': 0.2, 'curve_type': 'ease_out', 'intensity': 0.3}
            ]
        
        # === PROCESS TRACK A ===
        
        half_transition = transition_duration / 2
        
        # Extract harmonic and percussive
        harm_a, perc_a = self.get_harmonic_content(track_a[:int((half_transition + crossfade_duration) * self.sr)])
        
        # Phase 1: Tempo transition
        phase1_len = int(half_transition * self.sr)
        
        speed_a_to_meeting = meeting_tempo / tempo_a
        
        # Process harmonic and percussive separately
        harm_a_p1 = self.time_stretch_smooth(harm_a[:phase1_len], speed_a_to_meeting, use_phasevocoder=True)
        perc_a_p1 = self.time_stretch_smooth(perc_a[:phase1_len], speed_a_to_meeting, use_phasevocoder=False)
        
        # Match lengths before combining
        harm_a_p1, perc_a_p1 = self.match_lengths(harm_a_p1, perc_a_p1)
        
        # Create mix curve
        mix_curve = self.create_multi_stage_curve(len(harm_a_p1) / self.sr, stages_a)
        
        # Resample curve to match audio length
        if len(mix_curve) != len(harm_a_p1):
            mix_curve_resampled = np.interp(
                np.linspace(0, len(mix_curve)-1, len(harm_a_p1)),
                np.arange(len(mix_curve)),
                mix_curve
            )
        else:
            mix_curve_resampled = mix_curve
        
        track_a_p1 = harm_a_p1 * (1 - mix_curve_resampled * 0.3) + perc_a_p1 * (1 + mix_curve_resampled * 0.3)
        
        # Phase 2: Hold at meeting tempo with EQ sweep
        crossfade_len = int(crossfade_duration * self.sr)
        track_a_p2_base = self.time_stretch_smooth(
            track_a[phase1_len:phase1_len + crossfade_len],
            speed_a_to_meeting
        )
        
        # Apply gradual high-cut during crossfade (simpler approach)
        track_a_p2_filtered = track_a_p2_base.copy()
        n_chunks = 20  # Fewer chunks for stability
        chunk_size = max(1, len(track_a_p2_filtered) // n_chunks)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(track_a_p2_filtered))
            if start_idx >= len(track_a_p2_filtered):
                break
                
            strength = (i / n_chunks) * 0.5  # Max 50% strength
            chunk = track_a_p2_filtered[start_idx:end_idx]
            track_a_p2_filtered[start_idx:end_idx] = self.apply_eq_curve(chunk, 'low_pass', strength)
        
        track_a_full = np.concatenate([track_a_p1, track_a_p2_filtered])
        
        # === PROCESS TRACK B ===
        
        harm_b, perc_b = self.get_harmonic_content(track_b[:int((half_transition + crossfade_duration) * self.sr)])
        
        # Phase 1: Start at meeting tempo with EQ sweep
        speed_meeting_to_b = tempo_b / meeting_tempo
        
        track_b_p1_base = self.time_stretch_smooth(
            track_b[:crossfade_len],
            1.0 / speed_meeting_to_b
        )
        
        # Apply gradual low-cut at start
        track_b_p1_filtered = track_b_p1_base.copy()
        chunk_size = max(1, len(track_b_p1_filtered) // n_chunks)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(track_b_p1_filtered))
            if start_idx >= len(track_b_p1_filtered):
                break
                
            strength = (1.0 - i / n_chunks) * 0.5  # Decreasing strength
            chunk = track_b_p1_filtered[start_idx:end_idx]
            track_b_p1_filtered[start_idx:end_idx] = self.apply_eq_curve(chunk, 'high_pass', strength)
        
        # Phase 2: Transition to natural tempo
        harm_b_p2 = self.time_stretch_smooth(
            harm_b[crossfade_len:crossfade_len + phase1_len],
            speed_meeting_to_b,
            use_phasevocoder=True
        )
        perc_b_p2 = self.time_stretch_smooth(
            perc_b[crossfade_len:crossfade_len + phase1_len],
            speed_meeting_to_b,
            use_phasevocoder=False
        )
        
        # Match lengths
        harm_b_p2, perc_b_p2 = self.match_lengths(harm_b_p2, perc_b_p2)
        
        # Mix with curve
        mix_curve_b = self.create_multi_stage_curve(len(harm_b_p2) / self.sr, stages_b)
        
        if len(mix_curve_b) != len(harm_b_p2):
            mix_curve_b_resampled = np.interp(
                np.linspace(0, len(mix_curve_b)-1, len(harm_b_p2)),
                np.arange(len(mix_curve_b)),
                mix_curve_b
            )
        else:
            mix_curve_b_resampled = mix_curve_b
        
        track_b_p2 = harm_b_p2 * (1 + mix_curve_b_resampled * 0.3) + perc_b_p2 * (1 - mix_curve_b_resampled * 0.3)
        
        track_b_full = np.concatenate([track_b_p1_filtered, track_b_p2])
        
        # === MULTI-LAYERED CROSSFADE ===
        
        # Detect beats for effects
        beats_a = self.detect_beats(track_a_full)
        beats_b = self.detect_beats(track_b_full)
        
        # Apply beat-synced effects
        if transition_style in ['complex', 'ultra_complex']:
            track_a_crossfade = track_a_full[len(track_a_p1):]
            track_b_crossfade = track_b_full[:len(track_a_crossfade)]
            
            # Match lengths
            track_a_crossfade, track_b_crossfade = self.match_lengths(track_a_crossfade, track_b_crossfade)
            
            # Add subtle effects
            track_a_crossfade = self.apply_beat_synced_effect(track_a_crossfade, beats_a, 'filter_sweep')
            track_b_crossfade = self.apply_beat_synced_effect(track_b_crossfade, beats_b, 'echo')
            
            # Multi-layer frequency crossfade
            crossfaded = self.create_layered_crossfade(
                track_a_crossfade,
                track_b_crossfade,
                crossfade_duration,
                layers=4 if transition_style == 'ultra_complex' else 3
            )
        else:
            # Simple crossfade
            fade_start = len(track_a_p1)
            fade_a = track_a_full[fade_start:]
            fade_b = track_b_full[:len(fade_a)]
            
            # Match lengths
            fade_a, fade_b = self.match_lengths(fade_a, fade_b)
            min_len = len(fade_a)
            
            fade_out = np.cos(np.linspace(0, np.pi/2, min_len)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi/2, min_len)) ** 2
            
            crossfaded = fade_a * fade_out + fade_b * fade_in
        
        # === ASSEMBLE ===
        
        transition = np.concatenate([
            track_a_full[:len(track_a_p1)],
            crossfaded,
            track_b_full[len(crossfaded):]
        ])
        
        return transition
    
    def get_transition_for_stream(self,
                                   track_a_audio,
                                   track_b_audio,
                                   transition_point_a,
                                   transition_duration=12.0,
                                   crossfade_duration=6.0,
                                   transition_style='complex'):
        """
        Get a complex transition segment for streaming.
        
        Args:
            track_a_audio: Full audio of track A
            track_b_audio: Full audio of track B
            transition_point_a: Time in track A to start transition (seconds)
            transition_duration: Duration of tempo transition
            crossfade_duration: Duration of crossfade
            transition_style: 'simple', 'complex', 'ultra_complex'
            
        Returns:
            Dictionary with transition segments
        """
        # Detect tempos
        tempo_a = self.detect_tempo(track_a_audio)
        tempo_b = self.detect_tempo(track_b_audio)
        
        print(f"Track A tempo: {tempo_a:.1f} BPM")
        print(f"Track B tempo: {tempo_b:.1f} BPM")
        print(f"Transition style: {transition_style}")
        
        # Extract segments
        transition_start_sample = int(transition_point_a * self.sr)
        buffer_samples = int((transition_duration + crossfade_duration + 4) * self.sr)
        
        track_a_segment = track_a_audio[transition_start_sample:
                                       transition_start_sample + buffer_samples]
        track_b_segment = track_b_audio[:buffer_samples]
        
        # Create complex transition
        transition = self.create_complex_transition(
            track_a_segment,
            track_b_segment,
            tempo_a,
            tempo_b,
            transition_duration=transition_duration,
            crossfade_duration=crossfade_duration,
            transition_style=transition_style
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


# Example usage
if __name__ == "__main__":
    # Initialize advanced DJ engine
    dj = AdvancedTempoTransitionDJ(sample_rate=44100)
    
    # Load tracks
    print("Loading tracks...")
    track_a, sr = librosa.load("C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/let-me-love-you_dj-snake-justin-bieber.wav", sr=44100, mono=True)
    track_b, _ = librosa.load("C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/memories_david-guetta-kid-cudi.wav", sr=44100, mono=True)

    # Get complex transition
    result = dj.get_transition_for_stream(
        track_a_audio=track_a,
        track_b_audio=track_b,
        transition_point_a=30.0,
        transition_duration=12.0,
        crossfade_duration=6.0,
        transition_style='ultra_complex'  # Try: 'simple', 'complex', 'ultra_complex'
    )
    
    # Assemble and save
    final_mix = np.concatenate([
        result['pre_transition'],
        result['transition'],
        result['post_transition'][:int(30 * sr)]
    ])
    
    # Normalize
    final_mix = final_mix / np.max(np.abs(final_mix)) * 0.95
    
    sf.write("complex_transition2.wav", final_mix, sr)
    print("\n✓ Complex transition complete!")
    print(f"Transition starts at: {result['transition_start_time']:.1f}s")
    print(f"Transition ends at: {result['transition_end_time']:.1f}s")
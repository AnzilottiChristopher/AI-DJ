"""
Audio Processing Module

Handles tempo adjustment, filtering, and other DSP operations.
Uses audiotsm for high-quality time-stretching and scipy for filtering.
"""

import numpy as np
import soundfile as sf
from typing import Optional, Tuple
import audiotsm
from scipy import signal


class AudioProcessor:
    def _slow_down_tempo_mono(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Time-stretch mono audio using audiotsm.
        """
        return self._apply_tsm(audio, speed_factor)
    def _normalize_audio_shape(self, audio: np.ndarray) -> np.ndarray:
        """
        Ensure audio is (N,2) for stereo or (N,1) for mono. Handles (N,1), (1,N), (2,N), (N,2), (N,), (2,), etc.
        """
        arr = np.asarray(audio)
        # If 1D, treat as mono and reshape to (N,1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        # If shape is (N,1) or (1,N), reshape to (N,1)
        if arr.ndim == 2 and (arr.shape[1] == 1 or arr.shape[0] == 1):
            arr = arr.squeeze()
            return arr.reshape(-1, 1)
        # If shape is (2,N), transpose to (N,2)
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        # If shape is (N,2), return as is
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        # If shape is (2,), treat as mono and reshape to (2,1)
        if arr.ndim == 1 and arr.shape[0] == 2:
            return arr.reshape(-1, 1)
        # Fallback: reshape to (N,1)
        return arr.reshape(-1, 1)
    """Process audio: tempo adjustment, filtering, normalization."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Sample rate in Hz (default 44100)
        """
        self.sample_rate = sample_rate
    
    # ==================== TEMPO ADJUSTMENT ====================
    
    def slow_down_tempo(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Slow down audio tempo using TSMD (Time-Stretch using Mellin Transform).
        """
        print(f"[DEBUG] slow_down_tempo raw input shape: {audio.shape}, dtype: {audio.dtype}")
        if audio.ndim == 1:
            print("[DEBUG] Reshaping 1D input to (N,1) for mono consistency.")
            audio = audio.reshape(-1, 1)
        if speed_factor <= 0:
            raise ValueError("speed_factor must be > 0")
        # Handle stereo by processing each channel separately
        audio = self._normalize_audio_shape(audio)
        if audio.ndim != 2:
            raise ValueError(f"[slow_down_tempo] Expected 2D array, got shape {audio.shape}")
            if audio.shape[1] == 1:
                # Mono: pass 1D array to _slow_down_tempo_mono
                mono_1d = np.ascontiguousarray(audio[:, 0]).astype(np.float64).reshape(-1)
                slowed = self._slow_down_tempo_mono(mono_1d, speed_factor)
                slowed = slowed.reshape(-1, 1)
                print(f"[DEBUG] slow_down_tempo output shape (mono): {slowed.shape}, dtype: {slowed.dtype}")
                return self._normalize_audio_shape(slowed)
        elif audio.shape[1] == 2:
            left = self._apply_tsm(audio[:, 0], speed_factor)
            right = self._apply_tsm(audio[:, 1], speed_factor)
            left = left.reshape(-1)
            right = right.reshape(-1)
            out = np.column_stack((left, right))
            print(f"[DEBUG] slow_down_tempo output shape (stereo): {out.shape}, dtype: {out.dtype}")
            return self._normalize_audio_shape(out)
        else:
            raise ValueError(f"[slow_down_tempo] Unsupported channel count: {audio.shape[1]}")
    
    def _apply_tsm(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Apply TSMD time-stretching to mono audio."""
        # Use audiotsm's TSMD algorithm (good balance of speed/quality)
        # Reshape for processing if needed
        audio = audio.reshape(-1, 1) if len(audio.shape) == 1 else audio

        # Apply time stretching
        print(f"[DEBUG] _apply_tsm input shape: {audio.shape}, dtype: {audio.dtype}, type: {type(audio)}")
        arr = np.asarray(audio)
        # Guarantee contiguous float64 arrays
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = np.ascontiguousarray(arr[:, 0]).astype(np.float64).reshape(-1)
            print(f"[DEBUG] _apply_tsm mono processed shape: {arr.shape}, dtype: {arr.dtype}, type: {type(arr)}")
        elif arr.ndim == 1:
            arr = np.ascontiguousarray(arr).astype(np.float64).reshape(-1)
            print(f"[DEBUG] _apply_tsm mono processed shape: {arr.shape}, dtype: {arr.dtype}, type: {type(arr)}")
        else:
            arr = np.ascontiguousarray(arr).astype(np.float64)
            print(f"[DEBUG] _apply_tsm stereo processed shape: {arr.shape}, dtype: {arr.dtype}, type: {type(arr)}")
        # Apply time stretching
        tsm = audiotsm.wsola(arr, speed_factor, self.sample_rate)
        # Return as 1D for mono, 2D for stereo
        if arr.ndim == 1:
            return np.asarray(tsm).reshape(-1, 1)
        return np.asarray(tsm)
    
    def speed_up_tempo(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """
        Speed up audio tempo.
        
        Args:
            audio: Audio array
            speed_factor: Speed multiplier (1.5 = 150% speed, 2.0 = double speed)
        
        Returns:
            Faster audio
        """
        return self.slow_down_tempo(audio, 1.0 / speed_factor)
    
    def adjust_bpm(self, audio: np.ndarray, current_bpm: float, target_bpm: float) -> np.ndarray:
        """
        Adjust audio tempo to match target BPM.
        """
        print(f"[DEBUG] adjust_bpm input shape: {audio.shape}, dtype: {audio.dtype}")
        audio = self._normalize_audio_shape(audio)
        if current_bpm <= 0 or target_bpm <= 0:
            raise ValueError("BPM values must be > 0")
        speed_factor = target_bpm / current_bpm
        if audio.shape[1] == 2:
            left = self._slow_down_tempo_mono(audio[:, 0], speed_factor)
            right = self._slow_down_tempo_mono(audio[:, 1], speed_factor)
            left = left.reshape(-1, 1)
            right = right.reshape(-1, 1)
            out = np.column_stack((left, right))
            print(f"[DEBUG] adjust_bpm output shape (stereo): {out.shape}, dtype: {out.dtype}")
        elif audio.shape[1] == 1:
            out = self.slow_down_tempo(audio, speed_factor)
            print(f"[DEBUG] adjust_bpm output shape (mono): {out.shape}, dtype: {out.dtype}")
        else:
            raise ValueError(f"[adjust_bpm] Unsupported channel count: {audio.shape[1]}")
        return self._normalize_audio_shape(out)
    
    # ==================== FILTERING ====================
    
    def apply_lowpass_filter(
        self, 
        audio: np.ndarray, 
        cutoff_freq: float, 
        order: int = 5
    ) -> np.ndarray:
        """
        Apply low-pass filter to remove high frequencies.
        
        Args:
            audio: Audio array
            cutoff_freq: Cutoff frequency in Hz (e.g., 5000 for 5kHz)
            order: Filter order (higher = steeper, default 5)
        
        Returns:
            Filtered audio
        
        Example:
            # Remove frequencies above 5kHz
            filtered = processor.apply_lowpass_filter(audio, 5000)
        """
        nyquist = self.sample_rate / 2
        if cutoff_freq >= nyquist:
            raise ValueError(f"Cutoff frequency must be < {nyquist} Hz")
        
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        return self._apply_filter(audio, b, a)
    
    def apply_highpass_filter(
        self, 
        audio: np.ndarray, 
        cutoff_freq: float, 
        order: int = 5
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low frequencies.
        
        Args:
            audio: Audio array
            cutoff_freq: Cutoff frequency in Hz (e.g., 100 for 100Hz)
            order: Filter order (default 5)
        
        Returns:
            Filtered audio
        
        Example:
            # Remove frequencies below 100Hz (remove bass)
            filtered = processor.apply_highpass_filter(audio, 100)
        """
        nyquist = self.sample_rate / 2
        if cutoff_freq >= nyquist:
            raise ValueError(f"Cutoff frequency must be < {nyquist} Hz")
        
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        
        return self._apply_filter(audio, b, a)
    
    def apply_bandpass_filter(
        self, 
        audio: np.ndarray, 
        low_freq: float, 
        high_freq: float, 
        order: int = 5
    ) -> np.ndarray:
        """
        Apply band-pass filter to isolate a frequency range.
        
        Args:
            audio: Audio array
            low_freq: Lower cutoff frequency in Hz
            high_freq: Upper cutoff frequency in Hz
            order: Filter order (default 5)
        
        Returns:
            Filtered audio
        
        Example:
            # Keep only 100Hz - 5kHz band (for vocals)
            filtered = processor.apply_bandpass_filter(audio, 100, 5000)
        """
        nyquist = self.sample_rate / 2
        if low_freq >= high_freq or high_freq >= nyquist:
            raise ValueError(f"Invalid frequencies: {low_freq} < {high_freq} < {nyquist}")
        
        normalized_low = low_freq / nyquist
        normalized_high = high_freq / nyquist
        b, a = signal.butter(order, [normalized_low, normalized_high], btype='band')
        
        return self._apply_filter(audio, b, a)
    
    def apply_notch_filter(
        self, 
        audio: np.ndarray, 
        freq: float, 
        Q: float = 30
    ) -> np.ndarray:
        """
        Apply notch filter to remove a specific frequency (e.g., hum).
        
        Args:
            audio: Audio array
            freq: Frequency to remove in Hz (e.g., 60 for AC hum)
            Q: Quality factor (higher = narrower notch, default 30)
        
        Returns:
            Filtered audio
        
        Example:
            # Remove 60Hz hum
            filtered = processor.apply_notch_filter(audio, 60)
        """
        nyquist = self.sample_rate / 2
        normalized_freq = freq / nyquist
        b, a = signal.iirnotch(normalized_freq, Q)
        
        return self._apply_filter(audio, b, a)
    
    def _apply_filter(self, audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Internal: apply filter coefficients to audio (handles stereo/mono robustly)."""
        arr = np.asarray(audio)
        # Handle stereo (N,2)
        if arr.ndim == 2 and arr.shape[1] == 2:
            left = signal.filtfilt(b, a, arr[:, 0])
            right = signal.filtfilt(b, a, arr[:, 1])
            return np.column_stack((left, right))
        # Handle mono (N,)
        elif arr.ndim == 1:
            return signal.filtfilt(b, a, arr)
        # Handle (N,1) or (1,N) as mono
        elif arr.ndim == 2 and (arr.shape[1] == 1 or arr.shape[0] == 1):
            arr = arr.squeeze()
            return signal.filtfilt(b, a, arr)
        # Fallback: flatten and filter as mono
        return signal.filtfilt(b, a, arr.flatten())
    
    # ==================== COMBINED EFFECTS ====================
    
    def vocal_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance vocals by boosting mid-range frequencies.
        Uses band-pass filter to emphasize vocal frequencies (100Hz - 8kHz).
        
        Example:
            enhanced = processor.vocal_enhancement(audio)
        """
        return self.apply_bandpass_filter(audio, 100, 8000)
    
    def bass_boost(self, audio: np.ndarray, boost_db: float = 6) -> np.ndarray:
        """
        Boost bass frequencies.
        
        Args:
            audio: Audio array
            boost_db: Boost amount in dB (default 6dB)
        
        Returns:
            Bass-boosted audio
        """
        # Create low-freq emphasis by combining original + boosted lowpass
        bass = self.apply_lowpass_filter(audio, 250)
        boost_factor = 10 ** (boost_db / 20.0)
        return audio + (bass * (boost_factor - 1.0) * 0.5)
    
    def treble_cut(self, audio: np.ndarray, reduction_db: float = 3) -> np.ndarray:
        """
        Reduce treble (high frequencies).
        
        Args:
            audio: Audio array
            reduction_db: Reduction amount in dB (default 3dB)
        
        Returns:
            Treble-reduced audio
        """
        # Low-pass filter removes treble
        filtered = self.apply_lowpass_filter(audio, 8000)
        reduction_factor = 10 ** (-reduction_db / 20.0)
        # Blend filtered with original
        return audio * reduction_factor + filtered * (1 - reduction_factor)
    
    # ==================== UTILITY ====================
    
    def normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target loudness.
        
        Args:
            audio: Audio array
            target_db: Target level in dB (default -3dB for headroom)
        
        Returns:
            Normalized audio
        """
        # Calculate current peak
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        
        # Calculate scaling factor
        target_linear = 10 ** (target_db / 20.0)
        scale = target_linear / peak
        
        return audio * scale
    
    def crossfade(
        self, 
        audio1: np.ndarray, 
        audio2: np.ndarray, 
        duration: float
    ) -> np.ndarray:
        """
        Crossfade between two audio tracks.
        
        Args:
            audio1: First audio track
            audio2: Second audio track  
            duration: Crossfade duration in seconds
        
        Returns:
            Crossfaded audio (audio1 -> audio2)
        
        Example:
            # 2-second crossfade
            mixed = processor.crossfade(song1, song2, 2.0)
        """
        fade_samples = int(duration * self.sample_rate)
        fade_samples = min(fade_samples, len(audio1), len(audio2))
        
        if fade_samples == 0:
            return audio1
        
        # Create fade curves
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        # Apply fades
        audio1_faded = audio1.copy()
        audio2_faded = audio2.copy()
        
        if len(audio1.shape) == 2:  # Stereo
            audio1_faded[:fade_samples] *= fade_out[:, np.newaxis]
            audio2_faded[:fade_samples] *= fade_in[:, np.newaxis]
        else:  # Mono
            audio1_faded[:fade_samples] *= fade_out
            audio2_faded[:fade_samples] *= fade_in
        
        # Mix
        result = np.zeros_like(audio1)
        result[:fade_samples] = audio1_faded[:fade_samples] + audio2_faded[:fade_samples]
        result[fade_samples:] = audio2[fade_samples:]
        
        return result


# Convenience functions for quick usage
def slow_down(audio: np.ndarray, factor: float, sr: int = 44100) -> np.ndarray:
    """Quick function to slow down audio."""
    processor = AudioProcessor(sr)
    return processor.slow_down_tempo(audio, factor)

def apply_lowpass(audio: np.ndarray, cutoff: float, sr: int = 44100) -> np.ndarray:
    """Quick function to apply low-pass filter."""
    processor = AudioProcessor(sr)
    return processor.apply_lowpass_filter(audio, cutoff)

"""
offline_edm_transition.py

Offline EDM style transition:
  • BPM estimate, meet in the middle target BPM
  • Time stretch using audiotsm phase vocoder (or optional STFT vocoder)
  • Beat aligned overlap, tries to align to downbeats if librosa is available
  • Warp style transition using HPF sweep on outgoing and LPF opening on incoming
  • Bar based transition length (8, 16, 32 bars)

Install:
  pip install numpy scipy soundfile audiotsm
Optional better beat grid:
  pip install librosa
"""



import math
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfilt_zi, stft, istft

from audiotsm import phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter


# ===================== CONFIG YOU EDIT =====================

FILE_A = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Ellie Goulding - Lights.wav"
FILE_B = "C:/Users/ryryf/OneDrive/Documents/VS Code/DJProj/AI DJ/music_data/audio/Major Lazer - Lean On.wav"
OUTPUT_FILE = "mix_edm.wav"

SAMPLE_RATE = 44100

USE_LIBROSA_IF_AVAILABLE = True
USE_STFT_VOCODER = False

FORCE_TARGET_BPM = None  # set like 128.0 if you want fixed
TRANSITION_BARS = 16      # EDM friendly: 8, 16, 32
BEATS_PER_BAR = 4

HP_START = 20.0
HP_END = 950.0
HP_ORDER = 4

LP_START = 950.0
LP_END = 18000.0
LP_ORDER = 4

BLOCK_SIZE = 2048

CURVE_POWER = 1.8        # stronger build feel for EDM
DROP_BOOST_DB = 1.5      # small perceived lift when incoming opens up, keep low to avoid clipping

# If beat alignment fails, fallback overlap duration in seconds
FALLBACK_TRANSITION_SECONDS = 12.0

# ============================================================


def ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[:, None]
    return x


def peak_normalize(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-9:
        return x
    return (x / m) * peak


def simple_bpm_estimate(x_mono: np.ndarray, sr: int) -> float:
    x = x_mono.astype(np.float32)
    x = x - float(np.mean(x))
    env = np.abs(x)

    sos = butter(2, 8.0, btype="low", fs=sr, output="sos")
    env = sosfilt(sos, env)

    env = env - float(np.mean(env))
    if float(np.max(np.abs(env))) < 1e-6:
        return 128.0

    ds = max(1, sr // 200)
    env_ds = env[::ds]
    sr_ds = sr // ds

    min_bpm = 80.0
    max_bpm = 170.0
    min_lag = int(sr_ds * 60.0 / max_bpm)
    max_lag = int(sr_ds * 60.0 / min_bpm)

    ac = np.correlate(env_ds, env_ds, mode="full")
    ac = ac[len(ac) // 2 :]

    search = ac[min_lag:max_lag]
    if search.size < 3:
        return 128.0

    lag = int(np.argmax(search) + min_lag)
    bpm = 60.0 * sr_ds / lag
    return float(np.clip(bpm, 80.0, 170.0))


def try_librosa_beats(x_mono: np.ndarray, sr: int):
    try:
        import librosa  # type: ignore
        tempo, beat_frames = librosa.beat.beat_track(y=x_mono, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        tempo = float(np.asarray(tempo).item())
        if not (80.0 <= tempo <= 170.0) or beat_times.size < 8:
            return None
        return tempo, beat_times.astype(np.float64)
    except Exception:
        return None


def estimate_bpm_and_beats(x: np.ndarray, sr: int):
    mono = np.mean(ensure_2d(x), axis=1).astype(np.float32)

    if USE_LIBROSA_IF_AVAILABLE:
        res = try_librosa_beats(mono, sr)
        if res is not None:
            tempo, beat_times = res
            return tempo, beat_times

    bpm = simple_bpm_estimate(mono, sr)
    return bpm, None


def audiotsm_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """
    audiotsm expects audio shaped (channels, samples).
    Our code uses (samples, channels), so we transpose in and out.
    """
    from audiotsm import phasevocoder
    from audiotsm.io.array import ArrayReader, ArrayWriter

    y = ensure_2d(y).astype(np.float32)   # (samples, channels)
    n_channels = y.shape[1]

    y_cs = y.T                           # (channels, samples)

    reader = ArrayReader(y_cs)           # <-- your version: only 1 arg
    writer = ArrayWriter(n_channels)

    tsm = phasevocoder(channels=n_channels, speed=rate)
    tsm.run(reader, writer)

    out_cs = np.asarray(writer.data, dtype=np.float32)  # (channels, samples_out)
    out = out_cs.T                                      # (samples_out, channels)
    return out





def stft_vocoder_stretch(y: np.ndarray, rate: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    y = ensure_2d(y).astype(np.float32)
    outs = []
    for ch in range(y.shape[1]):
        f, t, Z = stft(y[:, ch], nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
        mag = np.abs(Z)
        phase = np.angle(Z)

        n_frames = Z.shape[1]
        t_out = np.arange(0, n_frames, rate)
        t_out = t_out[t_out < n_frames - 1]
        n_out = len(t_out)

        phase_acc = phase[:, 0].copy()
        out_spec = np.zeros((Z.shape[0], n_out), dtype=np.complex64)

        omega = np.angle(Z[:, 1:] * np.conj(Z[:, :-1]))
        omega = np.concatenate([omega[:, :1], omega], axis=1)

        for i, ti in enumerate(t_out):
            i0 = int(math.floor(ti))
            frac = float(ti - i0)

            m = (1.0 - frac) * mag[:, i0] + frac * mag[:, i0 + 1]
            dphi = (1.0 - frac) * omega[:, i0] + frac * omega[:, i0 + 1]
            phase_acc = phase_acc + dphi
            out_spec[:, i] = (m * np.exp(1j * phase_acc)).astype(np.complex64)

        _, x_out = istft(out_spec, nperseg=n_fft, noverlap=n_fft - hop, input_onesided=True, boundary=None)
        outs.append(x_out.astype(np.float32))

    return np.stack(outs, axis=1)


def smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def edm_curve(n: int, power: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    t = smoothstep(t)
    t = np.power(t, power).astype(np.float32)
    return t


def apply_time_varying_filter(x: np.ndarray, sr: int, kind: str, start: float, end: float, order: int) -> np.ndarray:
    x = ensure_2d(x).astype(np.float32)
    n = x.shape[0]
    n_blocks = int(math.ceil(n / BLOCK_SIZE))

    start_c = float(np.clip(start, 5.0, sr * 0.49))
    sos0 = butter(order, start_c, btype=kind, fs=sr, output="sos")

    zis = []
    for _ in range(x.shape[1]):
        zis.append(sosfilt_zi(sos0).astype(np.float32))
    zis = np.stack(zis, axis=0)

    out = np.zeros_like(x)

    for bi in range(n_blocks):
        i0 = bi * BLOCK_SIZE
        i1 = min(n, (bi + 1) * BLOCK_SIZE)
        chunk = x[i0:i1, :]

        u = bi / max(1, n_blocks - 1)
        cutoff = float((1.0 - u) * start + u * end)
        cutoff = float(np.clip(cutoff, 5.0, sr * 0.49))

        sos = butter(order, cutoff, btype=kind, fs=sr, output="sos")

        for ch in range(x.shape[1]):
            ych, zis[ch] = sosfilt(sos, chunk[:, ch], zi=zis[ch])
            out[i0:i1, ch] = ych.astype(np.float32)

    return out


def pick_downbeat_index(beat_times: np.ndarray, beats_per_bar: int, choose_from_end: bool, bars: int) -> int:
    if beat_times is None or beat_times.size < beats_per_bar * bars + 4:
        return -1

    downbeats = np.arange(0, beat_times.size, beats_per_bar)
    if downbeats.size < bars + 2:
        return -1

    if choose_from_end:
        return int(downbeats[-(bars + 1)])
    return int(downbeats[1])


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def main():
    a_raw, sr_a = sf.read(FILE_A, always_2d=False)
    b_raw, sr_b = sf.read(FILE_B, always_2d=False)

    if sr_a != SAMPLE_RATE or sr_b != SAMPLE_RATE:
        raise ValueError("Set SAMPLE_RATE to match both files, or resample them first to the same rate.")

    a = peak_normalize(ensure_2d(a_raw.astype(np.float32)))
    b = peak_normalize(ensure_2d(b_raw.astype(np.float32)))

    bpm_a, beats_a = estimate_bpm_and_beats(a, SAMPLE_RATE)
    bpm_b, beats_b = estimate_bpm_and_beats(b, SAMPLE_RATE)

    target_bpm = float(FORCE_TARGET_BPM) if FORCE_TARGET_BPM else float((bpm_a + bpm_b) / 2.0)

    rate_a = bpm_a / target_bpm
    rate_b = bpm_b / target_bpm

    if USE_STFT_VOCODER:
        a_t = stft_vocoder_stretch(a, rate=rate_a)
        b_t = stft_vocoder_stretch(b, rate=rate_b)
    else:
        a_t = audiotsm_stretch(a, rate=rate_a)
        b_t = audiotsm_stretch(b, rate=rate_b)

    a_t = peak_normalize(a_t)
    b_t = peak_normalize(b_t)

    beats_to_seconds = (60.0 / target_bpm)
    transition_beats = TRANSITION_BARS * BEATS_PER_BAR
    transition_seconds = transition_beats * beats_to_seconds
    xfade = int(round(transition_seconds * SAMPLE_RATE))

    if beats_a is not None and beats_b is not None:
        out_start_beat_i = pick_downbeat_index(beats_a, BEATS_PER_BAR, choose_from_end=True, bars=TRANSITION_BARS)
        in_start_beat_i = pick_downbeat_index(beats_b, BEATS_PER_BAR, choose_from_end=False, bars=TRANSITION_BARS)

        if out_start_beat_i >= 0 and in_start_beat_i >= 0:
            out_start_s = float(beats_a[out_start_beat_i])
            in_start_s = float(beats_b[in_start_beat_i])

            out_start = int(round(out_start_s * SAMPLE_RATE))
            in_start = int(round(in_start_s * SAMPLE_RATE))

            out_start = max(0, min(out_start, a_t.shape[0] - xfade))
            in_start = max(0, min(in_start, b_t.shape[0] - xfade))

            a_head = a_t[:out_start, :]
            a_tail = a_t[out_start:out_start + xfade, :]

            b_head = b_t[in_start:in_start + xfade, :]
            b_rest = b_t[in_start + xfade:, :]
        else:
            xfade = int(round(FALLBACK_TRANSITION_SECONDS * SAMPLE_RATE))
            xfade = min(xfade, a_t.shape[0], b_t.shape[0])
            a_head = a_t[:-xfade, :]
            a_tail = a_t[-xfade:, :]
            b_head = b_t[:xfade, :]
            b_rest = b_t[xfade:, :]
    else:
        xfade = int(round(FALLBACK_TRANSITION_SECONDS * SAMPLE_RATE))
        xfade = min(xfade, a_t.shape[0], b_t.shape[0])
        a_head = a_t[:-xfade, :]
        a_tail = a_t[-xfade:, :]
        b_head = b_t[:xfade, :]
        b_rest = b_t[xfade:, :]

    # EDM filter warp
    a_tail_f = apply_time_varying_filter(a_tail, SAMPLE_RATE, "high", HP_START, HP_END, HP_ORDER)
    b_head_f = apply_time_varying_filter(b_head, SAMPLE_RATE, "low", LP_START, LP_END, LP_ORDER)

    fade_in = edm_curve(xfade, CURVE_POWER)[:, None]
    fade_out = 1.0 - fade_in

    # equal power feel
    fi = np.sqrt(np.clip(fade_in, 0.0, 1.0)).astype(np.float32)
    fo = np.sqrt(np.clip(fade_out, 0.0, 1.0)).astype(np.float32)

    # small lift as incoming opens, feels like a drop without smashing levels
    boost = 1.0 + (db_to_gain(DROP_BOOST_DB) - 1.0) * fade_in
    xfade_mix = a_tail_f * fo + (b_head_f * fi) * boost

    out = np.concatenate([a_head, xfade_mix, b_rest], axis=0)
    out = peak_normalize(out, 0.98)

    sf.write(OUTPUT_FILE, out, SAMPLE_RATE)

    print("BPM A", round(bpm_a, 2))
    print("BPM B", round(bpm_b, 2))
    print("Target BPM", round(target_bpm, 2))
    print("Transition bars", TRANSITION_BARS)
    print("Wrote", OUTPUT_FILE)


if __name__ == "__main__":
    main()

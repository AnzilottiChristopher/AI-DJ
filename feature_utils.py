import math
import re
from typing import Any, Dict


DEFAULT_SONG_FEATURES: Dict[str, Any] = {
    "bpm": 120.0,
    "key": "C",
    "scale": "major",
    "key_strength": 0.8,
    "loudness": 0.8,
    "danceability": 1.0,
    "spectral_centroid": 1500.0,
    "spectral_rolloff": 2000.0,
    "dissonance": 0.45,
    "onset_rate": 3.5,
}

NUMERIC_FEATURE_KEYS = (
    "bpm",
    "key_strength",
    "loudness",
    "danceability",
    "spectral_centroid",
    "spectral_rolloff",
    "dissonance",
    "onset_rate",
)


def coerce_scalar_float(value: Any) -> float | None:
    """Return a finite float from scalars or 1-item containers."""
    if value is None:
        return None

    candidate = value
    for _ in range(4):
        if hasattr(candidate, "tolist") and not isinstance(candidate, (str, bytes, dict)):
            candidate = candidate.tolist()
            continue
        if isinstance(candidate, (list, tuple)):
            if not candidate:
                return None
            candidate = candidate[0]
            continue
        break

    try:
        number = float(candidate)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None
    return number


def sanitize_song_features(features: Dict[str, Any] | None) -> Dict[str, Any]:
    """Normalize song feature payloads so downstream math always has safe values."""
    raw = features if isinstance(features, dict) else {}
    sanitized = dict(raw)

    for key in NUMERIC_FEATURE_KEYS:
        coerced = coerce_scalar_float(raw.get(key))
        sanitized[key] = coerced if coerced is not None else DEFAULT_SONG_FEATURES[key]

    key = raw.get("key")
    if isinstance(key, str) and key.strip():
        sanitized["key"] = key.strip()
    else:
        sanitized["key"] = DEFAULT_SONG_FEATURES["key"]

    scale = raw.get("scale")
    if isinstance(scale, str) and scale.strip().lower() in {"major", "minor"}:
        sanitized["scale"] = scale.strip().lower()
    else:
        sanitized["scale"] = DEFAULT_SONG_FEATURES["scale"]

    return sanitized


def features_need_backfill(features: Dict[str, Any] | None) -> bool:
    """Return True when a feature payload is missing or contains unusable values."""
    raw = features if isinstance(features, dict) else {}
    if not raw:
        return True

    for key in NUMERIC_FEATURE_KEYS:
        if coerce_scalar_float(raw.get(key)) is None:
            return True

    key = raw.get("key")
    if not isinstance(key, str) or not key.strip():
        return True

    scale = raw.get("scale")
    if not isinstance(scale, str) or scale.strip().lower() not in {"major", "minor"}:
        return True

    return False


def normalize_search_text(text: str) -> str:
    """Normalize human-entered song text for tolerant matching."""
    normalized = text.lower().replace("-", " ").replace("_", " ")
    normalized = normalized.replace("'", "").replace("\u2019", "")
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

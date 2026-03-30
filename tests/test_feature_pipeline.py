import json

import numpy as np

from feature_utils import features_need_backfill, sanitize_song_features
from music_library import MusicLibrary


def test_sanitize_song_features_handles_singleton_arrays_and_none_values():
    sanitized = sanitize_song_features(
        {
            "bpm": np.array([129.19921875]),
            "key": None,
            "scale": None,
            "key_strength": None,
            "loudness": None,
            "danceability": None,
            "spectral_centroid": None,
            "spectral_rolloff": None,
            "dissonance": None,
            "onset_rate": None,
        }
    )

    assert sanitized["bpm"] == 129.19921875
    assert sanitized["key"] == "C"
    assert sanitized["scale"] == "major"
    assert sanitized["danceability"] == 1.0
    assert not features_need_backfill(sanitized)


def test_music_library_search_handles_apostrophes_and_sanitized_defaults(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    song_name = "I'm-Good_David-Guetta.wav"
    (audio_dir / song_name).write_bytes(b"")

    metadata_path = tmp_path / "segmented_songs.json"
    metadata_path.write_text(
        json.dumps(
            {
                "songs": [
                    {
                        "song_name": song_name,
                        "features": {
                            "bpm": None,
                            "key": None,
                            "scale": None,
                            "key_strength": None,
                            "loudness": None,
                            "danceability": None,
                            "spectral_centroid": None,
                            "spectral_rolloff": None,
                            "dissonance": None,
                            "onset_rate": None,
                        },
                        "segments": [],
                    }
                ]
            }
        )
    )

    library = MusicLibrary(audio_dir, metadata_path)

    assert library.search("I'm Good", "David Guetta") is not None
    assert library.search("Im Good", "David Guetta") is not None
    assert '"bpm": 120.0' in library.get_library_for_llm()

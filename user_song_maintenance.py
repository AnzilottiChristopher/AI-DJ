import json
from pathlib import Path

from database import get_connection
from feature_utils import features_need_backfill, sanitize_song_features


def repair_incomplete_user_song_features(audio_dir: str | Path = "music_data/audio") -> int:
    """Repair persisted user-song rows whose extracted features are incomplete."""
    audio_root = Path(audio_dir)
    conn = get_connection()
    rows = conn.execute("SELECT id, song_name, song_data FROM user_songs").fetchall()
    repaired = 0

    try:
        for row in rows:
            song_data = json.loads(row["song_data"])
            if not features_need_backfill(song_data.get("features")):
                continue

            audio_path = audio_root / row["song_name"]
            if not audio_path.exists():
                print(f"[STARTUP] Cannot repair user song without audio file: {row['song_name']}")
                song_data["features"] = sanitize_song_features(song_data.get("features"))
            else:
                from user_songs.extract_features import extract_features

                print(f"[STARTUP] Repairing features for user song: {row['song_name']}")
                song_data["features"] = sanitize_song_features(extract_features(audio_path))

            conn.execute(
                "UPDATE user_songs SET song_data = ? WHERE id = ?",
                (json.dumps(song_data), row["id"]),
            )
            repaired += 1

        if repaired:
            conn.commit()
    finally:
        conn.close()

    return repaired

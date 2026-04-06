import json
import re
from pathlib import Path
from typing import Optional

from feature_utils import normalize_search_text, sanitize_song_features

class MusicLibrary:
    def __init__(self, audio_path, metadata_path):
        self.audio_path = Path(audio_path)
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata(metadata_path)
        self.index = self._build_idx()
        # Track which normalized keys belong to the base library vs user uploads
        self.base_song_keys: set = set(self.index.keys())
        # Maps normalized key -> user_id for user-uploaded songs
        self._user_song_owner: dict = {}

    def _load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return data

    def _normalize_lookup_key(self, name: str) -> str:
        """Normalize a song name for index lookups."""
        return name.lower().replace('-', ' ').replace('_', ' ')

    def _normalize_filename_key(self, name: str) -> str:
        """Normalize filenames so we can recover from minor naming differences."""
        return re.sub(r'[^a-z0-9]+', '', name.lower())

    def _normalize_search_key(self, name: str) -> str:
        """Normalize user-facing text for more forgiving title/artist lookups."""
        return normalize_search_text(name)

    def _scan_audio_files(self) -> dict[str, Path]:
        """Map normalized filenames to the files that actually exist on disk."""
        files: dict[str, Path] = {}
        if not self.audio_path.exists():
            return files

        for path in self.audio_path.iterdir():
            if path.is_file():
                files[self._normalize_filename_key(path.name)] = path
        return files

    def _resolve_audio_path(self, song_name: str, available_files: Optional[dict[str, Path]] = None) -> Optional[Path]:
        """Return the best on-disk path for a metadata entry, if one exists."""
        expected = self.audio_path / song_name
        if expected.exists():
            return expected

        available_files = available_files if available_files is not None else self._scan_audio_files()
        resolved = available_files.get(self._normalize_filename_key(song_name))
        if resolved:
            print(f"[LIBRARY] Resolved '{song_name}' to on-disk file '{resolved.name}'")
            return resolved
        return None

    def _index_entry(self, song: dict, available_files: Optional[dict[str, Path]] = None):
        """Build one index entry, skipping songs whose audio is unavailable."""
        song_name = song['song_name']
        resolved_path = self._resolve_audio_path(song_name, available_files)
        if not resolved_path:
            print(f"[LIBRARY] Skipping missing audio file: {song_name}")
            return None, None

        name = song_name.replace('.wav', '').replace('.mp3', '')
        normalized = self._normalize_lookup_key(name)
        print(f"entering the name: {normalized} as a key")
        return normalized, {
            'filename': song_name,
            'path': resolved_path,
            'features': sanitize_song_features(song.get('features')),
            'segments': song.get('segments', [])
        }

    def _build_idx(self):
        idx = {}
        available_files = self._scan_audio_files()
        for song in self.metadata["songs"]:
            normalized, entry = self._index_entry(song, available_files)
            if entry:
                idx[normalized] = entry
        return idx

    def reload(self):
        """Reload the base library from disk (preserves user songs in memory)."""
        print("[LIBRARY] Reloading from disk...")
        new_metadata = self._load_metadata(self.metadata_path)
        new_index = {}
        available_files = self._scan_audio_files()
        for song in new_metadata["songs"]:
            normalized, entry = self._index_entry(song, available_files)
            if entry:
                new_index[normalized] = entry

        # Re-merge user songs after reload
        for key, data in self.index.items():
            if key in self._user_song_owner:
                new_index[key] = data

        self.metadata = new_metadata
        self.index = new_index
        self.base_song_keys = set(k for k in self.index if k not in self._user_song_owner)

        print(f"[LIBRARY] Reloaded {len(self.index)} songs ({len(self.base_song_keys)} base)")
        return len(self.index)

    def add_song_hot(self, song_data):
        """Hot-add a base library song without a full reload."""
        normalized, entry = self._index_entry(song_data)
        if not entry:
            raise FileNotFoundError(f"Audio file not found for '{song_data['song_name']}'")

        self.index[normalized] = entry
        self.base_song_keys.add(normalized)

        if song_data not in self.metadata['songs']:
            self.metadata['songs'].append(song_data)

        print(f"[LIBRARY] Hot-added (base): {normalized} ({len(self.index)} songs total)")
        return normalized

    def add_user_song_hot(self, song_data: dict, user_id: int) -> str:
        """Hot-add a user-owned song to the in-memory library."""
        normalized, entry = self._index_entry(song_data)
        if not entry:
            raise FileNotFoundError(f"Audio file not found for '{song_data['song_name']}'")

        self.index[normalized] = entry
        self._user_song_owner[normalized] = user_id

        print(f"[LIBRARY] Hot-added (user {user_id}): {normalized} ({len(self.index)} songs total)")
        return normalized

    def get_by_filename(self, filename):
        normalized = filename.replace(".wav",'').lower().replace('-','')
        return self.index.get(normalized)

    def search(self, title, artist):
        print(f"search input - title of the song is {title}, artist is {artist}")
        if not title:
            return None

        raw_query = f"{title} {artist}".lower() if artist else title.lower()
        query = self._normalize_search_key(raw_query)

        if raw_query in self.index:
            candidate = self.index[raw_query]
            if candidate['path'].exists():
                return candidate
            print(f"[LIBRARY] Indexed path missing at search time: {candidate['path']}")

        for key, data in self.index.items():
            normalized_key = self._normalize_search_key(key)
            if (query == normalized_key or query in normalized_key or normalized_key in query) and data['path'].exists():
                return data

        return None

    def get_all_songs(self):
        return list(self.index.keys())

    def get_library_for_llm(self, user_id: Optional[int] = None) -> str:
        """
        Return the song library as LLM-friendly JSON.
        Guests (user_id=None) see base songs only.
        Authenticated users see base songs + their own uploads.
        """
        songs = []
        for key, data in self.index.items():
            owner = self._user_song_owner.get(key)
            # Include if it's a base song or belongs to this user
            if owner is None or owner == user_id:
                features = sanitize_song_features(data.get('features'))
                songs.append({
                    "filename": data['filename'],
                    "title": key,
                    "bpm": round(features.get('bpm', 120), 1),
                    "key": features.get('key', 'C'),
                    "scale": features.get('scale', 'major'),
                    "danceability": round(features.get('danceability', 1.0), 2),
                    "loudness": round(features.get('loudness', 0.8), 2),
                    "energy": round(features.get('onset_rate', 3.5), 2),
                })
        return json.dumps(songs, indent=2)

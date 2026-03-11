import json
from pathlib import Path
from typing import Optional

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

    def _build_idx(self):
        idx = {}
        for song in self.metadata["songs"]:
            name = song['song_name'].replace('.wav', '')
            normalized = name.lower().replace('-',' ').replace('_', ' ')
            print(f"entering the name: {normalized} as a key")
            idx[normalized] = {
                'filename': song['song_name'],
                'path': self.audio_path / song['song_name'],
                'features': song['features'],
                'segments': song.get('segments', [])
            }
        return idx

    def reload(self):
        """Reload the base library from disk (preserves user songs in memory)."""
        print("[LIBRARY] Reloading from disk...")
        new_metadata = self._load_metadata(self.metadata_path)
        new_index = {}
        for song in new_metadata["songs"]:
            name = song['song_name'].replace('.wav', '')
            normalized = name.lower().replace('-',' ').replace('_', ' ')
            print(f"entering the name: {normalized} as a key")
            new_index[normalized] = {
                'filename': song['song_name'],
                'path': self.audio_path / song['song_name'],
                'features': song['features'],
                'segments': song.get('segments', [])
            }

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
        name = song_data['song_name'].replace('.wav', '')
        normalized = name.lower().replace('-',' ').replace('_', ' ')

        self.index[normalized] = {
            'filename': song_data['song_name'],
            'path': self.audio_path / song_data['song_name'],
            'features': song_data.get('features', {}),
            'segments': song_data.get('segments', [])
        }
        self.base_song_keys.add(normalized)

        if song_data not in self.metadata['songs']:
            self.metadata['songs'].append(song_data)

        print(f"[LIBRARY] Hot-added (base): {normalized} ({len(self.index)} songs total)")
        return normalized

    def add_user_song_hot(self, song_data: dict, user_id: int) -> str:
        """Hot-add a user-owned song to the in-memory library."""
        name = song_data['song_name'].replace('.wav', '')
        normalized = name.lower().replace('-',' ').replace('_', ' ')

        self.index[normalized] = {
            'filename': song_data['song_name'],
            'path': self.audio_path / song_data['song_name'],
            'features': song_data.get('features', {}),
            'segments': song_data.get('segments', [])
        }
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

        query = f"{title} {artist}".lower() if artist else title.lower()
        query = query.replace('-', ' ').replace("'", "").replace("\u2019", "")

        if query in self.index:
            return self.index[query]

        for key, data in self.index.items():
            if query in key or key in query:
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
                features = data.get('features', {})
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

# so when the LLM requests a specific song, this class will search the files and load it to be played
# so when the LLM requests a specific song, this class will search the files and load it to be played
import json
from pathlib import Path


class MusicLibrary:
    def __init__(self, audio_path, metadata_path):
        self.audio_path = Path(audio_path)
        self.metadata = self._load_metadata(metadata_path)
        self.index = self._build_idx()

    def _load_metadata(self, metadata_path):
        # just load the json
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return data

    def _build_idx(self):
        idx = {}

        for song in self.metadata["songs"]:
            # remove the wav, and remove the hypen with a dash
            name = song["song_name"].replace(".wav", "")

            # words in the song name are spaced by -, and the title and artist are separated by _
            normalized = name.lower().replace("-", " ").replace("_", " ")
            print(f"entering the name: {normalized} as a key")
            idx[normalized] = {
                "filename": song["song_name"],
                "path": self.audio_path / song["song_name"],
                "features": song["features"],
                "segments": song.get("segments", []),
            }
        return idx

    def get_by_filename(self, filename):
        normalized = filename.replace(".wav", "").lower().replace("-", "")
        return self.index.get(normalized)

    def search(self, title, artist):
        print(f"search input - title of the song is {title}, artist is {artist}")
        if not title:
            return None

        query = f"{title} {artist}".lower() if artist else title.lower()
        query = query.replace("-", " ")

        # return exact match
        if query in self.index:
            return self.index[query]

        # fall back to partial match, if song name matches or arists matchs
        # this will need to be refined
        for key, data in self.index.items():
            if query in key or key in query:
                return data

        return None

    def get_all_songs(self):
        """Return all songs in the library."""
        return list(self.index.keys())

"""
Upload Handler API - Matches Original JSON Structure with Features

Users upload: "Artist - Song Title.wav"
Backend stores: "Song-Title_Artist.wav"
Appends to segmented_songs.json with structure:
  {
    "song_name": "...",
    "features": {},
    "segments": [...]
  }

FastAPI Integration:
    from upload_handler import router as upload_router
    app.include_router(upload_router)
"""

import re
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Paths
MUSIC_DATA_DIR = Path("music_data")
AUDIO_DIR = MUSIC_DATA_DIR / "audio"
SEGMENTED_SONGS_FILE = MUSIC_DATA_DIR / "segmented_songs.json"
SEGMENT_SCRIPT = Path("user_songs/segment_new_song.py")


def extract_artist_title_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract artist and title from "Artist - Song Title.wav" format."""
    name_without_ext = filename.rsplit('.', 1)[0]
    
    if ' - ' not in name_without_ext:
        return (None, None)
    
    parts = name_without_ext.split(' - ', 1)
    if len(parts) != 2:
        return (None, None)
    
    artist = parts[0].strip()
    title = parts[1].strip()
    
    return (artist, title) if artist and title else (None, None)


def normalize_filename(artist: str, title: str) -> str:
    """Convert to "Song-Title_Artist.wav" format."""
    title_clean = re.sub(r'[^\w\s-]', '', title)
    title_clean = re.sub(r'\s+', '-', title_clean.strip())
    
    artist_clean = re.sub(r'[^\w\s-]', '', artist)
    artist_clean = re.sub(r'\s+', '-', artist_clean.strip())
    
    return f"{title_clean}_{artist_clean}.wav"


def ensure_directories():
    """Create required directories."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MUSIC_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_segmented_songs() -> dict:
    """Load existing segmented songs."""
    if SEGMENTED_SONGS_FILE.exists():
        with open(SEGMENTED_SONGS_FILE, 'r') as f:
            return json.load(f)
    return {"songs": []}


def save_segmented_songs(data: dict):
    """Save segmented songs data."""
    with open(SEGMENTED_SONGS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def run_segmentation(audio_path: Path) -> Optional[dict]:
    """Run segment_new_song.py on uploaded audio."""
    try:
        result = subprocess.run(
            ["python", str(SEGMENT_SCRIPT), str(audio_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=SEGMENT_SCRIPT.parent
        )
        
        if result.returncode != 0:
            print(f"Segmentation failed: {result.stderr}")
            return None
        
        segments_file = SEGMENT_SCRIPT.parent / f"{audio_path.stem}_segments.json"
        
        if not segments_file.exists():
            return None
        
        with open(segments_file, 'r') as f:
            segments_data = json.load(f)
        
        segments_file.unlink()
        return segments_data
        
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None


def convert_segment_names(segments: list) -> list:
    """
    Convert AI segment names to original convention and add numbers to duplicates.
    
    This is ONLY for newly uploaded songs (AI-segmented).
    Existing songs keep their original names (verse, chorus, etc.)
    
    AI names: intro, buildup, drop, cooloff, outro
    Convert to: intro, build-up, beat-drop, cool-down, outro
    
    Also adds numbers to duplicates:
    ['build-up', 'beat-drop', 'build-up'] -> ['build-up', 'beat-drop', 'build-up1']
    """
    # Mapping from AI names to original names
    name_map = {
        'intro': 'intro',
        'buildup': 'build-up',
        'drop': 'beat-drop',
        'cooloff': 'cool-down',
        'outro': 'outro'
    }
    
    # Convert names
    converted = []
    for seg in segments:
        ai_name = seg['name']
        original_name = name_map.get(ai_name, ai_name)
        converted.append({
            'name': original_name,
            'start': seg['start'],
            'end': seg['end']
        })
    
    # Add numbers to duplicates
    label_counts = {}
    numbered_segments = []
    
    for seg in converted:
        label = seg['name']
        
        if label not in label_counts:
            label_counts[label] = 0
            numbered_label = label
        else:
            label_counts[label] += 1
            numbered_label = f"{label}{label_counts[label]}"
        
        numbered_segments.append({
            'name': numbered_label,
            'start': seg['start'],
            'end': seg['end']
        })
    
    return numbered_segments


@router.post("/song")
async def upload_song(file: UploadFile = File(...)):
    """
    Upload song with format: "Artist - Song Title.wav"
    
    Returns:
        {
            "success": true,
            "filename": "A-Sky-Full-Of-Stars_Coldplay.wav",
            "artist": "Coldplay",
            "title": "A Sky Full of Stars",
            "segments": [...],
            "segment_count": 9
        }
    """
    try:
        if not file.filename.endswith(('.wav', '.mp3')):
            raise HTTPException(400, "Only .wav and .mp3 files supported")
        
        artist, title = extract_artist_title_from_filename(file.filename)
        
        if not artist or not title:
            raise HTTPException(
                400,
                'Invalid filename. Use: "Artist - Song Title.wav"'
            )
        
        ensure_directories()
        
        normalized_name = normalize_filename(artist, title)
        audio_path = AUDIO_DIR / normalized_name
        
        segmented_data = load_segmented_songs()
        if any(s['song_name'] == normalized_name for s in segmented_data.get('songs', [])):
            raise HTTPException(409, f"Song exists: {title} by {artist}")
        
        with open(audio_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        segments_result = run_segmentation(audio_path)
        
        if not segments_result:
            if audio_path.exists():
                audio_path.unlink()
            raise HTTPException(500, "Segmentation failed")
        
        # Convert AI segments to original naming convention and add numbers
        converted_segments = convert_segment_names(segments_result['segments'])
        
        # Add to segmented_songs.json - MATCH ORIGINAL STRUCTURE
        new_song = {
            "song_name": normalized_name,
            "features": {},  # Empty for now - can be computed later if needed
            "segments": converted_segments
        }
        
        segmented_data['songs'].append(new_song)
        save_segmented_songs(segmented_data)
        
        return JSONResponse({
            "success": True,
            "filename": normalized_name,
            "artist": artist,
            "title": title,
            "segments": segments_result['segments'],
            "segment_count": len(segments_result['segments']),
            "message": f"'{title}' by {artist} uploaded successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@router.get("/songs")
async def list_songs():
    """List all segmented songs."""
    try:
        data = load_segmented_songs()
        songs = []
        
        for s in data.get('songs', []):
            # Extract artist and title from song_name
            song_name = s['song_name']
            name_without_ext = song_name.replace('.wav', '')
            
            if '_' in name_without_ext:
                parts = name_without_ext.split('_')
                title = '_'.join(parts[:-1]).replace('-', ' ').title()
                artist = parts[-1].replace('-', ' ').title()
            else:
                title = name_without_ext.replace('-', ' ').title()
                artist = 'Unknown'
            
            songs.append({
                "song_name": song_name,
                "artist": artist,
                "title": title,
                "segment_count": len(s.get('segments', []))
            })
        
        return JSONResponse({"songs": songs, "total": len(songs)})
    except Exception as e:
        raise HTTPException(500, f"Failed to load songs: {str(e)}")


@router.delete("/song/{filename}")
async def delete_song(filename: str):
    """Delete a song."""
    try:
        data = load_segmented_songs()
        original_count = len(data['songs'])
        
        data['songs'] = [s for s in data['songs'] if s['song_name'] != filename]
        
        if len(data['songs']) == original_count:
            raise HTTPException(404, f"Song not found: {filename}")
        
        save_segmented_songs(data)
        
        audio_path = AUDIO_DIR / filename
        if audio_path.exists():
            audio_path.unlink()
        
        return JSONResponse({"success": True, "message": f"Deleted: {filename}"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to delete: {str(e)}")
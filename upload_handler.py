import re
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import sys

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Paths - Relative to src/ directory
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
    # Remove apostrophes and other special characters that cause search issues
    title_clean = re.sub(r'[^\w\s-]', '', title)  # Removes apostrophes, quotes, etc.
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
        # Run from user_songs directory
        script_dir = Path("user_songs")
        script_name = "segment_new_song.py"
        
        # Make audio path absolute
        audio_path_abs = Path(audio_path).resolve()
        
        result = subprocess.run(
            [sys.executable, script_name, str(audio_path_abs)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(script_dir)  # Run from user_songs/ directory
        )
        
        if result.returncode != 0:
            print(f"Segmentation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Segments file created in user_songs/ directory
        segments_file = script_dir / f"{audio_path.stem}_segments.json"
        
        if not segments_file.exists():
            print(f"Segments file not found: {segments_file}")
            return None
        
        with open(segments_file, 'r') as f:
            segments_data = json.load(f)
        
        # Clean up temp file
        segments_file.unlink()
        return segments_data
        
    except Exception as e:
        print(f"Segmentation error: {e}")
        import traceback
        traceback.print_exc()
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



async def process_upload_background(audio_path: Path, normalized_name: str, artist: str, title: str):
    """
    Background task to process uploaded song without blocking.
    
    Runs segmentation, feature extraction, and library reload.
    """
    import asyncio
    
    try:
        print(f"\n[BACKGROUND] Processing: {normalized_name}")
        
        # Run segmentation in executor (CPU-intensive, runs in thread pool)
        loop = asyncio.get_event_loop()
        segments_result = await loop.run_in_executor(
            None,  # Use default executor
            run_segmentation,
            audio_path
        )
        
        if not segments_result:
            print(f"[BACKGROUND] Segmentation failed for {normalized_name}")
            if audio_path.exists():
                audio_path.unlink()
            return
        
        print(f"[BACKGROUND] Segmentation complete: {len(segments_result['segments'])} segments")
        
        # Convert AI segments to original naming convention
        converted_segments = convert_segment_names(segments_result['segments'])
        
        # Use extracted features from segmentation
        extracted_features = segments_result.get('features', {})
        
        # Add to segmented_songs.json
        new_song = {
            "song_name": normalized_name,
            "features": extracted_features,
            "segments": converted_segments
        }
        
        segmented_data = load_segmented_songs()
        segmented_data['songs'].append(new_song)
        save_segmented_songs(segmented_data)
        
        print(f"[BACKGROUND] Saved to database")
        
        # Trigger library reload (now safe during playback)
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post("http://localhost:8000/api/library/reload", timeout=5.0)
                print(f"[BACKGROUND] Library reloaded - '{title}' by {artist} is now available!")
        except Exception as e:
            print(f"[BACKGROUND] Warning: Could not trigger library reload: {e}")
        
        print(f"[BACKGROUND] ✓ COMPLETE: {normalized_name}\n")
        
    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        import traceback
        traceback.print_exc()


@router.post("/song")
async def upload_song(file: UploadFile = File(...)):
    """Upload song - returns immediately, processes in background."""
    try:
        if not file.filename.endswith(('.wav', '.mp3')):
            raise HTTPException(400, "Only .wav and .mp3 files supported")
        
        artist, title = extract_artist_title_from_filename(file.filename)
        
        if not artist or not title:
            raise HTTPException(400, 'Invalid filename. Use: "Artist - Song Title.wav"')
        
        ensure_directories()
        
        normalized_name = normalize_filename(artist, title)
        audio_path = AUDIO_DIR / normalized_name
        
        # Check if song already exists
        segmented_data = load_segmented_songs()
        if any(s['song_name'] == normalized_name for s in segmented_data.get('songs', [])):
            raise HTTPException(409, f"Song already exists: '{title}' by {artist}")
        
        if audio_path.exists():
            raise HTTPException(409, f"Audio file already exists: {normalized_name}")
        
        # Save uploaded file (fast, ~1 second)
        with open(audio_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[UPLOAD] File saved: {normalized_name}")
        print(f"[UPLOAD] Starting background processing...")
        
        # Process in background (don't wait!)
        import asyncio
        asyncio.create_task(process_upload_background(
            audio_path=audio_path,
            normalized_name=normalized_name,
            artist=artist,
            title=title
        ))
        
        # Return immediately - processing continues in background
        return JSONResponse({
            "success": True,
            "filename": normalized_name,
            "artist": artist,
            "title": title,
            "message": f"Upload started! Processing in background (~60 seconds)...",
            "processing": True
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

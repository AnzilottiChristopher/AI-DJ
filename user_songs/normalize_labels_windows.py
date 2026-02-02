import json
from pathlib import Path


def normalize_label(original_label):
    # Remove numeric suffixes (verse1 -> verse, cool-down1 -> cool-down)
    label = original_label.rstrip('0123456789').rstrip('-').strip()
    
    # Convert to lowercase for matching
    label_lower = label.lower()
    
    # INTRO
    if 'intro' in label_lower:
        return 'intro'
    
    # OUTRO
    if 'outro' in label_lower:
        return 'outro'
    
    # DROP (beat-drop, drop, etc.)
    if 'drop' in label_lower:
        return 'drop'
    
    # BUILDUP (build-up, buildup, etc.)
    if 'build' in label_lower:
        return 'buildup'
    
    # COOLOFF (cool-down, cool-off, cooldown, breakdown, etc.)
    if any(word in label_lower for word in ['cool', 'breakdown', 'break']):
        return 'cooloff'
    
    # VERSE -> buildup (verses build toward chorus/drop)
    if 'verse' in label_lower:
        return 'buildup'
    
    # CHORUS -> drop (choruses are peak energy in pop-EDM)
    if 'chorus' in label_lower:
        return 'drop'
    
    # BRIDGE -> cooloff
    if 'bridge' in label_lower:
        return 'cooloff'
    
    # Unknown - default to buildup
    print(f"  ⚠️  Unknown label: '{original_label}' -> mapping to 'buildup'")
    return 'buildup'


def load_and_merge_json_files(music_data_dir):
    music_data_path = Path(music_data_dir)

    if not music_data_path.exists():
        print(f"❌ Directory not found: {music_data_path}")
        return None

    # Find all converted JSON files
    json_files = list(music_data_path.glob("*_converted.json"))

    if not json_files:
        print(f"❌ No *_converted.json files found in {music_data_path}")
        return None

    all_songs = []

    print(f"Looking for JSON files in: {music_data_path}")
    print()

    for json_file in json_files:
        print(f"📁 {json_file.name}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        songs = data.get('songs', [])
        all_songs.extend(songs)

        print(f"   Loaded: {len(songs)} songs")

    return all_songs


def normalize_all_labels(songs):
    normalized_songs = []

    for song in songs:
        normalized_song = {
            'song_name': song['song_name'],
            'segments': []
        }

        # First pass: normalize all labels
        temp_segments = []
        for segment in song['segments']:
            normalized_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'name': normalize_label(segment['name']),
                'original_name': segment['name']  # Keep for reference
            }
            temp_segments.append(normalized_segment)
        
        # Second pass: merge consecutive segments with same label
        merged_segments = []
        for segment in temp_segments:
            if merged_segments and merged_segments[-1]['name'] == segment['name']:
                # Extend the previous segment
                merged_segments[-1]['end'] = segment['end']
                merged_segments[-1]['original_name'] += f" + {segment['original_name']}"
            else:
                merged_segments.append(segment)
        
        normalized_song['segments'] = merged_segments
        normalized_songs.append(normalized_song)

    return normalized_songs


def main():
    print("="*60)
    print("STEP 1: NORMALIZE LABELS (CORRECTED)")
    print("="*60)
    print()

    # Paths (relative to user_songs folder)
    music_data_dir = "../music_data"
    output_file = "../music_data/normalized_labels.json"

    # Load all JSON files
    all_songs = load_and_merge_json_files(music_data_dir)

    if all_songs is None:
        return

    print()
    print("="*60)
    print(f"Total songs loaded: {len(all_songs)}")
    print("="*60)
    print()

    # Normalize labels
    print("Normalizing labels...")
    print()
    
    normalized_songs = normalize_all_labels(all_songs)
    
    print()
    print(f"✓ Processed {len(normalized_songs)}/{len(all_songs)} songs")

    # Save output
    output_data = {'songs': normalized_songs}
    
    # Create directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Statistics
    print()
    print("="*60)
    print("✅ NORMALIZATION COMPLETE")
    print("="*60)
    print(f"Output: {output_file}")

    # Count segments
    total_segments = sum(len(song['segments']) for song in normalized_songs)
    print()
    print("Dataset:")
    print(f"  Total songs: {len(normalized_songs)}")
    print(f"  Total segments: {total_segments}")

    # Count label distribution
    label_counts = {}
    for song in normalized_songs:
        for segment in song['segments']:
            label = segment['name']
            label_counts[label] = label_counts.get(label, 0) + 1

    print()
    print("Segment distribution:")
    for label in ['intro', 'buildup', 'drop', 'cooloff', 'outro']:
        count = label_counts.get(label, 0)
        pct = (count / total_segments) * 100 if total_segments > 0 else 0
        print(f"  {label:10s}: {count:3d} ({pct:5.1f}%)")

    # Show sample normalization
    if normalized_songs and normalized_songs[0]['segments']:
        print()
        print("Sample normalization (first song):")
        for seg in normalized_songs[0]['segments'][:5]:
            if 'original_name' in seg:
                print(f"  {seg['original_name']:20s} -> {seg['name']}")

    print()
    print("="*60)
    print("✅ Ready for feature extraction!")
    print("="*60)
    print()
    print("Next step: Run extract_features_windows.py")


if __name__ == "__main__":
    main()
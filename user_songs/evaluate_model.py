"""
Run from user_songs/ folder: python evaluate_model.py
"""

import json
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_model(model_dir):
    """Load trained model and metadata."""
    model_dir = Path(model_dir)
    
    print("Loading model...")
    
    # Load Random Forest
    with open(model_dir / "rf_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(model_dir / "feature_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    with open(model_dir / "model_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    print("✓ Model loaded")
    return model, scaler, metadata


def load_test_data(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def evaluate_song(song_windows, model, scaler, metadata):
    feature_names = metadata['feature_names']
    
    # Extract features
    X = []
    ground_truth = []
    
    for window in song_windows:
        features = window['features']
        feature_vector = [features[name] for name in feature_names]
        X.append(feature_vector)
        ground_truth.append(window['label'])
    
    X = np.array(X)
    
    # Normalize features
    X_scaled = scaler.transform(X)
    
    # Predict
    predicted_labels = model.predict(X_scaled)
    
    # Convert to segments with timestamps
    segments = []
    current_label = None
    segment_start = None
    
    for i, (label, window) in enumerate(zip(predicted_labels, song_windows)):
        if label != current_label:
            # End previous segment
            if current_label is not None:
                segments.append({
                    'label': current_label,
                    'start': segment_start,
                    'end': window['features']['window_start']
                })
            
            # Start new segment
            current_label = label
            segment_start = window['features']['window_start']
    
    # Add final segment
    if current_label is not None:
        segments.append({
            'label': current_label,
            'start': segment_start,
            'end': song_windows[-1]['features']['window_end']
        })
    
    # Calculate accuracy
    correct = sum(1 for gt, pred in zip(ground_truth, predicted_labels) if gt == pred)
    accuracy = correct / len(ground_truth)
    
    return segments, predicted_labels, ground_truth, accuracy


def compare_segments(predicted, ground_truth_windows):
    # Compare predicted segments to ground truth
    print("\nPREDICTED SEGMENTS:")
    print("-" * 60)
    for seg in predicted:
        duration = seg['end'] - seg['start']
        print(f"{seg['start']:6.1f}s - {seg['end']:6.1f}s  ({duration:5.1f}s)  {seg['label']:10s}")
    
    print("\nGROUND TRUTH SEGMENTS:")
    print("-" * 60)
    
    # Get unique ground truth segments
    gt_segments = []
    current_label = None
    segment_start = None
    
    for window in ground_truth_windows:
        label = window['label']
        if label != current_label:
            if current_label is not None:
                gt_segments.append({
                    'label': current_label,
                    'start': segment_start,
                    'end': window['features']['window_start']
                })
            current_label = label
            segment_start = window['features']['window_start']
    
    # Add final segment
    if current_label is not None:
        gt_segments.append({
            'label': current_label,
            'start': segment_start,
            'end': ground_truth_windows[-1]['features']['window_end']
        })
    
    for seg in gt_segments:
        duration = seg['end'] - seg['start']
        print(f"{seg['start']:6.1f}s - {seg['end']:6.1f}s  ({duration:5.1f}s)  {seg['label']:10s}")


def main():
    print("="*60)
    print("PHASE 3 - STEP 2: EVALUATE MODEL")
    print("="*60)
    print()
    
    # Paths
    model_dir = "./models"
    training_data_file = "../music_data/training_features.json"
    
    # Check if model exists
    if not Path(model_dir).exists():
        print(f"Model directory not found: {model_dir}")
        print("\nRun train_rf.py first!")
        return
    
    # Load model
    model, scaler, metadata = load_model(model_dir)
    
    # Load test data
    print("\nLoading test data...")
    all_data = load_test_data(training_data_file)
    
    # Group windows by song
    songs = {}
    for window in all_data:
        song_name = window['song_name']
        if song_name not in songs:
            songs[song_name] = []
        songs[song_name].append(window)
    
    print(f"Found {len(songs)} songs")
    
    # Evaluate a few songs
    print("\n" + "="*60)
    print("EVALUATING SAMPLE SONGS")
    print("="*60)
    
    # Pick 3 songs to evaluate in detail
    sample_songs = list(songs.keys())[:3]
    
    overall_accuracies = []
    
    for song_name in sample_songs:
        print("\n" + "-"*60)
        print(f"Song: {song_name}")
        print("-"*60)
        
        song_windows = songs[song_name]
        segments, predictions, ground_truth, accuracy = evaluate_song(
            song_windows, model, scaler, metadata
        )
        
        overall_accuracies.append(accuracy)
        
        print(f"\nWindow-level accuracy: {accuracy*100:.2f}%")
        print(f"Total windows: {len(predictions)}")
        
        # Show comparison
        compare_segments(segments, song_windows)
        
        # Confusion for this song
        print("\nConfusion:")
        unique_labels = ['intro', 'buildup', 'drop', 'cooloff', 'outro']
        for true_label in unique_labels:
            pred_for_label = [p for t, p in zip(ground_truth, predictions) if t == true_label]
            if pred_for_label:
                from collections import Counter
                counts = Counter(pred_for_label)
                print(f"  {true_label:10s}: ", end="")
                for label in unique_labels:
                    count = counts.get(label, 0)
                    if count > 0:
                        print(f"{label}={count} ", end="")
                print()
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL EVALUATION")
    print("="*60)
    print(f"\nSongs evaluated: {len(sample_songs)}")
    print(f"Average accuracy: {np.mean(overall_accuracies)*100:.2f}%")
    print(f"Accuracy range: {np.min(overall_accuracies)*100:.2f}% - {np.max(overall_accuracies)*100:.2f}%")
    
    # Evaluate all songs quickly
    print("\n" + "="*60)
    print("QUICK EVALUATION ON ALL SONGS")
    print("="*60)
    
    all_accuracies = []
    for song_name, song_windows in songs.items():
        _, _, _, accuracy = evaluate_song(song_windows, model, scaler, metadata)
        all_accuracies.append((song_name, accuracy))
    
    # Sort by accuracy
    all_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 best segmented:")
    for i, (song, acc) in enumerate(all_accuracies[:10]):
        print(f"  {i+1:2d}. {acc*100:5.2f}%  {song}")
    
    print("\nBottom 5 (need improvement):")
    for i, (song, acc) in enumerate(all_accuracies[-5:]):
        print(f"  {i+1:2d}. {acc*100:5.2f}%  {song}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nOverall accuracy across all songs: {np.mean([a[1] for a in all_accuracies])*100:.2f}%")


if __name__ == "__main__":
    main()
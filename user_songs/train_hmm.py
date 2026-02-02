"""
Run from user_songs/ folder: python train_hmm.py
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


def load_training_data(data_file):
    # Load and organize training data by song.
    print("="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total windows: {len(data)}")
    
    # Group windows by song
    songs = {}
    feature_names = None
    
    for item in data:
        song_name = item['song_name']
        features = item['features']
        label = item['label']
        
        # Get feature names from first item
        if feature_names is None:
            feature_names = sorted(features.keys())
        
        # Extract features in consistent order
        feature_vector = [features[name] for name in feature_names]
        
        if song_name not in songs:
            songs[song_name] = {'features': [], 'labels': []}
        
        songs[song_name]['features'].append(feature_vector)
        songs[song_name]['labels'].append(label)
    
    print(f"Total songs: {len(songs)}")
    print(f"Features per window: {len(feature_names)}")
    
    # Convert to numpy arrays
    for song_name in songs:
        songs[song_name]['features'] = np.array(songs[song_name]['features'])
        songs[song_name]['labels'] = np.array(songs[song_name]['labels'])
    
    # Get label distribution
    all_labels = []
    for song in songs.values():
        all_labels.extend(song['labels'])
    
    unique, counts = np.unique(all_labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        pct = (count / len(all_labels)) * 100
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")
    
    return songs, feature_names


def create_label_mappings():
    # Create mappings between labels and integers.
    labels = ['intro', 'buildup', 'drop', 'cooloff', 'outro']
    label_to_int = {label: i for i, label in enumerate(labels)}
    int_to_label = {i: label for i, label in enumerate(labels)}
    return label_to_int, int_to_label


def split_songs(songs, test_size=0.2, random_seed=42):
    # Split songs into train and test sets.
    song_names = list(songs.keys())
    np.random.seed(random_seed)
    np.random.shuffle(song_names)
    
    split_idx = int(len(song_names) * (1 - test_size))
    train_songs = {name: songs[name] for name in song_names[:split_idx]}
    test_songs = {name: songs[name] for name in song_names[split_idx:]}
    
    return train_songs, test_songs


def prepare_sequences(songs, label_to_int):
    # Prepare sequences for HMM training.
    all_features = []
    all_labels_int = []
    lengths = []
    
    for song_name, song_data in songs.items():
        features = song_data['features']
        labels = song_data['labels']
        
        # Convert labels to integers
        labels_int = [label_to_int[label] for label in labels]
        
        all_features.append(features)
        all_labels_int.extend(labels_int)
        lengths.append(len(features))
    
    # Concatenate all features
    X = np.vstack(all_features)
    y = np.array(all_labels_int)
    
    return X, y, lengths


def train_hmm_model(X_train, y_train, lengths_train, label_to_int):
    print("\n" + "="*60)
    print("TRAINING HMM MODEL")
    print("="*60)
    
    n_states = 5
    
    # Define transition matrix with DJ mixing constraints
    # States: 0=intro, 1=buildup, 2=drop, 3=cooloff, 4=outro
    transition_matrix = np.array([
        # from: intro    buildup  drop    cooloff  outro
        [0.05,  0.95,    0.0,    0.0,     0.0],    # intro -> buildup
        [0.0,   0.2,     0.8,    0.0,     0.0],    # buildup -> drop
        [0.0,   0.1,     0.2,    0.6,     0.1],    # drop -> cooloff/buildup/outro
        [0.0,   0.7,     0.0,    0.1,     0.2],    # cooloff -> buildup/outro
        [0.0,   0.0,     0.0,    0.0,     1.0]     # outro -> outro (absorbing)
    ])
    
    # Set initial state probabilities (always start with intro)
    start_prob = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    # Create HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=100,
        random_state=42,
        init_params='mc',
        params='mc'
    )
    
    # Set our custom matrices
    model.startprob_ = start_prob
    model.transmat_ = transition_matrix
    
    print("Training HMM...")
    print(f"  States: {n_states}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training sequences: {len(lengths_train)}")
    print(f"  Total training windows: {len(X_train)}")
    print(f"  Avg sequence length: {np.mean(lengths_train):.1f} windows")
    
    # Fit model with sequence lengths
    model.fit(X_train, lengths=lengths_train)
    
    print("✓ Training complete")
    
    return model


def evaluate_model(model, test_songs, scaler, label_to_int, int_to_label):
    # Evaluate model on test songs.
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    all_true = []
    all_pred = []
    
    print(f"\nEvaluating {len(test_songs)} test songs...")
    
    for song_name, song_data in test_songs.items():
        features = song_data['features']
        true_labels = song_data['labels']
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        predictions = model.predict(features_scaled)
        pred_labels = [int_to_label[pred] for pred in predictions]
        
        all_true.extend(true_labels)
        all_pred.extend(pred_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_true, all_pred)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_true, all_pred, labels=['intro', 'buildup', 'drop', 'cooloff', 'outro'])
    
    labels = ['intro', 'buildup', 'drop', 'cooloff', 'outro']
    print("\n" + " "*12 + "  ".join(f"{l:8s}" for l in labels))
    for i, label in enumerate(labels):
        row = cm[i]
        print(f"{label:10s}  " + "  ".join(f"{v:8d}" for v in row))
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    report = classification_report(all_true, all_pred, labels=labels, output_dict=True, zero_division=0)
    for label in labels:
        if label in report:
            metrics = report[label]
            print(f"  {label:10s}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    return accuracy


def save_model(model, scaler, label_to_int, int_to_label, feature_names, model_dir):
    # Save trained model and metadata
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save HMM model
    model_path = model_dir / "hmm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model: {model_path}")
    
    # Save scaler
    scaler_path = model_dir / "feature_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_path}")
    
    # Save metadata
    metadata = {
        'label_to_int': label_to_int,
        'int_to_label': int_to_label,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_states': 5
    }
    metadata_path = model_dir / "model_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {metadata_path}")


def main():
    print("="*60)
    print("PHASE 3 - STEP 1: TRAIN HMM MODEL")
    print("="*60)
    print()
    
    # Paths
    training_data_file = "../music_data/training_features.json"
    model_dir = "./models"
    
    # Check if training data exists
    if not Path(training_data_file).exists():
        print(f"❌ Training data not found: {training_data_file}")
        print("\nMake sure you've run extract_features_windows.py first!")
        return
    
    # Load data organized by song
    songs, feature_names = load_training_data(training_data_file)
    
    # Create label mappings
    label_to_int, int_to_label = create_label_mappings()
    
    # Split into train and test SONGS (not windows)
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    train_songs, test_songs = split_songs(songs, test_size=0.2)
    
    # Count windows
    train_windows = sum(len(s['features']) for s in train_songs.values())
    test_windows = sum(len(s['features']) for s in test_songs.values())
    
    print(f"Training songs: {len(train_songs)} ({train_windows} windows)")
    print(f"Test songs: {len(test_songs)} ({test_windows} windows)")
    
    # Prepare sequences
    X_train, y_train, lengths_train = prepare_sequences(train_songs, label_to_int)
    X_test, y_test, lengths_test = prepare_sequences(test_songs, label_to_int)
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features normalized")
    
    # Train model
    model = train_hmm_model(X_train_scaled, y_train, lengths_train, label_to_int)
    
    # Evaluate model
    accuracy = evaluate_model(model, test_songs, scaler, label_to_int, int_to_label)
    
    # Save model
    save_model(model, scaler, label_to_int, int_to_label, feature_names, model_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {model_dir}/")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("\nYour model is ready!")
    print("\nNext steps:")
    print("  1. Run evaluate_model.py to test on specific songs")
    print("  2. Run segment_new_song.py to segment new songs")


if __name__ == "__main__":
    try:
        import hmmlearn
    except ImportError:
        print("hmmlearn not installed!")
        print("\nInstall it with:")
        print("  pip install hmmlearn scikit-learn")
        exit(1)
    
    main()
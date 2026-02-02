"""
Run from user_songs/ folder: python train_hmm.py
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_training_data(data_file):
    # Load and prepare training data.
    print("="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total windows: {len(data)}")
    
    # Extract features and labels
    feature_names = None
    X = []
    y = []
    
    for item in data:
        features = item['features']
        
        if feature_names is None:
            feature_names = sorted(features.keys())
            print(f"Features per window: {len(feature_names)}")
        
        feature_vector = [features[name] for name in feature_names]
        X.append(feature_vector)
        y.append(item['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        pct = (count / len(y)) * 100
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")
    
    return X, y, feature_names


def train_model(X_train, y_train):
    # Train Random Forest classifier.
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Random Forest works much better than HMM for this problem
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    print(f"  Trees: 200")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {len(X_train)}")
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    # Feature importance
    importances = model.feature_importances_
    return model, importances


def evaluate_model(model, X_test, y_test, feature_names):
    # Evaluate model on test set
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    labels = ['intro', 'buildup', 'drop', 'cooloff', 'outro']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    print("\n" + " "*12 + "  ".join(f"{l:8s}" for l in labels))
    for i, label in enumerate(labels):
        row = cm[i]
        print(f"{label:10s}  " + "  ".join(f"{v:8d}" for v in row))
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    for label in labels:
        if label in report:
            metrics = report[label]
            print(f"  {label:10s}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Top 10 most important features
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    print("\nTop 10 Most Important Features:")
    for i, idx in enumerate(indices):
        print(f"  {i+1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    return accuracy


def save_model(model, scaler, feature_names, model_dir):
    # Save trained model and metadata.
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "rf_model.pkl"
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
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_type': 'RandomForest',
        'labels': ['intro', 'buildup', 'drop', 'cooloff', 'outro']
    }
    metadata_path = model_dir / "model_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {metadata_path}")


def main():
    print("="*60)
    print("PHASE 3 - STEP 1: TRAIN MODEL")
    print("="*60)
    print()
    
    # Paths
    training_data_file = "../music_data/training_features.json"
    model_dir = "./models"
    
    # Check if training data exists
    if not Path(training_data_file).exists():
        print(f"Training data not found: {training_data_file}")
        print("\nMake sure you've run extract_features_windows.py first!")
        return
    
    # Load data
    X, y, feature_names = load_training_data(training_data_file)
    
    # Split into train and test
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features normalized")
    
    # Train model
    model, importances = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test_scaled, y_test, feature_names)
    
    # Save model
    save_model(model, scaler, feature_names, model_dir)
    
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
    print("\nNote: This uses Random Forest instead of HMM")
    print("      (Much better accuracy with limited training data)")


if __name__ == "__main__":
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("scikit-learn not installed!")
        print("\nInstall it with:")
        print("  pip install scikit-learn")
        exit(1)
    
    main()
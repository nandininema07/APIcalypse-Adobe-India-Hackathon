# src/heading_classifier.py (PLACEHOLDER VERSION FOR LOCAL DEVELOPMENT)

import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # A simple, fast classifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib # For saving/loading the scikit-learn model

# Add project root to sys.path to find data directories
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.insert(0, project_root)

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed', 'heading_classification')
MODEL_DIR = os.path.join(project_root, 'models') # Directory to save trained models
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure model directory exists

TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train.txt')
VAL_FILE = os.path.join(PROCESSED_DATA_DIR, 'val.txt')
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'test.txt') # For final evaluation

# Placeholder model path
PLACEHOLDER_MODEL_PATH = os.path.join(MODEL_DIR, 'heading_classifier_placeholder.pkl')

def load_data(file_path: str):
    """Loads data from a FastText-formatted text file."""
    texts = []
    labels = []
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at {file_path}.")
        return [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1) # Split only on the first space
                if len(parts) == 2:
                    label = parts[0].replace('__label__', '')
                    text = parts[1]
                    texts.append(text)
                    labels.append(label)
    return texts, labels

def train_placeholder_model(train_file: str, model_path: str, val_file: str = None):
    """
    Trains a placeholder scikit-learn Logistic Regression model.
    """
    print(f"\nINFO: Starting Placeholder Model training using {train_file}...")
    
    X_train, y_train = load_data(train_file)
    if not X_train:
        print("ERROR: No training data loaded. Cannot train model.")
        return None

    print(f"INFO: Loaded {len(X_train)} training samples.")

    # Create a simple TF-IDF + Logistic Regression pipeline
    model_pipeline = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 3), # Consider word n-grams up to 3
            min_df=5,           # Ignore terms that appear in too few documents
            max_features=10000, # Max features to keep model size small
            analyzer='word'     # Analyze words (including our feature tokens)
        ),
        LogisticRegression(max_iter=500, solver='liblinear', multi_class='auto', random_state=42)
    )

    try:
        model_pipeline.fit(X_train, y_train)
        print(f"INFO: Placeholder model training complete. Saving model to {model_path}...")
        joblib.dump(model_pipeline, model_path)
        print("INFO: Model saved.")

        # Evaluate on validation set if provided
        if val_file and os.path.exists(val_file):
            X_val, y_val = load_data(val_file)
            if X_val:
                print(f"\nINFO: Evaluating model on validation set: {val_file}")
                y_pred = model_pipeline.predict(X_val)
                print("Validation Results:")
                print(classification_report(y_val, y_pred, zero_division=0))
        
        return model_pipeline

    except Exception as e:
        print(f"ERROR: An error occurred during placeholder model training: {e}")
        return None

def evaluate_placeholder_model(model_path: str, test_file: str):
    """
    Loads a trained placeholder model and evaluates it on a test set.
    """
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please train it first.")
        return

    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found at {test_file}.")
        return

    print(f"\nINFO: Loading model from {model_path}...")
    model_pipeline = joblib.load(model_path)
    print("INFO: Model loaded. Evaluating on test set...")

    X_test, y_test = load_data(test_file)
    if not X_test:
        print("ERROR: No test data loaded. Cannot evaluate.")
        return

    y_pred = model_pipeline.predict(X_test)
    print("Test Results:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Placeholder model size check (TF-IDF vectorizer + Logistic Regression is usually small)
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"INFO: Trained placeholder model size: {model_size_mb:.2f} MB")
    if model_size_mb <= 200:
        print("INFO: Placeholder model size is within the <= 200MB constraint for Round 1A. ")
    else:
        print("WARNING: Placeholder model size exceeds 200MB. Consider reducing `max_features` in TfidfVectorizer.")


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure processed data exists before training
    if not (os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE) and os.path.exists(TEST_FILE)):
        print(f"ERROR: Data files not found in {PROCESSED_DATA_DIR}. Please run data_processor.py first to generate them.")
    else:
        # Train the placeholder model
        trained_model = train_placeholder_model(TRAIN_FILE, PLACEHOLDER_MODEL_PATH, VAL_FILE)
        
        # Evaluate the trained model on the test set
        if trained_model:
            evaluate_placeholder_model(PLACEHOLDER_MODEL_PATH, TEST_FILE)
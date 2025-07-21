# download_model.py
import torch
import torch.nn as nn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Re-define mock classes here or import them if in a common place for model definitions
# For simplicity, let's redefine minimal structure to save dummy files.
class MockLiLTBaseForSaving(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_layer = nn.Linear(10, 10) # Minimal layer to have a state_dict

class MockLanguageAdapterForSaving(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_layer = nn.Linear(5, 5)

class MockHierarchyClassifierForSaving(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 4)

def simulate_model_download_and_save():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Simulating download and saving of mock ML models...")

    # LiLT Base Model
    lilt_base = MockLiLTBaseForSaving()
    torch.save(lilt_base.state_dict(), os.path.join(model_dir, "lilt_base_model.pt"))
    logger.info("Mock LiLT Base Model saved.")

    # Language Adapters
    for lang in ['en', 'ja', 'zh', 'universal']:
        adapter = MockLanguageAdapterForSaving()
        torch.save(adapter.state_dict(), os.path.join(model_dir, f"lilt_adapter_{lang}.pt"))
        logger.info(f"Mock LiLT {lang} Adapter saved.")

    # Hierarchy Classifier
    classifier = MockHierarchyClassifierForSaving()
    torch.save(classifier.state_dict(), os.path.join(model_dir, "hierarchy_classifier.pt"))
    logger.info("Mock Hierarchy Classifier saved.")

    # Update config.yaml with these mock paths (optional, but good practice if paths change)
    # For now, we assume config.yaml has these paths correct.

    logger.info("All mock ML models simulated and saved to 'models/' directory.")

if __name__ == "__main__":
    simulate_model_download_and_save()
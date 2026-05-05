"""
Learned router — replaces zero-shot classifier from Phase 3.

Loads the trained logistic regression model and predicts which
model to route to based on the prompt text.

Falls back to zero-shot classification if the trained model
is not found (i.e. train_router.py hasn't been run yet).
"""

import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "router_model.pkl")

_model = None


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run: python scripts/train_router.py"
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict_model(prompt: str) -> str:
    """
    Returns the model name the classifier thinks should handle this prompt.
    """
    model = _load_model()
    return model.predict([prompt])[0]


def is_trained_model_available() -> bool:
    return os.path.exists(MODEL_PATH)

"""
Prediction service - runs ML model inference on feature vectors.

This service:
- Loads trained model bundle (pipeline + metadata)
- Accepts FeatureVector from feature_engine
- Returns prediction with confidence and SHAP explanation
- Publishes prediction events for downstream services
"""

import numpy as np
import joblib
import shap
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .feature_engine import FeatureVector
from events.publisher import event_publisher


# Path to model bundle
# Check Docker path first (/ml), then local path (../ml)
_DOCKER_MODEL_PATH = Path("/ml/models/model_bundle_latest.joblib")
_LOCAL_MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "models" / "model_bundle_latest.joblib"
MODEL_PATH = _DOCKER_MODEL_PATH if _DOCKER_MODEL_PATH.exists() else _LOCAL_MODEL_PATH


@dataclass
class Prediction:
    """ML model prediction result."""
    timestamp: int
    price: float
    direction: str           # "UP" or "DOWN"
    confidence: float        # 0.0 to 1.0
    shap_explanation: dict   # Top contributing features

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "shap_explanation": self.shap_explanation,
        }


class PredictionService:
    """
    Runs ML model inference on feature vectors.

    Loads the trained XGBoost model and provides predictions
    with confidence scores and SHAP explanations.
    """

    def __init__(self):
        self.model = None
        self.metadata = None
        self.explainer = None
        self.feature_order: list[str] = []
        self.latest_prediction: Optional[Prediction] = None

    def load_model(self, model_path: Path = MODEL_PATH) -> bool:
        """
        Load model bundle from disk.

        Returns True if successful, False otherwise.
        """
        if not model_path.exists():
            print(f"Model not found at {model_path}")
            return False

        try:
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.metadata = bundle["metadata"]
            self.feature_order = self.metadata["features"]

            # Create SHAP explainer from the XGBoost model
            self.explainer = shap.TreeExplainer(self.model)

            print(f"Model loaded: v{self.metadata['version']}")
            print(f"  Accuracy: {self.metadata['results']['accuracy']:.2%}")
            print(f"  Features: {self.feature_order}")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(self, features: FeatureVector) -> Optional[Prediction]:
        """
        Make prediction from feature vector.

        Args:
            features: FeatureVector from feature_engine

        Returns:
            Prediction with direction, confidence, and SHAP explanation
        """
        if self.model is None:
            print("Model not loaded")
            return None

        # Extract features in correct order (matching training)
        feature_values = np.array([[
            features.rsi,
            features.macd,
            features.macd_signal,
            features.macd_histogram,
            features.ema_ratio,
            features.volatility,
            features.volume_spike,
            features.momentum,
            features.bollinger_position,
        ]])

        # Get prediction and probability (no scaling needed for XGBoost)
        pred_class = self.model.predict(feature_values)[0]
        pred_proba = self.model.predict_proba(feature_values)[0]

        direction = "UP" if pred_class == 1 else "DOWN"
        confidence = float(pred_proba[1] if pred_class == 1 else pred_proba[0])

        # SHAP explanation (no scaling needed)
        shap_values = self.explainer.shap_values(feature_values)[0]

        # Get top 3 contributing features
        shap_pairs = list(zip(self.feature_order, shap_values))
        shap_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)[:3]

        shap_explanation = {
            feat: {
                "value": round(float(shap_val), 4),
                "direction": "pushes UP" if shap_val > 0 else "pushes DOWN"
            }
            for feat, shap_val in shap_sorted
        }

        prediction = Prediction(
            timestamp=features.timestamp,
            price=features.price,
            direction=direction,
            confidence=confidence,
            shap_explanation=shap_explanation,
        )

        self.latest_prediction = prediction
        return prediction

    async def predict_and_publish(self, features: FeatureVector) -> Optional[Prediction]:
        """
        Make prediction and publish to Redis.

        Args:
            features: FeatureVector from feature_engine

        Returns:
            Prediction result
        """
        prediction = self.predict(features)

        if prediction is None:
            return None

        # Cache in Redis
        await event_publisher.set_json(
            "prediction:latest",
            prediction.to_dict(),
            expire_seconds=120,
        )

        # Publish event for downstream services (trade executor)
        await event_publisher.publish(
            "event:prediction_ready",
            prediction.to_dict(),
        )

        # Add to prediction history stream
        await event_publisher.add_to_stream(
            "stream:predictions",
            prediction.to_dict(),
            maxlen=1000,
        )

        return prediction

    def get_model_info(self) -> dict:
        """Get model metadata for API responses."""
        if self.metadata is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "version": self.metadata["version"],
            "accuracy": self.metadata["results"]["accuracy"],
            "features": self.feature_order,
            "feature_importance": self.metadata["feature_importance"],
        }


# Global singleton
prediction_service = PredictionService()

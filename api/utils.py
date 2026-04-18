import io
from typing import Dict

from PIL import Image, UnidentifiedImageError


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load image bytes into a normalized RGB PIL image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Archivo de imagen invalido") from exc

    return image.convert("RGB")


def get_risk_level(probability: float) -> str:
    """Map probability to risk bucket used by the Streamlit UI."""
    if probability < 0.40:
        return "Bajo Riesgo (Patron Neurotipico)"
    if probability <= 0.60:
        return "Indeterminado / Zona Limítrofe"
    return "Alto Riesgo (Firma Estructural Asociada a TEA)"


def build_prediction_response(probability: float) -> Dict[str, float | str]:
    return {
        "probability": round(float(probability), 6),
        "probability_percent": round(float(probability) * 100.0, 2),
        "risk_level": get_risk_level(float(probability)),
        "thresholds": {
            "low_max": 0.40,
            "indeterminate_max": 0.60,
        },
    }

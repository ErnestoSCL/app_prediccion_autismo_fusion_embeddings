from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tv_models
from torchvision import transforms

try:
    from .utils import build_prediction_response
except ImportError:
    from utils import build_prediction_response


class EfficientNetBaseline(nn.Module):
    def __init__(self, embedding_dim: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        backbone = tv_models.efficientnet_b0(pretrained=False)
        self.features = backbone.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = nn.Linear(1280, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.embedding_layer(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.encode(x))
        out = torch.sigmoid(self.output_layer(x))
        return out


class MultimodalFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalPredictor:
    def __init__(self, models_dir: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = models_dir

        self.model_sagittal = EfficientNetBaseline().to(self.device)
        self.model_coronal = EfficientNetBaseline().to(self.device)
        self.model_axial = EfficientNetBaseline().to(self.device)
        self.model_fusion = MultimodalFusion().to(self.device)

        self._load_weights()

        self.model_sagittal.eval()
        self.model_coronal.eval()
        self.model_axial.eval()
        self.model_fusion.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_weights(self) -> None:
        self.model_sagittal.load_state_dict(
            torch.load(
                self.models_dir / "baseline" / "model_sagittal_best.pt",
                map_location=self.device,
            )
        )
        self.model_coronal.load_state_dict(
            torch.load(
                self.models_dir / "baseline" / "model_coronal_best.pt",
                map_location=self.device,
            )
        )
        self.model_axial.load_state_dict(
            torch.load(
                self.models_dir / "baseline" / "model_axial_best.pt",
                map_location=self.device,
            )
        )
        self.model_fusion.load_state_dict(
            torch.load(
                self.models_dir / "multimodal" / "multimodal_best.pt",
                map_location=self.device,
            )
        )

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, img_sagittal: Image.Image, img_coronal: Image.Image, img_axial: Image.Image) -> float:
        t_sag = self._process_image(img_sagittal)
        t_cor = self._process_image(img_coronal)
        t_axi = self._process_image(img_axial)

        emb_sag = self.model_sagittal.encode(t_sag)
        emb_cor = self.model_coronal.encode(t_cor)
        emb_axi = self.model_axial.encode(t_axi)

        multimodal_vector = torch.cat([emb_sag, emb_cor, emb_axi], dim=1)
        logits = self.model_fusion(multimodal_vector)
        probability = torch.sigmoid(logits).item()
        return float(probability)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PREDICTOR = MultimodalPredictor(MODELS_DIR)


def predict_multimodal(img_sagittal: Image.Image, img_coronal: Image.Image, img_axial: Image.Image) -> dict:
    probability = PREDICTOR.predict(img_sagittal, img_coronal, img_axial)
    return build_prediction_response(probability)

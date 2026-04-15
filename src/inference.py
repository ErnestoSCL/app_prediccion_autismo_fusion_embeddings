import torch
import torch.nn as nn
from torchvision import models as tv_models
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ==========================================
# 1. ARQUITECTURAS
# ==========================================
class EfficientNetBaseline(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.5):
        super(EfficientNetBaseline, self).__init__()
        # Usamos architecture default
        backbone = tv_models.efficientnet_b0(pretrained=False) # Se cargan pesos guardados despues
        self.features        = backbone.features
        self.gap             = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = nn.Linear(1280, embedding_dim)
        self.dropout         = nn.Dropout(dropout)     
        self.output_layer    = nn.Linear(embedding_dim, 1)

    def encode(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.embedding_layer(x))
        return x

    def forward(self, x):
        x   = self.dropout(self.encode(x))
        out = torch.sigmoid(self.output_layer(x))
        return out


class MultimodalFusion(nn.Module):
    """ Variante C (2 Capas Ocultas, guardada en mejores pesos) """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. INFERENCE MANAGER
# ==========================================
class MultimodalPredictor:
    def __init__(self, models_dir: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = Path(models_dir)
        
        # Iniciar arquitectura base
        self.model_sagittal = EfficientNetBaseline().to(self.device)
        self.model_coronal  = EfficientNetBaseline().to(self.device)
        self.model_axial    = EfficientNetBaseline().to(self.device)
        self.model_fusion   = MultimodalFusion().to(self.device)
        
        # Cargar pesos validos
        self.load_weights()
        
        # Modo evaluación
        self.model_sagittal.eval()
        self.model_coronal.eval()
        self.model_axial.eval()
        self.model_fusion.eval()
        
        # Trasnformaciones Estandar de Evaluacion
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_weights(self):
        """ Carga los pesos en las 4 redes, ignorando los map_location exactos si cambia de CPU/GPU """
        self.model_sagittal.load_state_dict(
            torch.load(self.models_dir / 'baseline' / 'model_sagittal_best.pt', map_location=self.device)
        )
        self.model_coronal.load_state_dict(
            torch.load(self.models_dir / 'baseline' / 'model_coronal_best.pt', map_location=self.device)
        )
        self.model_axial.load_state_dict(
            torch.load(self.models_dir / 'baseline' / 'model_axial_best.pt', map_location=self.device)
        )
        self.model_fusion.load_state_dict(
            torch.load(self.models_dir / 'multimodal' / 'multimodal_best.pt', map_location=self.device)
        )

    def process_image(self, image: Image.Image):
        """ Asegura 3 canales (RGB) y aplica transformaciones """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0) # (1, 3, 224, 224)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, img_sagittal: Image.Image, img_coronal: Image.Image, img_axial: Image.Image):
        """
        Calcula embeddings, concatena, y lanza prediccion multimodal.
        """
        # 1. Trannsform
        t_sag = self.process_image(img_sagittal)
        t_cor = self.process_image(img_coronal)
        t_axi = self.process_image(img_axial)
        
        # 2. Embeddings (256-d cada uno)
        emb_sag = self.model_sagittal.encode(t_sag)
        emb_cor = self.model_coronal.encode(t_cor)
        emb_axi = self.model_axial.encode(t_axi)
        
        # 3. Concatenacion Fusión temprana (768-d)
        multimodal_vector = torch.cat([emb_sag, emb_cor, emb_axi], dim=1) # Shape: (1, 768)
        
        # 4. Inferencia
        logits = self.model_fusion(multimodal_vector)
        prob_autism = torch.sigmoid(logits).item() # Valor flotante [0.0, 1.0]
        
        return prob_autism

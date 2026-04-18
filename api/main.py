from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

try:
    from .model import predict_multimodal
    from .utils import load_image_from_bytes
    from .database import engine, get_db
    from .models_db import Base, Prediction
except ImportError:
    from model import predict_multimodal
    from utils import load_image_from_bytes
    from database import engine, get_db
    from models_db import Base, Prediction

# Crea automáticamente la tabla si no existe en la Base de Datos
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Autism Risk Multimodal API",
    description="API de prediccion multimodal con vistas sagital, coronal y axial.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "project": "app_prediccion_autismo_fusion_embeddings",
        "architecture": {
            "backbone": "EfficientNetB0 x3",
            "fusion": "MLP 768->128->32->1",
            "views": ["sagittal", "coronal", "axial"],
        },
        "output": {
            "type": "binary_probability",
            "target": "autism_risk",
        },
        "thresholds": {
            "low_max": 0.40,
            "indeterminate_max": 0.60,
        },
    }


@app.post("/predict")
async def predict(
    sagittal_file: UploadFile = File(...),
    coronal_file: UploadFile = File(...),
    axial_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    files = {
        "sagittal": sagittal_file,
        "coronal": coronal_file,
        "axial": axial_file,
    }

    for view_name, image_file in files.items():
        if not image_file.content_type or not image_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"{view_name}_file debe ser una imagen valida",
            )

    try:
        sag_bytes = await sagittal_file.read()
        cor_bytes = await coronal_file.read()
        axi_bytes = await axial_file.read()

        img_sag = load_image_from_bytes(sag_bytes)
        img_cor = load_image_from_bytes(cor_bytes)
        img_axi = load_image_from_bytes(axi_bytes)

        # 1. Hacer la predicción
        result = predict_multimodal(img_sag, img_cor, img_axi)
        
        # 2. Guardar en PostgreSQL
        riesgo = str(result.get("risk_category", "Unknown"))
        prob = float(result.get("probability", 0.0))
        img_names = f"{sagittal_file.filename} | {coronal_file.filename} | {axial_file.filename}"
        
        db_prediction = Prediction(
            project="app_prediccion_autismo_multimodal",
            image_name=img_names,
            predicted_label=riesgo,
            confidence=prob,
            probabilities=result,  # Guardamos todo el JSON del resultado
            is_correct=None
        )
        
        db.add(db_prediction)
        db.commit()

        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {exc}") from exc

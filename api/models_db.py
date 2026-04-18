from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from api.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, index=True)
    project         = Column(String(50))   # cuál de los 3 proyectos
    image_name      = Column(String(255))
    predicted_label = Column(String(100))
    confidence      = Column(Float)
    probabilities   = Column(JSON)         # todas las clases con sus scores
    is_correct      = Column(Boolean, nullable=True)  # feedback del usuario
    created_at      = Column(DateTime, server_default=func.now())

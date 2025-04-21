from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import models
from database import SessionLocal, engine
from pydantic import BaseModel
from datetime import datetime
import numpy as np
from pathlib import Path
import shutil
import os
from ai_model import get_model

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Eye Examination AI System",
    description="API for analyzing eye images and detecting various conditions using AI",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    """
    Root endpoint that provides basic API information.
    """
    return {
        "message": "Welcome to the Eye Examination AI System API",
        "version": "1.0.0",
        "endpoints": {
            "POST /patients/": "Create a new patient",
            "POST /examinations/{patient_id}": "Create a new examination with AI analysis",
            "GET /patients/{patient_id}/examinations": "Get all examinations for a patient"
        }
    }

# Pydantic models for request/response
class PatientBase(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: datetime
    address: str
    email: str

class PatientCreate(PatientBase):
    pass

class Patient(PatientBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/patients/", response_model=Patient)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    db_patient = models.Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.post("/examinations/{patient_id}")
async def create_examination(
    patient_id: int,
    od_image: UploadFile = File(...),
    os_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Verify patient exists
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Create examination record
    examination = models.Examination(
        patient_id=patient_id,
        date=datetime.utcnow()
    )

    # Save images
    save_dir = Path("uploads") / str(patient_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process right eye (OD)
    od_path = save_dir / f"od_{datetime.utcnow().timestamp()}.jpg"
    with od_path.open("wb") as buffer:
        shutil.copyfileobj(od_image.file, buffer)
    
    # Process left eye (OS)
    os_path = save_dir / f"os_{datetime.utcnow().timestamp()}.jpg"
    with os_path.open("wb") as buffer:
        shutil.copyfileobj(os_image.file, buffer)

    # Get AI predictions
    model = get_model()
    predictions = model.predict(str(od_path), str(os_path))

    # Update examination with AI predictions
    examination.has_glaucoma = predictions['has_glaucoma']
    examination.has_cataract = predictions['has_cataract']
    examination.has_scarring = predictions['has_scarring']
    examination.has_cardiovascular_disease = predictions['has_cardiovascular_disease']
    examination.has_diabetes = predictions['has_diabetes']
    examination.is_healthy = predictions['is_healthy']
    examination.ai_confidence_score = max(predictions['confidence_scores'].values())

    # Save paths for future reference
    examination.path = str(save_dir)
    examination.name = f"Examination_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    db.add(examination)
    db.commit()
    db.refresh(examination)

    return {
        "message": "Examination created successfully",
        "examination_id": examination.id,
        "results": {
            "glaucoma": examination.has_glaucoma,
            "cataract": examination.has_cataract,
            "scarring": examination.has_scarring,
            "cardiovascular_disease": examination.has_cardiovascular_disease,
            "diabetes": examination.has_diabetes,
            "is_healthy": examination.is_healthy,
            "confidence_scores": predictions['confidence_scores']
        }
    }

@app.get("/patients/{patient_id}/examinations")
def get_patient_examinations(patient_id: int, db: Session = Depends(get_db)):
    examinations = db.query(models.Examination).filter(
        models.Examination.patient_id == patient_id
    ).all()
    return examinations 
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
from ai_model import EyeDiseaseModel
from tempfile import NamedTemporaryFile
import logging
import traceback
import json

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize the model
model = EyeDiseaseModel()

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
            "GET /patients/{patient_id}/examinations": "Get all examinations for a patient",
            "POST /predict/": "Analyze a single eye image",
            "GET /health": "Check API health status"
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
    try:
        # Verify patient exists
        patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Create examination record
        examination = models.Examination(
            patient_id=patient_id,
            date=datetime.utcnow()
        )

        # Create uploads directory if it doesn't exist
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

        # Get AI predictions for both eyes
        od_predictions = model.predict(str(od_path))
        os_predictions = model.predict(str(os_path))

        # Combine predictions from both eyes
        combined_predictions = {
            'has_glaucoma': od_predictions['has_glaucoma'] or os_predictions['has_glaucoma'],
            'has_cataract': od_predictions['has_cataract'] or os_predictions['has_cataract'],
            'has_scarring': od_predictions['has_scarring'] or os_predictions['has_scarring'],
            'has_cardiovascular_disease': od_predictions['has_cardiovascular_disease'] or os_predictions['has_cardiovascular_disease'],
            'has_diabetes': od_predictions['has_diabetes'] or os_predictions['has_diabetes'],
            'is_healthy': od_predictions['is_healthy'] and os_predictions['is_healthy'],
            'confidence_scores': {
                'glaucoma': max(od_predictions['confidence_scores']['glaucoma'], os_predictions['confidence_scores']['glaucoma']),
                'cataract': max(od_predictions['confidence_scores']['cataract'], os_predictions['confidence_scores']['cataract']),
                'scarring': max(od_predictions['confidence_scores']['scarring'], os_predictions['confidence_scores']['scarring']),
                'cardiovascular': max(od_predictions['confidence_scores']['cardiovascular'], os_predictions['confidence_scores']['cardiovascular']),
                'diabetes': max(od_predictions['confidence_scores']['diabetes'], os_predictions['confidence_scores']['diabetes']),
                'healthy': min(od_predictions['confidence_scores']['healthy'], os_predictions['confidence_scores']['healthy'])
            }
        }

        # Update examination with AI predictions
        examination.has_glaucoma = combined_predictions['has_glaucoma']
        examination.has_cataract = combined_predictions['has_cataract']
        examination.has_scarring = combined_predictions['has_scarring']
        examination.has_cardiovascular_disease = combined_predictions['has_cardiovascular_disease']
        examination.has_diabetes = combined_predictions['has_diabetes']
        examination.is_healthy = combined_predictions['is_healthy']
        examination.ai_confidence_score = max(combined_predictions['confidence_scores'].values())

        # Save paths for future reference
        examination.path = str(save_dir)
        examination.name = f"Examination_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        db.add(examination)
        db.commit()
        db.refresh(examination)

        return {
            "message": "Examination created successfully",
            "examination_id": examination.id,
            "results": combined_predictions
        }

    except Exception as e:
        logger.error(f"Error processing examination: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}/examinations")
def get_patient_examinations(patient_id: int, db: Session = Depends(get_db)):
    examinations = db.query(models.Examination).filter(
        models.Examination.patient_id == patient_id
    ).all()
    return examinations

@app.post("/predict/")
async def predict_disease(
    file: UploadFile = File(...),
    patient_id: int = None,
    db: Session = Depends(get_db)
):
    """
    Endpoint to predict eye disease from an uploaded image and save the record
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
            
        logger.info(f"Saved file temporarily at: {temp_file_path}")

        # Make prediction
        try:
            prediction = model.predict(temp_file_path)
            logger.info(f"Prediction successful: {prediction}")
            
            # Convert NumPy boolean values to Python native booleans
            prediction = {
                'has_glaucoma': bool(prediction['has_glaucoma']),
                'has_cataract': bool(prediction['has_cataract']),
                'has_scarring': bool(prediction['has_scarring']),
                'has_cardiovascular_disease': bool(prediction['has_cardiovascular_disease']),
                'has_diabetes': bool(prediction['has_diabetes']),
                'is_healthy': bool(prediction['is_healthy']),
                'confidence_scores': {
                    'glaucoma': float(prediction['confidence_scores']['glaucoma']),
                    'cataract': float(prediction['confidence_scores']['cataract']),
                    'scarring': float(prediction['confidence_scores']['scarring']),
                    'cardiovascular': float(prediction['confidence_scores']['cardiovascular']),
                    'diabetes': float(prediction['confidence_scores']['diabetes']),
                    'healthy': float(prediction['confidence_scores']['healthy'])
                }
            }
            
            # Create image record
            image_record = models.ImageRecord(
                DATE=datetime.utcnow(),
                DEMOGRAPHIC=False,  # Default to False
                ERRORCODE=None,  # No error
                PATUNIQUE=patient_id if patient_id else 0,  # Use provided patient ID or 0
                NAME=file.filename,
                PATH=temp_file_path,
                TYPE="fundus",  # Assuming all uploads are fundus images
                VENDORGRAPHUNIQUE=1  # Default API identifier
            )
            db.add(image_record)
            db.flush()  # Get the ID without committing
            
            # Create AI prediction record
            ai_prediction = models.AIPrediction(
                image_record_id=image_record.EXAMUNIQUE,
                predicted_class="Multiple",  # Since we now predict multiple conditions
                confidence=max(prediction['confidence_scores'].values()),
                clinical_features=json.dumps(prediction)
            )
            db.add(ai_prediction)
            db.commit()
            
            # Format response to match frontend expectations
            response = {
                'status': 'success',
                'results': {
                    'glaucoma': prediction['has_glaucoma'],
                    'cataract': prediction['has_cataract'],
                    'scarring': prediction['has_scarring'],
                    'cardiovascular_disease': prediction['has_cardiovascular_disease'],
                    'diabetes': prediction['has_diabetes'],
                    'is_healthy': prediction['is_healthy'],
                    'confidence_scores': prediction['confidence_scores']
                }
            }
            
            return response
            
        except Exception as model_error:
            logger.error(f"Model prediction error: {str(model_error)}")
            logger.error(traceback.format_exc())
            
            # Create error record
            error_record = models.ImageRecord(
                DATE=datetime.utcnow(),
                DEMOGRAPHIC=False,
                ERRORCODE='E',  # E for error
                PATUNIQUE=patient_id if patient_id else 0,
                NAME=file.filename,
                PATH=temp_file_path,
                TYPE="fundus",
                VENDORGRAPHUNIQUE=1
            )
            db.add(error_record)
            db.commit()
            
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(model_error)}")

        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"} 
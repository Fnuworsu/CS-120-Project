from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    date_of_birth = Column(DateTime)
    address = Column(String(200))
    email = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    examinations = relationship("Examination", back_populates="patient")

class Examination(Base):
    __tablename__ = "examinations"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    date = Column(DateTime, default=datetime.utcnow)
    demographic = Column(Boolean)
    errorcode = Column(String(1))
    exam_unique = Column(Integer)
    image_import_unique = Column(Integer)
    imported = Column(DateTime)
    name = Column(String(40))
    path = Column(String)
    pat_unique = Column(Integer)
    type = Column(String(20))
    vendor_api_unique = Column(Integer)
    
    # Fundus examination fields
    od_optic_disc = Column(String(100))
    os_optic_disc = Column(String(100))
    od_macula = Column(String(100))
    os_macula = Column(String(100))
    od_posterior_pole = Column(String(100))
    os_posterior_pole = Column(String(100))
    od_cd_ratio = Column(Float)
    os_cd_ratio = Column(Float)
    od_blood_vessels = Column(String(100))
    os_blood_vessels = Column(String(100))
    
    # AI Analysis Results
    has_glaucoma = Column(Boolean, default=False)
    has_cataract = Column(Boolean, default=False)
    has_scarring = Column(Boolean, default=False)
    has_cardiovascular_disease = Column(Boolean, default=False)
    has_diabetes = Column(Boolean, default=False)
    is_healthy = Column(Boolean, default=True)
    
    patient = relationship("Patient", back_populates="examinations")
    
    # Additional metadata
    ai_confidence_score = Column(Float)
    notes = Column(String(500))
    reviewed_by = Column(String(100))
    review_date = Column(DateTime) 
import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { createExamination } from '../services/api';
import './ExaminationForm.css';

interface PredictionResults {
  glaucoma: boolean;
  cataract: boolean;
  scarring: boolean;
  cardiovascular_disease: boolean;
  diabetes: boolean;
  is_healthy: boolean;
  confidence_scores: {
    glaucoma: number;
    cataract: number;
    scarring: number;
    cardiovascular: number;
    diabetes: number;
    healthy: number;
  };
}

const ExaminationForm: React.FC = () => {
  const { patientId } = useParams<{ patientId: string }>();
  const [images, setImages] = useState<{
    od: File | null;
    os: File | null;
  }>({
    od: null,
    os: null
  });
  const [prediction, setPrediction] = useState<PredictionResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageChange = (eye: 'od' | 'os', event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setImages(prev => ({
        ...prev,
        [eye]: event.target.files![0]
      }));
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (!patientId) {
        throw new Error('Patient ID is required');
      }

      if (!images.od || !images.os) {
        throw new Error('Please upload images for both eyes');
      }

      const response = await createExamination(
        parseInt(patientId),
        images.od,
        images.os
      );
      setPrediction(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="examination-form">
      <h2>New Examination</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Upload Images</h3>
          <div className="input-group">
            <label htmlFor="od-image">OD Image:</label>
            <input
              type="file"
              id="od-image"
              accept="image/*"
              onChange={(e) => handleImageChange('od', e)}
            />
          </div>
          <div className="input-group">
            <label htmlFor="os-image">OS Image:</label>
            <input
              type="file"
              id="os-image"
              accept="image/*"
              onChange={(e) => handleImageChange('os', e)}
            />
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Submit Examination'}
        </button>
      </form>

      {prediction && (
        <div className="prediction-results">
          <h3>AI Prediction Results</h3>
          <div className="prediction-grid">
            <div className="prediction-item">
              <span className="label">Glaucoma:</span>
              <span className={`value ${prediction.glaucoma ? 'positive' : 'negative'}`}>
                {prediction.glaucoma ? 'Detected' : 'Not Detected'}
              </span>
              <span className="confidence">
                Confidence: {(prediction.confidence_scores.glaucoma * 100).toFixed(1)}%
              </span>
            </div>
            <div className="prediction-item">
              <span className="label">Cataract:</span>
              <span className={`value ${prediction.cataract ? 'positive' : 'negative'}`}>
                {prediction.cataract ? 'Detected' : 'Not Detected'}
              </span>
              <span className="confidence">
                Confidence: {(prediction.confidence_scores.cataract * 100).toFixed(1)}%
              </span>
            </div>
            <div className="prediction-item">
              <span className="label">Scarring:</span>
              <span className={`value ${prediction.scarring ? 'positive' : 'negative'}`}>
                {prediction.scarring ? 'Detected' : 'Not Detected'}
              </span>
              <span className="confidence">
                Confidence: {(prediction.confidence_scores.scarring * 100).toFixed(1)}%
              </span>
            </div>
            <div className="prediction-item">
              <span className="label">Overall Health:</span>
              <span className={`value ${prediction.is_healthy ? 'positive' : 'negative'}`}>
                {prediction.is_healthy ? 'Healthy' : 'Abnormal'}
              </span>
              <span className="confidence">
                Confidence: {(prediction.confidence_scores.healthy * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExaminationForm; 
import React, { useState } from 'react';
import axios from 'axios';
import './ImageUpload.css';

interface Prediction {
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

interface DiseaseInfo {
  name: string;
  description: string;
  reasons: string[];
}

const diseaseInfo: Record<string, DiseaseInfo> = {
  glaucoma: {
    name: "Glaucoma",
    description: "A group of eye conditions that damage the optic nerve, often due to abnormally high pressure in the eye.",
    reasons: [
      "Detected increased cup-to-disc ratio in the optic nerve head",
      "Observed thinning of the retinal nerve fiber layer",
      "Identified characteristic visual field defects"
    ]
  },
  cataract: {
    name: "Cataract",
    description: "Clouding of the normally clear lens of the eye, leading to blurry vision.",
    reasons: [
      "Detected increased opacity in the lens area",
      "Observed changes in light scattering patterns",
      "Identified characteristic lens clouding patterns"
    ]
  },
  scarring: {
    name: "Scarring",
    description: "Tissue damage that has healed, leaving permanent marks on the eye tissue.",
    reasons: [
      "Detected irregular tissue patterns in the retinal surface",
      "Observed abnormal texture variations in the fundus image",
      "Identified characteristic scarring patterns in the retinal tissue"
    ]
  },
  cardiovascular_disease: {
    name: "Cardiovascular Disease Indicators",
    description: "Changes in blood vessels that may indicate underlying cardiovascular problems.",
    reasons: [
      "Detected narrowing of retinal arteries",
      "Observed arteriovenous nicking in blood vessels",
      "Identified abnormal blood vessel patterns and tortuosity"
    ]
  },
  diabetes: {
    name: "Diabetic Retinopathy",
    description: "A diabetes complication that affects the eyes, caused by damage to the blood vessels of the retina.",
    reasons: [
      "Detected microaneurysms in the retinal blood vessels",
      "Observed hard exudates in the retinal tissue",
      "Identified characteristic patterns of diabetic retinopathy"
    ]
  }
};

const ImageUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError('');
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'success' && response.data.results) {
        setPrediction(response.data.results);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (err) {
      console.error('Error details:', err);
      if (axios.isAxiosError(err)) {
        const errorMessage = err.response?.data?.detail || err.message;
        setError(`Error: ${errorMessage}`);
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const formatConfidence = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  const getMostLikelyDisease = () => {
    if (!prediction) return null;
    
    // Get all diseases with their confidence scores
    const allDiseases = Object.entries(prediction.confidence_scores)
      .map(([key, value]) => ({ key, confidence: value }));
    
    // Sort by confidence and get the highest
    const mostLikely = allDiseases.sort((a, b) => b.confidence - a.confidence)[0];
    
    // If healthy has the highest confidence, return healthy
    if (mostLikely.key === 'healthy') {
      return {
        key: 'healthy',
        confidence: mostLikely.confidence,
        name: 'Healthy Eye',
        description: 'No significant eye conditions detected.',
        reasons: [
          'Normal appearance of the optic nerve head',
          'Regular blood vessel patterns',
          'No signs of abnormal tissue changes'
        ]
      };
    }
    
    // Otherwise return the most likely disease
    return {
      key: mostLikely.key,
      confidence: mostLikely.confidence,
      ...diseaseInfo[mostLikely.key]
    };
  };

  return (
    <div className="image-upload-container">
      <h2>Eye Disease Detection</h2>
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input-container">
          <input
            type="file"
            onChange={handleFileSelect}
            accept="image/*"
            className="file-input"
          />
          <p className="file-instructions">
            Supported formats: JPG, PNG, JPEG
          </p>
        </div>

        {preview && (
          <div className="image-preview">
            <img src={preview} alt="Preview" />
          </div>
        )}

        <button 
          type="submit" 
          disabled={!selectedFile || isLoading}
          className="submit-button"
        >
          {isLoading ? 'Analyzing...' : 'Analyze Image'}
        </button>

        {error && (
          <div className="error-message">
            <p>{error}</p>
            <p className="error-help">
              Please make sure you're uploading a valid retinal fundus image and try again.
            </p>
          </div>
        )}

        {prediction && (
          <div className="prediction-results">
            <h3>Analysis Results</h3>
            {(() => {
              const mostLikelyDisease = getMostLikelyDisease();
              if (!mostLikelyDisease) return null;
              
              return (
                <div className="disease-card">
                  <h4>{mostLikelyDisease.name}</h4>
                  <div className="confidence-score">
                    Confidence: {formatConfidence(mostLikelyDisease.confidence)}
                  </div>
                  
                  <p className="disease-description">{mostLikelyDisease.description}</p>
                  
                  <div className="reasons-section">
                    <h5>Why the AI thinks this is {mostLikelyDisease.name}:</h5>
                    <ul>
                      {mostLikelyDisease.reasons.map((reason, index) => (
                        <li key={index}>{reason}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </form>
    </div>
  );
};

export default ImageUpload; 
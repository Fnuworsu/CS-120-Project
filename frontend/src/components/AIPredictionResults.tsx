import React from 'react';
import './AIPredictionResults.css';

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

interface Props {
  predictions: Prediction | null;
}

const AIPredictionResults: React.FC<Props> = ({ predictions }) => {
  if (!predictions) return null;

  const getStatusColor = (isPositive: boolean) => {
    return isPositive ? '#dc3545' : '#28a745';
  };

  const formatConfidence = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  return (
    <div className="ai-predictions">
      <h3>AI Analysis Results</h3>
      <div className="predictions-grid">
        <div className="prediction-item">
          <span className="condition">Glaucoma</span>
          <span 
            className="status"
            style={{ color: getStatusColor(predictions.glaucoma) }}
          >
            {predictions.glaucoma ? 'Detected' : 'Not Detected'}
          </span>
          <span className="confidence">
            Confidence: {formatConfidence(predictions.confidence_scores.glaucoma)}
          </span>
        </div>

        <div className="prediction-item">
          <span className="condition">Cataract</span>
          <span 
            className="status"
            style={{ color: getStatusColor(predictions.cataract) }}
          >
            {predictions.cataract ? 'Detected' : 'Not Detected'}
          </span>
          <span className="confidence">
            Confidence: {formatConfidence(predictions.confidence_scores.cataract)}
          </span>
        </div>

        <div className="prediction-item">
          <span className="condition">Scarring</span>
          <span 
            className="status"
            style={{ color: getStatusColor(predictions.scarring) }}
          >
            {predictions.scarring ? 'Detected' : 'Not Detected'}
          </span>
          <span className="confidence">
            Confidence: {formatConfidence(predictions.confidence_scores.scarring)}
          </span>
        </div>

        <div className="prediction-item">
          <span className="condition">Cardiovascular Disease</span>
          <span 
            className="status"
            style={{ color: getStatusColor(predictions.cardiovascular_disease) }}
          >
            {predictions.cardiovascular_disease ? 'Detected' : 'Not Detected'}
          </span>
          <span className="confidence">
            Confidence: {formatConfidence(predictions.confidence_scores.cardiovascular)}
          </span>
        </div>

        <div className="prediction-item">
          <span className="condition">Diabetes</span>
          <span 
            className="status"
            style={{ color: getStatusColor(predictions.diabetes) }}
          >
            {predictions.diabetes ? 'Detected' : 'Not Detected'}
          </span>
          <span className="confidence">
            Confidence: {formatConfidence(predictions.confidence_scores.diabetes)}
          </span>
        </div>

        <div className="prediction-item">
          <span className="condition">Overall Health</span>
          <span 
            className="status"
            style={{ color: getStatusColor(!predictions.is_healthy) }}
          >
            {predictions.is_healthy ? 'Healthy' : 'Requires Attention'}
          </span>
          <span className="confidence">
            Confidence: {formatConfidence(predictions.confidence_scores.healthy)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default AIPredictionResults; 
import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import './ExaminationForm.css';
import AIPredictionResults from './AIPredictionResults';
import axios from 'axios';

interface FundusData {
  opticDisc: {
    od: string;
    os: string;
  };
  macula: {
    od: string;
    os: string;
  };
  posteriorPole: {
    od: string;
    os: string;
  };
  cdRatio: {
    od: string;
    os: string;
  };
  bloodVessels: {
    od: string;
    os: string;
  };
}

interface AIPrediction {
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
  const [fundusData, setFundusData] = useState<FundusData>({
    opticDisc: { od: '', os: 'Normal' },
    macula: { od: '', os: '' },
    posteriorPole: { od: '', os: '' },
    cdRatio: { od: '', os: '0.3' },
    bloodVessels: { od: '', os: 'Normal' },
  });

  const [images, setImages] = useState<{
    od: File | null;
    os: File | null;
  }>({
    od: null,
    os: null,
  });

  const [predictions, setPredictions] = useState<AIPrediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (eye: 'od' | 'os') => (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setImages(prev => ({
        ...prev,
        [eye]: event.target.files![0],
      }));
    }
  };

  const handleInputChange = (
    field: keyof FundusData,
    eye: 'od' | 'os',
    value: string
  ) => {
    setFundusData(prev => ({
      ...prev,
      [field]: {
        ...prev[field],
        [eye]: value,
      },
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      if (!images.od || !images.os) {
        throw new Error('Please upload both OD and OS images');
      }

      if (!patientId) {
        throw new Error('Patient ID is required');
      }

      const formData = new FormData();
      formData.append('od_image', images.od);
      formData.append('os_image', images.os);

      const response = await axios.post(
        `http://localhost:8000/examinations/${patientId}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setPredictions(response.data.results);
    } catch (err) {
      console.error('Error submitting examination:', err);
      setError(err instanceof Error ? err.message : 'An error occurred while submitting the examination');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="examination-form">
      <h2>Ocular Examination</h2>
      {error && <div className="error-message">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="section">
          <h3>Fundus</h3>
          
          <div className="field-group">
            <label>Optic Disc</label>
            <div className="eye-inputs">
              <div>
                <span>OD</span>
                <input
                  type="text"
                  value={fundusData.opticDisc.od}
                  onChange={(e) => handleInputChange('opticDisc', 'od', e.target.value)}
                  placeholder="No pics"
                />
              </div>
              <div>
                <span>OS</span>
                <input
                  type="text"
                  value={fundusData.opticDisc.os}
                  onChange={(e) => handleInputChange('opticDisc', 'os', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="field-group">
            <label>C/D Ratio</label>
            <div className="eye-inputs">
              <div>
                <span>OD</span>
                <input
                  type="text"
                  value={fundusData.cdRatio.od}
                  onChange={(e) => handleInputChange('cdRatio', 'od', e.target.value)}
                />
              </div>
              <div>
                <span>OS</span>
                <input
                  type="text"
                  value={fundusData.cdRatio.os}
                  onChange={(e) => handleInputChange('cdRatio', 'os', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="field-group">
            <label>Macula</label>
            <div className="eye-inputs">
              <div>
                <span>OD</span>
                <input
                  type="text"
                  value={fundusData.macula.od}
                  onChange={(e) => handleInputChange('macula', 'od', e.target.value)}
                />
              </div>
              <div>
                <span>OS</span>
                <input
                  type="text"
                  value={fundusData.macula.os}
                  onChange={(e) => handleInputChange('macula', 'os', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="field-group">
            <label>Blood Vessels</label>
            <div className="eye-inputs">
              <div>
                <span>OD</span>
                <input
                  type="text"
                  value={fundusData.bloodVessels.od}
                  onChange={(e) => handleInputChange('bloodVessels', 'od', e.target.value)}
                />
              </div>
              <div>
                <span>OS</span>
                <input
                  type="text"
                  value={fundusData.bloodVessels.os}
                  onChange={(e) => handleInputChange('bloodVessels', 'os', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="field-group">
            <label>Posterior Pole</label>
            <div className="eye-inputs">
              <div>
                <span>OD</span>
                <input
                  type="text"
                  value={fundusData.posteriorPole.od}
                  onChange={(e) => handleInputChange('posteriorPole', 'od', e.target.value)}
                />
              </div>
              <div>
                <span>OS</span>
                <input
                  type="text"
                  value={fundusData.posteriorPole.os}
                  onChange={(e) => handleInputChange('posteriorPole', 'os', e.target.value)}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="image-upload-section">
          <div>
            <label>Upload OD Image:</label>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageUpload('od')} 
              required
            />
          </div>
          <div>
            <label>Upload OS Image:</label>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageUpload('os')} 
              required
            />
          </div>
        </div>

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Processing...' : 'Submit Examination'}
        </button>
      </form>

      {predictions && <AIPredictionResults predictions={predictions} />}
    </div>
  );
};

export default ExaminationForm; 
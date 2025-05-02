import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { 
  Eye, 
  Upload, 
  Loader2, 
 
  AlertCircle
  
 
} from 'lucide-react';
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

interface ClinicalFeatures {
  opticDisc: string;
  cdRatio: number;
  macula: string;
  bloodVessels: string;
  periphery: string;
  other: string;
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
  const [selectedFiles, setSelectedFiles] = useState<{ left: File | null; right: File | null }>({
    left: null,
    right: null
  });
  const [previews, setPreviews] = useState<{ left: string; right: string }>({
    left: '',
    right: ''
  });
  const [predictions, setPredictions] = useState<{ left: Prediction | null; right: Prediction | null }>({
    left: null,
    right: null
  });
  const [clinicalFeatures, setClinicalFeatures] = useState<{ left: ClinicalFeatures | null; right: ClinicalFeatures | null }>({
    left: null,
    right: null
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const onDrop = useCallback((acceptedFiles: File[], eye: 'left' | 'right') => {
    const file = acceptedFiles[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      setSelectedFiles(prev => ({ ...prev, [eye]: file }));
      setPreviews(prev => ({ ...prev, [eye]: URL.createObjectURL(file) }));
      setPredictions(prev => ({ ...prev, [eye]: null }));
      setError('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1,
    onDrop: (files) => onDrop(files, 'left')
  });

  const { getRootProps: getRootPropsRight, getInputProps: getInputPropsRight, isDragActive: isDragActiveRight } = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1,
    onDrop: (files) => onDrop(files, 'right')
  });

  const getRandomOpticDiscStatus = () => {
    const statuses = [
      "Normal appearance",
      "Pale disc margins",
      "Increased cup-to-disc ratio",
      "Blurred disc margins",
      "Disc hemorrhage present",
      "Neuroretinal rim thinning"
    ];
    return statuses[Math.floor(Math.random() * statuses.length)];
  };

  const getRandomMaculaStatus = () => {
    const statuses = [
      "No abnormalities detected",
      "Mild macular edema",
      "Drusen present",
      "Normal foveal reflex",
      "Pigmentary changes",
      "Early AMD changes"
    ];
    return statuses[Math.floor(Math.random() * statuses.length)];
  };

  const getRandomBloodVesselStatus = () => {
    const statuses = [
      "Normal caliber and distribution",
      "Mild arterial narrowing",
      "AV nicking observed",
      "Increased tortuosity",
      "Venous beading present",
      "Copper/silver wiring"
    ];
    return statuses[Math.floor(Math.random() * statuses.length)];
  };

  const getRandomPeripheryStatus = () => {
    const statuses = [
      "Within normal limits",
      "Peripheral hemorrhages",
      "Lattice degeneration",
      "Retinal thinning",
      "Scattered microaneurysms",
      "Pigmentary changes"
    ];
    return statuses[Math.floor(Math.random() * statuses.length)];
  };

  const getRandomOtherFindings = () => {
    const findings = [
      "No additional findings",
      "Cotton wool spots present",
      "Hard exudates observed",
      "Retinal scarring",
      "Vitreous floaters",
      "Early lens opacity"
    ];
    return findings[Math.floor(Math.random() * findings.length)];
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFiles.left && !selectedFiles.right) {
      setError('Please select at least one image');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const results: { left: Prediction | null; right: Prediction | null } = {
        left: null,
        right: null
      };

      // Generate random clinical features
      const mockFeatures: { left: ClinicalFeatures | null; right: ClinicalFeatures | null } = {
        left: selectedFiles.left ? {
          opticDisc: getRandomOpticDiscStatus(),
          cdRatio: Number((Math.random() * 0.8 + 0.2).toFixed(2)), // Random between 0.2 and 1.0
          macula: getRandomMaculaStatus(),
          bloodVessels: getRandomBloodVesselStatus(),
          periphery: getRandomPeripheryStatus(),
          other: getRandomOtherFindings()
        } : null,
        right: selectedFiles.right ? {
          opticDisc: getRandomOpticDiscStatus(),
          cdRatio: Number((Math.random() * 0.8 + 0.2).toFixed(2)), // Random between 0.2 and 1.0
          macula: getRandomMaculaStatus(),
          bloodVessels: getRandomBloodVesselStatus(),
          periphery: getRandomPeripheryStatus(),
          other: getRandomOtherFindings()
        } : null
      };

      if (selectedFiles.left) {
        const formDataLeft = new FormData();
        formDataLeft.append('file', selectedFiles.left);
        const responseLeft = await axios.post('http://localhost:8000/predict/', formDataLeft, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        if (responseLeft.data.status === 'success' && responseLeft.data.results) {
          results.left = responseLeft.data.results;
        }
      }

      if (selectedFiles.right) {
        const formDataRight = new FormData();
        formDataRight.append('file', selectedFiles.right);
        const responseRight = await axios.post('http://localhost:8000/predict/', formDataRight, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        if (responseRight.data.status === 'success' && responseRight.data.results) {
          results.right = responseRight.data.results;
        }
      }

      setPredictions(results);
      setClinicalFeatures(mockFeatures);
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

  const getMostLikelyCondition = (prediction: Prediction | null) => {
    if (!prediction) return null;

    const conditions = [
      { name: 'Glaucoma', value: prediction.confidence_scores.glaucoma },
      { name: 'Cataract', value: prediction.confidence_scores.cataract },
      { name: 'Scarring', value: prediction.confidence_scores.scarring },
      { name: 'Cardiovascular', value: prediction.confidence_scores.cardiovascular },
      { name: 'Diabetes', value: prediction.confidence_scores.diabetes },
      { name: 'Healthy', value: prediction.confidence_scores.healthy }
    ];

    return conditions.reduce((max, current) => 
      current.value > max.value ? current : max
    );
  };

  const renderEyeUpload = (eye: 'left' | 'right') => {
    const dropzoneProps = eye === 'left' ? getRootProps() : getRootPropsRight();
    const inputProps = eye === 'left' ? getInputProps() : getInputPropsRight();
    const isDragActiveState = eye === 'left' ? isDragActive : isDragActiveRight;

    return (
      <div className={`eye-upload ${eye}-eye`}>
        <h3>
          <Eye className="eye-icon" />
          {eye === 'left' ? 'Left Eye' : 'Right Eye'}
        </h3>
        
        <div {...dropzoneProps} className={`dropzone ${isDragActiveState ? 'active' : ''}`}>
          <input {...inputProps} />
          {!previews[eye] ? (
            <div className="dropzone-content">
              <Upload className="upload-icon" />
              <p>Drag & drop an image here, or click to select</p>
              <p className="file-types">Supported: JPG, PNG, JPEG</p>
            </div>
          ) : (
            <div className="image-preview">
              <img src={previews[eye]} alt={`${eye} eye preview`} />
              <button 
                className="remove-image"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedFiles(prev => ({ ...prev, [eye]: null }));
                  setPreviews(prev => ({ ...prev, [eye]: '' }));
                  setPredictions(prev => ({ ...prev, [eye]: null }));
                }}
              >
                ×
              </button>
            </div>
          )}
        </div>

        {predictions[eye] && (
          <div className="prediction-results">
            <h4>Analysis Results</h4>
            <div className="most-likely-condition">
              {getMostLikelyCondition(predictions[eye])?.name}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderResults = () => {
    if (!clinicalFeatures.left && !clinicalFeatures.right) return null;

    const renderTableCell = (parameter: string, value: any) => {
      const isAbnormal = isAbnormalValue(parameter, value);
      return (
        <td className={isAbnormal ? 'abnormal' : 'normal'}>
          {value || 'N/A'}
        </td>
      );
    };

    return (
      <div className="results-container">
        <h2>Retinal Analysis Results</h2>
        <h3>Ocular Examination Report</h3>
        
        <div className="examination-results">
          <div className="results-table">
            <table>
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>OD (Right Eye)</th>
                  <th>OS (Left Eye)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Optic Disc</td>
                  {renderTableCell('Optic Disc', clinicalFeatures.right?.opticDisc)}
                  {renderTableCell('Optic Disc', clinicalFeatures.left?.opticDisc)}
                </tr>
                <tr>
                  <td>C/D Ratio</td>
                  {renderTableCell('C/D Ratio', clinicalFeatures.right?.cdRatio)}
                  {renderTableCell('C/D Ratio', clinicalFeatures.left?.cdRatio)}
                </tr>
                <tr>
                  <td>Macula</td>
                  {renderTableCell('Macula', clinicalFeatures.right?.macula)}
                  {renderTableCell('Macula', clinicalFeatures.left?.macula)}
                </tr>
                <tr>
                  <td>Blood Vessels</td>
                  {renderTableCell('Blood Vessels', clinicalFeatures.right?.bloodVessels)}
                  {renderTableCell('Blood Vessels', clinicalFeatures.left?.bloodVessels)}
                </tr>
                <tr>
                  <td>Periphery</td>
                  {renderTableCell('Periphery', clinicalFeatures.right?.periphery)}
                  {renderTableCell('Periphery', clinicalFeatures.left?.periphery)}
                </tr>
                <tr>
                  <td>Other</td>
                  {renderTableCell('Other', clinicalFeatures.right?.other)}
                  {renderTableCell('Other', clinicalFeatures.left?.other)}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  const resetAll = () => {
    setSelectedFiles({ left: null, right: null });
    setPreviews({ left: '', right: '' });
    setPredictions({ left: null, right: null });
    setClinicalFeatures({ left: null, right: null });
    setError('');
  };

  const isAbnormalValue = (parameter: string, value: any): boolean => {
    switch (parameter) {
      case 'C/D Ratio':
        return typeof value === 'number' && value > 0.5;
      case 'Optic Disc':
        return value !== 'Normal appearance';
      case 'Macula':
        return value !== 'No abnormalities detected';
      case 'Blood Vessels':
        return value !== 'Normal caliber and distribution';
      case 'Periphery':
        return value !== 'Within normal limits';
      case 'Other':
        return value !== 'No additional findings';
      default:
        return false;
    }
  };

  return (
    <div className="image-upload-container">
      <h2>Eye Disease Detection</h2>
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="eyes-container">
          {renderEyeUpload('left')}
          {renderEyeUpload('right')}
        </div>

        {(clinicalFeatures.left || clinicalFeatures.right) ? (
          <button 
            type="button"
            onClick={resetAll}
            className="submit-button cancel"
          >
            Cancel
          </button>
        ) : (
          <button 
            type="submit" 
            disabled={(!selectedFiles.left && !selectedFiles.right) || isLoading}
            className="submit-button"
          >
            {isLoading ? (
              <>
                <Loader2 className="spinner" />
                Analyzing...
              </>
            ) : (
              'Analyze Images'
            )}
          </button>
        )}

        {error && (
          <div className="error-message">
            <AlertCircle className="error-icon" />
            {error}
          </div>
        )}

        {(clinicalFeatures.left || clinicalFeatures.right) && renderResults()}
      </form>
    </div>
  );
};

export default ImageUpload; 
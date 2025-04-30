import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
import logging
import cv2
from torchvision.models import resnet50, ResNet50_Weights
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeImageDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Custom dataset for eye images
        Args:
            data_dir: Directory containing the images organized by condition
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['glaucoma', 'cataract', 'scarring', 'cardiovascular', 'diabetes', 'healthy']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images and their labels
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory {class_dir} does not exist")
                continue
                
            for img_path in class_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
                
        logger.info(f"Loaded {len(self.images)} images from {len(self.classes)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class EyeDiseaseModel:
    def __init__(self):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify the final layer for our specific classes
        num_classes = 6  # Healthy, Glaucoma, Cataract, Scarring, Cardiovascular, Diabetes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define class labels
        self.classes = ['healthy', 'glaucoma', 'cataract', 'scarring', 'cardiovascular', 'diabetes']

    def preprocess_image(self, image_path):
        """
        Preprocess the image for model input
        """
        try:
            # Open and convert image to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image)
            
            # Ensure correct shape [batch_size, channels, height, width]
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Predict eye conditions from an image
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            if image_tensor is None:
                raise ValueError("Failed to preprocess image")

            # Move to device
            image_tensor = image_tensor.to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]

            # Convert to numpy for easier handling
            probabilities = probabilities.cpu().numpy()

            # Get predicted class and confidence
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])

            # Generate clinical features
            clinical_features = self._generate_clinical_features(
                predicted_class, confidence, probabilities
            )

            # Analyze additional clinical features
            additional_features = self._analyze_clinical_features(image)

            # Combine all features
            combined_features = {**clinical_features, **additional_features}

            # Adjust predictions based on clinical features
            final_predictions = self._adjust_predictions(probabilities, combined_features)

            # Format response with confidence scores
            response = {
                'has_glaucoma': final_predictions['glaucoma'] > 0.5,
                'has_cataract': final_predictions['cataract'] > 0.5,
                'has_scarring': final_predictions['scarring'] > 0.5,
                'has_cardiovascular_disease': final_predictions['cardiovascular'] > 0.5,
                'has_diabetes': final_predictions['diabetes'] > 0.5,
                'is_healthy': final_predictions['healthy'] > 0.5,
                'confidence_scores': {
                    'glaucoma': float(final_predictions['glaucoma']),
                    'cataract': float(final_predictions['cataract']),
                    'scarring': float(final_predictions['scarring']),
                    'cardiovascular': float(final_predictions['cardiovascular']),
                    'diabetes': float(final_predictions['diabetes']),
                    'healthy': float(final_predictions['healthy'])
                }
            }

            return response

        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _generate_clinical_features(self, predicted_class, confidence, class_probabilities):
        """
        Generate clinical features based on the model's prediction
        """
        features = {
            'primary_diagnosis': predicted_class,
            'confidence_score': confidence,
            'differential_diagnoses': [],
            'risk_factors': [],
            'severity': 'Unknown'
        }
        
        # Add differential diagnoses based on class probabilities
        for prob, class_name in zip(class_probabilities, self.classes):
            if class_name != predicted_class and prob > 0.1:  # 10% threshold
                features['differential_diagnoses'].append({
                    'condition': class_name,
                    'probability': prob
                })
        
        # Add condition-specific features
        if predicted_class == 'glaucoma':
            features.update({
                'risk_factors': ['High intraocular pressure', 'Age', 'Family history'],
                'severity': 'Moderate' if confidence < 0.8 else 'High',
                'recommended_tests': ['Tonometry', 'Visual field test', 'Optic nerve examination']
            })
        elif predicted_class == 'cataract':
            features.update({
                'risk_factors': ['Age', 'UV exposure', 'Diabetes'],
                'severity': 'Mild' if confidence < 0.7 else 'Moderate',
                'recommended_tests': ['Visual acuity test', 'Slit lamp examination']
            })
        elif predicted_class == 'scarring':
            features.update({
                'risk_factors': ['Previous injury', 'Inflammation', 'Surgery'],
                'severity': 'Moderate',
                'recommended_tests': ['Corneal topography', 'Slit lamp examination']
            })
        elif predicted_class == 'cardiovascular':
            features.update({
                'risk_factors': ['High blood pressure', 'Smoking', 'Diabetes'],
                'severity': 'Moderate',
                'recommended_tests': ['Echocardiogram', '24-hour blood pressure monitor']
            })
        elif predicted_class == 'diabetes':
            features.update({
                'risk_factors': ['Family history', 'Obesity', 'Physical inactivity'],
                'severity': 'Pre-diabetes' if confidence < 0.7 else 'Diabetes',
                'recommended_tests': ['Fasting blood glucose test', 'HbA1c test']
            })
        else:  # healthy
            features.update({
                'risk_factors': [],
                'severity': 'None',
                'recommended_tests': ['Routine eye examination']
            })
        
        return features

    def _analyze_clinical_features(self, image):
        """Analyze clinical features from the image"""
        try:
            # Convert image to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate opacity score (reduced sensitivity)
            opacity_score = np.mean(gray) / 255.0
            opacity_score = min(max(opacity_score, 0.0), 1.0)
            
            # Calculate texture score for scarring
            texture_score = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
            texture_score = min(max(texture_score, 0.0), 1.0)
            
            # Calculate vessel density and tortuosity
            vessel_density = self._calculate_vessel_density(gray)
            vessel_tortuosity = self._calculate_vessel_tortuosity(gray)
            
            # Calculate cup-to-disc ratio (simplified)
            cup_to_disc_ratio = self._estimate_cup_to_disc_ratio(gray)
            
            # Calculate hemorrhages score
            hemorrhages_score = self._detect_hemorrhages(gray)
            
            return {
                'opacity_score': float(opacity_score),
                'texture_score': float(texture_score),
                'vessel_density': float(vessel_density),
                'vessel_tortuosity': float(vessel_tortuosity),
                'cup_to_disc_ratio': float(cup_to_disc_ratio),
                'hemorrhages': float(hemorrhages_score)
            }
        except Exception as e:
            logger.error(f"Error analyzing clinical features: {str(e)}")
            return {
                'opacity_score': 0.0,
                'texture_score': 0.0,
                'vessel_density': 0.5,
                'vessel_tortuosity': 0.5,
                'cup_to_disc_ratio': 0.5,
                'hemorrhages': 0.0
            }

    def _adjust_predictions(self, model_probabilities, clinical_features):
        """Adjust model predictions based on clinical features"""
        adjusted_probs = {
            'healthy': 0.0,
            'glaucoma': 0.0,
            'cataract': 0.0,
            'scarring': 0.0,
            'cardiovascular': 0.0,
            'diabetes': 0.0
        }
        
        # Map model probabilities to our conditions
        # Assuming model_probabilities has 6 values: [healthy, glaucoma, cataract, scarring, cardiovascular, diabetes]
        adjusted_probs['healthy'] = float(model_probabilities[0])
        adjusted_probs['glaucoma'] = float(model_probabilities[1])
        adjusted_probs['cataract'] = float(model_probabilities[2])
        adjusted_probs['scarring'] = float(model_probabilities[3])
        adjusted_probs['cardiovascular'] = float(model_probabilities[4])
        adjusted_probs['diabetes'] = float(model_probabilities[5])
        
        # Glaucoma adjustment based on cup-to-disc ratio
        if clinical_features['cup_to_disc_ratio'] > 0.6:
            adjusted_probs['glaucoma'] *= 1.5
            # Reduce other probabilities proportionally
            total_other = sum(adjusted_probs.values()) - adjusted_probs['glaucoma']
            for condition in adjusted_probs:
                if condition != 'glaucoma':
                    adjusted_probs[condition] *= (1 - 0.3)
        
        # Cataract adjustment based on opacity score
        if clinical_features['opacity_score'] > 0.7:
            adjusted_probs['cataract'] *= 1.3  # Reduced from 1.5 to 1.3
            # Reduce other probabilities proportionally
            total_other = sum(adjusted_probs.values()) - adjusted_probs['cataract']
            for condition in adjusted_probs:
                if condition != 'cataract':
                    adjusted_probs[condition] *= (1 - 0.2)  # Reduced from 0.3 to 0.2
        
        # Scarring adjustment based on texture analysis
        if clinical_features['texture_score'] > 0.8:
            adjusted_probs['scarring'] *= 1.4
            # Reduce other probabilities proportionally
            total_other = sum(adjusted_probs.values()) - adjusted_probs['scarring']
            for condition in adjusted_probs:
                if condition != 'scarring':
                    adjusted_probs[condition] *= (1 - 0.25)
        
        # Cardiovascular adjustment based on vessel analysis
        if clinical_features['vessel_density'] < 0.3:
            adjusted_probs['cardiovascular'] *= 1.4
            # Reduce other probabilities proportionally
            total_other = sum(adjusted_probs.values()) - adjusted_probs['cardiovascular']
            for condition in adjusted_probs:
                if condition != 'cardiovascular':
                    adjusted_probs[condition] *= (1 - 0.25)
        
        # Diabetes adjustment based on vessel tortuosity
        if clinical_features['vessel_tortuosity'] > 0.7:
            adjusted_probs['diabetes'] *= 1.4
            # Reduce other probabilities proportionally
            total_other = sum(adjusted_probs.values()) - adjusted_probs['diabetes']
            for condition in adjusted_probs:
                if condition != 'diabetes':
                    adjusted_probs[condition] *= (1 - 0.25)
        
        # Normalize probabilities
        total = sum(adjusted_probs.values())
        if total > 0:
            for condition in adjusted_probs:
                adjusted_probs[condition] /= total
        
        return adjusted_probs
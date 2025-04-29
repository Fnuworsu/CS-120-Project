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
        num_classes = 4  # Normal, Glaucoma, Cataract, Scarring
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
        self.classes = ['Normal', 'Glaucoma', 'Cataract', 'Scarring']

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
            additional_features = self._analyze_clinical_features(image_path)

            # Combine all features
            combined_features = {**clinical_features, **additional_features}

            # Adjust predictions based on clinical features
            final_predictions = self._adjust_predictions(probabilities, combined_features)

            return {
                'has_glaucoma': final_predictions['glaucoma'],
                'has_cataract': final_predictions['cataract'],
                'has_scarring': final_predictions['scarring'],
                'has_cardiovascular_disease': final_predictions['cardiovascular'],
                'has_diabetes': final_predictions['diabetes'],
                'is_healthy': final_predictions['healthy'],
                'confidence_scores': {
                    'glaucoma': float(final_predictions['glaucoma_confidence']),
                    'cataract': float(final_predictions['cataract_confidence']),
                    'scarring': float(final_predictions['scarring_confidence']),
                    'cardiovascular': float(final_predictions['cardiovascular_confidence']),
                    'diabetes': float(final_predictions['diabetes_confidence']),
                    'healthy': float(final_predictions['healthy_confidence'])
                }
            }

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
        if predicted_class == 'Glaucoma':
            features.update({
                'risk_factors': ['High intraocular pressure', 'Age', 'Family history'],
                'severity': 'Moderate' if confidence < 0.8 else 'High',
                'recommended_tests': ['Tonometry', 'Visual field test', 'Optic nerve examination']
            })
        elif predicted_class == 'Cataract':
            features.update({
                'risk_factors': ['Age', 'UV exposure', 'Diabetes'],
                'severity': 'Mild' if confidence < 0.7 else 'Moderate',
                'recommended_tests': ['Visual acuity test', 'Slit lamp examination']
            })
        elif predicted_class == 'Scarring':
            features.update({
                'risk_factors': ['Previous injury', 'Inflammation', 'Surgery'],
                'severity': 'Moderate',
                'recommended_tests': ['Corneal topography', 'Slit lamp examination']
            })
        else:  # Normal
            features.update({
                'risk_factors': [],
                'severity': 'None',
                'recommended_tests': ['Routine eye examination']
            })
        
        return features

    def _analyze_clinical_features(self, image_path):
        """Analyze specific clinical features in the fundus image"""
        # Load image for OpenCV processing
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image for clinical analysis")
        
        # Convert to RGB for consistent processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize feature dictionary
        features = {
            'cup_to_disc_ratio': 0.0,
            'vessel_abnormalities': 0.0,
            'opacity_score': 0.0,
            'hemorrhages': 0.0,
            'scarring_score': 0.0
        }
        
        try:
            # Extract optic disc features (for glaucoma)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Detect circular structures (optic disc)
            circles = cv2.HoughCircles(
                enhanced, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=100
            )
            
            if circles is not None:
                # Analyze cup-to-disc ratio
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    roi = enhanced[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
                    if roi.size > 0:
                        # Estimate cup-to-disc ratio using intensity analysis
                        threshold = np.mean(roi) + np.std(roi)
                        cup_area = np.sum(roi > threshold)
                        disc_area = np.pi * (i[2]**2)
                        features['cup_to_disc_ratio'] = min(cup_area / disc_area if disc_area > 0 else 0, 1.0)
            
            # Analyze blood vessels (for cardiovascular and diabetic conditions)
            green_channel = img[:,:,1]  # Green channel shows vessels best
            vessel_mask = cv2.adaptiveThreshold(
                green_channel,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            features['vessel_abnormalities'] = np.sum(vessel_mask > 0) / vessel_mask.size
            
            # Analyze overall image clarity (for cataracts)
            features['opacity_score'] = 1.0 - cv2.Laplacian(gray, cv2.CV_64F).var() / 10000
            
            # Detect dark spots (hemorrhages for diabetic retinopathy)
            dark_threshold = np.percentile(green_channel, 20)
            dark_regions = np.sum(green_channel < dark_threshold) / green_channel.size
            features['hemorrhages'] = dark_regions
            
            # Analyze texture variations (for scarring)
            texture = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
            features['scarring_score'] = np.std(texture) / 100
            
        except Exception as e:
            logger.error(f"Error in clinical feature analysis: {str(e)}")
            # Return default features if analysis fails
            return features
        
        logger.info(f"Clinical features extracted: {features}")
        return features

    def _adjust_predictions(self, model_probabilities, clinical_features):
        """Adjust model predictions based on clinical features"""
        # Initialize prediction probabilities
        adjusted_probs = {
            'glaucoma': 0.0,
            'cataract': 0.0,
            'scarring': 0.0,
            'cardiovascular': 0.0,
            'diabetes': 0.0,
            'healthy': 0.0
        }
        
        # Map model probabilities to our conditions
        # Assuming model_probabilities has 4 values: [normal, glaucoma, cataract, scarring]
        adjusted_probs['healthy'] = float(model_probabilities[0])
        adjusted_probs['glaucoma'] = float(model_probabilities[1])
        adjusted_probs['cataract'] = float(model_probabilities[2])
        adjusted_probs['scarring'] = float(model_probabilities[3])
        
        # Glaucoma adjustment
        if clinical_features.get('cup_to_disc_ratio', 0) > 0.5:
            adjusted_probs['glaucoma'] = max(
                adjusted_probs['glaucoma'],
                clinical_features['cup_to_disc_ratio'] * 0.7
            )
        
        # Cardiovascular/Diabetic adjustment
        if clinical_features.get('vessel_abnormalities', 0) > 0.15:
            vessel_score = clinical_features['vessel_abnormalities']
            adjusted_probs['cardiovascular'] = max(
                adjusted_probs['cardiovascular'],
                vessel_score * 0.6
            )
            adjusted_probs['diabetes'] = max(
                adjusted_probs['diabetes'],
                vessel_score * 0.6
            )
        
        # Cataract adjustment
        if clinical_features.get('opacity_score', 0) > 0.5:
            adjusted_probs['cataract'] = max(
                adjusted_probs['cataract'],
                clinical_features['opacity_score'] * 0.8
            )
        
        # Diabetic retinopathy adjustment
        if clinical_features.get('hemorrhages', 0) > 0.1:
            adjusted_probs['diabetes'] = max(
                adjusted_probs['diabetes'],
                clinical_features['hemorrhages'] * 0.7
            )
        
        # Scarring adjustment
        if clinical_features.get('scarring_score', 0) > 0.3:
            adjusted_probs['scarring'] = max(
                adjusted_probs['scarring'],
                clinical_features['scarring_score'] * 0.6
            )
        
        # Normalize probabilities
        total = sum(adjusted_probs.values())
        if total > 0:
            normalized_probs = {k: v/total for k, v in adjusted_probs.items()}
        else:
            normalized_probs = adjusted_probs
        
        # Add confidence scores
        result = {
            'glaucoma': normalized_probs['glaucoma'] > 0.5,
            'cataract': normalized_probs['cataract'] > 0.5,
            'scarring': normalized_probs['scarring'] > 0.5,
            'cardiovascular': normalized_probs['cardiovascular'] > 0.5,
            'diabetes': normalized_probs['diabetes'] > 0.5,
            'healthy': normalized_probs['healthy'] > 0.5,
            'glaucoma_confidence': normalized_probs['glaucoma'],
            'cataract_confidence': normalized_probs['cataract'],
            'scarring_confidence': normalized_probs['scarring'],
            'cardiovascular_confidence': normalized_probs['cardiovascular'],
            'diabetes_confidence': normalized_probs['diabetes'],
            'healthy_confidence': normalized_probs['healthy']
        }
        
        return result
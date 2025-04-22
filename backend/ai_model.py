import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class EyeDiseaseModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Modify the final layer for our multi-label classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6)  # 6 outputs: glaucoma, cataract, scarring, cardiovascular, diabetes, healthy
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_path):
        """Preprocess the image for model input."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, od_image_path, os_image_path):
        """
        Predict eye diseases from both OD and OS images.
        Returns predictions and confidence scores.
        """
        with torch.no_grad():
            # Process both images
            od_input = self.preprocess_image(od_image_path)
            os_input = self.preprocess_image(os_image_path)
            
            # Get predictions for both eyes
            od_output = torch.sigmoid(self.model(od_input))
            os_output = torch.sigmoid(self.model(os_input))
            
            # Average the predictions from both eyes
            combined_output = (od_output + os_output) / 2
            predictions = combined_output.squeeze().cpu().numpy()

        # Convert predictions to dictionary
        results = {
            'has_glaucoma': bool(predictions[0] > 0.5),
            'has_cataract': bool(predictions[1] > 0.5),
            'has_scarring': bool(predictions[2] > 0.5),
            'has_cardiovascular_disease': bool(predictions[3] > 0.5),
            'has_diabetes': bool(predictions[4] > 0.5),
            'is_healthy': bool(predictions[5] > 0.5),
            'confidence_scores': {
                'glaucoma': float(predictions[0]),
                'cataract': float(predictions[1]),
                'scarring': float(predictions[2]),
                'cardiovascular': float(predictions[3]),
                'diabetes': float(predictions[4]),
                'healthy': float(predictions[5])
            }
        }
        
        return results

# Function to get model instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = EyeDiseaseModel()
    return _model_instance 
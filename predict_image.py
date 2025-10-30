#!/usr/bin/env python3
"""
Wildlife Classification Inference Script
Usage: python predict_image.py <image_path>
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from pathlib import Path

def load_model(model_path, device):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device):
    """Predict wildlife presence in an image"""
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        predicted_class = predicted_class.item()
        confidence = confidence.item()
        probabilities = probabilities.squeeze().cpu().numpy()

    return predicted_class, confidence, probabilities

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'best_wildlife_model.pth'

    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found")
        print("Please make sure you have trained the model first")
        sys.exit(1)

    # Load model and predict
    model = load_model(model_path, device)
    predicted_class, confidence, probabilities = predict_image(image_path, model, device)

    class_names = ['Empty', 'Wildlife']

    print(f"\n=== Wildlife Classification Result ===")
    print(f"Image: {Path(image_path).name}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities:")
    print(f"  Empty: {probabilities[0]:.4f}")
    print(f"  Wildlife: {probabilities[1]:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()

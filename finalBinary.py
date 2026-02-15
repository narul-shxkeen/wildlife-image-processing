import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# === 1. Load model ===
def load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# === 2. Prediction function ===
def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item(), confidence.item()


# === 3. CONFIG ===
source_dir = Path("source")
result_dir = Path("binaryResult")

wildlife_dir = result_dir / "wildlife"
empty_dir = result_dir / "empty"

model_path = Path("best_wildlife_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
class_names = ["empty", "wildlife"]  # 0 = empty, 1 = wildlife

# === 4. Load model ===
model = load_model(model_path, device)
print("Model loaded.")


# === 5. Recursively go through all images ===
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if Path(file).suffix.lower() not in image_exts:
            continue

        img_path = Path(root) / file
        pred_class, conf = predict_image(img_path, model, device)

        # relative path from `source/`
        rel_path = img_path.relative_to(source_dir)

        # choose output folder
        if pred_class == 1:  # Wildlife
            dest = wildlife_dir / rel_path
        else:  # Empty
            dest = empty_dir / rel_path

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest)

        print(f"{img_path} → {class_names[pred_class]} (conf={conf:.3f})")


print("\nDone! Results saved inside 'binaryResult/' directory.")

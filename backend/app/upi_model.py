# backend/app/upi_model.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Definition ===
class UPIModel(nn.Module):
    def __init__(self):
        super(UPIModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: real, fake

    def forward(self, x):
        return self.model(x)

# === Load model weights ===
model = UPIModel().to(device)
model.load_state_dict(torch.load("models/upi_model.pth", map_location=device))
model.eval()

# === Image Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Inference Function ===
def predict_upi(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, float(probs[0][pred])

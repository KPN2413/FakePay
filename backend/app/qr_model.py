# backend/app/qr_model.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pyzbar.pyzbar import decode
import io
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === ResNet18 Architecture ===
class QRModel(nn.Module):
    def __init__(self):
        super(QRModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# === Load Model ===
model = QRModel().to(device)
model.load_state_dict(torch.load("models/qr_model.pth", map_location=device))
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Decode QR Code ===
def decode_qr_from_bytes(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        decoded_objs = decode(img)
        for obj in decoded_objs:
            return obj.data.decode("utf-8")
    except:
        return None

# === Extract UPI handle from QR Text ===
def extract_upi_handle(qr_text):
    match = re.search(r'pa=([a-zA-Z0-9._\-]+@[a-zA-Z]+)', qr_text)
    return match.group(1) if match else None

# === Predict Real vs Fake ===
def predict_qr(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, float(probs[0][pred])

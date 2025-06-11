
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from pyzbar.pyzbar import decode
from PIL import Image
import io
import re

app = FastAPI()

# Allow CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === AutoEncoder Model Definition ===
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
model.load_state_dict(torch.load("qr_autoencoder.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === UPI Blacklist ===
blacklist = {'fraud123@paytm', 'donation@upi', 'moneyhelp@okaxis'}

# === QR Decode ===
def decode_qr_from_bytes(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        decoded_objs = decode(img)
        for obj in decoded_objs:
            return obj.data.decode("utf-8")
    except:
        return None

# === UPI Handle Extract ===
def extract_upi_handle(qr_text):
    match = re.search(r'pa=([a-zA-Z0-9._\-]+@[a-zA-Z]+)', qr_text)
    return match.group(1) if match else None

# === Visual Score ===
def calculate_visual_score(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img_tensor)
        loss = nn.functional.mse_loss(recon, img_tensor)
        return loss.item()

# === API ===
@app.post("/scan_qr/")
async def scan_qr(file: UploadFile = File(...)):
    file_bytes = await file.read()

    qr_text = decode_qr_from_bytes(file_bytes)
    upi_handle = extract_upi_handle(qr_text) if qr_text else None
    is_blacklisted = upi_handle in blacklist if upi_handle else False
    visual_score = calculate_visual_score(file_bytes)

    result = {
        "upi_handle": upi_handle,
        "is_blacklisted": is_blacklisted,
        "visual_anomaly_score": round(visual_score, 5),
        "qr_type": "anomaly" if visual_score > 0.01 else "known-fake",
        "final_verdict": "Suspicious" if is_blacklisted or visual_score > 0.01 else "Clean"
    }

    return result

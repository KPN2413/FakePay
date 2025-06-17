from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pyzbar.pyzbar import decode
from pydantic import BaseModel
import io
import re

app = FastAPI()

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load ResNet18 Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(num_classes=2).to(device)
model.load_state_dict(torch.load("resnet18_qr_classifier.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

blacklist = {'fraud123@paytm', 'donation@upi', 'moneyhelp@okaxis'}
class_names = ['fake', 'real']

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

# === Predict QR Real or Fake ===
def predict_qr(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return class_names[pred], float(probs[0][pred])

# === API: Scan QR ===
@app.post("/scan_qr/")
async def scan_qr(file: UploadFile = File(...)):
    file_bytes = await file.read()
    label, confidence = predict_qr(file_bytes)
    qr_text = decode_qr_from_bytes(file_bytes)
    upi_handle = extract_upi_handle(qr_text) if qr_text else None
    is_blacklisted = upi_handle in blacklist if upi_handle else False

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "upi_handle": upi_handle,
        "is_blacklisted": is_blacklisted,
        "riskLevel": "HIGH" if is_blacklisted or label == "fake" else "LOW",
        "riskScore": 90 if is_blacklisted else (80 if label == "fake" else 10),
        "details": {
            "merchantName": "Merchant XYZ" if upi_handle else None,
            "warnings": ["Blacklisted UPI"] if is_blacklisted else [],
            "recommendations": ["Avoid payment"] if is_blacklisted else ["Looks safe"]
        },
        "final_verdict": "Suspicious" if is_blacklisted or label == "fake" else "Clean"
    }

# === API: Check UPI ID ===
@app.get("/check_upi/")
async def check_upi(upiId: str):
    is_blacklisted = upiId in blacklist
    return {
        "id": "upi-" + upiId,
        "upiId": upiId,
        "riskLevel": "HIGH" if is_blacklisted else "LOW",
        "riskScore": 75 if is_blacklisted else 10,
        "details": {
            "providerVerified": not is_blacklisted,
            "providerName": "Paytm" if "paytm" in upiId else "Unknown",
            "registeredName": None if is_blacklisted else "Merchant XYZ",
            "warnings": ["Blacklisted UPI"] if is_blacklisted else [],
            "recommendations": ["Avoid payment"] if is_blacklisted else ["Proceed with caution"]
        }
    }

# === API: Submit Feedback ===
class Feedback(BaseModel):
    resultId: str
    wasHelpful: bool
    comments: str | None = None

@app.post("/submit_feedback/")
async def submit_feedback(data: Feedback):
    print("Feedback received:", data.dict())
    return {"success": True}

# === Uvicorn Dev Runner ===
import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)

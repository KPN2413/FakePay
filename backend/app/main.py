# backend/app/main.py
from app.upi_model import predict_upi
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.qr_model import predict_qr, decode_qr_from_bytes, extract_upi_handle

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Static data ===
blacklist = {'fraud123@paytm', 'donation@upi', 'moneyhelp@okaxis'}
class_names = ['fake', 'real']

# === QR Scan Endpoint ===
@app.post("/scan_qr/")
async def scan_qr(file: UploadFile = File(...)):
    file_bytes = await file.read()

    label_index, confidence = predict_qr(file_bytes)
    label = class_names[label_index]

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
@app.post("/scan_upi/")
async def scan_upi(file: UploadFile = File(...)):
    file_bytes = await file.read()
    label_index, confidence = predict_upi(file_bytes)
    class_names = ["fake", "real"]
    label = class_names[label_index]

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "riskLevel": "HIGH" if label == "fake" else "LOW",
        "riskScore": 80 if label == "fake" else 10,
        "final_verdict": "Suspicious" if label == "fake" else "Clean"
    }

# === Check UPI Handle Endpoint ===
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

# === Feedback API ===
class Feedback(BaseModel):
    resultId: str
    wasHelpful: bool
    comments: str | None = None

@app.post("/submit_feedback/")
async def submit_feedback(data: Feedback):
    print("Feedback received:", data.dict())
    return {"success": True}

# === Local Dev Only ===
import uvicorn
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=10000, reload=False)

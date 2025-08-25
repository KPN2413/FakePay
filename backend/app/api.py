# backend/app/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import io, os
from PIL import Image
import torch
import torchvision.transforms as T

router = APIRouter()

DEVICE = "cpu"

# Lazy singletons
_qr_model = None
_upi_model = None

# Basic ImageNet-style preprocessing; adjust if your training used different stats/size
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _model_path(filename: str) -> str:
    # models/ relative to this file
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "models", filename))

def _load_resnet18(num_classes: int, weights_path: str):
    """
    Generic loader that works with a plain state_dict or a jit/pt file.
    Adjust if you saved with a custom wrapper.
    """
    try:
        # If scripted/traced
        if weights_path.endswith((".pt", ".pth")):
            # Try torch.jit first
            try:
                m = torch.jit.load(weights_path, map_location=DEVICE)
                m.eval()
                return m
            except Exception:
                pass

        import torchvision.models as models
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        sd = torch.load(weights_path, map_location=DEVICE)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # strict=False allows loading when keys are prefixed/suffixed
        m.load_state_dict(sd, strict=False)
        m.eval()
        return m
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {weights_path}: {e}")

def _ensure_qr_model():
    global _qr_model
    if _qr_model is None:
        path = _model_path("qr_model.pth")
        _qr_model = _load_resnet18(num_classes=2, weights_path=path)

def _ensure_upi_model():
    global _upi_model
    if _upi_model is None:
        path = _model_path("upi_model.pth")
        _upi_model = _load_resnet18(num_classes=2, weights_path=path)

def _infer(model, img: Image.Image) -> Dict:
    x = _transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        class_id = int(torch.tensor(probs).argmax().item())
        score = float(max(probs))
    # assume class 0 = real, 1 = fake (change if your labels differ)
    label = "fake" if class_id == 0 else "real"
    return {"class_id": class_id, "label": label, "score": score}

@router.post("/predict/qr")
async def predict_qr(file: UploadFile = File(...)):
    try:
        _ensure_qr_model()
        img = Image.open(io.BytesIO(await file.read()))
        return _infer(_qr_model, img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/upi")
async def predict_upi(file: UploadFile = File(...)):
    try:
        _ensure_upi_model()
        img = Image.open(io.BytesIO(await file.read()))
        return _infer(_upi_model, img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

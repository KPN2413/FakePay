# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# we'll plug routes from api.py in the next step
try:
    from .api import router as api_router  # will exist in Step 2
except Exception:
    api_router = None

app = FastAPI(title="FakePay Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # later restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# include routes only if api_router is present
if api_router is not None:
    app.include_router(api_router)

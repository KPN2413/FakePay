// lib/mock-api.ts
import { ImageAnalysisResult, QrCodeResult } from "@/lib/types";

/**
 * Small helper to POST an image as multipart/form-data with field name "file".
 * The Next.js rewrite proxies /api/* to your Render backend.
 */
async function postImage(endpoint: string, file: File) {
  const fd = new FormData();
  fd.append("file", file); // MUST be "file" (FastAPI parameter name)

  const res = await fetch(endpoint, {
    method: "POST",
    body: fd,
    cache: "no-store",
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`Backend ${res.status}: ${msg || "request failed"}`);
  }
  return res.json();
}

/**
 * Screenshot analysis (full app UI). We map backend {class_id,label,score}
 * to your ImageAnalysisResult shape.
 */
export async function analyzeImage(image: File): Promise<ImageAnalysisResult> {
  // if your screenshot model is the "UPI" detector, use /api/predict/upi
  const data = await postImage("/api/predict/upi", image);
  // backend returns: { class_id, label, score } where score is 0–1
  const label: "real" | "fake" =
    (data.label as "real" | "fake") ??
    (data.class_id === 1 ? "fake" : "real");
  const riskScore = Math.round(((data.score ?? 0) as number) * 100);
  const riskLevel = label === "fake" ? "HIGH" : "LOW";

  return {
    id: `img-${Date.now().toString(36)}`,
    imageUrl: URL.createObjectURL(image),
    detectedElements: {
      isUpiInterface: true,
      appName: "Unknown",
      upiId: undefined,
      amount: undefined,
      merchantName: undefined,
      timestamp: new Date().toISOString(),
    },
    riskLevel,
    riskScore,
    analysisDetails: {
      isKnownInterface: true,
      interfaceAnomalies: [],
      warnings: [],
      recommendations:
        label === "fake"
          ? ["Do not proceed", "Verify the source before paying"]
          : ["No issues detected"],
    },
  };
}

/**
 * QR image verification. Returns your QrCodeResult shape.
 */
export async function verifyQrCode(image: File): Promise<QrCodeResult> {
  const data = await postImage("/api/predict/qr", image);
  const label: "real" | "fake" =
    (data.label as "real" | "fake") ??
    (data.class_id === 1 ? "fake" : "real");
  const riskScore = Math.round(((data.score ?? 0) as number) * 100);

  return {
    id: `qr-${Date.now().toString(36)}`,
    upiId: data.upi_handle ?? "N/A", // keep if UI expects it; backend may not send
    amount: undefined,
    riskLevel: label === "fake" ? "HIGH" : "LOW",
    riskScore,
    isValid: label === "real",
    createdAt: new Date(),
    details: {
      isStaticQR: true,
      merchantName: "Unverified Service",
      warnings: label === "fake" ? ["Potentially fraudulent QR"] : [],
      recommendations:
        label === "fake"
          ? ["Do not proceed", "Verify merchant first"]
          : ["This QR appears safe"],
    },
  };
}

/**
 * No backend route yet; keep as a no-op so UI doesn't break during demo.
 * If you later add a FastAPI route, swap this to POST /api/feedback (etc).
 */
export async function submitFeedback(_data: {
  resultId: string;
  wasHelpful: boolean;
  comments?: string;
}): Promise<{ success: boolean }> {
  // For now, succeed locally.
  return { success: true };
}

/**
 * Your current backend doesn’t expose a text-based UPI check endpoint.
 * Return a minimal object so /upi-check page can render without crashing.
 * Replace with a real fetch when you add an API route.
 */
export async function checkUpiId(upiId: string): Promise<any> {
  return {
    upiId,
    exists: true,
    isBlacklisted: false,
    lastChecked: new Date().toISOString(),
  };
}

/**
 * Contact form: keep local success for now.
 * Replace with a POST to a backend route or email service later.
 */
export async function submitContactForm(_data: any): Promise<{ success: boolean }> {
  return { success: true };
}

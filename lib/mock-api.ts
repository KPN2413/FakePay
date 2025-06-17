import { ImageAnalysisResult, QrCodeResult } from "@/lib/types";

export async function analyzeImage(image: File): Promise<ImageAnalysisResult> {
  const formData = new FormData();
  formData.append("file", image);

  const response = await fetch(
    "https://fakepay-backend.onrender.com/scan_qr/",
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error("Failed to analyze screenshot");
  }

  const data = await response.json();

  return {
    id: `img-${Date.now().toString(36)}`,
    imageUrl: URL.createObjectURL(image),
    detectedElements: {
      isUpiInterface: true,
      appName: "Unknown",
      upiId: data.upi_handle,
      amount: undefined,
      merchantName: undefined,
      timestamp: new Date().toISOString(),
    },
    riskLevel:
      data.riskLevel ?? (data.final_verdict === "Suspicious" ? "HIGH" : "LOW"),
    riskScore: data.riskScore ?? Math.round(data.visual_anomaly_score * 100),
    analysisDetails: {
      isKnownInterface: true,
      interfaceAnomalies: [],
      warnings: data.details?.warnings ?? [],
      recommendations: data.details?.recommendations ?? [],
    },
  };
}

export async function verifyQrCode(image: File): Promise<QrCodeResult> {
  const formData = new FormData();
  formData.append("file", image);

  const response = await fetch(
    "https://fakepay-backend.onrender.com/scan_qr/",
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) throw new Error("Failed to analyze QR code");

  const data = await response.json();

  return {
    id: `qr-${Date.now().toString(36)}`,
    upiId: data.upi_handle ?? "N/A",
    amount: data.amount ?? undefined,
    riskLevel:
      data.riskLevel ?? (data.final_verdict === "Suspicious" ? "HIGH" : "LOW"),
    riskScore: data.riskScore ?? Math.round(data.visual_anomaly_score * 100),
    isValid: data.final_verdict === "Clean",
    createdAt: new Date(),
    details: {
      isStaticQR: data.qr_type === "static",
      merchantName:
        data.details?.merchantName ?? data.merchant ?? "Unverified Service",
      warnings:
        data.details?.warnings ??
        (data.is_blacklisted ? ["UPI ID is blacklisted"] : []),
      recommendations:
        data.details?.recommendations ??
        (data.final_verdict === "Suspicious"
          ? ["Do not proceed", "Verify merchant first"]
          : ["This QR appears safe"]),
    },
  };
}

export async function submitFeedback(data: {
  resultId: string;
  wasHelpful: boolean;
  comments?: string;
}): Promise<{ success: boolean }> {
  const response = await fetch(
    "https://fakepay-backend.onrender.com/submit_feedback/",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }
  );

  if (!response.ok) throw new Error("Failed to submit feedback");

  return await response.json();
}

export async function checkUpiId(upiId: string): Promise<any> {
  const response = await fetch(
    `https://fakepay-backend.onrender.com/check_upi/?upiId=${upiId}`
  );

  if (!response.ok) throw new Error("Failed to check UPI");

  return await response.json();
}

export async function submitContactForm(
  data: any
): Promise<{ success: boolean }> {
  console.log("Contact form submitted:", data);
  return { success: true };
}

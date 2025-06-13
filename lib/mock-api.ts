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
      appName: "Unknown", // You can fill these if backend returns more info
      upiId: data.upi_handle,
      amount: undefined, // If you're not detecting amount yet
      merchantName: undefined,
      timestamp: new Date().toISOString(),
    },
    riskLevel: data.final_verdict === "Suspicious" ? "HIGH" : "LOW",
    riskScore: Math.round(data.visual_anomaly_score * 100), // scale to 0–100
    analysisDetails: {
      isKnownInterface: true,
      interfaceAnomalies: [],
      warnings: data.is_blacklisted
        ? ["UPI ID is blacklisted"]
        : data.visual_anomaly_score > 0.01
        ? ["Visual tampering detected"]
        : [],
      recommendations:
        data.final_verdict === "Suspicious"
          ? [
              "Do not proceed with payment",
              "Verify with sender before trusting",
            ]
          : ["This QR appears safe"],
    },
  };
}

export async function submitContactForm(
  data: any
): Promise<{ success: boolean }> {
  console.log("Received contact form data:", data);
  return { success: true };
}

// Application wide constants

export const APP_NAME = "FakePay"

// Common UPI applications
export const UPI_APPS = [
  "Google Pay",
  "PhonePe",
  "Paytm",
  "Amazon Pay",
  "BHIM",
  "WhatsApp Pay", 
  "SBI Pay",
  "ICICI iMobile",
  "HDFC PayZapp",
  "Axis Pay",
]

// Common UPI scam types
export const SCAM_TYPES = [
  {
    id: "fake-app",
    name: "Fake UPI App",
    description: "Scammers create fake UPI apps that mimic genuine ones to steal your credentials and money.",
  },
  {
    id: "phishing-link",
    name: "Phishing Links",
    description: "Fraudulent links that appear legitimate but steal your payment details or install malware.",
  },
  {
    id: "qr-code",
    name: "Malicious QR Codes",
    description: "QR codes that direct to fraudulent websites or automatically initiate payments to scammers.",
  },
  {
    id: "remote-access",
    name: "Remote Access Scams",
    description: "Scammers request remote access to your device to supposedly help resolve issues.",
  },
  {
    id: "kyc-update",
    name: "Fake KYC Update",
    description: "Messages claiming your UPI account will be blocked unless you update your KYC details.",
  },
  {
    id: "lottery-prize",
    name: "Lottery/Prize Scams",
    description: "Fake messages about winning prizes requiring small payments to claim larger rewards.",
  },
]

// UPI ID validation pattern
export const UPI_ID_PATTERN = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9]+$/

// Risk levels
export const RISK_LEVELS = {
  HIGH: {
    label: "High Risk",
    color: "destructive",
    description: "This is likely a scam. Do not proceed with any payment or share any information.",
  },
  MEDIUM: {
    label: "Medium Risk",
    color: "warning",
    description: "There are suspicious elements. Verify through official channels before proceeding.",
  },
  LOW: {
    label: "Low Risk",
    color: "success",
    description: "This appears legitimate, but always exercise caution with online payments.",
  },
  UNKNOWN: {
    label: "Unknown",
    color: "secondary",
    description: "We couldn't determine the risk level. Proceed with extreme caution.",
  },
}

// Safety tips
export const SAFETY_TIPS = [
  "Never share your UPI PIN, OTP, or password with anyone",
  "Verify the receiver's UPI ID before making a payment",
  "Be cautious of offers that seem too good to be true",
  "Avoid scanning QR codes from untrusted sources",
  "Check payment receipts and statements regularly",
  "Only download UPI apps from official app stores",
  "Enable additional security features like biometric authentication",
  "Report suspicious activities immediately to your bank and NPCI",
]

// Mock API response delay (in ms)
export const API_RESPONSE_DELAY = 1500

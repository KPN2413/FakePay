// Common Types for FakePay application

export interface UpiCheckResult {
  id: string;
  upiId: string;
  isValid: boolean;
  riskLevel: RiskLevel;
  riskScore: number; // 0-100
  createdAt: Date;
  details: {
    providerVerified: boolean; 
    providerName?: string;
    registeredName?: string;
    accountType?: string;
    warnings?: string[];
    recommendations?: string[];
  };
}

export type RiskLevel = 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN';

export interface QrCodeResult {
  id: string;
  url?: string;
  upiId?: string;
  amount?: number;
  isValid: boolean;
  riskLevel: RiskLevel;
  riskScore: number; // 0-100
  createdAt: Date;
  details: {
    isStaticQR: boolean;
    merchantName?: string;
    merchantCategory?: string;
    warnings?: string[];
    recommendations?: string[];
  };
}

export interface ImageAnalysisResult {
  id: string;
  imageUrl: string;
  detectedElements: {
    isUpiInterface: boolean;
    appName?: string;
    upiId?: string;
    amount?: number;
    merchantName?: string;
    timestamp?: string;
  };
  riskLevel: RiskLevel;
  riskScore: number;
  analysisDetails: {
    isKnownInterface: boolean;
    interfaceAnomalies: string[];
    warnings?: string[];
    recommendations?: string[];
  };
}

export interface UserFeedback {
  resultId: string;
  wasHelpful: boolean;
  comments?: string;
  userAction?: 'PROCEEDED' | 'AVOIDED' | 'REPORTED' | 'OTHER';
}

export interface ContactFormData {
  name: string;
  email: string;
  subject: string;
  message: string;
}

export interface FeatureCard {
  icon: React.ReactNode;
  title: string;
  description: string;
  href: string;
}

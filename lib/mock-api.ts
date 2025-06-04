// Mock API services for FakePay application
import { 
  UpiCheckResult, 
  QrCodeResult, 
  ImageAnalysisResult,
  RiskLevel 
} from '@/lib/types';
import { API_RESPONSE_DELAY, UPI_ID_PATTERN } from '@/lib/constants';

// Helper to simulate API delay
const simulateApiDelay = () => 
  new Promise(resolve => setTimeout(resolve, API_RESPONSE_DELAY));

// UPI ID check mock service
export async function checkUpiId(upiId: string): Promise<UpiCheckResult> {
  await simulateApiDelay();
  
  // Basic validation
  const isValidFormat = UPI_ID_PATTERN.test(upiId);
  
  // Extract provider from UPI ID
  const providerPart = upiId.split('@')[1];
  
  // Mock risk assessment
  let riskLevel: RiskLevel = 'UNKNOWN';
  let riskScore = 50;
  let warnings: string[] = [];
  let recommendations: string[] = [];
  let providerVerified = false;
  let providerName: string | undefined;
  let registeredName: string | undefined;
  
  if (isValidFormat) {
    // Simulate known vs unknown providers
    const knownProviders = ['okaxis', 'oksbi', 'okicici', 'okhdfcbank', 'paytm', 'ybl', 'upi'];
    providerVerified = knownProviders.some(p => providerPart?.includes(p));
    
    if (providerVerified) {
      riskLevel = 'LOW';
      riskScore = Math.floor(Math.random() * 20) + 10; // 10-30
      providerName = knownProviders.find(p => providerPart?.includes(p))?.toUpperCase();
      registeredName = `User ${Math.floor(Math.random() * 1000)}`;
    } else {
      // Unknown or suspicious provider
      riskLevel = 'MEDIUM';
      riskScore = Math.floor(Math.random() * 30) + 40; // 40-70
      warnings.push('This UPI provider is not recognized as a mainstream provider');
      recommendations.push('Verify this UPI ID through official channels before proceeding');
    }
    
    // Specific suspicious patterns
    if (upiId.includes('refund') || upiId.includes('help') || upiId.includes('support')) {
      riskLevel = 'HIGH';
      riskScore = Math.floor(Math.random() * 20) + 80; // 80-100
      warnings.push('This UPI ID contains suspicious keywords often used in scams');
      recommendations.push('Do not make any payment to this UPI ID');
    }
  } else {
    riskLevel = 'HIGH';
    riskScore = 95;
    warnings.push('This is not a valid UPI ID format');
    recommendations.push('A valid UPI ID should be in the format username@provider');
  }
  
  return {
    id: `upi-${Date.now().toString(36)}`,
    upiId,
    isValid: isValidFormat,
    riskLevel,
    riskScore,
    createdAt: new Date(),
    details: {
      providerVerified,
      providerName,
      registeredName,
      accountType: providerVerified ? 'Personal' : undefined,
      warnings,
      recommendations,
    }
  };
}

// QR code verification mock service
export async function verifyQrCode(qrImageOrUrl: File | string): Promise<QrCodeResult> {
  await simulateApiDelay();
  
  // Mock QR code data
  const mockQrData = [
    { 
      upiId: 'merchant@okaxis', 
      riskLevel: 'LOW' as RiskLevel, 
      merchantName: 'Verified Merchant',
      isStaticQR: true
    },
    { 
      upiId: 'store123@paytm', 
      riskLevel: 'LOW' as RiskLevel, 
      merchantName: 'Local Store',
      isStaticQR: true
    },
    { 
      upiId: 'refund.help@unknown', 
      riskLevel: 'HIGH' as RiskLevel, 
      merchantName: undefined,
      isStaticQR: false
    },
    { 
      upiId: 'payment@ybl', 
      riskLevel: 'MEDIUM' as RiskLevel, 
      merchantName: 'Unverified Service',
      isStaticQR: true
    },
  ];
  
  // Randomly select one of the mock QR data scenarios
  const randomIndex = Math.floor(Math.random() * mockQrData.length);
  const selectedMockData = mockQrData[randomIndex];
  
  const riskScore = selectedMockData.riskLevel === 'LOW' 
    ? Math.floor(Math.random() * 20) + 10 // 10-30
    : selectedMockData.riskLevel === 'MEDIUM'
      ? Math.floor(Math.random() * 30) + 40 // 40-70
      : Math.floor(Math.random() * 20) + 80; // 80-100
      
  let warnings: string[] = [];
  let recommendations: string[] = [];
  
  if (selectedMockData.riskLevel === 'HIGH') {
    warnings = [
      'This QR code contains suspicious elements',
      'The linked UPI ID contains keywords often used in scams'
    ];
    recommendations = [
      'Do not scan this QR code with your UPI app',
      'Report this QR code if you received it unexpectedly'
    ];
  } else if (selectedMockData.riskLevel === 'MEDIUM') {
    warnings = ['This QR code links to a less common payment provider'];
    recommendations = ['Verify the merchant before proceeding with payment'];
  }
  
  return {
    id: `qr-${Date.now().toString(36)}`,
    url: typeof qrImageOrUrl === 'string' ? qrImageOrUrl : undefined,
    upiId: selectedMockData.upiId,
    amount: Math.random() > 0.5 ? Math.floor(Math.random() * 1000) + 100 : undefined,
    isValid: selectedMockData.riskLevel !== 'HIGH',
    riskLevel: selectedMockData.riskLevel,
    riskScore,
    createdAt: new Date(),
    details: {
      isStaticQR: selectedMockData.isStaticQR,
      merchantName: selectedMockData.merchantName,
      merchantCategory: selectedMockData.merchantName ? 'Retail' : undefined,
      warnings,
      recommendations,
    }
  };
}

// Image analysis mock service
export async function analyzeImage(image: File): Promise<ImageAnalysisResult> {
  await simulateApiDelay();
  
  // Mock image analysis scenarios
  const analysisScenarios = [
    {
      isUpiInterface: true,
      appName: 'Google Pay',
      upiId: 'merchant@okicici',
      amount: 1299,
      merchantName: 'Genuine Store',
      riskLevel: 'LOW' as RiskLevel,
      isKnownInterface: true,
      interfaceAnomalies: [],
    },
    {
      isUpiInterface: true,
      appName: 'Unknown App',
      upiId: 'support@refund',
      amount: 1999,
      merchantName: undefined,
      riskLevel: 'HIGH' as RiskLevel,
      isKnownInterface: false,
      interfaceAnomalies: ['Suspicious app name', 'Interface elements don\'t match known UPI apps'],
    },
    {
      isUpiInterface: true,
      appName: 'PhonePe',
      upiId: 'merchant@paytm',
      amount: 499,
      merchantName: 'Local Service',
      riskLevel: 'MEDIUM' as RiskLevel,
      isKnownInterface: true,
      interfaceAnomalies: ['Mismatched provider branding'],
    },
    {
      isUpiInterface: false,
      riskLevel: 'UNKNOWN' as RiskLevel,
      isKnownInterface: false,
      interfaceAnomalies: ['No payment interface detected'],
    },
  ];
  
  // Randomly select one of the analysis scenarios
  const randomIndex = Math.floor(Math.random() * analysisScenarios.length);
  const selectedScenario = analysisScenarios[randomIndex];
  
  const riskScore = selectedScenario.riskLevel === 'LOW' 
    ? Math.floor(Math.random() * 20) + 10 // 10-30
    : selectedScenario.riskLevel === 'MEDIUM'
      ? Math.floor(Math.random() * 30) + 40 // 40-70
      : selectedScenario.riskLevel === 'UNKNOWN'
        ? 50
        : Math.floor(Math.random() * 20) + 80; // 80-100
      
  let warnings: string[] = [];
  let recommendations: string[] = [];
  
  if (selectedScenario.riskLevel === 'HIGH') {
    warnings = [
      'This appears to be a fraudulent payment interface',
      'The UPI ID contains suspicious elements'
    ];
    recommendations = [
      'Do not proceed with this payment',
      'Report this to your bank immediately if you\'ve already made a payment'
    ];
  } else if (selectedScenario.riskLevel === 'MEDIUM') {
    warnings = ['Some elements of this payment screen are unusual'];
    recommendations = ['Cross-verify the merchant details before proceeding'];
  } else if (selectedScenario.riskLevel === 'UNKNOWN') {
    warnings = ['Unable to detect a payment interface in this image'];
    recommendations = ['Upload a clearer image of the payment screen'];
  }
  
  return {
    id: `img-${Date.now().toString(36)}`,
    imageUrl: URL.createObjectURL(image),
    detectedElements: {
      isUpiInterface: selectedScenario.isUpiInterface,
      appName: selectedScenario.appName,
      upiId: selectedScenario.upiId,
      amount: selectedScenario.amount,
      merchantName: selectedScenario.merchantName,
      timestamp: new Date().toISOString(),
    },
    riskLevel: selectedScenario.riskLevel,
    riskScore,
    analysisDetails: {
      isKnownInterface: selectedScenario.isKnownInterface,
      interfaceAnomalies: selectedScenario.interfaceAnomalies,
      warnings,
      recommendations,
    }
  };
}

// Submit user feedback
export async function submitFeedback(feedback: { resultId: string; wasHelpful: boolean; comments?: string }): Promise<{ success: boolean }> {
  await simulateApiDelay();
  return { success: true };
}

// Submit contact form
export async function submitContactForm(formData: { name: string; email: string; subject: string; message: string }): Promise<{ success: boolean }> {
  await simulateApiDelay();
  return { success: true };
}
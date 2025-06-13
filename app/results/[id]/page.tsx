"use client"
 
import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { motion } from "framer-motion"
import { AlertTriangle, CheckCircle, Info, ArrowLeft, ThumbsUp, ThumbsDown, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ImageUploadPreview } from "@/components/features/image-upload-preview"
import { RiskLevelBadge } from "@/components/features/risk-level-badge"
import { ResultDetailItem } from "@/components/features/result-detail-item"
import { submitFeedback } from "@/lib/mock-api"
import { ImageAnalysisResult, QrCodeResult, UpiCheckResult } from "@/lib/types"
import { Textarea } from "@/components/ui/textarea"
import { Separator } from "@/components/ui/separator"

type ResultType = ImageAnalysisResult | QrCodeResult | UpiCheckResult

export default function ResultsPage() {
  const router = useRouter()
  const params = useParams()
  const [result, setResult] = useState<ResultType | null>(null)
  const [loading, setLoading] = useState(true)
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false)
  const [showFeedbackForm, setShowFeedbackForm] = useState(false)
  const [feedbackComment, setFeedbackComment] = useState("")
  const [submittingFeedback, setSubmittingFeedback] = useState(false)
  
  useEffect(() => {
    // Get stored result from sessionStorage
    const storedResult = sessionStorage.getItem('analysisResult') || 
                        sessionStorage.getItem('qrVerifyResult') || 
                        sessionStorage.getItem('upiCheckResult')
    
    if (storedResult) {
      try {
        const parsedResult = JSON.parse(storedResult)
        
        // Verify this is the correct result based on the ID in the URL
        if (parsedResult.id === params.id) {
          setResult(parsedResult)
        } else {
          // Wrong result ID, redirect to home
          toast.error("Result not found")
          router.push('/')
        }
      } catch (error) {
        console.error("Error parsing result:", error)
        toast.error("Something went wrong")
      }
    } else {
      // No result found, redirect to home
      toast.error("No analysis result found")
      router.push('/')
    }
    
    setLoading(false)
  }, [params.id, router])
  
  const handleFeedback = async (wasHelpful: boolean) => {
    if (result) {
      setSubmittingFeedback(true)
      
      try {
        await submitFeedback({
          resultId: result.id,
          wasHelpful,
          comments: feedbackComment
        })
        
        setFeedbackSubmitted(true)
        toast.success("Thank you for your feedback!")
      } catch (error) {
        console.error("Error submitting feedback:", error)
        toast.error("Failed to submit feedback")
      } finally {
        setSubmittingFeedback(false)
        setShowFeedbackForm(false)
      }
    }
  }
  
  const isImageResult = (result: any): result is ImageAnalysisResult => {
    return result && 'imageUrl' in result
  }
  
  const isQrResult = (result: any): result is QrCodeResult => {
    return result && 'details' in result && 'isStaticQR' in result.details
  }
  
  const isUpiResult = (result: any): result is UpiCheckResult => {
    return result && 'upiId' in result && !('imageUrl' in result) && !('isStaticQR' in result?.details)
  }
  
  if (loading) {
    return (
      <div className="container py-12 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <h3 className="text-lg font-medium">Loading result...</h3>
        </div>
      </div>
    )
  }
  
  if (!result) {
    return (
      <div className="container py-12">
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl">Result Not Found</CardTitle>
            <CardDescription>
              We couldn't find the analysis result you're looking for.
            </CardDescription>
          </CardHeader>
          <CardFooter>
            <Button onClick={() => router.push('/')}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Return Home
            </Button>
          </CardFooter>
        </Card>
      </div>
    )
  }
  
  // Determine risk color based on risk level
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'HIGH':
        return 'destructive'
      case 'MEDIUM':
        return 'amber-500'
      case 'LOW':
        return 'green-500'
      default:
        return 'secondary'
    }
  }
  
  // Get icon based on risk level
  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'HIGH':
        return <AlertTriangle className="h-5 w-5 text-destructive" />
      case 'MEDIUM':
        return <AlertTriangle className="h-5 w-5 text-amber-500" />
      case 'LOW':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      default:
        return <Info className="h-5 w-5 text-muted-foreground" />
    }
  }
  
  return (
    <div className="flex items-center justify-center min-h-screen px-4">
      <div className="w-full max-w-3xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-2xl">Analysis Result</CardTitle>
                <RiskLevelBadge riskLevel={result.riskLevel} />
              </div>
              <CardDescription>
                {isImageResult(result) && "Analysis of your payment screenshot"}
                {isQrResult(result) && "Analysis of the QR code"}
                {isUpiResult(result) && "Analysis of the UPI ID"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Preview image if available */}
              {isImageResult(result) && result.imageUrl && (
                <ImageUploadPreview url={result.imageUrl} />
              )}
              
              {/* Risk Score */}
              <div className="w-full bg-muted rounded-full h-4 overflow-hidden">
                <div
                  className={`h-full ${
                    result.riskLevel === "HIGH"
                      ? "bg-destructive"
                      : result.riskLevel === "MEDIUM"
                      ? "bg-amber-500"
                      : "bg-green-500"
                  }`}
                  style={{ width: `${result.riskScore}%` }}
                ></div>
              </div>
              <div className="flex justify-between text-sm">
                <span>Safe</span>
                <span className="font-medium">
                  Risk Score: {result.riskScore}%
                </span>
                <span>Risky</span>
              </div>
              
              {/* Result Details */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Details</h3>
                
                {isImageResult(result) && (
                  <>
                    <ResultDetailItem 
                      label="UPI Interface Detected" 
                      value={result.detectedElements.isUpiInterface ? "Yes" : "No"} 
                    />
                    {result.detectedElements.appName && (
                      <ResultDetailItem 
                        label="App Name" 
                        value={result.detectedElements.appName} 
                      />
                    )}
                    {result.detectedElements.upiId && (
                      <ResultDetailItem 
                        label="UPI ID" 
                        value={result.detectedElements.upiId} 
                      />
                    )}
                    {result.detectedElements.amount && (
                      <ResultDetailItem 
                        label="Amount" 
                        value={`₹${result.detectedElements.amount.toLocaleString()}`}
                        />
                    )}
                    {result.detectedElements.merchantName && (
                      <ResultDetailItem 
                        label="Merchant Name" 
                        value={result.detectedElements.merchantName} 
                      />
                    )}
                  </>
                )}
                
                {isQrResult(result) && (
                  <>
                    {result.upiId && (
                      <ResultDetailItem 
                        label="UPI ID" 
                        value={result.upiId} 
                      />
                    )}
                    <ResultDetailItem 
                      label="QR Type" 
                      value={result.details.isStaticQR ? "Static" : "Dynamic"} 
                    />
                    {result.amount && (
                      <ResultDetailItem 
                        label="Amount" 
                        value={`₹${result.amount.toLocaleString()}`}
                        />
                    )}
                    {result.details.merchantName && (
                      <ResultDetailItem 
                        label="Merchant Name" 
                        value={result.details.merchantName} 
                      />
                    )}
                  </>
                )}
                
                {isUpiResult(result) && (
                  <>
                    <ResultDetailItem 
                      label="UPI ID" 
                      value={result.upiId} 
                    />
                    <ResultDetailItem 
                      label="Valid Format" 
                      value={result.isValid ? "Yes" : "No"} 
                    />
                    <ResultDetailItem 
                      label="Provider Verified" 
                      value={result.details.providerVerified ? "Yes" : "No"} 
                    />
                    {result.details.providerName && (
                      <ResultDetailItem 
                        label="Provider" 
                        value={result.details.providerName} 
                      />
                    )}
                    {result.details.registeredName && (
                      <ResultDetailItem 
                        label="Registered Name" 
                        value={result.details.registeredName} 
                      />
                    )}
                  </>
                )}
              </div>
              
              {/* Warnings */}
              {((isImageResult(result) && result.analysisDetails.warnings?.length) ||
               (isQrResult(result) && result.details.warnings?.length) ||
               (isUpiResult(result) && result.details.warnings?.length)) && (
                <div className="bg-destructive/10 p-4 rounded-md">
                  <h3 className="text-lg font-medium flex items-center text-destructive mb-2">
                    <AlertTriangle className="h-5 w-5 mr-2" />
                    Warnings
                  </h3>
                  <ul className="list-disc list-inside space-y-1">
                    {isImageResult(result) && result.analysisDetails.warnings?.map((warning, index) => (
                      <li key={index} className="text-sm">{warning}</li>
                    ))}
                    {isQrResult(result) && result.details.warnings?.map((warning, index) => (
                      <li key={index} className="text-sm">{warning}</li>
                    ))}
                    {isUpiResult(result) && result.details.warnings?.map((warning, index) => (
                      <li key={index} className="text-sm">{warning}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Recommendations */}
              {((isImageResult(result) && result.analysisDetails.recommendations?.length) ||
               (isQrResult(result) && result.details.recommendations?.length) ||
               (isUpiResult(result) && result.details.recommendations?.length)) && (
                <div className="bg-primary/10 p-4 rounded-md">
                  <h3 className="text-lg font-medium flex items-center text-primary mb-2">
                    <Info className="h-5 w-5 mr-2" />
                    Recommendations
                  </h3>
                  <ul className="list-disc list-inside space-y-1">
                    {isImageResult(result) && result.analysisDetails.recommendations?.map((rec, index) => (
                      <li key={index} className="text-sm">{rec}</li>
                    ))}
                    {isQrResult(result) && result.details.recommendations?.map((rec, index) => (
                      <li key={index} className="text-sm">{rec}</li>
                    ))}
                    {isUpiResult(result) && result.details.recommendations?.map((rec, index) => (
                      <li key={index} className="text-sm">{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Feedback Form */}
              {!feedbackSubmitted && !showFeedbackForm && (
                <div className="pt-4">
                  <p className="text-sm text-muted-foreground mb-2">Was this analysis helpful?</p>
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => {
                        setShowFeedbackForm(true)
                        setFeedbackComment("")
                      }}
                    >
                      <ThumbsUp className="mr-2 h-4 w-4" />
                      Yes
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        setShowFeedbackForm(true)
                        setFeedbackComment("")
                      }}
                    >
                      <ThumbsDown className="mr-2 h-4 w-4" />
                      No
                    </Button>
                  </div>
                </div>
              )}
              
              {showFeedbackForm && (
                <div className="pt-2">
                  <Textarea
                    placeholder="Add additional comments (optional)"
                    value={feedbackComment}
                    onChange={(e) => setFeedbackComment(e.target.value)}
                    rows={3}
                    className="mb-2"
                  />
                  <div className="flex gap-2">
                    <Button 
                      variant="default" 
                      size="sm" 
                      onClick={() => handleFeedback(true)}
                      disabled={submittingFeedback}
                    >
                      {submittingFeedback ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <ThumbsUp className="mr-2 h-4 w-4" />
                      )}
                      Submit
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => setShowFeedbackForm(false)}
                      disabled={submittingFeedback}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
              
              {feedbackSubmitted && (
                <p className="text-sm text-muted-foreground pt-2">
                  Thank you for your feedback!
                </p>
              )}
            </CardContent>
            <Separator />
            <CardFooter className="flex justify-between pt-6">
              <Button variant="outline" onClick={() => router.back()}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back
              </Button>
              <Button onClick={() => router.push("/")}>Return Home</Button>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </div>
  )
}
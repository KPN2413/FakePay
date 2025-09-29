"use client"

import { useEffect, useMemo, useState } from "react"
import { useParams, useRouter, useSearchParams } from "next/navigation"
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
  const search = useSearchParams()
  const debugMode = useMemo(() => search.get("debug") === "1", [search])

  const [result, setResult] = useState<ResultType | null>(null)
  const [loading, setLoading] = useState(true)
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false)
  const [showFeedbackForm, setShowFeedbackForm] = useState(false)
  const [feedbackComment, setFeedbackComment] = useState("")
  const [submittingFeedback, setSubmittingFeedback] = useState(false)
  const [lastError, setLastError] = useState<string | null>(null)
  const [sourceKey, setSourceKey] = useState<string | null>(null)

  // ---- Robust reader: try all keys; don't hard-fail on ID mismatch; surface errors; normalise riskLevel.
  useEffect(() => {
    try {
      const keys = ["analysisResult", "qrVerifyResult", "upiCheckResult"] as const
      let chosen: any = null
      let chosenKey: string | null = null

      // Optional: read any previously stored error from the upload/API page
      const possibleError = sessionStorage.getItem("lastAnalysisError")
      if (possibleError) {
        try { setLastError(JSON.parse(possibleError)?.message ?? String(possibleError)) } catch { setLastError(possibleError) }
      }

      for (const k of keys) {
        const raw = sessionStorage.getItem(k)
        if (!raw) continue
        try {
          const obj = JSON.parse(raw)
          // minimal shape check
          if (obj && (obj.riskLevel !== undefined || obj.riskScore !== undefined || obj.details || obj.analysisDetails)) {
            chosen = obj
            chosenKey = k
            break
          }
        } catch {
          // keep scanning next key
        }
      }

      if (!chosen) {
        setLoading(false)
        // Don't redirect; show a friendly message in UI.
        toast.error("No analysis result found")
        return
      }

      // normalise risk level
      if (typeof chosen.riskLevel === "string") {
        chosen.riskLevel = chosen.riskLevel.toUpperCase()
      }

      // keep info for debug
      setSourceKey(chosenKey)
      setResult(chosen)
    } catch (err) {
      console.error("Result read error:", err)
      toast.error("Something went wrong while reading the result")
    } finally {
      setLoading(false)
    }
  }, [params?.id])

  const handleFeedback = async (wasHelpful: boolean) => {
    if (result) {
      setSubmittingFeedback(true)
      try {
        await submitFeedback({
          resultId: (result as any).id,
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

  const isImageResult = (r: any): r is ImageAnalysisResult => r && "imageUrl" in r
  const isQrResult = (r: any): r is QrCodeResult => r && "details" in r && "isStaticQR" in r.details
  const isUpiResult = (r: any): r is UpiCheckResult => r && "upiId" in r && !("imageUrl" in r) && !("isStaticQR" in r?.details)

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

  // ---------- Swap logic (your requirement) ----------
  // 1) Swap HIGH <-> LOW for displayed risk level (MEDIUM stays)
  const displayRiskLevel =
    (result?.riskLevel === "HIGH") ? "LOW"
    : (result?.riskLevel === "LOW") ? "HIGH"
    : (result?.riskLevel ?? "LOW")

  // 2) Score bar colours swapped: HIGH→green, LOW→red, MEDIUM stays amber
  const scoreBarClass =
    displayRiskLevel === "HIGH"
      ? "bg-green-500"
      : displayRiskLevel === "MEDIUM"
      ? "bg-amber-500"
      : "bg-destructive"

  // 3) Top-right badge is fed with the inverted level via RiskLevelBadge below

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
          {lastError && (
            <CardContent>
              <div className="text-sm text-destructive">Error: {lastError}</div>
            </CardContent>
          )}
          <CardFooter className="flex gap-2">
            <Button onClick={() => router.push('/')}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Return Home
            </Button>
            <Button variant="outline" onClick={() => location.reload()}>
              Retry
            </Button>
          </CardFooter>
        </Card>
      </div>
    )
  }

  return (
    <div className="container py-12 max-w-3xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-2xl">Analysis Result</CardTitle>
              {/* Inverted level as requested */}
              <RiskLevelBadge riskLevel={displayRiskLevel} />
            </div>
            <CardDescription>
              {isImageResult(result) && "Analysis of your payment screenshot"}
              {isQrResult(result) && "Analysis of the QR code"}
              {isUpiResult(result) && "Analysis of the UPI ID"}
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* DEBUG BLOCk (only when ?debug=1) */}
            {debugMode && (
              <div className="text-xs rounded-md border p-3">
                <div><b>Debug:</b></div>
                <div>params.id: {String((params as any)?.id ?? "")}</div>
                <div>sourceKey: {String(sourceKey ?? "")}</div>
                <div>riskLevel(raw): {String((result as any)?.riskLevel)}</div>
                <div>riskLevel(displayed): {displayRiskLevel}</div>
                <div>riskScore: {String((result as any)?.riskScore)}</div>
                {lastError && <div className="text-destructive">lastError: {lastError}</div>}
              </div>
            )}

            {/* Preview image if available */}
            {isImageResult(result) && (result as ImageAnalysisResult).imageUrl && (
              <ImageUploadPreview url={(result as ImageAnalysisResult).imageUrl} />
            )}

            {/* Risk Score bar with swapped colours */}
            <div className="w-full bg-muted rounded-full h-4 overflow-hidden">
              <div
                className={`h-full ${scoreBarClass}`}
                style={{ width: `${(result as any).riskScore ?? 0}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-sm">
              <span>Risky</span>
              <span className="font-medium">Score: {(result as any).riskScore ?? 0}%</span>
              <span>Safe</span>
            </div>

            {/* Details */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Details</h3>

              {isImageResult(result) && (
                <>
                  <ResultDetailItem 
                    label="UPI Interface Detected" 
                    value={(result as ImageAnalysisResult).detectedElements.isUpiInterface ? "Yes" : "No"} 
                  />
                  {(result as ImageAnalysisResult).detectedElements.appName && (
                    <ResultDetailItem 
                      label="App Name" 
                      value={(result as ImageAnalysisResult).detectedElements.appName} 
                    />
                  )}
                  {(result as ImageAnalysisResult).detectedElements.upiId && (
                    <ResultDetailItem 
                      label="UPI ID" 
                      value={(result as ImageAnalysisResult).detectedElements.upiId} 
                    />
                  )}
                  {(result as ImageAnalysisResult).detectedElements.amount && (
                    <ResultDetailItem 
                      label="Amount" 
                      value={`₹${(result as ImageAnalysisResult).detectedElements.amount.toLocaleString()}`} 
                    />
                  )}
                  {(result as ImageAnalysisResult).detectedElements.merchantName && (
                    <ResultDetailItem 
                      label="Merchant Name" 
                      value={(result as ImageAnalysisResult).detectedElements.merchantName} 
                    />
                  )}
                </>
              )}

              {isQrResult(result) && (
                <>
                  {(result as QrCodeResult).upiId && (
                    <ResultDetailItem 
                      label="UPI ID" 
                      value={(result as QrCodeResult).upiId} 
                    />
                  )}
                  <ResultDetailItem 
                    label="QR Type" 
                    value={(result as QrCodeResult).details.isStaticQR ? "Static" : "Dynamic"} 
                  />
                  {(result as QrCodeResult).amount && (
                    <ResultDetailItem 
                      label="Amount" 
                      value={`₹${(result as QrCodeResult).amount!.toLocaleString()}`} 
                    />
                  )}
                  {(result as QrCodeResult).details.merchantName && (
                    <ResultDetailItem 
                      label="Merchant Name" 
                      value={(result as QrCodeResult).details.merchantName!} 
                    />
                  )}
                </>
              )}

              {isUpiResult(result) && (
                <>
                  <ResultDetailItem 
                    label="UPI ID" 
                    value={(result as UpiCheckResult).upiId} 
                  />
                  <ResultDetailItem 
                    label="Valid Format" 
                    value={(result as UpiCheckResult).isValid ? "Yes" : "No"} 
                  />
                  <ResultDetailItem 
                    label="Provider Verified" 
                    value={(result as UpiCheckResult).details.providerVerified ? "Yes" : "No"} 
                  />
                  {(result as UpiCheckResult).details.providerName && (
                    <ResultDetailItem 
                      label="Provider" 
                      value={(result as UpiCheckResult).details.providerName!} 
                    />
                  )}
                  {(result as UpiCheckResult).details.registeredName && (
                    <ResultDetailItem 
                      label="Registered Name" 
                      value={(result as UpiCheckResult).details.registeredName!} 
                    />
                  )}
                </>
              )}
            </div>

            {/* Warnings */}
            {((isImageResult(result) && (result as ImageAnalysisResult).analysisDetails.warnings?.length) ||
             (isQrResult(result) && (result as QrCodeResult).details.warnings?.length) ||
             (isUpiResult(result) && (result as UpiCheckResult).details.warnings?.length)) && (
              <div className="bg-destructive/10 p-4 rounded-md">
                <h3 className="text-lg font-medium flex items-center text-destructive mb-2">
                  <AlertTriangle className="h-5 w-5 mr-2" />
                  Warnings
                </h3>
                <ul className="list-disc list-inside space-y-1">
                  {isImageResult(result) && (result as ImageAnalysisResult).analysisDetails.warnings?.map((warning, index) => (
                    <li key={index} className="text-sm">{warning}</li>
                  ))}
                  {isQrResult(result) && (result as QrCodeResult).details.warnings?.map((warning, index) => (
                    <li key={index} className="text-sm">{warning}</li>
                  ))}
                  {isUpiResult(result) && (result as UpiCheckResult).details.warnings?.map((warning, index) => (
                    <li key={index} className="text-sm">{warning}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Recommendations */}
            {((isImageResult(result) && (result as ImageAnalysisResult).analysisDetails.recommendations?.length) ||
             (isQrResult(result) && (result as QrCodeResult).details.recommendations?.length) ||
             (isUpiResult(result) && (result as UpiCheckResult).details.recommendations?.length)) && (
              <div className="bg-primary/10 p-4 rounded-md">
                <h3 className="text-lg font-medium flex items-center text-primary mb-2">
                  <Info className="h-5 w-5 mr-2" />
                  Recommendations
                </h3>
                <ul className="list-disc list-inside space-y-1">
                  {isImageResult(result) && (result as ImageAnalysisResult).analysisDetails.recommendations?.map((rec, index) => (
                    <li key={index} className="text-sm">{rec}</li>
                  ))}
                  {isQrResult(result) && (result as QrCodeResult).details.recommendations?.map((rec, index) => (
                    <li key={index} className="text-sm">{rec}</li>
                  ))}
                  {isUpiResult(result) && (result as UpiCheckResult).details.recommendations?.map((rec, index) => (
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
            <Button onClick={() => router.push('/')}>
              Return Home
            </Button>
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  )
}

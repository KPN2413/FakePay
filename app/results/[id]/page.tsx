"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowLeft, Loader2, ThumbsDown, ThumbsUp } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ImageUploadPreview } from "@/components/features/image-upload-preview";
import { RiskLevelBadge } from "@/components/features/risk-level-badge";
import { ResultDetailItem } from "@/components/features/result-detail-item";
import { submitFeedback } from "@/lib/mock-api";
import { ImageAnalysisResult, QrCodeResult, UpiCheckResult } from "@/lib/types";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";

type ResultType = ImageAnalysisResult | QrCodeResult | UpiCheckResult;

export default function ResultsPage() {
  const router = useRouter();
  const params = useParams();
  const [result, setResult] = useState<ResultType | null>(null);
  const [loading, setLoading] = useState(true);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [submittingFeedback, setSubmittingFeedback] = useState(false);

  useEffect(() => {
    const storedResult =
      sessionStorage.getItem("analysisResult") ||
      sessionStorage.getItem("qrVerifyResult") ||
      sessionStorage.getItem("upiCheckResult");

    if (storedResult) {
      try {
        const parsedResult = JSON.parse(storedResult);
        if (parsedResult?.id === params.id) {
          setResult(parsedResult);
        } else {
          toast.error("Result not found");
          router.push("/");
        }
      } catch (error) {
        console.error("Error parsing result:", error);
        toast.error("Something went wrong while loading the result.");
        router.push("/");
      }
    } else {
      toast.error("No analysis result found");
      router.push("/");
    }

    setLoading(false);
  }, [params.id, router]);

  const handleFeedback = async (wasHelpful: boolean) => {
    if (result) {
      setSubmittingFeedback(true);
      try {
        await submitFeedback({
          resultId: result.id,
          wasHelpful,
          comments: feedbackComment,
        });
        setFeedbackSubmitted(true);
        toast.success("Thank you for your feedback!");
      } catch (error) {
        console.error("Error submitting feedback:", error);
        toast.error("Failed to submit feedback");
      } finally {
        setSubmittingFeedback(false);
        setShowFeedbackForm(false);
      }
    }
  };

  const isImageResult = (result: any): result is ImageAnalysisResult => {
    return result && "imageUrl" in result;
  };

  const isQrResult = (result: any): result is QrCodeResult => {
    return result && "details" in result && "isStaticQR" in result.details;
  };

  const isUpiResult = (result: any): result is UpiCheckResult => {
    return (
      result &&
      "upiId" in result &&
      !("imageUrl" in result) &&
      !("isStaticQR" in result?.details)
    );
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "HIGH":
        return "bg-destructive";
      case "MEDIUM":
        return "bg-amber-500";
      case "LOW":
        return "bg-green-500";
      default:
        return "bg-secondary";
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <h3 className="text-lg font-medium">Loading result...</h3>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl">Result Not Found</CardTitle>
            <CardDescription>
              We couldn't find the analysis result you're looking for.
            </CardDescription>
          </CardHeader>
          <CardFooter>
            <Button onClick={() => router.push("/")}>Return Home</Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen px-4 py-12">
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
              {isImageResult(result) && result.imageUrl && (
                <ImageUploadPreview url={result.imageUrl} />
              )}

              <div className="relative w-full h-4 bg-muted rounded-md">
                <div
                  className={`absolute top-0 left-0 h-full ${getRiskColor(
                    result.riskLevel
                  )}`}
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

              {/* Result details */}
              {result && "detectedElements" in result && (
                <div className="space-y-4">
                  {Object.entries(result.detectedElements || {}).map(
                    ([label, value]) => (
                      <ResultDetailItem
                        key={label}
                        label={label}
                        value={value || "—"}
                      />
                    )
                  )}
                </div>
              )}

              {/* Warnings */}
              {result.analysisDetails?.warnings?.length > 0 && (
                <div>
                  <h4 className="font-semibold text-red-600">Warnings</h4>
                  <ul className="list-disc list-inside text-sm">
                    {result.analysisDetails.warnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Recommendations */}
              {result.analysisDetails?.recommendations?.length > 0 && (
                <div>
                  <h4 className="font-semibold text-green-600">
                    Recommendations
                  </h4>
                  <ul className="list-disc list-inside text-sm">
                    {result.analysisDetails.recommendations.map((r, idx) => (
                      <li key={idx}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Feedback Form */}
              {!feedbackSubmitted && (
                <div className="pt-4 border-t mt-6">
                  <h4 className="font-medium mb-2">
                    Was this analysis helpful?
                  </h4>
                  <div className="flex gap-2">
                    <Button
                      onClick={() => handleFeedback(true)}
                      disabled={submittingFeedback}
                    >
                      <ThumbsUp className="w-4 h-4 mr-1" />
                      Yes
                    </Button>
                    <Button
                      onClick={() => handleFeedback(false)}
                      disabled={submittingFeedback}
                      variant="outline"
                    >
                      <ThumbsDown className="w-4 h-4 mr-1" />
                      No
                    </Button>
                  </div>
                  {showFeedbackForm && (
                    <div className="mt-4">
                      <Textarea
                        value={feedbackComment}
                        onChange={(e) => setFeedbackComment(e.target.value)}
                        placeholder="Any comments or suggestions?"
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Thank You Message */}
              {feedbackSubmitted && (
                <div className="text-sm text-green-600 font-medium mt-4">
                  ✅ Thank you for your feedback!
                </div>
              )}
            </CardContent>

            <Separator />

            <CardFooter className="flex justify-between pt-6">
              <Button variant="outline" onClick={() => router.back()}>
                <ArrowLeft className="mr-2 h-4 w-4" /> Back
              </Button>
              <Button onClick={() => router.push("/")}>Return Home</Button>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}

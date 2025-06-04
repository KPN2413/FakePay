"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useForm } from "react-hook-form"
import { motion } from "framer-motion"
import { QrCode, Upload, AlertTriangle, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ImageUploadPreview } from "@/components/features/image-upload-preview"
import { verifyQrCode } from "@/lib/mock-api"

interface FormData {
  qrImage: FileList
}

export default function QrVerifyPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  
  const { register, handleSubmit, formState: { errors }, watch } = useForm<FormData>()
  
  const qrImageFile = watch("qrImage")?.[0]
  
  const onImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    } else {
      setPreviewUrl(null)
    }
  }
  
  const onSubmit = async (data: FormData) => {
    if (!data.qrImage?.[0]) {
      toast.error("Please select an image to analyze")
      return
    }
    
    setIsLoading(true)
    
    try {
      const result = await verifyQrCode(data.qrImage[0])
      
      // Store result in sessionStorage
      sessionStorage.setItem('qrVerifyResult', JSON.stringify(result))
      
      // Navigate to results page
      router.push(`/results/${result.id}`)
    } catch (error) {
      console.error("Error verifying QR code:", error)
      toast.error("Failed to verify the QR code. Please try again.")
    } finally {
      setIsLoading(false)
    }
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
            <CardTitle className="text-2xl">QR Code Verification</CardTitle>
            <CardDescription>
              Scan and analyze QR codes to detect potentially malicious payment requests.
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit(onSubmit)}>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="qrImage">Upload QR Code Image</Label>
                <Input
                  id="qrImage"
                  type="file"
                  accept="image/*"
                  {...register("qrImage", { required: "Please select an image" })}
                  onChange={onImageChange}
                  disabled={isLoading}
                  className="cursor-pointer"
                />
                {errors.qrImage && (
                  <p className="text-sm font-medium text-destructive flex items-center mt-2">
                    <AlertTriangle className="h-4 w-4 mr-1" />
                    {errors.qrImage.message}
                  </p>
                )}
              </div>
              
              {previewUrl && (
                <ImageUploadPreview url={previewUrl} />
              )}
              
              {!previewUrl && (
                <div className="border rounded-md p-8 flex flex-col items-center justify-center text-center">
                  <QrCode className="h-10 w-10 text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Upload a clear image of the QR code you want to verify
                  </p>
                </div>
              )}
              
              <div className="bg-muted p-4 rounded-md text-sm">
                <p className="font-medium mb-2 flex items-center">
                  <AlertTriangle className="h-4 w-4 mr-2 text-amber-500" />
                  QR Code Safety Tips:
                </p>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                  <li>Always verify the merchant before scanning a QR code</li>
                  <li>Be cautious of QR codes in unexpected places or from unknown sources</li>
                  <li>Check the URL or UPI ID the QR code redirects to before proceeding</li>
                  <li>Be wary of QR codes promising rewards, cashbacks, or urgent action</li>
                </ul>
              </div>
            </CardContent>
            <CardFooter>
              <Button type="submit" disabled={isLoading} className="w-full">
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Verifying QR Code...
                  </>
                ) : (
                  <>
                    <QrCode className="mr-2 h-4 w-4" />
                    Verify QR Code
                  </>
                )}
              </Button>
            </CardFooter>
          </form>
        </Card>
      </motion.div>
    </div>
  )
}
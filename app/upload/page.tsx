"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useForm } from "react-hook-form"
import { motion } from "framer-motion"
import { FileImage, Upload, AlertTriangle, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ImageUploadPreview } from "@/components/features/image-upload-preview"
import { analyzeImage } from "@/lib/mock-api"
import { ImageAnalysisResult } from "@/lib/types"

interface FormData {
  image: FileList
}

export default function UploadPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  
  const { register, handleSubmit, formState: { errors }, watch } = useForm<FormData>()
  
  const imageFile = watch("image")?.[0]
  
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
    if (!data.image?.[0]) {
      toast.error("Please select an image to analyze")
      return
    }
    
    setIsLoading(true)
    
    try {
      const result = await analyzeImage(data.image[0])
      
      // Store result in sessionStorage to access it on results page
      sessionStorage.setItem('analysisResult', JSON.stringify(result))
      
      // Navigate to results page
      router.push(`/results/${result.id}`)
    } catch (error) {
      console.error("Error analyzing image:", error)
      toast.error("Failed to analyze the image. Please try again.")
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
            <CardTitle className="text-2xl">Analyze Payment Screenshots</CardTitle>
            <CardDescription>
              Upload screenshots of UPI payment interfaces or messages to identify potential scams.
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit(onSubmit)}>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="image">Upload Payment Screenshot</Label>
                <Input
                  id="image"
                  type="file"
                  accept="image/*"
                  {...register("image", { required: "Please select an image" })}
                  onChange={onImageChange}
                  disabled={isLoading}
                  className="cursor-pointer"
                />
                {errors.image && (
                  <p className="text-sm font-medium text-destructive flex items-center mt-2">
                    <AlertTriangle className="h-4 w-4 mr-1" />
                    {errors.image.message}
                  </p>
                )}
              </div>
              
              {previewUrl && (
                <ImageUploadPreview url={previewUrl} />
              )}
              
              {!previewUrl && (
                <div className="border rounded-md p-8 flex flex-col items-center justify-center text-center">
                  <FileImage className="h-10 w-10 text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Upload a clear screenshot of a UPI payment interface, QR code, or payment message
                  </p>
                </div>
              )}
              
              <div className="bg-muted p-4 rounded-md text-sm">
                <p className="font-medium mb-2 flex items-center">
                  <AlertTriangle className="h-4 w-4 mr-2 text-amber-500" />
                  For best results:
                </p>
                <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                  <li>Upload clear, readable screenshots</li>
                  <li>Include the entire payment interface in the image</li>
                  <li>Make sure the UPI ID and amount are visible</li>
                  <li>Avoid cropping important details</li>
                </ul>
              </div>
            </CardContent>
            <CardFooter>
              <Button type="submit" disabled={isLoading} className="w-full">
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing Image...
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Analyze Screenshot
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
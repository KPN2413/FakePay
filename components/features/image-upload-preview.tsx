"use client"
 
import Image from "next/image"
import { motion } from "framer-motion"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ImageUploadPreviewProps {
  url: string
  onRemove?: () => void
}

export function ImageUploadPreview({ url, onRemove }: ImageUploadPreviewProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative rounded-md overflow-hidden border"
    >
      <div className="relative aspect-video w-full">
        <Image
          src={url}
          alt="Preview"
          fill
          className="object-contain"
        />
      </div>
      
      {onRemove && (
        <Button
          size="icon"
          variant="destructive"
          className="absolute top-2 right-2 h-8 w-8 rounded-full"
          onClick={onRemove}
        >
          <X className="h-4 w-4" />
        </Button>
      )}
    </motion.div>
  )
}

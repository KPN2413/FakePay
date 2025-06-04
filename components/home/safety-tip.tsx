"use client"

import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Shield } from "lucide-react"

interface SafetyTipProps {
  tip: string
  index: number
}

export function SafetyTip({ tip, index }: SafetyTipProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.05 * index }}
    >
      <Card className="h-full transition-all hover:shadow-md">
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <Shield className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm">{tip}</p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
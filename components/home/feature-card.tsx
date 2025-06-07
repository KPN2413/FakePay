"use client"

import Link from "next/link"
import { motion } from "framer-motion"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { type FeatureCard as FeatureCardType } from "@/lib/types"

interface FeatureCardProps {
  feature: FeatureCardType
  index: number
}

export function FeatureCard({ feature, index }: FeatureCardProps) { 
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 * index }}
    >
      <Card className="h-full transition-all hover:shadow-lg">
        <CardHeader className="flex items-center justify-center pb-2">
          {feature.icon}
        </CardHeader>
        <CardContent className="text-center">
          <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
          <p className="text-muted-foreground">{feature.description}</p>
        </CardContent>
        <CardFooter className="flex justify-center pt-0">
          <Button variant="ghost" asChild className="group">
            <Link href={feature.href}>
              Try Now
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  )
}

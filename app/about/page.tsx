"use client"

import { motion } from "framer-motion"
import { Shield, AlertTriangle, CheckCircle, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { SCAM_TYPES } from "@/lib/constants"
import Link from "next/link"

export default function AboutPage() {
  return (
    <div className="container py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto"
      >
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight">About FakePay</h1>
          <p className="mt-4 text-xl text-muted-foreground">
            FakePay helps you stay safe from UPI payment scams through advanced detection and verification.
          </p>
        </div>
        
        <Tabs defaultValue="about" className="mb-12">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="about">About Us</TabsTrigger>
            <TabsTrigger value="scams">Common Scams</TabsTrigger>
            <TabsTrigger value="how">How It Works</TabsTrigger>
          </TabsList>
          
          <TabsContent value="about" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Our Mission</CardTitle>
                <CardDescription>
                  Making digital payments safer for everyone
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p>
                  FakePay was created with a simple mission: to protect users from the growing threat of UPI payment scams. 
                  As digital payments become increasingly popular in India, scammers have developed sophisticated techniques 
                  to trick users into making fraudulent payments.
                </p>
                <p>
                  Our team of security experts and developers has built FakePay to provide a comprehensive solution for 
                  verifying payment requests, analyzing suspicious UPI IDs, and detecting fraudulent QR codes before you 
                  make a payment.
                </p>
                <p>
                  We believe that education and awareness are crucial in the fight against digital fraud. That's why 
                  FakePay not only detects potential scams but also provides explanations and recommendations to help 
                  you understand the risks and make safer payment decisions.
                </p>
                <div className="flex justify-center mt-6">
                  <Button asChild>
                    <Link href="/contact">Contact Us</Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="scams" className="mt-6 space-y-6">
            {SCAM_TYPES.map((scam) => (
              <motion.div
                key={scam.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xl flex items-center gap-2">
                      <AlertTriangle className="h-5 w-5 text-destructive" />
                      {scam.name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p>{scam.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </TabsContent>
          
          <TabsContent value="how" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>How FakePay Works</CardTitle>
                <CardDescription>
                  Our technology and verification processes
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <h3 className="text-lg font-medium flex items-center gap-2">
                    <Shield className="h-5 w-5 text-primary" />
                    UPI ID Verification
                  </h3>
                  <p className="text-muted-foreground">
                    Our system analyzes UPI IDs to verify their format, check if they belong to known 
                    payment providers, and identify suspicious patterns often used in scams. We compare 
                    UPI IDs against our database of known legitimate providers and flag potential risks.
                  </p>
                </div>
                
                <div className="space-y-2">
                  <h3 className="text-lg font-medium flex items-center gap-2">
                    <Shield className="h-5 w-5 text-primary" />
                    QR Code Analysis
                  </h3>
                  <p className="text-muted-foreground">
                    When you upload a QR code, we decode it to extract the payment information and analyze 
                    the destination UPI ID or URL. We check for suspicious redirect patterns, unknown payment 
                    providers, and other indicators of potential fraud.
                  </p>
                </div>
                
                <div className="space-y-2">
                  <h3 className="text-lg font-medium flex items-center gap-2">
                    <Shield className="h-5 w-5 text-primary" />
                    Screenshot Analysis
                  </h3>
                  <p className="text-muted-foreground">
                    Our advanced image processing technology can analyze screenshots of payment interfaces to detect 
                    if they match legitimate UPI apps. We identify inconsistencies in the interface, suspicious elements, 
                    and potential fake apps designed to steal your payment information.
                  </p>
                </div>
                
                <div className="space-y-2">
                  <h3 className="text-lg font-medium flex items-center gap-2">
                    <Shield className="h-5 w-5 text-primary" />
                    Risk Assessment
                  </h3>
                  <p className="text-muted-foreground">
                    Based on our analysis, we provide a risk assessment ranging from Low to High, along with 
                    detailed explanations of any identified issues and recommendations for safe action. Our 
                    risk scores are calculated using multiple factors to give you a comprehensive understanding 
                    of potential threats.
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardHeader className="text-center pb-2">
              <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-500" />
              <CardTitle className="text-xl">Secure & Private</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-muted-foreground">
                Your payment information is never stored permanently and all analysis is performed securely.
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="text-center pb-2">
              <Info className="h-8 w-8 mx-auto mb-2 text-blue-500" />
              <CardTitle className="text-xl">Educational</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-muted-foreground">
                Learn about different types of UPI scams and how to protect yourself from them.
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="text-center pb-2">
              <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-amber-500" />
              <CardTitle className="text-xl">Preventative</CardTitle>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-muted-foreground">
                Identify potential scams before they happen and prevent financial loss.
              </p>
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  )
}
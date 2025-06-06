"use client" 

import Link from "next/link"
import Image from "next/image"
import { motion } from "framer-motion"
import { ArrowRight, Shield, Smartphone, FileSearch, QrCode } from "lucide-react"
import { Button } from "@/components/ui/button"
import { FeatureCard } from "@/components/home/feature-card"
import { SafetyTip } from "@/components/home/safety-tip"
import { SAFETY_TIPS } from "@/lib/constants"
import { type FeatureCard as FeatureCardType } from "@/lib/types"

export default function Home() {
  const features: FeatureCardType[] = [
    {
      icon: <Smartphone className="h-10 w-10 text-primary" />,
      title: "UPI ID Verification",
      description: "Verify the legitimacy of UPI payment addresses before sending money.",
      href: "/upi-check",
    },
    {
      icon: <QrCode className="h-10 w-10 text-primary" />,
      title: "QR Code Scanner",
      description: "Scan and analyze QR codes to detect potentially malicious payment requests.",
      href: "/qr-verify",
    },
    {
      icon: <FileSearch className="h-10 w-10 text-primary" />,
      title: "Screenshot Analysis",
      description: "Upload payment screenshots to identify potential fraud indicators.",
      href: "/upload",
    },
  ]

  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-gradient-to-b from-background to-muted">
        <div className="container px-4 md:px-6">
          <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 items-center">
            <motion.div 
              className="flex flex-col justify-center space-y-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="inline-block px-3 py-1 mb-2 text-sm font-medium text-primary bg-primary/10 rounded-full">
                Stay protected online
              </div>
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                Detect & Prevent UPI Payment Scams
              </h1>
              <p className="max-w-[600px] text-muted-foreground md:text-xl">
                FakePay helps you verify the legitimacy of UPI payment requests, scan QR codes for threats, and analyze screenshots to identify potential scams.
              </p>
              <div className="flex flex-col gap-2 min-[400px]:flex-row">
                <Button asChild size="lg">
                  <Link href="/upload">
                    Analyze Payment <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link href="/about">Learn More</Link>
                </Button>
              </div>
            </motion.div>
            <motion.div
              className="flex items-center justify-center"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="relative w-full max-w-[500px] aspect-square">
                <Image
                  src="https://images.pexels.com/photos/6347729/pexels-photo-6347729.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                  alt="Secure mobile payment"
                  fill
                  className="object-cover rounded-lg"
                  priority
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent rounded-lg"></div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="w-full py-12 md:py-24 bg-card">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <div className="inline-block px-3 py-1 mb-2 text-sm font-medium text-primary bg-primary/10 rounded-full">
                Our Features
              </div>
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                Comprehensive Scam Protection
              </h2>
              <p className="max-w-[700px] text-muted-foreground md:text-xl">
                FakePay offers multiple ways to verify payment requests and protect yourself from UPI scams.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-6 mt-12 md:grid-cols-3">
            {features.map((feature, index) => (
              <FeatureCard key={index} feature={feature} index={index} />
            ))}
          </div>
        </div>
      </section>

      {/* Safety Tips Section */}
      <section className="w-full py-12 md:py-24">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="inline-block px-3 py-1 mb-2 text-sm font-medium text-primary bg-primary/10 rounded-full">
              Stay Safe
            </div>
            <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl">
              Essential UPI Safety Tips
            </h2>
            <p className="max-w-[700px] text-muted-foreground md:text-xl">
              Follow these important safety practices to protect yourself from UPI payment scams.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 mt-12 md:grid-cols-2 lg:grid-cols-4">
            {SAFETY_TIPS.slice(0, 8).map((tip, index) => (
              <SafetyTip key={index} tip={tip} index={index} />
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full py-12 md:py-24 bg-primary text-primary-foreground">
        <div className="container px-4 md:px-6">
          <div className="grid gap-6 lg:grid-cols-2 items-center">
            <div className="flex flex-col justify-center space-y-4">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl">
                Ready to secure your UPI payments?
              </h2>
              <p className="max-w-[600px] md:text-xl">
                Start using FakePay today to protect yourself from UPI scams and fraudulent payment requests.
              </p>
            </div>
            <div className="flex flex-col gap-2 min-[400px]:flex-row justify-center lg:justify-end">
              <Button size="lg" variant="secondary" asChild>
                <Link href="/upload">
                  Analyze Payment <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="bg-primary-foreground text-primary" asChild>
                <Link href="/about">Learn More</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

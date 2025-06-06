"use client"
 
import { useState } from "react"
import { useRouter } from "next/navigation"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"
import { motion } from "framer-motion"
import { Check, Loader2, AlertTriangle } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { UPI_ID_PATTERN } from "@/lib/constants"
import { checkUpiId } from "@/lib/mock-api"
import { UpiIdExample } from "@/components/features/upi-id-example"

// Form validation schema
const formSchema = z.object({
  upiId: z
    .string()
    .min(1, "UPI ID is required")
    .regex(UPI_ID_PATTERN, "Invalid UPI ID format. It should be in the format username@provider")
})

type FormData = z.infer<typeof formSchema>

export default function UpiCheckPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  
  const form = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      upiId: ""
    }
  })
  
  const onSubmit = async (data: FormData) => {
    setIsLoading(true)
    
    try {
      const result = await checkUpiId(data.upiId)
      
      // Store result in sessionStorage
      sessionStorage.setItem('upiCheckResult', JSON.stringify(result))
      
      // Navigate to results page
      router.push(`/results/${result.id}`)
    } catch (error) {
      console.error("Error checking UPI ID:", error)
      toast.error("Failed to check UPI ID. Please try again.")
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
            <CardTitle className="text-2xl">UPI ID Verification</CardTitle>
            <CardDescription>
              Verify the legitimacy of UPI payment addresses before sending money.
            </CardDescription>
          </CardHeader>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)}>
              <CardContent className="space-y-6">
                <FormField
                  control={form.control}
                  name="upiId"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>UPI ID</FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="example@ybl" 
                          {...field} 
                          disabled={isLoading}
                        />
                      </FormControl>
                      <FormDescription>
                        Enter the UPI ID you want to verify
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <UpiIdExample name="Google Pay" id="username@okicici" />
                  <UpiIdExample name="PhonePe" id="username@ybl" />
                  <UpiIdExample name="Paytm" id="username@paytm" />
                  <UpiIdExample name="BHIM" id="username@upi" />
                </div>
                
                <div className="bg-muted p-4 rounded-md text-sm">
                  <p className="font-medium mb-2 flex items-center">
                    <AlertTriangle className="h-4 w-4 mr-2 text-amber-500" />
                    Important Information
                  </p>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>A valid UPI ID should be in the format username@provider</li>
                    <li>Be cautious of UPI IDs containing words like "refund", "help", or "support"</li>
                    <li>Verify the UPI ID matches the person or business you intend to pay</li>
                    <li>Official bank UPI handles include okaxis, oksbi, okicici, okhdfcbank, etc.</li>
                  </ul>
                </div>
              </CardContent>
              <CardFooter>
                <Button type="submit" disabled={isLoading} className="w-full">
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Verifying...
                    </>
                  ) : (
                    <>
                      <Check className="mr-2 h-4 w-4" />
                      Verify UPI ID
                    </>
                  )}
                </Button>
              </CardFooter>
            </form>
          </Form>
        </Card>
      </motion.div>
    </div>
  )
}

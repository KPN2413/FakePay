"use client"

import { Copy } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"

interface UpiIdExampleProps {
  name: string
  id: string
}

export function UpiIdExample({ name, id }: UpiIdExampleProps) {
  const copyToClipboard = () => {
    navigator.clipboard.writeText(id)
    toast.success(`Copied ${id} to clipboard`)
  }
  
  return (
    <div className="flex items-center justify-between p-3 border rounded-md bg-card">
      <div>
        <p className="text-sm font-medium">{name}</p>
        <p className="text-xs text-muted-foreground">{id}</p>
      </div>
      <Button variant="ghost" size="icon" onClick={copyToClipboard} title="Copy to clipboard">
        <Copy className="h-4 w-4" />
      </Button>
    </div>
  )
}
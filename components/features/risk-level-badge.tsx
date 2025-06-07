import { AlertTriangle, CheckCircle, Info } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { RiskLevel } from "@/lib/types"
import { RISK_LEVELS } from "@/lib/constants"

interface RiskLevelBadgeProps {
  riskLevel: RiskLevel
}
 
export function RiskLevelBadge({ riskLevel }: RiskLevelBadgeProps) {
  let icon = <Info className="h-4 w-4 mr-1" />
  let variant: "default" | "destructive" | "outline" | "secondary" = "secondary"
  let label = "Unknown"
  
  switch (riskLevel) {
    case "HIGH":
      icon = <AlertTriangle className="h-4 w-4 mr-1" />
      variant = "destructive"
      label = "High Risk"
      break
    case "MEDIUM":
      icon = <AlertTriangle className="h-4 w-4 mr-1" />
      variant = "outline" // Custom styling below
      label = "Medium Risk"
      break
    case "LOW":
      icon = <CheckCircle className="h-4 w-4 mr-1" />
      variant = "outline" // Custom styling below
      label = "Low Risk"
      break
    default:
      break
  }
  
  // Custom colors for medium and low risk
  const customClass = 
    riskLevel === "MEDIUM" 
      ? "border-amber-500 bg-amber-500/10 text-amber-500 hover:bg-amber-500/20"
      : riskLevel === "LOW"
      ? "border-green-500 bg-green-500/10 text-green-500 hover:bg-green-500/20"
      : ""
  
  return (
    <Badge 
      variant={variant} 
      className={`flex items-center ${customClass}`}
    >
      {icon}
      {label}
    </Badge>
  )
}

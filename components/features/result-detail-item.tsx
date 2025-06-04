interface ResultDetailItemProps {
  label: string
  value: string | number | boolean
}

export function ResultDetailItem({ label, value }: ResultDetailItemProps) {
  // Convert boolean values to Yes/No
  const displayValue = typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value
  
  return (
    <div className="flex justify-between py-2 border-b">
      <span className="font-medium">{label}</span>
      <span className="text-right">{displayValue}</span>
    </div>
  )
}
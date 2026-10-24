import * as React from "react"
import { Input as InputPrimitive } from "@base-ui/react/input"
import { cn } from "cn"

// Shared by Input and Textarea: hairline as an inset shadow; focus and invalid share the ring shape.
const fieldChrome =
  "w-full min-w-0 rounded-control bg-surface text-sm text-fg shadow-[inset_0_0_0_1px_var(--border)] outline-none transition-shadow duration-100 placeholder:text-fg-3 focus-visible:shadow-focus disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:shadow-[inset_0_0_0_1px_var(--danger)] aria-invalid:focus-visible:shadow-invalid"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <InputPrimitive
      type={type}
      data-slot="input"
      className={cn(fieldChrome, "h-8 px-2", className)}
      {...props}
    />
  )
}

export { Input, fieldChrome }
